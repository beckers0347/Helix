"""Speed and VRAM benchmark harness for Ogun.

Measures:
- Model load time (cold load)
- Prompt-eval tok/s
- Generation tok/s
- Time-to-first-token (TTFT)
- Peak VRAM consumption
- CPU/GPU layer offload split (via ollama ps)
- Standalone builder wall-clock time, call counts, tokens, model vs test time
- Independent verification of delivered tools (flags, total, unicode, memory)

Does not modify project code.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ogun_core.config import MODEL_NAME, OLLAMA_HOST
from ogun_core.paths import WORKSPACE_DIR
from ogun_core.engine import CoreEngine, OllamaClient, unload_model as engine_unload
from ogun_core.standalone_builder import build_standalone_project_staged


def get_vram_info() -> Tuple[float, float, float]:
    """Return (total_mb, used_mb, free_mb) from nvidia-smi or NVML."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=3.0,
        )
        line = proc.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            tot = float(parts[0])
            used = float(parts[1])
            free = float(parts[2])
            if free <= tot:
                return tot, used, free
    except Exception:
        pass

    try:
        nvml = ctypes.CDLL("nvml.dll")
        nvml.nvmlInit_v2()
        handle = ctypes.c_void_p()
        nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle))

        class Memory(ctypes.Structure):
            _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

        mem = Memory()
        nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem))
        tot = mem.total / (1024.0 * 1024.0)
        free = mem.free / (1024.0 * 1024.0)
        used = mem.used / (1024.0 * 1024.0)
        nvml.nvmlShutdown()
        return tot, used, free
    except Exception:
        return 0.0, 0.0, 0.0


def get_ollama_ps(model_name: str = "qwen2.5-coder:14b", host: Optional[str] = None) -> Dict[str, str]:
    """Query `ollama ps` and return details dict."""
    try:
        env = dict(os.environ)
        if host:
            env["OLLAMA_HOST"] = host
        proc = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=5.0, env=env)
        lines = proc.stdout.strip().splitlines()
        if len(lines) > 1:
            for line in lines[1:]:
                parts = re.split(r"\s{2,}", line.strip())
                if parts and model_name.split(":")[0] in parts[0]:
                    processor = parts[3] if len(parts) > 3 else "Unknown"
                    size = parts[2] if len(parts) > 2 else "Unknown"
                    ctx = parts[4] if len(parts) > 4 else "Unknown"
                    return {"name": parts[0], "processor": processor, "size": size, "context": ctx, "raw": line}
    except Exception as e:
        return {"error": str(e)}
    return {"processor": "None", "size": "0", "context": "0", "raw": "Model not loaded"}


def unload_model(host: str = OLLAMA_HOST, model: str = "qwen2.5-coder:14b", timeout: float = 15.0) -> bool:
    """Explicitly unload the model from Ollama and wait until VRAM is freed."""
    import urllib.request
    try:
        url = f"{host.rstrip('/')}/api/generate"
        payload = json.dumps({"model": model, "keep_alive": 0}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            resp.read()
    except Exception:
        pass

    t0 = time.time()
    while time.time() - t0 < timeout:
        ps = get_ollama_ps(model, host=host)
        if ps.get("processor") == "None" or "Stopping" in ps.get("raw", ""):
            time.sleep(0.5)
            ps2 = get_ollama_ps(model, host=host)
            if ps2.get("processor") == "None":
                return True
        time.sleep(0.5)
    return True


class VramBudgetSimulator:
    """Simulates limited free VRAM by holding memory with PyTorch."""

    def __init__(self, target_free_mb: Optional[float] = None, host: str = OLLAMA_HOST, model: str = "qwen2.5-coder:14b"):
        self.target_free_mb = target_free_mb
        self.host = host
        self.model = model
        self.proc: Optional[subprocess.Popen] = None

    def __enter__(self):
        # 1. Unload model first to obtain true idle free VRAM
        unload_model(self.host, self.model)
        time.sleep(1.0)

        if self.target_free_mb is None:
            return self

        _, _, current_free = get_vram_info()
        hold_mb = int(current_free - self.target_free_mb)
        if hold_mb > 100:
            script = (
                "import torch, sys\n"
                f"t = torch.empty(({hold_mb}, 1024, 1024), dtype=torch.uint8, device='cuda')\n"
                "sys.stdout.write('READY\\n')\n"
                "sys.stdout.flush()\n"
                "sys.stdin.readline()\n"
            )
            self.proc = subprocess.Popen(
                [sys.executable, "-c", script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            ready = self.proc.stdout.readline()
            if "READY" in ready:
                time.sleep(0.5)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.proc:
            try:
                self.proc.stdin.write("\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=3.0)
            except Exception:
                self.proc.kill()
            self.proc = None
            time.sleep(0.5)
        unload_model(self.host, self.model)


def run_llm_benchmark(
    host: str = OLLAMA_HOST,
    model: str = "qwen2.5-coder:14b",
    num_ctx: int = 8192,
    prompt: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run single benchmark evaluation of prompt evaluation and token generation."""
    import urllib.request

    if prompt is None:
        prompt = (
            "Write a self-contained Python function that parses a list of CSV file paths, "
            "computes the sum, average, min, and max for all numeric columns, and returns "
            "the results structured as a dictionary. Include error handling for missing files."
        )

    _, _, initial_free = get_vram_info()

    opts = {"num_ctx": num_ctx}
    if options:
        opts.update(options)

    req_data = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": opts,
    }

    url = f"{host.rstrip('/')}/api/generate"
    req = urllib.request.Request(
        url,
        data=json.dumps(req_data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    t_start = time.time()
    t_first_token: Optional[float] = None
    last_chunk = {}
    peak_vram_mb = 0.0

    with urllib.request.urlopen(req, timeout=300.0) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            chunk = json.loads(line)
            if t_first_token is None and chunk.get("response"):
                t_first_token = time.time()
                _, used_mb, _ = get_vram_info()
                peak_vram_mb = max(peak_vram_mb, used_mb)
            if chunk.get("done", False):
                last_chunk = chunk

    t_end = time.time()
    ps_info = get_ollama_ps(model, host=host)

    load_duration_s = last_chunk.get("load_duration", 0) / 1e9
    prompt_eval_count = last_chunk.get("prompt_eval_count", 0)
    prompt_eval_duration_s = last_chunk.get("prompt_eval_duration", 1) / 1e9
    eval_count = last_chunk.get("eval_count", 0)
    eval_duration_s = last_chunk.get("eval_duration", 1) / 1e9

    prompt_tok_s = round(prompt_eval_count / prompt_eval_duration_s, 2) if prompt_eval_duration_s > 0 else 0.0
    eval_tok_s = round(eval_count / eval_duration_s, 2) if eval_duration_s > 0 else 0.0
    ttft_s = round((t_first_token - t_start), 3) if t_first_token else round(t_end - t_start, 3)

    return {
        "model": model,
        "num_ctx": num_ctx,
        "options": opts,
        "load_time_s": round(load_duration_s, 3),
        "ttft_s": ttft_s,
        "prompt_eval_tokens": prompt_eval_count,
        "prompt_eval_tok_s": prompt_tok_s,
        "eval_tokens": eval_count,
        "eval_tok_s": eval_tok_s,
        "wall_time_s": round(t_end - t_start, 2),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "initial_free_mb": round(initial_free, 1),
        "layer_split": ps_info.get("processor", "Unknown"),
        "raw_ps": ps_info.get("raw", ""),
    }


class InstrumentedOllamaClient:
    """Wraps OllamaClient to count calls, tokens, and model latency."""

    def __init__(self, real_client: OllamaClient):
        self._real = real_client
        self.call_count = 0
        self.prompt_tokens = 0
        self.eval_tokens = 0
        self.model_seconds = 0.0
        self.calls: List[Dict[str, Any]] = []

    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.call_count += 1
        t0 = time.time()

        # Call underlying with token capture
        resp = self._real.chat(messages, tools=tools, options=options)
        elapsed = time.time() - t0
        self.model_seconds += elapsed

        p_cnt = resp.get("prompt_eval_count", 0)
        e_cnt = resp.get("eval_count", 0)
        if not p_cnt:
            p_cnt = sum(len(str(m.get("content", ""))) for m in (messages or [])) // 4
        if not e_cnt:
            e_cnt = len(str(resp.get("content", ""))) // 4
        self.prompt_tokens += p_cnt
        self.eval_tokens += e_cnt

        self.calls.append({
            "call_index": self.call_count,
            "duration_s": round(elapsed, 3),
            "prompt_tokens": p_cnt,
            "eval_tokens": e_cnt,
        })
        return resp

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def run_standalone_builder_benchmark(
    project_name: str,
    request: str,
    model: Optional[str] = None,
    host: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute one full standalone builder run in Ogun, instrumenting calls."""
    if host:
        os.environ["OLLAMA_HOST"] = host
    base_client = OllamaClient(model=model or MODEL_NAME, host=host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    instrumented = InstrumentedOllamaClient(base_client)
    engine = CoreEngine(client=instrumented)

    t0 = time.time()
    success, msg, details = build_standalone_project_staged(
        engine=engine,
        request_text=request,
    )
    wall_clock = time.time() - t0
    model_time = instrumented.model_seconds
    test_time = max(0.0, wall_clock - model_time)

    proj_dir = WORKSPACE_DIR / project_name
    tool_file = proj_dir / f"{project_name}.py"

    verification = verify_colsum_tool(tool_file, proj_dir, project_name, request) if "colsum" in project_name else {"passed": tool_file.exists()}

    return {
        "project_name": project_name,
        "status": "success" if success else "failed",
        "message": msg,
        "wall_clock_s": round(wall_clock, 2),
        "llm_calls": instrumented.call_count,
        "prompt_tokens": instrumented.prompt_tokens,
        "generated_tokens": instrumented.eval_tokens,
        "model_time_s": round(model_time, 2),
        "test_time_s": round(test_time, 2),
        "tool_exists": tool_file.exists(),
        "tool_bytes": tool_file.stat().st_size if tool_file.exists() else 0,
        "verification": verification,
        "call_log": instrumented.calls,
    }


def verify_colsum_tool(
    tool_path: Path,
    proj_dir: Optional[Path] = None,
    project_name: str = "colsum",
    request: str = "",
) -> Dict[str, Any]:
    """Independently verify colsum / colsummary behaviors, including 50 MB memory budget."""
    if not tool_path.exists():
        return {"passed": False, "error": f"Tool not found: {tool_path}"}

    import tempfile
    td = Path(tempfile.mkdtemp(prefix="colsum_test_"))
    try:
        csv_file = td / "test.csv"
        csv_file.write_text("col_a,col_b,name\n10,100,item1\n20,200,item2\n30,300,item3\n", encoding="utf-8")
        tsv_file = td / "test.tsv"
        tsv_file.write_text("col_a\tcol_b\tname\n10\t100\titem1\n20\t200\titem2\n30\t300\titem3\n", encoding="utf-8")

        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

        # 1. Basic table mode
        p1 = subprocess.run([sys.executable, str(tool_path), str(csv_file)], capture_output=True, text=True, env=env)
        has_table = ("col_a" in p1.stdout and "Count" in p1.stdout and "-" in p1.stdout)

        # 2. --json flag
        p2 = subprocess.run([sys.executable, str(tool_path), str(csv_file), "--json"], capture_output=True, text=True, env=env)
        json_ok = False
        try:
            j = json.loads(p2.stdout.strip())
            json_ok = isinstance(j, (dict, list))
        except Exception:
            pass

        # 3. --column flag
        p3 = subprocess.run([sys.executable, str(tool_path), str(csv_file), "--column", "col_a"], capture_output=True, text=True, env=env)
        col_ok = ("col_a" in p3.stdout and "col_b" not in p3.stdout)

        # 4. --delimiter flag
        p4 = subprocess.run([sys.executable, str(tool_path), str(tsv_file), "--delimiter", "\t"], capture_output=True, text=True, env=env)
        delim_ok = (p4.returncode == 0 and "col_a" in p4.stdout)

        # 5. Unicode / cp1252 safety
        u_file = td / "unicode.csv"
        u_file.write_text("col_a,name\n15,Café crème naïve 🚀\n25,Multi-byte üñíçødé\n", encoding="utf-8")
        p5 = subprocess.run([sys.executable, str(tool_path), str(u_file)], capture_output=True, text=True, env=env)
        unicode_ok = p5.returncode == 0

        # 6. 50 MB memory budget check
        from ogun_core.quality import check_tool_memory_budget
        mem_ok, peak_rss_mb, _ = check_tool_memory_budget(
            proj_dir or td, tool_path, project_name,
            request or "Build a command-line tool colsum for CSV files with --json, --delimiter, --column.",
        )

        all_ok = has_table and json_ok and col_ok and delim_ok and unicode_ok and mem_ok
        return {
            "passed": all_ok,
            "has_table": has_table,
            "json_flag": json_ok,
            "column_flag": col_ok,
            "delimiter_flag": delim_ok,
            "unicode_safety": unicode_ok,
            "memory_50mb_ok": mem_ok,
            "peak_rss_mb": round(peak_rss_mb, 1),
        }
    finally:
        import shutil
        shutil.rmtree(td, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Ogun Speed & VRAM Benchmark Harness")
    parser.add_argument("--mode", choices=["llm", "builder", "all"], default="all")
    parser.add_argument("--budget", choices=["full", "8gb", "5gb", "all"], default="all")
    parser.add_argument("--model", default="qwen2.5-coder:14b")
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--project-name", default="colsum_b0")
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--auto-tune", action="store_true", default=False)
    args = parser.parse_args()

    budgets = []
    if args.budget in ("full", "all"):
        budgets.append(("Full GPU Free (~11 GB)", None))
    if args.budget in ("8gb", "all"):
        budgets.append(("~8 GB Free VRAM", 8192.0))
    if args.budget in ("5gb", "all"):
        budgets.append(("~5 GB Free VRAM", 5120.0))

    results = {"llm": [], "builder": []}

    print("=" * 75)
    print("                 OGUN SPEED ROUND BENCHMARK HARNESS")
    print(f"Model: {args.model} | Default Context: {args.ctx} | Host: {args.host}")
    print("=" * 75)

    if args.mode in ("llm", "all"):
        print("\n--- LLM RUNTIME BENCHMARK ---")
        for label, budget_mb in budgets:
            print(f"\n[+] Testing at VRAM budget: {label}...")
            with VramBudgetSimulator(budget_mb, host=args.host, model=args.model):
                opts = None
                eff_ctx = args.ctx
                eff_model = args.model
                if args.auto_tune:
                    from ogun_core.hardware import auto_configure, probe_hardware
                    hw = probe_hardware()
                    _, used_mb, free_mb = get_vram_info()
                    if hw.gpu and free_mb > 0:
                        from dataclasses import replace
                        g_adj = replace(hw.gpu, vram_free_gb=round(free_mb / 1024.0, 2))
                        hw = replace(hw, gpu=g_adj, gpus=[g_adj])
                    ov = {"model": args.model, "compact_ctx": True, "kv_cache_type": "q8_0", "reserve_gb": 0.5}
                    prof = auto_configure(hw, overrides=ov)
                    eff_ctx = prof.num_ctx
                    opts = {"num_thread": prof.num_thread, "num_batch": 128 if (prof.num_gpu is not None and prof.num_gpu < 49) else 512}
                    if prof.num_gpu is not None:
                        opts["num_gpu"] = prof.num_gpu
                bench = run_llm_benchmark(model=eff_model, num_ctx=eff_ctx, options=opts, host=args.host)
                bench["budget_label"] = label
                results["llm"].append(bench)
                print(f"    Load Time:       {bench['load_time_s']}s")
                print(f"    TTFT:            {bench['ttft_s']}s")
                print(f"    Prompt Eval:     {bench['prompt_eval_tok_s']} tok/s ({bench['prompt_eval_tokens']} tokens)")
                print(f"    Eval (Gen):      {bench['eval_tok_s']} tok/s ({bench['eval_tokens']} tokens)")
                print(f"    Peak VRAM:       {bench['peak_vram_mb']} MB")
                print(f"    Offload Split:   {bench['layer_split']}")

    if args.mode in ("builder", "all"):
        print("\n--- STANDALONE BUILDER BENCHMARK ---")
        req = (
            f"Build me a command-line tool called `{args.project_name}`. Give it a CSV file with a header row "
            "and it prints, for each numeric column, the count, minimum, maximum and mean, ignoring empty cells "
            "and non-numeric columns, as an aligned plain-text table with a header separator line. It needs a "
            "--json option, a --delimiter option (default comma) and a --column NAME option to show only that column. "
            "Write tests for it, run them, and make sure they pass."
        )
        print(f"[+] Starting build for project '{args.project_name}'...")
        b_res = run_standalone_builder_benchmark(
            project_name=args.project_name,
            request=req,
            model=args.model,
            host=args.host,
        )
        results["builder"].append(b_res)
        print(f"    Status:          {b_res['status']}")
        print(f"    Total Wall Clock:{b_res['wall_clock_s']}s")
        print(f"    LLM Calls:       {b_res['llm_calls']}")
        print(f"    Prompt Tokens:   {b_res['prompt_tokens']}")
        print(f"    Generated Tokens:{b_res['generated_tokens']}")
        print(f"    Model Wait Time: {b_res['model_time_s']}s")
        print(f"    Test/Smoke Time: {b_res['test_time_s']}s")
        print(f"    Independent Test:{b_res['verification']}")

    print("\n" + "=" * 75)
    print("BENCHMARK COMPLETED. RAW RESULTS:")
    print(json.dumps(results, indent=2))
    print("=" * 75)


if __name__ == "__main__":
    main()
