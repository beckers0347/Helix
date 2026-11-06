"""Offline unit tests for speed round optimizations in Ogun.

Covers:
1. KV cache quantization estimation (fp16, q8_0, q4_0) and compact_ctx (6144 under tight VRAM)
2. Static prompt-prefix ordering (STATIC_BUILDER_SYSTEM_PREFIX first, dynamic tail last)
3. 50 MB memory fixture chunked write + disk caching across repeated calls
4. Streaming chat (stream=True and on_token callback in OllamaClient.chat)
5. evict_other_loaded_models unloading non-active models before build
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

OGUN_ROOT = Path(__file__).resolve().parents[1]
if str(OGUN_ROOT) not in sys.path:
    sys.path.insert(0, str(OGUN_ROOT))

from ogun_core.engine import OllamaClient
from ogun_core.hardware import (
    GpuInfo,
    HardwareInfo,
    _estimate_kv_cache_gb,
    auto_configure,
)
import ogun_core.quality as oq
from ogun_core.quality import check_tool_memory_budget
from ogun_core.standalone_builder import (
    STATIC_BUILDER_SYSTEM_PREFIX,
    build_tool_generation_prompt,
    evict_other_loaded_models,
)


class TestOgunSpeedOptimizations(unittest.TestCase):
    def test_kv_cache_quantization_estimation(self):
        fp16_gb = _estimate_kv_cache_gb(48, 8, 128, 8192, "fp16")
        q8_gb = _estimate_kv_cache_gb(48, 8, 128, 8192, "q8_0")
        q4_gb = _estimate_kv_cache_gb(48, 8, 128, 8192, "q4_0")
        self.assertAlmostEqual(q8_gb, fp16_gb * 0.5, places=2)
        self.assertAlmostEqual(q4_gb, fp16_gb * 0.25, places=2)

    def test_auto_configure_compact_ctx_under_tight_vram(self):
        gpu = GpuInfo(
            name="NVIDIA GeForce RTX 5070",
            vendor="nvidia",
            vram_total_gb=12.0,
            vram_free_gb=7.5,
            unified_memory=False,
            driver="610.88",
            compute_capability="12.0",
            index=0,
        )
        hw_tight = HardwareInfo(
            os="windows",
            arch="amd64",
            cpu_name="AMD Ryzen 9 5900X",
            cpu_physical=12,
            cpu_logical=24,
            isa={"avx": True, "avx2": True, "avx512": False, "neon": False},
            ram_total_gb=64.0,
            ram_available_gb=48.0,
            gpus=[gpu],
            gpu=gpu,
            probe_seconds=0.01,
            notes=[],
        )
        with patch.dict(os.environ, {"OLLAMA_KV_CACHE_TYPE": "q8_0"}, clear=False):
            profile = auto_configure(hw=hw_tight, overrides={"compact_ctx": True})
        self.assertEqual(profile.num_ctx, 6144)
        self.assertEqual(profile.num_thread, 12)

    def test_static_prefix_ordering_across_builder_stages(self):
        req = "Build a command-line tool `colsum` for CSV files with `--json`, `--group-by`, and `--top`."
        p_tool = build_tool_generation_prompt(req, "colsum", ["--json", "--group-by", "--top"])
        self.assertTrue(p_tool.startswith(STATIC_BUILDER_SYSTEM_PREFIX))
        self.assertTrue(p_tool.rstrip().endswith("No explanations."))

    def test_memory_fixture_chunked_write_and_disk_caching(self):
        d = Path(tempfile.mkdtemp())
        try:
            tool = d / "colsum.py"
            tool.write_text("import sys\nprint('ok')\n", encoding="utf-8")
            req = "Build a command-line tool `colsum` for CSV files with `--json`."

            with patch.object(oq, "measure_process_peak_rss", return_value=14.0):
                ok1, _, _ = check_tool_memory_budget(d, tool, "colsum", req)
                fixture = d / ".build_log" / "_memory_fixture" / "large_50mb.csv"
                self.assertTrue(ok1 and fixture.exists())
                mtime1 = fixture.stat().st_mtime_ns
                # Second call reuses existing 50 MB fixture without rewriting
                ok2, _, _ = check_tool_memory_budget(d, tool, "colsum", req)
                mtime2 = fixture.stat().st_mtime_ns
                self.assertTrue(ok2)
                self.assertEqual(mtime1, mtime2)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_ollama_client_streaming_and_on_token(self):
        client = OllamaClient(model="qwen2.5-coder:14b", host="http://127.0.0.1:11434")
        fake_sdk_client = MagicMock()
        fake_sdk_client.chat.return_value = iter([
            {"message": {"role": "assistant", "content": "def "}},
            {"message": {"role": "assistant", "content": "main(): pass"}},
        ])

        tokens = []
        with patch("ollama.Client", return_value=fake_sdk_client):
            res = client.chat(
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
                on_token=lambda tok: tokens.append(tok),
            )
        self.assertEqual(res["content"], "def main(): pass")
        self.assertEqual(tokens, ["def ", "main(): pass"])

    def test_evict_other_loaded_models(self):
        ps_payload = json.dumps({
            "models": [
                {"name": "qwen2.5-coder:14b"},
                {"name": "llava:latest"},
            ]
        }).encode("utf-8")

        unloaded = []

        def fake_urlopen(req, timeout=10):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            resp = MagicMock()
            if url.endswith("/api/ps"):
                resp.read.return_value = ps_payload
            elif url.endswith("/api/generate"):
                body = json.loads(req.data.decode("utf-8"))
                unloaded.append((body.get("model"), body.get("keep_alive")))
                resp.read.return_value = b"{}"
            resp.__enter__.return_value = resp
            resp.__exit__.return_value = False
            return resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            evicted = evict_other_loaded_models("http://127.0.0.1:11434", "qwen2.5-coder:14b")
        self.assertEqual(evicted, ["llava:latest"])
        self.assertEqual(unloaded, [("llava:latest", 0)])


if __name__ == "__main__":
    unittest.main()
