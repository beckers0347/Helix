#!/usr/bin/env python3
"""Ogun Unsloth Direct Preference Optimization (DPO) Trainer.

Fine-tunes Qwen2.5-Coder models using 4-bit LoRA and DPO on pre-compiled datasets.
Designed for Windows 10/11 with NVIDIA RTX 5070 (12GB VRAM).

Features:
- Hardware Guard: Pre-flight VRAM and Disk checks before allocating resources.
- Merged Dataset Consumer: Reads pre-rendered ChatML DPO pairs directly from JSONL.
- Dry Run Mode: Lightweight validation and schema check without GPU or model weights.
- Windows Compatibility: Lazy imports, WinError handling, GGUF export with 16-bit HF fallback.
- Modelfile Generation: Automatically writes Ollama Modelfile next to the GGUF model.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Set, Tuple, Union

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [OGUN-DPO-TRAIN] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ogun_unsloth_trainer")

# Path resolvers
OGUN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_FILE = OGUN_ROOT / "training" / "data" / "merged_dpo_dataset.jsonl"
DEFAULT_OUTPUT_DIR = OGUN_ROOT / "training" / "data" / "exports"

# Model constants
MODEL_SIZE_MAP = {
    "14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
    "32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
}

REQUIRED_VRAM_BYTES = 11 * (1024**3)  # 11 GB minimum free VRAM for 14B 4-bit LoRA
REQUIRED_EXPORT_DISK_GB = 30.0         # ~30 GB required for 14B f16 GGUF export


# ==============================================================================
# 1. HARDWARE & DISK GUARDS
# ==============================================================================

def check_vram_preflight(
    model_size: str,
    force: bool = False,
    min_free_vram_bytes: int = REQUIRED_VRAM_BYTES,
    mem_get_info_fn: Optional[Callable[[], Tuple[int, int]]] = None,
    is_available_fn: Optional[Callable[[], bool]] = None,
) -> Tuple[bool, str]:
    """Perform pre-flight VRAM verification before attempting GPU allocations."""
    norm_size = model_size.lower().strip()

    # 1. Reject 32B model on 12GB hardware unless forced
    if norm_size == "32b" and not force:
        msg = (
            "Refusing to proceed: 32B model requested on 12GB VRAM card. "
            "4-bit quantized base weights for 32B require ~19GB VRAM alone, exceeding 12GB physical capacity. "
            "Use --model-size 14b or supply --force to bypass this safety guard."
        )
        return False, msg

    # Resolve CUDA detection functions (lazy-import torch if not provided)
    if is_available_fn is None:
        try:
            import torch

            is_available_fn = torch.cuda.is_available
        except Exception:
            is_available_fn = lambda: False

    if not is_available_fn():
        if force:
            return True, "CUDA is not available, but proceeding due to --force flag."
        return False, "Refusing to proceed: CUDA is not available. GPU is required for DPO training."

    # Inspect free VRAM
    if mem_get_info_fn is None:
        try:
            import torch

            mem_get_info_fn = torch.cuda.mem_get_info
        except Exception:
            mem_get_info_fn = lambda: (0, 0)

    try:
        free_bytes, total_bytes = mem_get_info_fn()
    except Exception as e:
        if force:
            return True, f"Failed to query VRAM ({e}), but proceeding due to --force flag."
        return False, f"Could not determine free VRAM: {e}. Use --force to override."

    free_gb = free_bytes / (1024**3)
    total_gb = total_bytes / (1024**3)
    min_gb = min_free_vram_bytes / (1024**3)

    if free_bytes < min_free_vram_bytes:
        msg = (
            f"Refusing to proceed: Insufficient free VRAM ({free_gb:.2f}GB free of {total_gb:.2f}GB total). "
            f"DPO training of {norm_size.upper()} 4-bit LoRA requires at least {min_gb:.1f}GB free VRAM. "
            "Close background GPU processes (Blender, Unreal Engine, Ollama) and retry, or use --force to override."
        )
        if not force:
            return False, msg
        return True, f"VRAM Warning bypassed via --force: {msg}"

    msg = f"VRAM Pre-Flight Passed: {free_gb:.2f}GB free of {total_gb:.2f}GB total (>= {min_gb:.1f}GB required)."
    return True, msg


def check_disk_space(
    target_path: Path,
    required_gb: float = REQUIRED_EXPORT_DISK_GB,
) -> Tuple[bool, float, str]:
    """Verify sufficient disk space exists on the target volume."""
    p = Path(target_path).resolve()
    # Find existing parent if path does not exist yet
    while not p.exists() and p != p.parent:
        p = p.parent

    try:
        usage = shutil.disk_usage(str(p))
        free_gb = usage.free / (1024**3)
    except (OSError, WindowsError) as e:
        return False, 0.0, f"Failed to check disk usage for {p}: {e}"

    if free_gb < required_gb:
        msg = (
            f"Insufficient disk space on {p}: {free_gb:.2f}GB free, "
            f"but {required_gb:.1f}GB is required for merged 16-bit / GGUF model export."
        )
        return False, free_gb, msg

    return True, free_gb, f"Disk check passed: {free_gb:.2f}GB free on {p} (>= {required_gb:.1f}GB required)."


def check_bf16_support(is_bf16_fn: Optional[Callable[[], bool]] = None) -> bool:
    """Check if hardware supports native bfloat16."""
    if is_bf16_fn is not None:
        return is_bf16_fn()
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:
        return False


# ==============================================================================
# 2. MESSAGE FORMATTING UTILITIES & AGENTFORGE PREFERENCE BUILDER
# ==============================================================================

_active_tokenizer: Any = None


def set_prompt_tokenizer(tok: Any) -> None:
    global _active_tokenizer
    _active_tokenizer = tok


def parse_tools(raw: Any) -> List[Dict[str, Any]]:
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(item)
    return out


def flat_schemas(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if "function" in t and isinstance(t["function"], dict):
            out.append(t["function"])
        elif "type" in t and t["type"] == "function" and "function" in t:
            out.append(t["function"])
        else:
            out.append(t)
    return out


def _calls_as_hf(calls_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hf_calls: List[Dict[str, Any]] = []
    for c in calls_list:
        if not isinstance(c, dict):
            continue
        if "function" in c and isinstance(c["function"], dict):
            hf_calls.append(c)
        elif "name" in c:
            hf_calls.append({
                "type": "function",
                "function": {
                    "name": c["name"],
                    "arguments": json.dumps(c.get("arguments", {}))
                    if isinstance(c.get("arguments"), dict)
                    else str(c.get("arguments", "{}")),
                },
            })
    return hf_calls


def _hf_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hf_msgs: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content") or ""
        msg: Dict[str, Any] = {"role": role, "content": content}
        if role == "assistant" and m.get("tool_calls"):
            msg["tool_calls"] = _calls_as_hf(m["tool_calls"])
        hf_msgs.append(msg)
    return hf_msgs


def _render_qwen_chatml(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> str:
    parts: List[str] = []
    system_rendered = False
    flat = flat_schemas(tools or [])

    first_msg = messages[0] if messages else None
    if first_msg and first_msg.get("role") == "system":
        sys_content = first_msg.get("content", "").strip()
        if flat:
            tools_block = (
                "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
                + "\n".join(json.dumps(t) for t in flat)
                + "\n</tools>\n\n"
                "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
            )
            sys_content += tools_block
        parts.append(f"<|im_start|>system\n{sys_content}<|im_end|>")
        system_rendered = True
    elif flat:
        sys_content = (
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
            + "\n".join(json.dumps(t) for t in flat)
            + "\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
        )
        parts.append(f"<|im_start|>system\n{sys_content}<|im_end|>")
        system_rendered = True

    remaining = messages[1:] if (messages and messages[0].get("role") == "system") else messages
    for m in remaining:
        role = m.get("role", "user")
        content = m.get("content") or ""
        if role == "tool":
            parts.append(f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>")
        elif role == "assistant":
            body = content
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function") if "function" in tc else tc
                    name = fn.get("name", "")
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass
                    tc_str = json.dumps({"name": name, "arguments": args})
                    body += f"\n<tool_call>\n{tc_str}\n</tool_call>"
            parts.append(f"<|im_start|>assistant\n{body.strip()}<|im_end|>")
        else:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def render_prompt(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> str:
    global _active_tokenizer
    if _active_tokenizer is not None and hasattr(_active_tokenizer, "apply_chat_template"):
        try:
            hf_msgs = _hf_messages(messages)
            rendered = _active_tokenizer.apply_chat_template(
                hf_msgs,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(rendered, str) and rendered.strip():
                return rendered
        except Exception:
            pass
    return _render_qwen_chatml(messages, tools=tools)


def is_tool_error(turn: Dict[str, Any]) -> bool:
    if turn.get("role") != "tool":
        return False
    content = (turn.get("content") or "").lower()
    return any(k in content for k in ["error", "exception", "failed", "traceback", "not found", "denied", "500"])


class AgentForgePreferenceBuilder:
    """Generates synthetic preference pairs from SFT tool-calling traces."""

    def __init__(self, seed: int = 3407):
        self.rng = copy.deepcopy(seed)
        self.counter = 0

    def _next_mode(self) -> int:
        self.counter += 1
        return (self.counter + self.rng) % 4

    def build_pair(
        self,
        messages: List[Dict[str, Any]],
        assistant_idx: int,
        tools: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        target_turn = messages[assistant_idx]
        tool_calls = target_turn.get("tool_calls", [])
        if not tool_calls:
            return None

        chosen_str = target_turn.get("content") or ""
        for tc in tool_calls:
            fn = tc.get("function") if "function" in tc else tc
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            tc_json = json.dumps({"name": name, "arguments": args})
            chosen_str += f"\n<tool_call>\n{tc_json}\n</tool_call>"
        chosen_str = chosen_str.strip()

        mode = self._next_mode()
        prev_is_err = (assistant_idx > 0 and is_tool_error(messages[assistant_idx - 1]))

        # Mode 4 (ignore tool error) allowed ONLY directly after a tool error
        if mode == 3 and not prev_is_err:
            mode = 0

        first_tc = tool_calls[0]
        fn = first_tc.get("function") if "function" in first_tc else first_tc
        orig_name = fn.get("name", "tool")
        orig_args = fn.get("arguments", {})
        if isinstance(orig_args, str):
            try:
                orig_args = json.loads(orig_args)
            except Exception:
                orig_args = {}

        if mode == 0:
            bad_tc = json.dumps({"name": f"invalid_{orig_name}_operation", "arguments": orig_args})
            rejected_str = f"<tool_call>\n{bad_tc}\n</tool_call>"
        elif mode == 1:
            bad_args = dict(orig_args)
            if bad_args:
                key = next(iter(bad_args.keys()))
                del bad_args[key]
            else:
                bad_args = {"invalid_param": 99999}
            bad_tc = json.dumps({"name": orig_name, "arguments": bad_args})
            rejected_str = f"<tool_call>\n{bad_tc}\n</tool_call>"
        elif mode == 2:
            rejected_str = f"I have successfully executed {orig_name} and finished your request."
        else:
            rejected_str = "Everything succeeded without any errors. Task complete."

        prompt_str = render_prompt(messages[:assistant_idx], tools=tools)

        return {
            "prompt": prompt_str,
            "chosen": chosen_str,
            "rejected": rejected_str,
            "source": "voxozi/agentforge-multiturn-toolcall",
            "pair_type": "synthetic",
        }


# ==============================================================================
# 3. DATASET FILTERING & MERGED DATASET CONSUMER
# ==============================================================================

def get_token_len(text: str, tokenizer: Optional[Any] = None) -> int:
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    return len(text.strip().split())


def filter_dpo_dataset(
    dataset: List[Dict[str, Any]],
    tokenizer: Optional[Any] = None,
    max_prompt_length: int = 512,
    max_seq_length: int = 1024,
) -> List[Dict[str, Any]]:
    """Apply token-length budget, identical-completion drop, and prompt-hash deduplication."""
    valid_pairs: List[Dict[str, Any]] = []
    seen_prompts: Set[str] = set()

    for row in dataset:
        prompt = row.get("prompt", "")
        chosen = row.get("chosen", "")
        rejected = row.get("rejected", "")

        # 1. Drop identical completions
        if chosen.strip() == rejected.strip():
            continue

        # 2. Token length budget
        prompt_len = get_token_len(prompt, tokenizer)
        chosen_len = get_token_len(chosen, tokenizer)
        rejected_len = get_token_len(rejected, tokenizer)

        if prompt_len > max_prompt_length:
            continue
        if chosen_len > max_prompt_length or rejected_len > max_prompt_length:
            continue
        if prompt_len + max(chosen_len, rejected_len) > max_seq_length:
            continue

        # 3. Deduplication via SHA-256 of prompt
        prompt_hash = hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()
        if prompt_hash in seen_prompts:
            continue
        seen_prompts.add(prompt_hash)

        valid_pairs.append(row)

    return valid_pairs


def load_dataset_file(
    filepath: Union[str, Path],
    max_train_rows: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Read merged DPO dataset from JSONL, validating required columns and split."""
    p = Path(filepath).resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset file '{p}' not found. Please run training\\run_data_prep.bat first."
        )

    train_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    required_cols = {"prompt", "chosen", "rejected"}
    warned_plain = False

    with open(p, "r", encoding="utf-8-sig") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = required_cols - set(row.keys())
            if missing:
                raise ValueError(f"Dataset line {idx} is missing required columns: {missing}")

            # Format check
            prompt_str = row["prompt"]
            if not warned_plain and not prompt_str.startswith("<|im_start|>"):
                logger.warning(
                    "Prompt on line %d appears to be in 'plain' format rather than Qwen ChatML template.",
                    idx,
                )
                warned_plain = True

            split = row.get("split", "train").lower()
            if split == "eval":
                eval_rows.append(row)
            else:
                train_rows.append(row)

    if max_train_rows is not None and max_train_rows > 0:
        train_rows = train_rows[:max_train_rows]

    logger.info(
        "Loaded dataset from %s: %d train rows, %d eval rows.",
        p,
        len(train_rows),
        len(eval_rows),
    )
    return train_rows, eval_rows


# ==============================================================================
# 4. MODEL INITIALIZATION & UNSLOTH INTEGRATION
# ==============================================================================

def load_model_and_tokenizer(
    model_size: str = "14b",
    model_name: Optional[str] = None,
    max_seq_length: int = 1024,
    lora_r: int = 32,
    lora_alpha: Optional[int] = None,
) -> Tuple[Any, Any]:
    """Lazy-load FastLanguageModel and apply 4-bit LoRA adapter."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError(
            "The 'unsloth' package is required for training. "
            "Install training dependencies via training\\requirements-train.txt."
        )

    hf_id = model_name or MODEL_SIZE_MAP.get(model_size.lower())
    if not hf_id:
        raise ValueError(f"Invalid model size '{model_size}'. Expected 14b or 32b.")

    alpha = lora_alpha or lora_r

    logger.info("Loading FastLanguageModel base weights for %s (load_in_4bit=True)...", hf_id)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=hf_id,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    logger.info("Configuring LoRA adapters (r=%d, alpha=%d)...", lora_r, alpha)
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    set_prompt_tokenizer(tokenizer)
    return model, tokenizer


def _supported_kwargs(target_callable: Any, candidates: Dict[str, Any]) -> Dict[str, Any]:
    import inspect

    try:
        sig = inspect.signature(target_callable)
    except (ValueError, TypeError):
        return candidates

    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return candidates

    accepted = set(sig.parameters.keys())
    return {k: v for k, v in candidates.items() if k in accepted}


def setup_dpo_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Optional[Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> Any:
    """Configure DPOTrainer with TRL version-compatible signatures."""
    try:
        from trl import DPOTrainer, DPOConfig
        from unsloth import PatchDPOTrainer
    except ImportError:
        raise ImportError(
            "The 'trl' and 'unsloth' packages are required for training. "
            "Install training dependencies via training\\requirements-train.txt."
        )

    PatchDPOTrainer()
    bf16_ok = check_bf16_support()

    config_candidates = {
        "output_dir": str(output_dir),
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optim": args.optim,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "logging_steps": 1,
        "fp16": not bf16_ok,
        "bf16": bf16_ok,
        "max_length": args.max_seq_length,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_prompt_length,
    }

    filtered_cfg = _supported_kwargs(DPOConfig.__init__, config_candidates)
    dpo_config = DPOConfig(**filtered_cfg)

    trainer_candidates = {
        "model": model,
        "ref_model": None,
        "args": dpo_config,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "tokenizer": tokenizer,
        "processing_class": tokenizer,
        "max_length": args.max_seq_length,
        "max_prompt_length": args.max_prompt_length,
    }

    filtered_trainer = _supported_kwargs(DPOTrainer.__init__, trainer_candidates)
    return DPOTrainer(**filtered_trainer)


# ==============================================================================
# 5. EXPORT, MODELFILE & WINDOWS CONVERSION
# ==============================================================================

def write_ollama_modelfile(output_dir: Path, gguf_path: Path) -> Path:
    """Generate Modelfile with Qwen ChatML template and stop token next to the GGUF model."""
    modelfile_path = output_dir / "Modelfile"
    content = f"""FROM {gguf_path.name}

TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ range .Messages }}}}<|im_start|>{{{{ .Role }}}}
{{{{ .Content }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
\"\"\"

PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.1
PARAMETER top_p 0.95
"""
    modelfile_path.write_text(content, encoding="utf-8")
    logger.info("Generated Ollama Modelfile at %s", modelfile_path)
    return modelfile_path


def print_manual_conversion_instructions(merged_hf_dir: Path, output_dir: Path) -> None:
    """Print exact llama.cpp CLI commands for Windows manual conversion and quantization."""
    model_stem = merged_hf_dir.name.replace("_merged_16bit", "")
    f16_gguf = output_dir / f"{model_stem}-f16.gguf"
    q4_gguf = output_dir / f"{model_stem}-Q4_K_M.gguf"

    instructions = f"""
================================================================================
>>> MANUAL GGUF CONVERSION INSTRUCTIONS (WINDOWS / LLAMA.CPP) <<<
================================================================================
The merged 16-bit Hugging Face model weights have been safely saved to:
  {merged_hf_dir}

Follow these steps to manually convert, quantize, and serve with llama.cpp:

1. Convert merged 16-bit HuggingFace weights to FP16 GGUF:
   python <LLAMA_CPP_PATH>\\convert_hf_to_gguf.py "{merged_hf_dir}" ^
     --outfile "{f16_gguf}" ^
     --outtype f16

2. Quantize FP16 GGUF to Q4_K_M (fits inside RTX 5070 12GB VRAM budget):
   llama-quantize.exe "{f16_gguf}" "{q4_gguf}" Q4_K_M

3. Serve via llama-server on port 8080:
   llama-server.exe -m "{q4_gguf}" -c 8192 -ngl 99 --jinja --host 127.0.0.1 --port 8080

4. Point Ogun at the quantized model in Ollama:
   ollama create ogun-coder -f "{output_dir}\\Modelfile"
================================================================================
"""
    print(instructions)


def export_model(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    model_name: str,
    quantization_method: str = "f16",
    required_disk_gb: float = REQUIRED_EXPORT_DISK_GB,
    force: bool = False,
) -> Path:
    """Save finetuned model as GGUF or fallback to merged 16-bit HF model."""
    output_path = Path(output_dir)
    has_space, free_gb, space_msg = check_disk_space(output_path, required_gb=required_disk_gb)
    if not has_space and not force:
        raise RuntimeError(f"Export aborted: {space_msg}")

    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Export pre-flight disk check passed (%s).", space_msg)

    # 1. Attempt GGUF Export
    try:
        logger.info("Attempting in-process GGUF export (quantization_method='%s')...", quantization_method)
        model.save_pretrained_gguf(
            str(output_path),
            tokenizer,
            quantization_method=quantization_method,
        )
        gguf_file = output_path / f"{model_name.replace('/', '_')}-{quantization_method}.gguf"
        write_ollama_modelfile(output_path, gguf_file)
        logger.info("Successfully exported GGUF to %s", output_path)
        return output_path
    except Exception as err:
        logger.warning(
            "In-process GGUF export failed (%s: %s). Falling back to merged_16bit HuggingFace export.",
            type(err).__name__,
            err,
        )

    # 2. Fallback: Save as merged_16bit HuggingFace model
    merged_dir = output_path / f"{model_name.replace('/', '_')}_merged_16bit"
    try:
        merged_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving merged 16-bit Hugging Face model to %s...", merged_dir)
        model.save_pretrained_merged(
            str(merged_dir),
            tokenizer,
            save_method="merged_16bit",
        )
        logger.info("Merged 16-bit model saved successfully to %s", merged_dir)
    except Exception as merge_err:
        logger.error("Failed to save merged 16-bit model: %s", merge_err)
        raise

    f16_gguf_name = output_path / f"{model_name.replace('/', '_')}-f16.gguf"
    write_ollama_modelfile(output_path, f16_gguf_name)
    print_manual_conversion_instructions(merged_dir, output_path)
    return merged_dir


# ==============================================================================
# 6. DRY RUN CONTROLLER
# ==============================================================================

def run_dry_run(args: argparse.Namespace) -> int:
    """Validate dataset file, display sample pairs, token stats, and preflight verdicts without GPU."""
    logger.info("=" * 70)
    logger.info("STARTING OGUN DRY RUN (Validation & Schema Check)")
    logger.info("Dataset File: %s", args.dataset_file)
    logger.info("Model: %s | Variant: %s", MODEL_SIZE_MAP.get(args.model_size), args.model_size.upper())
    logger.info("=" * 70)

    try:
        train_rows, eval_rows = load_dataset_file(args.dataset_file, max_train_rows=args.max_train_rows)
    except Exception as exc:
        logger.error("Dataset validation failed: %s", exc)
        return 1

    all_rows = train_rows + eval_rows
    if not all_rows:
        logger.error("Dataset file is empty.")
        return 1

    print("\n" + "=" * 70)
    print(">>> DRY RUN: RENDERED DPO SAMPLES (First 2 Examples) <<<")
    print("=" * 70)

    for i, sample in enumerate(all_rows[:2]):
        print(f"\n--- [EXAMPLE #{i + 1}] (split={sample.get('split', 'train')}, source={sample.get('source')}) ---")
        prompt_preview = sample["prompt"]
        if len(prompt_preview) > 300:
            prompt_preview = prompt_preview[:300] + " ... [TRUNCATED]"
        print(f"PROMPT ({len(sample['prompt'])} chars):\n{prompt_preview}\n")

        chosen_preview = sample["chosen"]
        if len(chosen_preview) > 300:
            chosen_preview = chosen_preview[:300] + " ... [TRUNCATED]"
        print(f"CHOSEN ({len(sample['chosen'])} chars):\n{chosen_preview}\n")

        rejected_preview = sample["rejected"]
        if len(rejected_preview) > 300:
            rejected_preview = rejected_preview[:300] + " ... [TRUNCATED]"
        print(f"REJECTED ({len(sample['rejected'])} chars):\n{rejected_preview}")
        print("-" * 50)

    # Token length statistics & drop warning
    dropped_prompt = 0
    dropped_seq = 0
    tool_dropped = 0
    total_checked = min(len(all_rows), 500)

    for row in all_rows[:total_checked]:
        p_len = get_token_len(row["prompt"], None)
        c_len = get_token_len(row["chosen"], None)
        r_len = get_token_len(row["rejected"], None)
        is_tool = bool(row.get("tools") or "<tool_call>" in row.get("prompt", ""))

        if p_len > args.max_prompt_length:
            dropped_prompt += 1
            if is_tool:
                tool_dropped += 1
        elif p_len + max(c_len, r_len) > args.max_seq_length:
            dropped_seq += 1
            if is_tool:
                tool_dropped += 1

    print("\n[Token Budget Pre-Flight Screen (Sampled 500 rows)]")
    print(f"  - Exceeding max-prompt-length ({args.max_prompt_length}): {dropped_prompt} rows")
    print(f"  - Exceeding max-seq-length ({args.max_seq_length}): {dropped_seq} rows")
    if tool_dropped > 0:
        logger.warning(
            "%d tool-calling pairs exceed token budget (prompt <= %d, total <= %d). "
            "Consider increasing --max-seq-length or --max-prompt-length for tool calling.",
            tool_dropped,
            args.max_prompt_length,
            args.max_seq_length,
        )

    # Hardware preflights
    vram_ok, vram_msg = check_vram_preflight(model_size=args.model_size, force=args.force)
    print(f"\n[Hardware Pre-Flight: VRAM] {'PASS' if vram_ok else 'FAIL'}: {vram_msg}")

    disk_ok, free_gb, disk_msg = check_disk_space(args.output_dir, required_gb=REQUIRED_EXPORT_DISK_GB)
    print(f"[Hardware Pre-Flight: Disk] {'PASS' if disk_ok else 'FAIL'}: {disk_msg}")

    # Planned output paths
    run_name = args.run_name or f"ogun-qwen2.5-coder-{args.model_size}-dpo"
    export_path = args.output_dir / run_name
    print(f"\n[Planned Output Paths]")
    print(f"  - LoRA Adapter Output: {export_path / 'adapter'}")
    print(f"  - GGUF Export Target:  {export_path / f'{run_name}-{args.gguf_quant}.gguf'}")
    print(f"  - Ollama Modelfile:    {export_path / 'Modelfile'}")

    print("\nDry run completed successfully. Exiting cleanly before GPU allocation.")
    return 0


# ==============================================================================
# 7. CLI ENTRY POINT & MAIN
# ==============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ogun Unsloth DPO Fine-Tuning Pipeline")
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=DEFAULT_DATASET_FILE,
        help=f"Path to pre-compiled DPO dataset JSONL. Default: {DEFAULT_DATASET_FILE}",
    )
    parser.add_argument(
        "--model-size",
        choices=["14b", "32b"],
        default="14b",
        help="Base model size variant: 14b or 32b. Default: 14b.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override Hugging Face model id (default selected by --model-size).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save adapters, checkpoints, and GGUF exports. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="ogun-qwen2.5-coder-14b-dpo",
        help="Subdirectory run name under output-dir.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Maximum total sequence length (prompt + completion). Default: 1024.",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=512,
        help="Maximum prompt length in tokens. Default: 512.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO beta regularization hyperparameter. Default: 0.1.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate for 4-bit LoRA DPO. Default: 5e-6.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device batch size. Default: 1.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
        help="Gradient accumulation steps. Default: 16.",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="paged_adamw_8bit",
        help="Optimizer name. Default: paged_adamw_8bit.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio. Default: 0.1.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm. Default: 1.0.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=1,
        help="Number of training epochs. Default: 1.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=32,
        help="LoRA rank dimension. Default: 32.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA alpha scale. Default: same as lora-r.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Max optimizer steps (-1 runs epochs). Default: -1.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Cap on number of training samples loaded from dataset file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed. Default: 3407.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Checkpoint every N steps. Default: 100.",
    )
    parser.add_argument(
        "--skip-gguf",
        action="store_true",
        help="Skip GGUF conversion and export LoRA adapters only.",
    )
    parser.add_argument(
        "--gguf-quant",
        type=str,
        default="f16",
        help="GGUF export quantization format (e.g. f16, q4_k_m). Default: f16.",
    )
    parser.add_argument(
        "--quantization-method",
        type=str,
        default="f16",
        help="Alias for --gguf-quant.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Override VRAM and disk space pre-flight guard refusals.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset file and preflight checks, then exit without GPU.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from.",
    )
    args = parser.parse_args(argv)
    if args.quantization_method != "f16" and args.gguf_quant == "f16":
        args.gguf_quant = args.quantization_method
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.dry_run:
        return run_dry_run(args)

    # 1. Hardware Guard Pre-Flight
    vram_ok, vram_msg = check_vram_preflight(model_size=args.model_size, force=args.force)
    if not vram_ok:
        logger.error("HARDWARE PRE-FLIGHT GUARD REJECTED EXECUTION:")
        logger.error(vram_msg)
        return 1
    logger.info(vram_msg)

    # 2. Disk Space Guard Check
    disk_ok, free_gb, disk_msg = check_disk_space(args.output_dir, required_gb=REQUIRED_EXPORT_DISK_GB)
    if not disk_ok and not args.force:
        logger.error("DISK SPACE PRE-FLIGHT REJECTED EXECUTION: %s", disk_msg)
        return 1
    logger.info(disk_msg)

    # 3. Load Dataset File
    train_rows, eval_rows = load_dataset_file(args.dataset_file, max_train_rows=args.max_train_rows)

    # 4. Load Base Model & Apply 4-bit LoRA
    model, tokenizer = load_model_and_tokenizer(
        model_size=args.model_size,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    # Filter dataset through tokenizer
    filtered_train = filter_dpo_dataset(
        train_rows,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_seq_length=args.max_seq_length,
    )
    filtered_eval = (
        filter_dpo_dataset(
            eval_rows,
            tokenizer=tokenizer,
            max_prompt_length=args.max_prompt_length,
            max_seq_length=args.max_seq_length,
        )
        if eval_rows
        else None
    )

    # 5. Convert to HuggingFace Dataset
    try:
        from datasets import Dataset

        train_ds = Dataset.from_list(filtered_train)
        eval_ds = Dataset.from_list(filtered_eval) if filtered_eval else None
    except ImportError:
        raise ImportError("The 'datasets' library is required to format batches for training.")

    # 6. Initialize DPOTrainer
    run_output_dir = args.output_dir / args.run_name
    trainer = setup_dpo_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        output_dir=run_output_dir,
        args=args,
    )

    # 7. Train
    logger.info("Beginning DPO fine-tuning loop...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 8. Save LoRA Adapter
    adapter_dir = run_output_dir / "adapter"
    logger.info("Saving trained LoRA adapter to %s...", adapter_dir)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # 9. GGUF Export
    if not args.skip_gguf:
        export_model(
            model=model,
            tokenizer=tokenizer,
            output_dir=run_output_dir,
            model_name=args.model_name or MODEL_SIZE_MAP.get(args.model_size, "qwen2.5-coder"),
            quantization_method=args.gguf_quant,
            required_disk_gb=REQUIRED_EXPORT_DISK_GB,
            force=args.force,
        )

    logger.info("DPO Fine-Tuning Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
