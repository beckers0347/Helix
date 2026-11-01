"""Helix configuration and prompt definitions.

Model options, generation hyperparameters, and system prompt templates.
"""
from __future__ import annotations

import os
import sys

# Model and host configuration
MODEL_NAME = os.environ.get("HELIX_MODEL", "qwen2.5-coder:14b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# Ollama generation hyperparameters (optimized for Qwen2.5-Coder)
TEMPERATURE = 0.1
TOP_P = 0.95
NUM_CTX = 8192
NUM_PREDICT = 3072
STOP_TOKENS = ["<|im_end|>"]
KEEP_ALIVE = os.environ.get("HELIX_KEEP_ALIVE", "2m")   # idle models must not hold GPU memory for long

# ---------------------------------------------------------------------------------------------------------------------
# Hardware allocation profiles: how many of the model's layers run on the GPU (Ollama option `num_gpu`).
# qwen2.5-coder:14b has 48 transformer blocks + 1 output layer = 49 offload units. Measured on the RTX 5070 (12 GB):
#   num_gpu=48  -> 6% of the model stays on the CPU (the output layer), ~40% SLOWER than everything on the GPU
#   num_gpu>=49 -> 100% GPU          num_gpu=24 -> 50%/50% CPU/GPU, ~8 tok/s, ~5-6 GB less VRAM
#   num_gpu=0   -> 100% CPU, ~4 tok/s, no VRAM at all
# so MAX_PERFORMANCE sends 99 ("every layer") and shows 48 as the layer count.
# ---------------------------------------------------------------------------------------------------------------------
VRAM_PROFILES = {
    "MAX_PERFORMANCE": {"num_gpu": 99, "layers": 48, "summary": "all layers on the GPU (fastest; uses ~10 GB of VRAM)"},
    "SHARED_VRAM": {"num_gpu": 24, "layers": 24, "summary": "half the layers on the GPU, half in system RAM (slower; leaves ~5-6 GB of VRAM for Unreal / Fusion 360)"},
    "CPU_ONLY": {"num_gpu": 0, "layers": 0, "summary": "no GPU at all (slowest; uses no VRAM)"},
}
DEFAULT_VRAM_PROFILE = "SHARED_VRAM"
_VRAM_ALIASES = {
    "max": "MAX_PERFORMANCE", "maximum": "MAX_PERFORMANCE", "performance": "MAX_PERFORMANCE", "max_performance": "MAX_PERFORMANCE",
    "shared": "SHARED_VRAM", "shared_vram": "SHARED_VRAM", "share": "SHARED_VRAM",
    "cpu": "CPU_ONLY", "cpu_only": "CPU_ONLY",
}


def resolve_vram_profile(name):
    """Profile key for a name/alias ('max', 'SHARED_VRAM', ...), or None when it is not a profile."""
    key = str(name or "").strip().replace("-", "_").replace(" ", "_").lower()
    if key in _VRAM_ALIASES:
        return _VRAM_ALIASES[key]
    upper = key.upper()
    return upper if upper in VRAM_PROFILES else None


def _parse_num_gpu(raw, default=24):
    """HELIX_NUM_GPU: a custom layer count. 'auto'/'all'/'-1' -> None (omit the option). Invalid -> default, with a warning."""
    if raw is None or str(raw).strip() == "":
        return default
    text = str(raw).strip().lower()
    if text in ("auto", "all", "-1"):
        return None
    try:
        value = int(text)
        if value >= 0:
            return value
    except ValueError:
        pass
    print(f"[Helix] ignoring invalid HELIX_NUM_GPU={raw!r}; using {default}.", file=sys.stderr)
    return default


_ACTIVE = {"profile": DEFAULT_VRAM_PROFILE, "num_gpu": VRAM_PROFILES[DEFAULT_VRAM_PROFILE]["num_gpu"],
           "num_ctx": None, "num_thread": None, "hardware": None}


def _init_vram_state():
    """Initial profile: HELIX_NUM_GPU (custom count / 'auto') wins, then HELIX_VRAM_PROFILE, then SHARED_VRAM."""
    custom = os.environ.get("HELIX_NUM_GPU")
    if custom is not None and custom.strip() != "":
        value = _parse_num_gpu(custom)
        _ACTIVE.update(profile="AUTO" if value is None else f"CUSTOM({value})", num_gpu=value)
        return
    wanted = os.environ.get("HELIX_VRAM_PROFILE")
    key = resolve_vram_profile(wanted) if wanted else DEFAULT_VRAM_PROFILE
    if wanted and key is None:
        print(f"[Helix] ignoring invalid HELIX_VRAM_PROFILE={wanted!r}; using {DEFAULT_VRAM_PROFILE}.", file=sys.stderr)
        key = DEFAULT_VRAM_PROFILE
    _ACTIVE.update(profile=key, num_gpu=VRAM_PROFILES[key]["num_gpu"])


_init_vram_state()


def get_vram_profile():
    """Name of the ACTIVE profile (MAX_PERFORMANCE / SHARED_VRAM / CPU_ONLY, or CUSTOM(n) / AUTO from the environment)."""
    return _ACTIVE["profile"]


def get_num_gpu():
    """The ACTIVE num_gpu (None = do not send the option). Read at CALL TIME: never copy this into a module constant."""
    return _ACTIVE["num_gpu"]


def set_vram_profile(name):
    """Switch profile at runtime. Returns (ok, changed, profile, num_gpu). Never raises; unknown names return ok=False."""
    key = resolve_vram_profile(name)
    if key is None:
        return False, False, _ACTIVE["profile"], _ACTIVE["num_gpu"]
    changed = _ACTIVE["profile"] != key or _ACTIVE["num_gpu"] != VRAM_PROFILES[key]["num_gpu"]
    _ACTIVE.update(profile=key, num_gpu=VRAM_PROFILES[key]["num_gpu"])
    return True, changed, key, _ACTIVE["num_gpu"]


def get_num_ctx():
    """Context window in tokens: the auto-tuned value once a hardware profile is applied, else NUM_CTX. Read at CALL TIME."""
    return _ACTIVE["num_ctx"] or NUM_CTX


def get_num_thread():
    """CPU threads for inference (None = let Ollama decide). Read at CALL TIME."""
    return _ACTIVE["num_thread"]


def get_hardware_profile():
    """The applied hardware profile dict (see helix_core.hardware.profile_to_dict) or None."""
    return _ACTIVE["hardware"]


def apply_hardware_profile(profile):
    """Adopt an auto-tuned hardware profile (dict) as the ACTIVE runtime settings: profile name AUTO, num_gpu, ctx, threads.

    Call once at startup (before any request) and again only when the user asks for /vram auto: Ollama reloads the model
    whenever num_gpu/num_ctx change, so these values must stay constant while the model is loaded. Never raises.
    """
    global MODEL_NAME
    if not isinstance(profile, dict):
        return False
    _ACTIVE["hardware"] = dict(profile)
    _ACTIVE["profile"] = "AUTO"
    _ACTIVE["num_gpu"] = profile.get("num_gpu")
    ctx = profile.get("num_ctx")
    _ACTIVE["num_ctx"] = int(ctx) if ctx else None
    thread = profile.get("num_thread")
    _ACTIVE["num_thread"] = int(thread) if thread else None
    tag = profile.get("model_tag")
    if tag:
        MODEL_NAME = str(tag)
    return True


def llm_options(**extra):
    """The Ollama `options` payload for EVERY chat/generate call, including the ACTIVE num_gpu.

    Ollama reloads the model whenever load-affecting options (num_ctx, num_gpu) differ between requests, so all call
    sites must build their options here and nowhere else.
    """
    options = {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "num_ctx": get_num_ctx(),
        "num_predict": NUM_PREDICT,
        "stop": list(STOP_TOKENS),
    }
    num_gpu = get_num_gpu()
    if num_gpu is not None:
        options["num_gpu"] = num_gpu
    num_thread = get_num_thread()
    if num_thread:
        options["num_thread"] = num_thread
    options.update(extra)
    return options
REQUEST_TIMEOUT = 300.0

# System instructions
SYSTEM_PROMPT = """You are Helix, an autonomous technical coding engine and developer assistant.
You specialize in writing clean, reliable, and production-ready Python code, inspecting directories,
diagnosing bugs, and authoring dynamic plugins.
When tool calling is required, output either native tool calls or fenced JSON tool calls.
Always ensure your code is syntactically valid and completely avoids placeholders or stubs.
"""

PLUGIN_BUILDER_PROMPT = """You are an expert Python engineer authoring a dynamic plugin for Helix.
Your response MUST contain EXACTLY TWO fenced ```python code blocks in this order:
1. The tool implementation block.
2. The unit test block.

### Contract Requirements:
- The tool MUST inherit from `helix_core.base_tool.BaseTool`.
- Define `name` (string, snake_case), `version` (string, e.g. "1.0.0"), `description` (string), and `parameters_schema` (JSON Schema dictionary).
- Implement `execute(self, **kwargs: Any) -> ToolResult`.
- Return `ToolResult.success(output=...)` on success, or `ToolResult.error(error_trace=...)` on failure.
- KNOWN MODEL FAULT: Treat an empty string as a VALID value! Test with `if param is None:`, NEVER with truthiness like `if not param:`.
- No mock stubs, no `# TODO`, no `# mock`, no dummy returns. Implement real operational logic using Python standard libraries.

### Unit Test Requirements:
- Standalone runnable via `python plugins/tests/test_<name>.py` (no pytest required).
- Add the Helix root to `sys.path` using `Path(__file__).resolve().parents[2]`.
- Verify execution on typical inputs, empty strings (if applicable), and edge cases.
- Print "PASS" and exit with code 0 on success, or print error to stderr and exit with non-zero code on failure.

### Template Skeleton:

```python
from __future__ import annotations
from typing import Any, Dict
from helix_core.base_tool import BaseTool, ToolResult

class SampleTool(BaseTool):
    name: str = "sample_tool"
    version: str = "1.0.0"
    description: str = "A sample tool description."
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "input_text": {
                "type": "string",
                "description": "Input text to process."
            }
        },
        "required": ["input_text"]
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        input_text = kwargs.get("input_text")
        if input_text is None:
            return ToolResult.error("Missing required parameter: input_text")
        try:
            # Real operational logic here
            result = input_text.upper()
            return ToolResult.success(output={"result": result})
        except Exception as exc:
            return ToolResult.error(error_trace=f"Execution failed: {exc}")
```

```python
from __future__ import annotations
import sys
from pathlib import Path

# Add Helix root to sys.path
HELIX_ROOT = Path(__file__).resolve().parents[2]
if str(HELIX_ROOT) not in sys.path:
    sys.path.insert(0, str(HELIX_ROOT))

from plugins.custom_tools.sample_tool import SampleTool

def test_sample_tool() -> None:
    tool = SampleTool()
    res = tool.execute(input_text="hello")
    assert res.status == "success", f"Expected success, got {res.status}: {res.error_trace}"
    assert res.output == {"result": "HELLO"}

    # Test empty string as valid input
    res_empty = tool.execute(input_text="")
    assert res_empty.status == "success"
    assert res_empty.output == {"result": ""}
    print("PASS")

if __name__ == "__main__":
    try:
        test_sample_tool()
        sys.exit(0)
    except AssertionError as ae:
        print(f"TEST ASSERTION FAILED: {ae}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"TEST RUNTIME ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
```
"""
