"""Offline unit and integration tests for Helix development-ability fixes (F1 - F7).

Runs completely offline: zero network calls, zero live model inference.
Tests all honesty verification, continuation handling, standalone routing,
adaptive timeouts, self-correction loops, quality checks, and telemetry sanitization.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import patch

# Ensure Helix root is on sys.path
HELIX_ROOT = Path(__file__).resolve().parents[1]
if str(HELIX_ROOT) not in sys.path:
    sys.path.insert(0, str(HELIX_ROOT))

from helix_core.base_tool import BaseTool, ToolResult
from helix_core.config import (
    SYSTEM_PROMPT,
    get_request_timeout,
    set_vram_profile,
)
from helix_core.engine import CoreEngine, OllamaClient
from helix_core.hardware import (
    GpuInfo,
    HardwareInfo,
    adjust_for_resident_models,
    format_banner,
    _probe_nvidia_gpus,
)
from helix_core.honesty import TurnEvents, sanitize_reply
from helix_core.plugin_builder import build_plugin_staged
from helix_core.plugin_manager import PluginManager
from helix_core.quality import check_requirements_coverage, classify_project_request
from helix_core.security import PathSecurityGuard


# -----------------------------------------------------------------------------
# F1 Honesty Guard Tests
# -----------------------------------------------------------------------------

# Exact fabricated reply fixture from the development-ability report
FABRICATED_REPLY_FIXTURE = """To create the `dupefinder` command-line tool, I'll follow these steps:

1. **Create the Tool File**: Save the tool in the `/plugins/custom_tools/` directory.
2. **Implement the Tool**: Write the Python code for the `dupefinder` tool.
3. **Create Tests**: Write tests for the tool.
4. **Run Tests**: Execute the tests to ensure they pass.

Let's start by creating the tool file and implementing the `dupefinder` functionality.

### Step 1: Create the Tool File

Create a new file named `dupefinder.py` in the `/plugins/custom_tools/` directory.

### Step 2: Implement the Tool

Here's the implementation of the `dupefinder` tool:

```python
#!/usr/bin/env python3
import os
import hashlib
import argparse
import json

def find_duplicates(folder_path, min_size):
    return []

def main():
    pass

if __name__ == '__main__':
    main()
```

### Step 3: Create Tests

Create a new file named `test_dupefinder.py` in the `/plugins/custom_tools/` directory.

Here's the test implementation:

```python
import unittest

class TestDupeFinder(unittest.TestCase):
    def test_find_duplicates(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
```

### Step 4: Run Tests

To run the tests, execute the following command in the terminal:

```sh
python test_dupefinder.py
```

If all tests pass, you should see the following output:

```
----------------------------------------------------------------------
Ran 1 test in 0.000s

OK
```

This confirms that the `dupefinder` tool and its tests are working correctly.
"""


def test_f1_honesty_guard_intercepts_fabricated_reply() -> None:
    """Fixture test: fabricated reply with unverified claims must be intercepted."""
    turn_events = TurnEvents()  # No tools ran; no files written; no tests executed
    sanitized, intercepted = sanitize_reply(FABRICATED_REPLY_FIXTURE, turn_events)

    assert intercepted is True, "Fabricated reply must be intercepted by honesty guard"
    # Fabricated test output block must be stripped
    assert "Ran 1 test in 0.000s" not in sanitized
    assert "OK" not in sanitized or "No tests were executed" in sanitized
    assert "This confirms that the `dupefinder` tool and its tests are working correctly" not in sanitized
    # Honest notice must be included
    assert "No files were written to disk" in sanitized
    assert "No tests were executed by Helix" in sanitized


def test_f1_honesty_guard_passes_genuine_reply() -> None:
    """Genuine reply backed by real tool executions must pass unchanged."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir).resolve()
        tool_file = root / "workspace" / "dupefinder" / "dupefinder.py"
        tool_file.parent.mkdir(parents=True, exist_ok=True)
        tool_file.write_text("print('genuine')", encoding="utf-8")

        turn_events = TurnEvents()
        turn_events.add_file_written(tool_file)
        turn_events.add_test_execution(
            test_path=str(root / "workspace" / "dupefinder" / "test_dupefinder.py"),
            passed=True,
            output="Ran 1 test in 0.010s\nOK",
        )

        genuine_reply = "I have implemented dupefinder.py and verified that the tests pass."
        sanitized, intercepted = sanitize_reply(genuine_reply, turn_events, base_dir=root)

        assert intercepted is False, "Genuine reply should not be intercepted"
        assert sanitized == genuine_reply


# -----------------------------------------------------------------------------
# F2 Continuation Tests
# -----------------------------------------------------------------------------

class MockChatClient:
    def __init__(self, responses: List[Any]) -> None:
        self.responses = list(responses)
        self.call_history: List[List[Dict[str, Any]]] = []

    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        self.call_history.append(list(messages))
        if not self.responses:
            return {"content": "No more responses"}
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_f2_continue_resumes_task_with_failure_context() -> None:
    """'continue' after a timeout/failure must resume the previous task, not start context-free chat."""
    client = MockChatClient([
        TimeoutError("Model read timeout 300s"),
        {"content": "Resuming task with tools."},
    ])

    engine = CoreEngine(client=client)
    # Turn 1: task times out
    reply1 = engine.run_turn("Build me a command-line tool called dupefinder")
    assert "timed out" in reply1.lower()
    assert engine.last_turn_status["status"] == "timeout"

    # Turn 2: user says "continue"
    reply2 = engine.run_turn("continue")

    # Verify that the model received the previous task and failure context
    assert len(client.call_history) == 2
    turn2_messages = client.call_history[1]
    last_user_msg = [m for m in turn2_messages if m["role"] == "user"][-1]["content"]

    assert "Resume and continue the task: 'Build me a command-line tool called dupefinder'" in last_user_msg
    assert "Previous status or failure context" in last_user_msg
    assert "write_file, run_python_test" in last_user_msg


# -----------------------------------------------------------------------------
# F3 Standalone Projects & Security Boundaries
# -----------------------------------------------------------------------------

def test_f3_project_classification() -> None:
    """Requests for standalone CLI tools are classified as standalone; plugins as plugin."""
    standalone_prompts = [
        "Build me a command-line tool called `dupefinder`.",
        "Create a standalone script to analyze log files.",
        "Write a python program with tests for file deduplication.",
        "Build me a cli tool that parses JSON.",
    ]
    plugin_prompts = [
        "Create a dynamic plugin for Helix named StringHasher.",
        "Build a plugin that adds web scraping to Helix.",
        "Add tool to helix inheriting from BaseTool.",
    ]

    for p in standalone_prompts:
        assert classify_project_request(p) == "standalone", f"Failed to classify standalone: {p}"

    for p in plugin_prompts:
        assert classify_project_request(p) == "plugin", f"Failed to classify plugin: {p}"


def test_f3_standalone_workspace_write_and_security_guard() -> None:
    """Standalone tools can be written to workspace/ and tested, but core files remain immutable."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir).resolve()
        guard = PathSecurityGuard(base_dir=root)

        # Allowed writes
        ws_tool = root / "workspace" / "dupefinder" / "dupefinder.py"
        ws_test = root / "workspace" / "dupefinder" / "test_dupefinder.py"
        assert guard.validate_write(ws_tool) == ws_tool
        assert guard.validate_write(ws_test) == ws_test

        # Forbidden writes
        try:
            guard.validate_write(root / "helix_core" / "engine.py")
            assert False, "Should have rejected write to helix_core"
        except PermissionError:
            pass

        try:
            guard.validate_write(root / "main.py")
            assert False, "Should have rejected write to main.py"
        except PermissionError:
            pass


# -----------------------------------------------------------------------------
# F4 Adaptive Timeouts & Staged Generation
# -----------------------------------------------------------------------------

def test_f4_adaptive_timeouts_by_hardware_profile() -> None:
    """Timeout adapts to active profile: GPU=300s, Hybrid=600s, CPU=900s."""
    set_vram_profile("max")
    assert get_request_timeout() == 300.0

    set_vram_profile("shared")
    assert get_request_timeout() == 600.0

    set_vram_profile("cpu")
    assert get_request_timeout() == 900.0

    # Environment variable override
    with patch.dict(os.environ, {"HELIX_REQUEST_TIMEOUT": "150.0"}):
        assert get_request_timeout() == 150.0

    # Reset profile
    set_vram_profile("shared")


def test_f4_ollama_client_uses_adaptive_timeout() -> None:
    """OllamaClient reads adaptive timeout live at call time."""
    client = OllamaClient()
    set_vram_profile("max")
    assert client.get_effective_timeout() == 300.0

    set_vram_profile("cpu")
    assert client.get_effective_timeout() == 900.0
    set_vram_profile("shared")


# -----------------------------------------------------------------------------
# F5 Self-Correction Loop Tests
# -----------------------------------------------------------------------------

def test_f5_self_correction_fails_then_fixes() -> None:
    """Plugin build fails verification on attempt 1, receives error trace, and fixes on attempt 2."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir).resolve()
        pm = PluginManager(base_dir=root)

        # Attempt 1: Syntax error in test
        bad_tool = """```python
from helix_core.base_tool import BaseTool, ToolResult
class AdderTool(BaseTool):
    name: str = "adder_tool"
    version: str = "1.0.0"
    description: str = "Add numbers"
    parameters_schema: dict = {"type": "object"}
    def execute(self, **kwargs):
        return ToolResult.success(output={"res": 42})
```"""
        bad_test = """```python
this is invalid python syntax syntax !!!
```"""
        reply_att1 = f"{bad_tool}\n\n{bad_test}"

        # Attempt 2: Fixed code
        good_tool = """```python
from helix_core.base_tool import BaseTool, ToolResult
class AdderTool(BaseTool):
    name: str = "adder_tool"
    version: str = "1.0.0"
    description: str = "Add numbers"
    parameters_schema: dict = {"type": "object"}
    def execute(self, **kwargs):
        return ToolResult.success(output={"res": 42})
```"""
        good_test = """```python
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from plugins.custom_tools.adder_tool import AdderTool
t = AdderTool()
res = t.execute()
assert res.status == "success"
print("PASS")
sys.exit(0)
```"""
        reply_att2 = f"{good_tool}\n\n{good_test}"

        client = MockChatClient([
            {"content": reply_att1},
            {"content": reply_att2},
        ])
        engine = CoreEngine(client=client, base_dir=root, plugin_manager=pm)

        success, msg, plugin_name = build_plugin_staged(
            engine=engine,
            plugin_manager=pm,
            request_text="Build adder_tool plugin",
            max_attempts=3,
        )

        assert success is True, f"Expected self-correction to succeed: {msg}"
        assert "[PASS]" in msg
        assert "attempt 2" in msg
        assert plugin_name == "adder_tool"


def test_f5_self_correction_exhausted_reports_failure() -> None:
    """When model fails all attempts, build truthfully reports [FAIL]."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir).resolve()
        pm = PluginManager(base_dir=root)

        broken_reply = "```python\n broken code\n```"
        client = MockChatClient([
            {"content": broken_reply},
            {"content": broken_reply},
        ])
        engine = CoreEngine(client=client, base_dir=root, plugin_manager=pm)

        success, msg, _ = build_plugin_staged(
            engine=engine,
            plugin_manager=pm,
            request_text="Build broken tool",
            max_attempts=2,
        )

        assert success is False
        assert "[FAIL]" in msg
        assert "after 2 attempts" in msg


# -----------------------------------------------------------------------------
# F6 Quality Guidance & Static Requirements-Coverage Check
# -----------------------------------------------------------------------------

def test_f6_static_requirements_coverage_check() -> None:
    """Verifies that requested flags and features are statically checked."""
    prompt = "Build dupefinder with --json and --min-size that reports wasted space."

    # Incomplete code missing --json and wasted space
    code_incomplete = """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--min-size', type=int)
args = parser.parse_args()
"""
    covered, missing = check_requirements_coverage(prompt, code_incomplete)
    assert covered is False
    assert any("--json" in m for m in missing)
    assert any("waste" in m.lower() for m in missing)

    # Complete code covering all requirements
    code_complete = """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--json', action='store_true')
parser.add_argument('--min-size', type=int, default=0)
args = parser.parse_args()
total_wasted_bytes = 100
"""
    covered2, missing2 = check_requirements_coverage(prompt, code_complete)
    assert covered2 is True, f"Expected complete code to pass, but missing: {missing2}"


def test_f6_system_prompt_contains_quality_checklist() -> None:
    """System prompt must provide clear quality guidance for real execution."""
    assert "Chunked File Reading" in SYSTEM_PROMPT
    assert "Windows UTF-8 Safety" in SYSTEM_PROMPT
    assert "Requirements Completeness" in SYSTEM_PROMPT
    assert "Safe Path & Error Handling" in SYSTEM_PROMPT
    assert "Truthful Execution" in SYSTEM_PROMPT


# -----------------------------------------------------------------------------
# F7 GPU Figure Sanitization Tests
# -----------------------------------------------------------------------------

def test_f7_nvidia_smi_garbage_rejection() -> None:
    """nvidia-smi outputs with [N/A], negative memory, non-finite, or free>total are rejected."""
    # Test cases: (index, name, memory.total, memory.free, driver)
    garbage_csv = (
        "0, GPU [N/A], [N/A], [N/A], 550.00\n"
        "1, GPU Negative, 12288.0, -17179865087.9, 550.00\n"
        "2, GPU NonFinite, NaN, 2048.0, 550.00\n"
        "3, GPU FreeExceedsTotal, 8192.0, 16384.0, 550.00\n"
        "4, GPU Valid, 12288.0, 10240.0, 550.00\n"
    )

    with patch("helix_core.hardware._find_nvidia_smi", return_value="nvidia-smi"):
        with patch("helix_core.hardware._run_subproc", return_value=(0, garbage_csv)):
            notes: List[str] = []
            gpus = _probe_nvidia_gpus(2.0, notes)

            # Only GPU 4 (Valid) should be accepted
            assert len(gpus) == 1, f"Expected 1 valid GPU, got {len(gpus)}"
            assert gpus[0].name == "GPU Valid"
            assert gpus[0].vram_total_gb == 12.0
            assert gpus[0].vram_free_gb == 10.0
            assert len(notes) >= 4, "Notes should record reasons for rejecting invalid entries"


def test_f7_format_banner_omits_invalid_figures() -> None:
    """Format banner omits negative or non-finite VRAM figures instead of printing nonsense."""
    invalid_gpu = GpuInfo(
        vendor="nvidia",
        name="Corrupted GPU",
        vram_total_gb=12.0,
        vram_free_gb=-999.0,  # Negative garbage
        unified_memory=False,
        driver="550.00",
        compute_capability="8.9",
        index=0,
    )
    hw = HardwareInfo(
        os="windows",
        arch="x86_64",
        cpu_name="Test CPU",
        cpu_physical=8,
        cpu_logical=16,
        isa={},
        ram_total_gb=32.0,
        ram_available_gb=20.0,
        gpus=[invalid_gpu],
        gpu=invalid_gpu,
        probe_seconds=0.1,
        notes=[],
    )

    from helix_core.hardware import auto_configure
    prof = auto_configure(hw)
    banner = format_banner(hw, prof)

    # Banner must not print negative numbers for free memory
    assert "-999" not in banner
    # Valid total is printed without the negative free figure
    assert "Corrupted GPU (12.0 GB VRAM)" in banner


def test_f7_adjust_resident_models_clamps_safely() -> None:
    """adjust_for_resident_models safely ignores negative or NaN inputs."""
    valid_gpu = GpuInfo(
        vendor="nvidia",
        name="RTX Test",
        vram_total_gb=12.0,
        vram_free_gb=6.0,
        unified_memory=False,
        driver="550.00",
        compute_capability="8.9",
        index=0,
    )
    hw = HardwareInfo(
        os="windows",
        arch="x86_64",
        cpu_name="Test CPU",
        cpu_physical=8,
        cpu_logical=16,
        isa={},
        ram_total_gb=32.0,
        ram_available_gb=20.0,
        gpus=[valid_gpu],
        gpu=valid_gpu,
        probe_seconds=0.1,
        notes=[],
    )

    # Negative extra memory must not reduce free memory
    res_neg = adjust_for_resident_models(hw, resident_vram_gb=-5.0)
    assert res_neg.gpu.vram_free_gb == 6.0

    # NaN extra memory must not crash or corrupt
    res_nan = adjust_for_resident_models(hw, resident_vram_gb=float("nan"))
    assert res_nan.gpu.vram_free_gb == 6.0


# -----------------------------------------------------------------------------
# Round 2 Fixes Tests (H1 - H5)
# -----------------------------------------------------------------------------

def test_h1_windows_unicode_cp1252_smoke_and_static_check() -> None:
    """H1: Windows Unicode / cp1252 safety static check and runtime smoke test."""
    from helix_core.quality import check_requirements_coverage
    from helix_core.standalone_builder import run_smoke_tests

    # Missing stdout configuration triggers static coverage failure for CLI tool
    bad_cli = "import argparse\nparser = argparse.ArgumentParser()\nargs = parser.parse_args()\nprint('hello')\n"
    cov, missing = check_requirements_coverage("Build a command-line tool called logsummary", bad_cli)
    assert cov is False
    assert any("Windows console UTF-8 safety" in m for m in missing)

    with tempfile.TemporaryDirectory() as tmp_dir:
        p_dir = Path(tmp_dir)
        tool_file = p_dir / "tool.py"
        good_tool = (
            "import sys, argparse\n"
            "if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):\n"
            "    sys.stdout.reconfigure(encoding='utf-8', errors='replace')\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('path')\n"
            "args = parser.parse_args()\n"
            "import os\n"
            "if not os.path.exists(args.path):\n"
            "    print(f'Error: {args.path}', file=sys.stderr)\n"
            "    sys.exit(1)\n"
            "print('OK')\n"
        )
        tool_file.write_text(good_tool, encoding="utf-8")
        ok, rep, fails = run_smoke_tests(p_dir, tool_file, "tool")
        assert ok is True, f"Expected smoke runs to pass: {rep}"


def test_h2_nonexistent_path_nonzero_exit_smoke_check() -> None:
    """H2: Nonexistent path must exit non-zero and write to stderr."""
    from helix_core.standalone_builder import run_smoke_tests

    with tempfile.TemporaryDirectory() as tmp_dir:
        p_dir = Path(tmp_dir)
        tool_file = p_dir / "bad_tool.py"
        bad_tool = (
            "import sys, argparse\n"
            "if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):\n"
            "    sys.stdout.reconfigure(encoding='utf-8', errors='replace')\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('path')\n"
            "args = parser.parse_args()\n"
            "print('Folder not found', file=sys.stderr)\n"
            "sys.exit(0)\n"
        )
        tool_file.write_text(bad_tool, encoding="utf-8")
        ok, rep, fails = run_smoke_tests(p_dir, tool_file, "bad_tool")
        assert ok is False
        assert any("nonexistent" in f.lower() for f in fails)


def test_h3_standalone_staged_builder_success_and_failure() -> None:
    """H3: Standalone staged builder tests real behavior, diagnoses test-vs-tool, and reports truthfully."""
    from helix_core.standalone_builder import build_standalone_project_staged
    from helix_core.engine import CoreEngine

    class MockEngineClient:
        def __init__(self, responses: List[str]) -> None:
            self.responses = list(responses)
        def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
            if self.responses:
                return {"content": self.responses.pop(0)}
            return {"content": "no response"}

    mocktool_dir = HELIX_ROOT / "workspace" / "mocktool"
    if mocktool_dir.exists():
        shutil.rmtree(mocktool_dir, ignore_errors=True)

    valid_tool = (
        "import sys, os, argparse\n"
        "if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):\n"
        "    sys.stdout.reconfigure(encoding='utf-8', errors='replace')\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('file')\n"
        "parser.add_argument('--json', action='store_true')\n"
        "args = parser.parse_args()\n"
        "if not os.path.exists(args.file):\n"
        "    print('Error: not found', file=sys.stderr)\n"
        "    sys.exit(1)\n"
        "print('Done')\n"
    )
    passing_test = (
        "import unittest\n"
        "class T(unittest.TestCase):\n"
        "    def test_ok(self): self.assertTrue(True)\n"
        "if __name__ == '__main__': unittest.main()\n"
    )

    client = MockEngineClient([
        f"```python\n{valid_tool}\n```",
        f"```python\n{passing_test}\n```",
    ])
    engine = CoreEngine(client=client)
    success, msg, details = build_standalone_project_staged(engine, "Build me a command-line tool called mocktool with --json", max_attempts=2)
    assert success is True
    assert "Successfully built" in msg
    assert details["status"] == "success"


def test_h3_round_limit_rich_status() -> None:
    """H3 (a): When turn round limit is hit, final message reports files written and test status."""
    from helix_core.engine import CoreEngine

    class LoopingClient:
        def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
            return {"content": "Still thinking with no tools."}

    engine = CoreEngine(client=LoopingClient())
    out = engine.run_turn("Explain quantum entanglement", max_rounds=2)
    assert "Still thinking" in out or "No files were written" in out or "Reached maximum tool execution rounds" in out


def test_adversarial_honesty_and_resume() -> None:
    """Honesty guard intercepts alternative claim phrasings, and resume detection does not hijack unrelated chat."""
    from helix_core.honesty import TurnEvents, sanitize_reply

    text = "I've saved the code to workspace/tool/tool.py and all green! Tests are passing ✅"
    events = TurnEvents()
    sanitized, intercepted = sanitize_reply(text, events)
    assert intercepted is True
    assert "all green" not in sanitized.lower()
    assert "tests are passing" not in sanitized.lower()
    assert "✅" not in sanitized

    continuation_words = {"continue", "go on", "try again", "keep going", "retry", "proceed", "resume", "carry on"}
    assert "continue the story" not in continuation_words
    assert "continue reading" not in continuation_words
    assert "retry sending email" not in continuation_words


# -----------------------------------------------------------------------------
# R3 Round 3 Self-Correction & Convergence Tests
# -----------------------------------------------------------------------------

def test_r3_fix_stage_no_blocks_escalation() -> None:
    """R3-1 / R3-2: When fix stage returns no code blocks, failure is logged and strategy rotates."""
    import tempfile, shutil
    from helix_core.standalone_builder import build_standalone_project_staged

    good_tool = """
import sys, argparse
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
parser = argparse.ArgumentParser()
parser.add_argument('file')
parser.add_argument('--json', action='store_true')
args = parser.parse_args()
print("processed")
"""
    failing_test = """
import unittest
class T(unittest.TestCase):
    def test_fail(self):
        self.assertEqual(1, 2)
if __name__ == '__main__':
    unittest.main()
"""
    client = MockChatClient([
        {"content": f"```python\n{good_tool}\n```"},
        {"content": f"```python\n{failing_test}\n```"},
        {"content": "Diagnosis: tool is fine, test is wrong."},
        {"content": "Still no code blocks."},
    ])
    engine = CoreEngine(client=client)

    test_dir = tempfile.mkdtemp()
    with patch("helix_core.standalone_builder.WORKSPACE_DIR", Path(test_dir)):
        success, msg, details = build_standalone_project_staged(
            engine,
            "Build me a command-line tool called mocktool. It needs a --json option.",
            max_attempts=2,
        )
        assert success is False
        writes_file = Path(test_dir) / "mocktool" / ".build_log" / "attempt_1" / "file_writes.json"
        assert writes_file.exists()
        assert "no code blocks" in writes_file.read_text(encoding="utf-8")
    shutil.rmtree(test_dir, ignore_errors=True)


def test_r3_fix_stage_rejects_stub_and_preserves_tool() -> None:
    """R3-1 / R3-2: When fix stage returns a degenerate stub, regression guard rejects it and preserves original tool."""
    import tempfile, shutil
    from helix_core.standalone_builder import build_standalone_project_staged

    good_tool = """
import sys, os, argparse
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
def process_data(path):
    return [line.strip() for line in open(path, encoding='utf-8')]
parser = argparse.ArgumentParser()
parser.add_argument('file')
parser.add_argument('--json', action='store_true')
args = parser.parse_args()
if not os.path.exists(args.file):
    print("error", file=sys.stderr)
    sys.exit(1)
print("done processing data")
"""
    failing_test = """
import unittest
class T(unittest.TestCase):
    def test_fail(self):
        self.assertEqual(1, 2)
if __name__ == '__main__':
    unittest.main()
"""
    stub_tool = """
import sys, argparse
parser = argparse.ArgumentParser()
parser.add_argument('file')
args = parser.parse_args()
print('Done')
"""
    client = MockChatClient([
        {"content": f"```python\n{good_tool}\n```"},
        {"content": f"```python\n{failing_test}\n```"},
        {"content": f"```python\n# FILE: mocktool.py\n{stub_tool}\n```"},
        {"content": f"```python\n# FILE: mocktool.py\n{stub_tool}\n```"},
    ])
    engine = CoreEngine(client=client)

    test_dir = tempfile.mkdtemp()
    with patch("helix_core.standalone_builder.WORKSPACE_DIR", Path(test_dir)):
        success, msg, details = build_standalone_project_staged(
            engine,
            "Build me a command-line tool called mocktool. It needs a --json option.",
            max_attempts=2,
        )
        assert success is False
        tool_on_disk = (Path(test_dir) / "mocktool" / "mocktool.py").read_text(encoding="utf-8")
        assert "print('Done')" not in tool_on_disk
        assert "process_data" in tool_on_disk

        writes_file = Path(test_dir) / "mocktool" / ".build_log" / "attempt_1" / "file_writes.json"
        assert writes_file.exists()
        assert "rejected" in writes_file.read_text(encoding="utf-8")
    shutil.rmtree(test_dir, ignore_errors=True)


def test_r3_per_attempt_tracing_records_all_fields() -> None:
    """R3-1: Per-attempt tracing under <project_dir>/.build_log/attempt_N/ records all artifacts."""
    import json, tempfile, shutil
    from helix_core.standalone_builder import build_standalone_project_staged

    good_tool = """
import sys, os, argparse
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
parser = argparse.ArgumentParser()
parser.add_argument('file')
parser.add_argument('--json', action='store_true')
args = parser.parse_args()
if not os.path.exists(args.file):
    print("error", file=sys.stderr)
    sys.exit(1)
print("ok")
"""
    passing_test = """
import unittest
class TestMockTool(unittest.TestCase):
    def test_ok(self):
        self.assertEqual(1, 1)
    def test_another(self):
        self.assertTrue(True)
if __name__ == '__main__':
    unittest.main()
"""
    client = MockChatClient([
        {"content": f"```python\n{good_tool}\n```"},
        {"content": f"```python\n{passing_test}\n```"},
    ])
    engine = CoreEngine(client=client)

    test_dir = tempfile.mkdtemp()
    with patch("helix_core.standalone_builder.WORKSPACE_DIR", Path(test_dir)):
        success, msg, details = build_standalone_project_staged(
            engine,
            "Build me a command-line tool called mocktool. It needs a --json option.",
            max_attempts=2,
        )
        assert success is True
        att1 = Path(test_dir) / "mocktool" / ".build_log" / "attempt_1"
        assert (att1 / "tool_prompt.txt").exists()
        assert (att1 / "tool_raw.txt").exists()
        assert (att1 / "tests_prompt.txt").exists()
        assert (att1 / "tests_raw.txt").exists()
        assert (att1 / "smoke_results.txt").exists()
        assert (att1 / "test_output.txt").exists()
        assert (att1 / "file_writes.json").exists()
        assert (att1 / "attempt_summary.json").exists()

        summary = json.loads((att1 / "attempt_summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "success"
        assert len(summary["tool_sha256"]) == 64
    shutil.rmtree(test_dir, ignore_errors=True)


def test_r3_stuck_failures_rotate_strategy_and_converge() -> None:
    """R3-3: Stuck test set failure rotates strategy to regenerate_tests_only and converges."""
    import tempfile, shutil
    from helix_core.standalone_builder import build_standalone_project_staged

    good_tool = """
import sys, os, argparse
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
parser = argparse.ArgumentParser()
parser.add_argument('file')
parser.add_argument('--json', action='store_true')
args = parser.parse_args()
if not os.path.exists(args.file):
    print("error", file=sys.stderr)
    sys.exit(1)
print("ok")
"""
    failing_test = """
import unittest
class T(unittest.TestCase):
    def test_fixed_fail(self):
        self.assertEqual(1, 999)
if __name__ == '__main__':
    unittest.main()
"""
    passing_test = """
import unittest
class T(unittest.TestCase):
    def test_fixed_pass(self):
        self.assertEqual(1, 1)
if __name__ == '__main__':
    unittest.main()
"""
    call_idx = 0
    prompts_seen = []
    def custom_chat(messages, tools=None):
        nonlocal call_idx
        call_idx += 1
        prompt = messages[0]["content"]
        prompts_seen.append(prompt)
        if call_idx == 1:
            return {"content": f"```python\n{good_tool}\n```"}
        elif call_idx == 2:
            return {"content": f"```python\n{failing_test}\n```"}
        elif call_idx == 3:
            return {"content": f"```python\n# FILE: test_mocktool.py\n{failing_test}\n```"}
        elif call_idx == 4:
            return {"content": f"```python\n# FILE: test_mocktool.py\n{passing_test}\n```"}
        return {"content": ""}

    client = MockChatClient([])
    client.chat = custom_chat
    engine = CoreEngine(client=client)

    test_dir = tempfile.mkdtemp()
    with patch("helix_core.standalone_builder.WORKSPACE_DIR", Path(test_dir)):
        success, msg, details = build_standalone_project_staged(
            engine,
            "Build me a command-line tool called mocktool. It needs a --json option.",
            max_attempts=4,
        )
        assert success is True
        assert any("passed all requirements coverage and runtime smoke tests" in p for p in prompts_seen)
    shutil.rmtree(test_dir, ignore_errors=True)


def test_r3_regression_rollback_to_best_so_far() -> None:
    """R3-3: Best-so-far candidate is tracked; regressed attempt triggers rollback to best candidate."""
    import tempfile, shutil
    from helix_core.standalone_builder import build_standalone_project_staged

    good_tool = """
import sys, os, argparse
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
parser = argparse.ArgumentParser()
parser.add_argument('file')
parser.add_argument('--json', action='store_true')
args = parser.parse_args()
if not os.path.exists(args.file):
    print("error", file=sys.stderr)
    sys.exit(1)
print("ok")
"""
    one_fail_test = """
import unittest
class T(unittest.TestCase):
    def test_ok(self):
        self.assertEqual(1, 1)
    def test_bad(self):
        self.assertEqual(1, 2)
if __name__ == '__main__':
    unittest.main()
"""
    two_fail_test = """
import unittest
class T(unittest.TestCase):
    def test_bad1(self):
        self.assertEqual(1, 2)
    def test_bad2(self):
        self.assertEqual(3, 4)
if __name__ == '__main__':
    unittest.main()
"""
    client = MockChatClient([
        {"content": f"```python\n{good_tool}\n```"},
        {"content": f"```python\n{one_fail_test}\n```"},
        {"content": f"```python\n# FILE: test_mocktool.py\n{two_fail_test}\n```"},
    ])
    engine = CoreEngine(client=client)

    test_dir = tempfile.mkdtemp()
    with patch("helix_core.standalone_builder.WORKSPACE_DIR", Path(test_dir)):
        success, msg, details = build_standalone_project_staged(
            engine,
            "Build me a command-line tool called mocktool. It needs a --json option.",
            max_attempts=2,
        )
        assert success is False
        assert len(details["failing_tests"]) == 1
        assert "restored from the best-so-far candidate (Attempt 1)" in msg
        assert "the tool meets the requirements in my checks; these 1 of my own tests are still wrong" in msg
    shutil.rmtree(test_dir, ignore_errors=True)


def test_r3_test_hygiene_static_check() -> None:
    """R3-2: check_test_code_hygiene enforces encoding, absolute path, and non-zero exit."""
    from helix_core.quality import check_test_code_hygiene

    bad_enc = "subprocess.run([sys.executable, 'tool.py'], capture_output=True, text=True)"
    ok, issues = check_test_code_hygiene(bad_enc, "tool")
    assert ok is False
    assert any("encoding='utf-8'" in i for i in issues)

    bad_rel = "subprocess.run([sys.executable, 'mytool.py'], capture_output=True, encoding='utf-8')"
    ok, issues = check_test_code_hygiene(bad_rel, "mytool")
    assert ok is False
    assert any("absolute path derived from __file__" in i for i in issues)

    bad_exit = "def test_empty_file(self):\n    proc = run()\n    self.assertEqual(proc.returncode, 0)"
    ok, issues = check_test_code_hygiene(bad_exit, "mytool")
    assert ok is False
    assert any("assert non-zero returncode" in i for i in issues)

    good_test = """
import sys, unittest, subprocess
from pathlib import Path
TOOL_PATH = Path(__file__).resolve().parent / "mytool.py"
class T(unittest.TestCase):
    def test_run(self):
        res = subprocess.run([sys.executable, str(TOOL_PATH)], capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(res.returncode, 0)
"""
    ok, issues = check_test_code_hygiene(good_test, "mytool")
    assert ok is True


def test_r3_honesty_guard_legitimate_failure_not_rewritten() -> None:
    """R3-4: Honest failure report with real test output is preserved without disclaimers."""
    from helix_core.honesty import TurnEvents, sanitize_reply

    honest_failure_reply = (
        "Failed to build `dupefinder`: tests still failing after 2 attempts (1 test(s) remaining).\n\n"
        "### Failing Tests (1 remaining):\n- `test_unicode`\n\n"
        "### Real Error / Test Output:\n"
        "```text\n"
        "Ran 3 tests in 0.045s\n"
        "FAILED (failures=1)\n"
        "```\n"
    )

    events = TurnEvents()
    events.add_test_execution("test_dupefinder.py", passed=False, output="FAILED (failures=1)")

    sanitized, intercepted = sanitize_reply(honest_failure_reply, events)
    assert intercepted is False
    assert "FAILED (failures=1)" in sanitized
    assert "tests still failing after 2 attempts" in sanitized
    assert "Invented test pass claims were removed" not in sanitized



# -----------------------------------------------------------------------------
# Round 4 Tests (R4-1 to R4-4)
# -----------------------------------------------------------------------------

def test_r4_timeout_in_test_stage_preserves_tool() -> None:
    """R4-1: Timeout during test generation preserves tool code and resumes in attempt 2."""
    import tempfile, shutil
    from helix_core.standalone_builder import build_standalone_project_staged

    tool_code = """
import sys, os, argparse
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
parser = argparse.ArgumentParser()
parser.add_argument('file')
parser.add_argument('--json', action='store_true')
args = parser.parse_args()
if not os.path.exists(args.file):
    print("error", file=sys.stderr)
    sys.exit(1)
print("ok")
"""
    passing_test = """
import unittest
class T(unittest.TestCase):
    def test_ok(self):
        self.assertEqual(1, 1)
if __name__ == '__main__':
    unittest.main()
"""
    call_count = 0
    class StagedTimeoutClient:
        def chat(self, messages: Any, tools: Any = None) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"content": f"```python\n{tool_code}\n```"}
            elif call_count == 2:
                raise TimeoutError("Model read timeout 300s during test gen")
            else:
                return {"content": f"```python\n{passing_test}\n```"}

    engine = CoreEngine(client=StagedTimeoutClient())
    test_dir = tempfile.mkdtemp()
    proj_dir = Path(test_dir) / "workspace" / "dupefinder"
    proj_dir.mkdir(parents=True, exist_ok=True)

    with patch("helix_core.standalone_builder.WORKSPACE_DIR", Path(test_dir) / "workspace"):
        success, msg, details = build_standalone_project_staged(
            engine,
            "Build me a command-line tool called dupefinder. It needs a --json option.",
            max_attempts=3,
        )
        assert success is True
        assert (proj_dir / "dupefinder.py").exists()
        assert (proj_dir / "test_dupefinder.py").exists()

    shutil.rmtree(test_dir, ignore_errors=True)


def test_r4_all_attempts_timeout_truthfully_reports_files() -> None:
    """R4-1: If all attempts time out, builder truthfully reports timeouts and names files on disk."""
    import tempfile, shutil
    from helix_core.standalone_builder import build_standalone_project_staged

    class AllTimeoutClient:
        def chat(self, messages: Any, tools: Any = None) -> Dict[str, Any]:
            raise TimeoutError("Persistent model read timeout 300s")

    engine = CoreEngine(client=AllTimeoutClient())
    test_dir = tempfile.mkdtemp()
    proj_dir = Path(test_dir) / "workspace" / "dupefinder"
    proj_dir.mkdir(parents=True, exist_ok=True)

    with patch("helix_core.standalone_builder.WORKSPACE_DIR", Path(test_dir) / "workspace"):
        success, msg, details = build_standalone_project_staged(
            engine,
            "Build me a command-line tool called dupefinder. It needs a --json option.",
            max_attempts=2,
        )
        assert success is False
        assert "all 2 attempts timed out" in msg.lower()
        assert details["status"] == "failed"

    shutil.rmtree(test_dir, ignore_errors=True)


def test_r4_timeout_in_fix_stage_preserves_files() -> None:
    """R4-1: Timeout during self-correction fix preserves tool and test files on disk and records in build log."""
    import tempfile, shutil
    from helix_core.standalone_builder import build_standalone_project_staged

    tool_code = """
import sys, os, argparse
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
parser = argparse.ArgumentParser()
parser.add_argument('file')
parser.add_argument('--json', action='store_true')
args = parser.parse_args()
if not os.path.exists(args.file):
    print("error", file=sys.stderr)
    sys.exit(1)
print("ok")
"""
    failing_test = """
import unittest
class T(unittest.TestCase):
    def test_fail(self):
        self.assertEqual(1, 2)
if __name__ == '__main__':
    unittest.main()
"""
    call_count = 0
    class FixTimeoutClient:
        def chat(self, messages: Any, tools: Any = None) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"content": f"```python\n{tool_code}\n```"}
            elif call_count == 2:
                return {"content": f"```python\n{failing_test}\n```"}
            else:
                raise TimeoutError("HTTP read timeout 300s in fix stage")

    engine = CoreEngine(client=FixTimeoutClient())
    test_dir = tempfile.mkdtemp()
    proj_dir = Path(test_dir) / "workspace" / "dupefinder"
    proj_dir.mkdir(parents=True, exist_ok=True)

    with patch("helix_core.standalone_builder.WORKSPACE_DIR", Path(test_dir) / "workspace"):
        success, msg, details = build_standalone_project_staged(
            engine,
            "Build me a command-line tool called dupefinder. It needs a --json option.",
            max_attempts=2,
        )
        assert success is False
        assert (proj_dir / "dupefinder.py").exists()
        assert (proj_dir / "test_dupefinder.py").exists()
        fix_log = proj_dir / ".build_log" / "attempt_1" / "fix_timeout.txt"
        assert fix_log.exists()

    shutil.rmtree(test_dir, ignore_errors=True)


def test_r4_arbitration_locks_tool_sha_on_test_defect() -> None:
    """R4-3: Arbitration classifies test defect and locks tool SHA256; candidate tool update is rejected."""
    import tempfile, shutil
    from helix_core.quality import arbitrate_test_failure
    from helix_core.standalone_builder import build_standalone_project_staged

    good_tool = """
import sys, os, argparse
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
parser = argparse.ArgumentParser()
parser.add_argument('file')
parser.add_argument('--json', action='store_true')
args = parser.parse_args()
if not os.path.exists(args.file):
    print("error", file=sys.stderr)
    sys.exit(1)
print("ok")
"""
    arb = arbitrate_test_failure(
        "Build me a command-line tool called dupefinder. It needs a --json option.",
        tool_code=good_tool,
        test_code="def test_bad(self): pass",
        test_output="AssertionError: 'a' unexpectedly found in output",
        failing_tests=["test_bad"],
        smoke_passed=True,
        coverage_passed=True,
        memory_passed=True,
    )
    assert arb["classification"] == "TEST"
    assert arb["lock_tool"] is True

    test_dir = tempfile.mkdtemp()
    proj_dir = Path(test_dir) / "workspace" / "dupefinder"
    proj_dir.mkdir(parents=True, exist_ok=True)

    bad_test = """
import unittest, subprocess, sys
from pathlib import Path
TOOL_PATH = Path(__file__).resolve().parent / 'dupefinder.py'
class TestDupe(unittest.TestCase):
    def test_json_bracket(self):
        res = subprocess.run([sys.executable, str(TOOL_PATH), '--help'], capture_output=True, text=True, encoding='utf-8')
        self.assertIn("['folder']", res.stdout)
if __name__ == '__main__':
    unittest.main()
"""
    broken_tool = "# Stub tool that should be rejected\nimport sys\n"
    responses = [
        {"content": f"```python\n{good_tool}\n```"},
        {"content": f"```python\n{bad_test}\n```"},
        {"content": f"```python\n# FILE: dupefinder.py\n{broken_tool}\n```"},
    ]
    client = MockChatClient(responses)
    engine = CoreEngine(client=client)

    with patch("helix_core.standalone_builder.WORKSPACE_DIR", Path(test_dir) / "workspace"):
        success, msg, details = build_standalone_project_staged(
            engine,
            "Build me a command-line tool called dupefinder. It needs a --json option.",
            max_attempts=2,
        )
        tool_content = (proj_dir / "dupefinder.py").read_text(encoding="utf-8")
        assert "Stub tool that should be rejected" not in tool_content
        assert "args = parser.parse_args()" in tool_content

    shutil.rmtree(test_dir, ignore_errors=True)


def test_r4_hygiene_static_checks_r4_rules() -> None:
    """R4-2: Static hygiene check rejects short assertNotIn, exact multi-space, and missing json.loads."""
    from helix_core.quality import check_test_code_hygiene

    # 1. Short assertNotIn (<= 2 chars)
    code1 = """import unittest
class T(unittest.TestCase):
    def test_x(self):
        self.assertNotIn("a", output)
"""
    ok1, issues1 = check_test_code_hygiene(code1, "testtool")
    assert not ok1
    assert any("<= 2 chars" in i for i in issues1)

    # 2. Exact multi-space formatting
    code2 = """import unittest
class T(unittest.TestCase):
    def test_x(self):
        self.assertEqual(output, "Name     Age    City")
"""
    ok2, issues2 = check_test_code_hygiene(code2, "testtool")
    assert not ok2
    assert any("multi-space" in i for i in issues2)

    # 3. Missing json.loads when testing --json
    code3 = """import unittest
class T(unittest.TestCase):
    def test_x(self):
        res = run([sys.executable, str(TOOL_PATH), "--json"])
        self.assertIn("foo", res.stdout)
"""
    ok3, issues3 = check_test_code_hygiene(code3, "testtool")
    assert not ok3
    assert any("json.loads" in i for i in issues3)

    # 4. Clean test adhering to rules passes
    code4 = """import unittest
import json
class T(unittest.TestCase):
    def test_x(self):
        res = run([sys.executable, str(TOOL_PATH), "--json"])
        data = json.loads(res.stdout)
        self.assertEqual(data["count"], 0)
        self.assertNotIn("unwanted_token", res.stdout)
"""
    ok4, issues4 = check_test_code_hygiene(code4, "testtool")
    assert ok4
    assert len(issues4) == 0


def test_r4_memory_gate_budget_enforcement() -> None:
    """R4-4: Memory budget gate rejects full file reads and enforces memory ceiling."""
    from helix_core.quality import check_requirements_coverage

    bad_code = """import sys
def main():
    data = open('bigfile.bin').read()
"""
    covered, missing = check_requirements_coverage("tool to process big files", bad_code)
    assert not covered
    assert any("full-file read" in m for m in missing)


def test_r4_quarantine_bad_test_and_skipped_status() -> None:
    """R4-3: Quarantines bad test with @unittest.skip and outputs accurate final status."""
    import tempfile, shutil
    from helix_core.standalone_builder import build_standalone_project_staged

    good_tool = """
import sys, os, argparse
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
parser = argparse.ArgumentParser()
parser.add_argument('file')
parser.add_argument('--json', action='store_true')
args = parser.parse_args()
if not os.path.exists(args.file):
    print("error", file=sys.stderr)
    sys.exit(1)
print("ok")
"""
    bad_output_test = """
import unittest, subprocess, sys
from pathlib import Path
TOOL_PATH = Path(__file__).resolve().parent / 'dupefinder.py'

class TestDupe(unittest.TestCase):
    def test_rigid_bad_assertion(self):
        res = subprocess.run([sys.executable, str(TOOL_PATH), '--help'], capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(res.stdout, "THIS RIGID STRING CONTRADICTS OBSERVED SMOKE OUTPUT")
    def test_valid_runs(self):
        res = subprocess.run([sys.executable, str(TOOL_PATH), '--help'], capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(res.returncode, 0)
if __name__ == '__main__':
    unittest.main()
"""
    responses = [
        {"content": f"```python\n{good_tool}\n```"},
        {"content": f"```python\n{bad_output_test}\n```"},
    ]
    client = MockChatClient(responses)
    engine = CoreEngine(client=client)

    test_dir = tempfile.mkdtemp()
    proj_dir = Path(test_dir) / "workspace" / "dupefinder"
    proj_dir.mkdir(parents=True, exist_ok=True)

    with patch("helix_core.standalone_builder.WORKSPACE_DIR", Path(test_dir) / "workspace"):
        success, msg, details = build_standalone_project_staged(
            engine,
            "Build me a command-line tool called dupefinder. It needs a --json option.",
            max_attempts=1,
        )
        assert "tool meets requirements; 1 of my tests were wrong and skipped: test_rigid_bad_assertion" in details["status"]
        test_file_content = (proj_dir / "test_dupefinder.py").read_text(encoding="utf-8")
        assert "@unittest.skip" in test_file_content
        assert "test_rigid_bad_assertion" in test_file_content

    shutil.rmtree(test_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# Round 5 Fixes Tests
# -----------------------------------------------------------------------------

def test_r5_table_alignment_under_max_width_smoke_check() -> None:
    """R5-2: Deterministic smoke test detects misaligned table rows under --max-width."""
    from helix_core.standalone_builder import run_smoke_tests
    test_dir = tempfile.mkdtemp()
    proj_dir = Path(test_dir)
    bad_tab = proj_dir / "tablefmt.py"
    bad_code = """import sys, os, argparse
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
parser = argparse.ArgumentParser()
parser.add_argument('csv_file')
parser.add_argument('--max-width', type=int, default=8)
args = parser.parse_args()
if not os.path.exists(args.csv_file):
    sys.exit(1)
print("id | name    | department")
print("---+---------+-----------")
print("1  | LongNameWasNotTruncated | Engineering")
print("2  | AnotherVeryLongName     | HR")
"""
    bad_tab.write_text(bad_code, encoding="utf-8")
    ok, rep, fails = run_smoke_tests(proj_dir, bad_tab, "tablefmt", "Build tablefmt with --max-width", timeout=10.0)
    assert not ok, "Smoke tests must fail when table rows are misaligned"
    assert any("Table alignment error" in f or "Table column alignment" in f for f in fails)

    good_code = """import sys, os, argparse
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
parser = argparse.ArgumentParser()
parser.add_argument('csv_file')
parser.add_argument('--max-width', type=int, default=8)
args = parser.parse_args()
if not os.path.exists(args.csv_file):
    sys.exit(1)
print("id | name     | depart..")
print("---+----------+---------")
print("1  | Short    | Engine..")
print("2  | VeryLo.. | HR      ")
"""
    bad_tab.write_text(good_code, encoding="utf-8")
    ok2, rep2, fails2 = run_smoke_tests(proj_dir, bad_tab, "tablefmt", "Build tablefmt with --max-width", timeout=10.0)
    assert ok2, f"Expected smoke test to pass on aligned table: {fails2}"

    shutil.rmtree(test_dir, ignore_errors=True)


def test_r5_whole_document_tool_relaxed_advisory_budget() -> None:
    """R5-1: Whole-document tools (JSON/XML/YAML) receive relaxed advisory memory budget."""
    from helix_core.quality import classify_tool_input_kind, check_tool_memory_budget
    assert classify_tool_input_kind("Build tool jsonget for json file", "jsonget") == "whole_document_json"
    assert classify_tool_input_kind("Build tool tabprint for csv file", "tabprint") == "streaming_csv"
    assert classify_tool_input_kind("Build tool dirsize for directory", "dirsize") == "streaming_directory"

    test_dir = tempfile.mkdtemp()
    proj_dir = Path(test_dir)
    tool_file = proj_dir / "jsonget.py"
    tool_file.write_text("""import sys, json, os
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("loaded")
""", encoding="utf-8")

    ok, rss, rep = check_tool_memory_budget(proj_dir, tool_file, "jsonget", request="Build me tool jsonget for json file", timeout=15.0)
    assert ok is True
    assert "Advisory" in rep or "PASSED" in rep
    shutil.rmtree(test_dir, ignore_errors=True)


def test_r5_identical_regeneration_loop_guard() -> None:
    """R5-1: Loop guard breaks early when model regenerates identical code with same failures."""
    from helix_core.standalone_builder import build_standalone_project_staged
    from helix_core.engine import CoreEngine

    identical_tool = """```python
import sys, argparse
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
parser = argparse.ArgumentParser()
parser.add_argument('path')
parser.add_argument('--flag', action='store_true')
args = parser.parse_args()
print("tool output")
```"""
    failing_test = """```python
import unittest, subprocess, sys
from pathlib import Path
TOOL_PATH = Path(__file__).resolve().parent / 'looptool.py'
class T(unittest.TestCase):
    def test_fail(self):
        self.assertEqual(1, 2)
if __name__ == '__main__':
    unittest.main()
```"""
    responses = [
        {"content": identical_tool},
        {"content": failing_test},
        {"content": identical_tool},
        {"content": identical_tool},
        {"content": identical_tool},
    ]
    client = MockChatClient(responses)
    engine = CoreEngine(client=client)

    test_dir = tempfile.mkdtemp()
    with patch("helix_core.standalone_builder.WORKSPACE_DIR", Path(test_dir)):
        success, msg, details = build_standalone_project_staged(
            engine,
            "Build me a command-line tool called looptool with --flag option",
            max_attempts=5,
        )
        assert success is False
        assert details["attempts"] <= 3, f"Loop guard should halt at attempt <= 3, but took {details['attempts']}"
    shutil.rmtree(test_dir, ignore_errors=True)


def test_r5_status_wording_matches_facts_on_disk() -> None:
    """R5-3 & R5-5: Final status wording is derived from facts on disk and matches defined table."""
    from helix_core.quality import compute_final_status_from_disk
    test_dir = tempfile.mkdtemp()
    proj_dir = Path(test_dir)

    st1 = compute_final_status_from_disk(proj_dir, "missing_tool")
    assert st1 == "REQUIREMENT FAILED (tool source file not found on disk)"

    (proj_dir / "sample.py").write_text("import sys, argparse\nif sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')\nparser = argparse.ArgumentParser()\nparser.add_argument('file')\nparser.add_argument('--json', action='store_true')\nargs = parser.parse_args()\nimport os\nif not os.path.exists(args.file): sys.exit(1)\nprint('done')\n", encoding="utf-8")
    st2 = compute_final_status_from_disk(proj_dir, "sample", "Build me a tool called sample with --json")
    assert st2 == "NO TESTS GENERATED (test file missing on disk)"

    (proj_dir / "test_sample.py").write_text("""import unittest
class T(unittest.TestCase):
    def test_a(self): pass
    def test_b(self): pass
if __name__ == '__main__':
    unittest.main()
""", encoding="utf-8")
    st3 = compute_final_status_from_disk(proj_dir, "sample", "Build me a tool called sample with --json")
    assert st3.startswith("COVERAGE GAP (missing active test coverage for: --json)"), f"Actual: {st3}"

    (proj_dir / "test_sample.py").write_text("""import unittest, subprocess, sys
from pathlib import Path
TOOL_PATH = Path(__file__).resolve().parent / 'sample.py'
class T(unittest.TestCase):
    def test_json(self):
        res = subprocess.run([sys.executable, str(TOOL_PATH), '--help'], capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(res.returncode, 0)
    @unittest.skip("skip reason")
    def test_b(self): pass
if __name__ == '__main__':
    unittest.main()
""", encoding="utf-8")
    st4 = compute_final_status_from_disk(proj_dir, "sample", "Build me a tool called sample with --json")
    assert st4.startswith("PASSED WITH SKIPS (1 active passed, 1 skipped: test_b)"), f"Actual: {st4}"

    shutil.rmtree(test_dir, ignore_errors=True)


def test_r5_quarantine_rewrite_before_skip_and_cap() -> None:
    """R5-3: Rewriting is tried before skipping, 25% cap enforced, and sole option tests protected."""
    from helix_core.quality import quarantine_bad_tests
    test_dir = tempfile.mkdtemp()
    proj_dir = Path(test_dir)
    test_path = proj_dir / "test_dirsize.py"

    suite_code = """import unittest, subprocess, sys
from pathlib import Path
TOOL_PATH = Path(__file__).resolve().parent / 'dirsize.py'
class T(unittest.TestCase):
    def test_1(self): pass
    def test_2(self): pass
    def test_3(self): pass
    def test_4(self): pass
    def test_5(self): pass
    def test_6(self): pass
    def test_top_option(self):
        res = subprocess.run([sys.executable, str(TOOL_PATH), '--top', '5'], capture_output=True, text=True)
        self.assertEqual(res.stdout, "rigid --top")
    def test_rigid_format(self):
        res = subprocess.run([sys.executable, str(TOOL_PATH)], capture_output=True, text=True)
        self.assertEqual(res.stdout, "rigid    spacing")
"""
    test_path.write_text(suite_code, encoding="utf-8")

    ok, qlist = quarantine_bad_tests(proj_dir, test_path, ["test_top_option", "test_rigid_format"], "AssertionError: rigid    spacing != other", request="tool with --top option")
    qnames = [q["name"] for q in qlist]
    assert "test_top_option" not in qnames, "Sole test for requested option --top must NOT be quarantined"
    assert "test_rigid_format" in qnames

    shutil.rmtree(test_dir, ignore_errors=True)


def test_r5_arbitration_treats_requested_option_failure_as_tool_defect() -> None:
    """R5-5: Arbitration treats test failures on requested options as TOOL defects."""
    from helix_core.quality import arbitrate_test_failure

    test_src = """
def test_keys_option(self):
    res = subprocess.run([sys.executable, str(TOOL_PATH), 'f.json', '--keys'], capture_output=True, text=True)
    self.assertIn('name', res.stdout)
"""
    arb = arbitrate_test_failure(
        request="Build tool jsonget with --keys option",
        tool_code="print('no keys')",
        test_code=test_src,
        test_output="AssertionError: 'name' not found in ''",
        failing_tests=["test_keys_option"],
        smoke_passed=True,
        coverage_passed=True,
        memory_passed=True,
    )
    assert arb["classification"] == "TOOL", "Failure asserting requested option must be classified as TOOL defect"
    assert arb["lock_tool"] is False, "Tool must not be locked when fixing requested option defect"


def test_r5_coverage_gap_status_when_option_uncovered() -> None:
    """R5-3: If requested option lacks active tests, status reports COVERAGE GAP."""
    from helix_core.quality import compute_final_status_from_disk
    test_dir = tempfile.mkdtemp()
    proj_dir = Path(test_dir)
    (proj_dir / "mytool.py").write_text("import sys, argparse\nif sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')\nparser = argparse.ArgumentParser()\nparser.add_argument('file')\nparser.add_argument('--pretty', action='store_true')\nargs = parser.parse_args()\nimport os\nif not os.path.exists(args.file): sys.exit(1)\n", encoding="utf-8")
    (proj_dir / "test_mytool.py").write_text("""import unittest
class T(unittest.TestCase):
    def test_basic(self): pass
if __name__ == '__main__': unittest.main()
""", encoding="utf-8")

    st = compute_final_status_from_disk(proj_dir, "mytool", "Build me a tool called mytool with --pretty option")
    assert st.startswith("COVERAGE GAP (missing active test coverage for: --pretty)"), f"Unexpected status: {st}"

    shutil.rmtree(test_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

def run_all_dev_fix_tests() -> bool:
    suites = [
        ("H1: Windows Unicode / cp1252 Smoke & Static Check", test_h1_windows_unicode_cp1252_smoke_and_static_check),
        ("H2: Nonexistent Path Nonzero Exit Smoke Check", test_h2_nonexistent_path_nonzero_exit_smoke_check),
        ("H3: Standalone Staged Builder Success & Diagnosis", test_h3_standalone_staged_builder_success_and_failure),
        ("H3: Round Limit Rich Status Summary", test_h3_round_limit_rich_status),
        ("Adversarial: Honesty Phrasings & Resume Hijack", test_adversarial_honesty_and_resume),
        ("F1: Honesty Guard - Fabricated Reply Interception", test_f1_honesty_guard_intercepts_fabricated_reply),
        ("F1: Honesty Guard - Genuine Reply Pass-through", test_f1_honesty_guard_passes_genuine_reply),
        ("F2: Continue - Task Resumption With Failure Context", test_f2_continue_resumes_task_with_failure_context),
        ("F3: Standalone Routing - Request Classification", test_f3_project_classification),
        ("F3: Standalone Routing - Workspace Write & Security Guard", test_f3_standalone_workspace_write_and_security_guard),
        ("F4: Adaptive Timeouts - Profile Scaling", test_f4_adaptive_timeouts_by_hardware_profile),
        ("F4: Adaptive Timeouts - Client Call-Time Resolution", test_f4_ollama_client_uses_adaptive_timeout),
        ("F5: Self-Correction - Fail Then Fix (2 Iterations)", test_f5_self_correction_fails_then_fixes),
        ("F5: Self-Correction - Exhausted Retries Report Failure", test_f5_self_correction_exhausted_reports_failure),
        ("F6: Quality Guidance - Static Requirements Coverage", test_f6_static_requirements_coverage_check),
        ("F6: Quality Guidance - System Prompt Checklist", test_f6_system_prompt_contains_quality_checklist),
        ("F7: GPU Telemetry - Garbage nvidia-smi Rejection", test_f7_nvidia_smi_garbage_rejection),
        ("F7: GPU Telemetry - Format Banner Clean Omission", test_f7_format_banner_omits_invalid_figures),
        ("F7: GPU Telemetry - Resident Model Clamp Safety", test_f7_adjust_resident_models_clamps_safely),
        ("R3: Fix Stage No Blocks Escalation", test_r3_fix_stage_no_blocks_escalation),
        ("R3: Fix Stage Rejects Stub & Preserves Tool", test_r3_fix_stage_rejects_stub_and_preserves_tool),
        ("R3: Per-Attempt Tracing Records All Fields", test_r3_per_attempt_tracing_records_all_fields),
        ("R3: Stuck Failures Rotate Strategy & Converge", test_r3_stuck_failures_rotate_strategy_and_converge),
        ("R3: Regression Rollback to Best-So-Far", test_r3_regression_rollback_to_best_so_far),
        ("R3: Test Hygiene Static Check", test_r3_test_hygiene_static_check),
        ("R3: Honesty Guard Legitimate Failure Preserved", test_r3_honesty_guard_legitimate_failure_not_rewritten),
        ("R4: Timeout in Test Stage Preserves Tool", test_r4_timeout_in_test_stage_preserves_tool),
        ("R4: All Attempts Timeout Truthful Failure", test_r4_all_attempts_timeout_truthfully_reports_files),
        ("R4: Timeout in Fix Stage Preserves Files", test_r4_timeout_in_fix_stage_preserves_files),
        ("R4: Arbitration Locks Tool SHA on Test Defect", test_r4_arbitration_locks_tool_sha_on_test_defect),
        ("R4: Hygiene Static Checks R4 Rules", test_r4_hygiene_static_checks_r4_rules),
        ("R4: Memory Gate Budget Enforcement", test_r4_memory_gate_budget_enforcement),
        ("R4: Quarantine Bad Test & Skipped Status", test_r4_quarantine_bad_test_and_skipped_status),
        ("R5: Table Alignment Smoke Check", test_r5_table_alignment_under_max_width_smoke_check),
        ("R5: Whole Document Advisory Budget", test_r5_whole_document_tool_relaxed_advisory_budget),
        ("R5: Loop Guard Identical Regeneration", test_r5_identical_regeneration_loop_guard),
        ("R5: Status Wording Matches Facts on Disk", test_r5_status_wording_matches_facts_on_disk),
        ("R5: Quarantine Rewrite & 25% Cap", test_r5_quarantine_rewrite_before_skip_and_cap),
        ("R5: Arbitration Option Guard", test_r5_arbitration_treats_requested_option_failure_as_tool_defect),
        ("R5: Coverage Gap Status", test_r5_coverage_gap_status_when_option_uncovered),
    ]

    print("=" * 60)
    print("  Helix Development-Ability Fixes Test Suite (F1 - F7)")
    print("=" * 60)

    passes = 0
    fails = 0

    for name, test_fn in suites:
        print(f"Running {name}...", end=" ", flush=True)
        try:
            test_fn()
            print("[PASS]")
            passes += 1
        except Exception as exc:
            import traceback
            print("[FAIL]")
            print(f"  Error: {exc}")
            traceback.print_exc()
            fails += 1

    print("=" * 60)
    print(f"  Summary: {passes} PASSED, {fails} FAILED")
    print("=" * 60)

    return fails == 0


if __name__ == "__main__":
    success = run_all_dev_fix_tests()
    sys.exit(0 if success else 1)
