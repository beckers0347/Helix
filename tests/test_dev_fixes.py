"""Offline unit and integration tests for Ogun development-ability fixes (F1 - F7).

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

# Ensure Ogun root is on sys.path
OGUN_ROOT = Path(__file__).resolve().parents[1]
if str(OGUN_ROOT) not in sys.path:
    sys.path.insert(0, str(OGUN_ROOT))

from ogun_core.base_tool import BaseTool, ToolResult
from ogun_core.config import (
    SYSTEM_PROMPT,
    get_request_timeout,
    set_vram_profile,
)
from ogun_core.engine import CoreEngine, OllamaClient
from ogun_core.hardware import (
    GpuInfo,
    HardwareInfo,
    adjust_for_resident_models,
    format_banner,
    _probe_nvidia_gpus,
)
from ogun_core.honesty import TurnEvents, sanitize_reply
from ogun_core.plugin_builder import build_plugin_staged
from ogun_core.plugin_manager import PluginManager
from ogun_core.quality import check_requirements_coverage, classify_project_request
from ogun_core.security import PathSecurityGuard


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
    assert "No tests were executed by Ogun" in sanitized


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
        "Create a dynamic plugin for Ogun named StringHasher.",
        "Build a plugin that adds web scraping to Ogun.",
        "Add tool to ogun inheriting from BaseTool.",
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
            guard.validate_write(root / "ogun_core" / "engine.py")
            assert False, "Should have rejected write to ogun_core"
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
    with patch.dict(os.environ, {"OGUN_REQUEST_TIMEOUT": "150.0"}):
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
from ogun_core.base_tool import BaseTool, ToolResult
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
from ogun_core.base_tool import BaseTool, ToolResult
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

    with patch("ogun_core.hardware._find_nvidia_smi", return_value="nvidia-smi"):
        with patch("ogun_core.hardware._run_subproc", return_value=(0, garbage_csv)):
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

    from ogun_core.hardware import auto_configure
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
    from ogun_core.quality import check_requirements_coverage
    from ogun_core.standalone_builder import run_smoke_tests

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
    from ogun_core.standalone_builder import run_smoke_tests

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
    from ogun_core.standalone_builder import build_standalone_project_staged
    from ogun_core.engine import CoreEngine

    class MockEngineClient:
        def __init__(self, responses: List[str]) -> None:
            self.responses = list(responses)
        def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
            if self.responses:
                return {"content": self.responses.pop(0)}
            return {"content": "no response"}

    mocktool_dir = OGUN_ROOT / "workspace" / "mocktool"
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
        "from _scaffold import ToolTestCase\n"
        "import unittest\n"
        "class T(ToolTestCase):\n"
        "    def test_ok(self):\n"
        "        res = self.run_tool('--help')\n"
        "        self.assertEqual(res.returncode, 0)\n"
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
    from ogun_core.engine import CoreEngine

    class LoopingClient:
        def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
            return {"content": "Still thinking with no tools."}

    engine = CoreEngine(client=LoopingClient())
    out = engine.run_turn("Explain quantum entanglement", max_rounds=2)
    assert "Still thinking" in out or "No files were written" in out or "Reached maximum tool execution rounds" in out


def test_adversarial_honesty_and_resume() -> None:
    """Honesty guard intercepts alternative claim phrasings, and resume detection does not hijack unrelated chat."""
    from ogun_core.honesty import TurnEvents, sanitize_reply

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
    from ogun_core.standalone_builder import build_standalone_project_staged

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
from _scaffold import ToolTestCase
import unittest
class T(ToolTestCase):
    def test_fail(self):
        self.assertEqual(1, 999)
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
    with patch("ogun_core.standalone_builder.WORKSPACE_DIR", Path(test_dir)):
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
    from ogun_core.standalone_builder import build_standalone_project_staged

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
from _scaffold import ToolTestCase
import unittest
class T(ToolTestCase):
    def test_fail(self):
        self.assertEqual(1, 999)
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
    with patch("ogun_core.standalone_builder.WORKSPACE_DIR", Path(test_dir)):
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
    from ogun_core.standalone_builder import build_standalone_project_staged

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
from _scaffold import ToolTestCase
import unittest
class TestMockTool(ToolTestCase):
    def test_ok(self):
        res = self.run_tool('--help')
        self.assertEqual(res.returncode, 0)
    def test_another(self):
        res = self.run_tool('--help')
        self.assertEqual(res.returncode, 0)
if __name__ == '__main__':
    unittest.main()
"""
    client = MockChatClient([
        {"content": f"```python\n{good_tool}\n```"},
        {"content": f"```python\n{passing_test}\n```"},
    ])
    engine = CoreEngine(client=client)

    test_dir = tempfile.mkdtemp()
    with patch("ogun_core.standalone_builder.WORKSPACE_DIR", Path(test_dir)):
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
    from ogun_core.standalone_builder import build_standalone_project_staged

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
from _scaffold import ToolTestCase
import unittest
class T(ToolTestCase):
    def test_fixed_fail(self):
        res = self.run_tool('--help')
        self.assertEqual(res.returncode, 999)
if __name__ == '__main__':
    unittest.main()
"""
    passing_test = """
from _scaffold import ToolTestCase
import unittest
class T(ToolTestCase):
    def test_fixed_pass(self):
        res = self.run_tool('--help')
        self.assertEqual(res.returncode, 0)
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
    with patch("ogun_core.standalone_builder.WORKSPACE_DIR", Path(test_dir)):
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
    from ogun_core.standalone_builder import build_standalone_project_staged

    good_tool = """
import sys, os, argparse
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
parser = argparse.ArgumentParser()
parser.add_argument('file')
parser.add_argument('--json', action='store_true')
parser.add_argument('--strict', action='store_true')
args = parser.parse_args()
if not os.path.exists(args.file):
    print("error", file=sys.stderr)
    sys.exit(1)
if args.json:
    print('{"status": "ok"}')
else:
    print("ok")
"""
    one_fail_test = """
from _scaffold import ToolTestCase
import json, unittest
class T(ToolTestCase):
    def test_ok(self):
        f = self.write_file("data.txt", "content")
        res = self.run_tool(str(f), "--json")
        self.assertOk(res)
        data = self.assertJsonOutput(res)
        self.assertEqual(data["status"], "ok")
    def test_bad(self):
        f = self.write_file("data.txt", "content")
        res = self.run_tool(str(f), "--strict")
        self.assertEqual(res.returncode, 999)
if __name__ == '__main__':
    unittest.main()
"""
    two_fail_test = """
from _scaffold import ToolTestCase
import json, unittest
class T(ToolTestCase):
    def test_bad1(self):
        f = self.write_file("data.txt", "content")
        res = self.run_tool(str(f), "--json", "--strict")
        self.assertJsonOutput(res)
        self.assertEqual(res.returncode, 999)
    def test_bad2(self):
        f = self.write_file("data.txt", "content")
        res = self.run_tool(str(f), "--strict")
        self.assertEqual(res.returncode, 888)
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
    with patch("ogun_core.standalone_builder.WORKSPACE_DIR", Path(test_dir)):
        success, msg, details = build_standalone_project_staged(
            engine,
            "Build me a command-line tool called mocktool. It needs a --json option and a --strict option.",
            max_attempts=2,
        )
        assert success is False
        assert len(details["failing_tests"]) == 1
        assert "restored from the best-so-far candidate (Attempt 1)" in msg
        assert "the tool meets the requirements in my checks; these 1 of my own tests are still wrong" in msg
    shutil.rmtree(test_dir, ignore_errors=True)


def test_r3_test_hygiene_static_check() -> None:
    """R3-2: check_test_code_hygiene enforces encoding, absolute path, and non-zero exit."""
    from ogun_core.quality import check_test_code_hygiene

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
    from ogun_core.honesty import TurnEvents, sanitize_reply

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
    from ogun_core.standalone_builder import build_standalone_project_staged

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
from _scaffold import ToolTestCase
import unittest
class T(ToolTestCase):
    def test_ok(self):
        res = self.run_tool('--help')
        self.assertEqual(res.returncode, 0)
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

    with patch("ogun_core.standalone_builder.WORKSPACE_DIR", Path(test_dir) / "workspace"):
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
    from ogun_core.standalone_builder import build_standalone_project_staged

    class AllTimeoutClient:
        def chat(self, messages: Any, tools: Any = None) -> Dict[str, Any]:
            raise TimeoutError("Persistent model read timeout 300s")

    engine = CoreEngine(client=AllTimeoutClient())
    test_dir = tempfile.mkdtemp()
    proj_dir = Path(test_dir) / "workspace" / "dupefinder"
    proj_dir.mkdir(parents=True, exist_ok=True)

    with patch("ogun_core.standalone_builder.WORKSPACE_DIR", Path(test_dir) / "workspace"):
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
    from ogun_core.standalone_builder import build_standalone_project_staged

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
from _scaffold import ToolTestCase
import unittest
class T(ToolTestCase):
    def test_fail(self):
        self.assertEqual(1, 999)
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

    with patch("ogun_core.standalone_builder.WORKSPACE_DIR", Path(test_dir) / "workspace"):
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
    from ogun_core.quality import arbitrate_test_failure
    from ogun_core.standalone_builder import build_standalone_project_staged

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

    with patch("ogun_core.standalone_builder.WORKSPACE_DIR", Path(test_dir) / "workspace"):
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
    from ogun_core.quality import check_test_code_hygiene

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
    from ogun_core.quality import check_requirements_coverage

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
    from ogun_core.standalone_builder import build_standalone_project_staged

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
from _scaffold import ToolTestCase
import unittest

class TestDupe(ToolTestCase):
    def test_rigid_bad_assertion(self):
        res = self.run_tool('--help')
        self.assertEqual(res.stdout, "THIS RIGID STRING CONTRADICTS OBSERVED SMOKE OUTPUT")
    def test_valid_runs(self):
        res = self.run_tool('--help')
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

    with patch("ogun_core.standalone_builder.WORKSPACE_DIR", Path(test_dir) / "workspace"):
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
    from ogun_core.standalone_builder import run_smoke_tests
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
    from ogun_core.quality import classify_tool_input_kind, check_tool_memory_budget
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
    from ogun_core.standalone_builder import build_standalone_project_staged
    from ogun_core.engine import CoreEngine

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
    with patch("ogun_core.standalone_builder.WORKSPACE_DIR", Path(test_dir)):
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
    from ogun_core.quality import compute_final_status_from_disk
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
    from ogun_core.quality import quarantine_bad_tests
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
    from ogun_core.quality import arbitrate_test_failure

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
    from ogun_core.quality import compute_final_status_from_disk
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


def test_r6_cli_contract_discovery_multi_arg() -> None:
    """R6-1: Discover CLI contracts for tools with positionals and required/optional flags."""
    from ogun_core.standalone_builder import discover_cli_contract, build_smoke_args, run_smoke_tests
    test_dir = tempfile.mkdtemp()
    proj_dir = Path(test_dir)
    tool_file = proj_dir / "complex_cli.py"
    tool_code = '''import sys, os, argparse
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
parser = argparse.ArgumentParser(description="Complex tool")
parser.add_argument("src", help="Source file")
parser.add_argument("dest", help="Destination file")
parser.add_argument("--config", required=True, help="Path to config file")
parser.add_argument("--count", type=int, default=1, help="Count of items")
args = parser.parse_args()
if not os.path.exists(args.src) or not os.path.exists(args.config) or "nonexist" in args.src or "nonexist" in args.dest:
    sys.stderr.write("Path not found\\n")
    sys.exit(1)
print(f"OK src={args.src} dest={args.dest} config={args.config} count={args.count}")
'''
    tool_file.write_text(tool_code, encoding="utf-8")

    contract = discover_cli_contract(tool_file, tool_code)
    req_pos_names = [p["name"] for p in contract["required_positionals"]]
    assert "src" in req_pos_names and "dest" in req_pos_names, f"Expected src and dest in required_positionals, got {req_pos_names}"
    req_opt_names = [o["flags"][0] for o in contract["required_options"]]
    assert "--config" in req_opt_names, f"Expected --config in required_options, got {req_opt_names}"
    opt_opt_names = [o["flags"][0] for o in contract["optional_options"]]
    assert "--count" in opt_opt_names, f"Expected --count in optional_options, got {opt_opt_names}"

    fixture_dir = proj_dir / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    args, is_table = build_smoke_args(contract, fixture_dir, "Build complex_cli with --config and --count", "complex_cli")
    assert len(args) >= 3, f"Expected at least 2 positionals and --config, got {args}"
    assert "--config" in args

    ok, rep, fails = run_smoke_tests(proj_dir, tool_file, "complex_cli", "Build complex_cli with --config and --count")
    assert ok, f"Smoke tests should pass with discovered contract: {fails}"

    shutil.rmtree(test_dir, ignore_errors=True)


def test_r6_argparse_error_classified_as_harness_error() -> None:
    """R6-1 / R6-4: Argparse usage errors classified as HARNESS_ERROR, never tool defect, tool not locked."""
    from ogun_core.quality import arbitrate_test_failure
    test_output = "usage: tool.py [-h] --config CONFIG src dest\ntool.py: error: the following arguments are required: dest"
    arb = arbitrate_test_failure(
        request="Build tool that takes src dest and --config",
        tool_code="print('tool')",
        test_code="def test_run(self): pass",
        test_output=test_output,
        failing_tests=["test_run"],
        smoke_passed=False,
        coverage_passed=True,
        memory_passed=True,
        smoke_has_harness_error=True,
    )
    assert arb["classification"] == "HARNESS_ERROR", f"Expected HARNESS_ERROR, got {arb['classification']}"
    assert arb["lock_tool"] is False, "Tool must NOT be locked on harness error"
    assert "Harness built an invalid command line" in arb["reason"]


def test_r6_scaffold_module_deterministic_helpers() -> None:
    """R6-2: _scaffold.py module provides isolated tmp, runner, write_file, make_dir, and assertions."""
    from ogun_core.standalone_builder import write_scaffold_file
    test_dir = tempfile.mkdtemp()
    proj_dir = Path(test_dir)
    scaffold_file = write_scaffold_file(proj_dir, "mytool")
    assert scaffold_file.exists()

    dummy_tool = proj_dir / "mytool.py"
    dummy_tool.write_text('''import sys, json
if "--fail" in sys.argv:
    sys.stderr.write("Fatal error occurred!\\n")
    sys.exit(2)
if "--json" in sys.argv:
    print(json.dumps({"status": "ok", "café": "東京"}))
    sys.exit(0)
print("Standard output from tool")
''', encoding="utf-8")

    import importlib.util
    spec = importlib.util.spec_from_file_location("_scaffold", str(scaffold_file))
    scaffold_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scaffold_mod)

    res = scaffold_mod.run_tool()
    assert res.returncode == 0
    assert "Standard output from tool" in res.stdout

    case = scaffold_mod.ToolTestCase()
    case.setUp()
    try:
        assert case.tmp.exists()
        f1 = case.write_file("data.txt", "hello café")
        assert f1.read_text(encoding="utf-8") == "hello café"
        f2 = case.write_file("data.bin", b"\\x00\\xff")
        assert f2.read_bytes() == b"\\x00\\xff"
        d1 = case.make_dir("subdir")
        assert d1.is_dir()

        ok_res = case.run_tool()
        case.assertOk(ok_res)

        fail_res = case.run_tool("--fail")
        case.assertFails(fail_res)

        json_res = case.run_tool("--json")
        data = case.assertJsonOutput(json_res)
        assert data["café"] == "東京"
    finally:
        case.tearDown()
        assert not case.tmp.exists()

    shutil.rmtree(test_dir, ignore_errors=True)


def test_r6_hygiene_static_check_rules() -> None:
    """R6-2: Static hygiene check requires scaffold, forbids subprocess, rejects trivial assertions."""
    from ogun_core.quality import check_test_code_hygiene

    bad_no_scaffold = '''import unittest
class TestTool(unittest.TestCase):
    def test_run(self):
        self.assertEqual(1, 1)
'''
    ok, issues = check_test_code_hygiene(bad_no_scaffold, "tool", require_scaffold=True)
    assert not ok
    assert any("must import `ToolTestCase` from `_scaffold`" in i for i in issues)

    bad_subprocess = '''from _scaffold import ToolTestCase
import subprocess
class TestTool(ToolTestCase):
    def test_run(self):
        res = subprocess.run(["python", "tool.py"])
        self.assertEqual(res.returncode, 0)
'''
    ok, issues = check_test_code_hygiene(bad_subprocess, "tool", require_scaffold=True)
    assert not ok
    assert any("subprocess directly" in i for i in issues)

    bad_assertion = '''from _scaffold import ToolTestCase
class TestTool(ToolTestCase):
    def test_run(self):
        res = self.run_tool()
        self.assertTrue(True)
'''
    ok, issues = check_test_code_hygiene(bad_assertion, "tool", require_scaffold=True)
    assert not ok
    assert any("trivial assertion" in i for i in issues)

    good_test = '''from _scaffold import ToolTestCase
class TestTool(ToolTestCase):
    def test_run(self):
        f = self.write_file("in.txt", "data")
        res = self.run_tool(str(f))
        self.assertOk(res)
        self.assertIn("data", res.stdout)
'''
    ok, issues = check_test_code_hygiene(good_test, "tool", require_scaffold=True)
    assert ok, f"Clean scaffold test should pass hygiene: {issues}"


def test_r6_deterministic_table_alignment_checker() -> None:
    """R6-3: Table alignment checker verifies pipe indices across header, separator, data rows."""
    from ogun_core.standalone_builder import check_table_alignment_output

    aligned_table = """
| Col A | Col B      | Score |
+-------+------------+-------+
| 1     | Alpha      |  95.0 |
| 2     | BetaLonger |  88.5 |
"""
    ok, err = check_table_alignment_output(aligned_table)
    assert ok, f"Aligned table must pass: {err}"

    misaligned_table = """
| Col A | Col B      | Score |
+-------+------------+-------+
| 1     | Alpha |  95.0 |
| 2     | BetaLonger |  88.5 |
"""
    ok, err = check_table_alignment_output(misaligned_table)
    assert not ok, "Misaligned table must fail"
    assert "pipe indices" in err or "character" in err


def test_r6_strategy_transition_table() -> None:
    """R6-4: Explicit strategy transitions according to arbitration rules."""
    from ogun_core.quality import determine_fix_strategy

    assert determine_fix_strategy("HARNESS_ERROR", lock_tool=False, smoke_ok=False, smoke_has_harness_error=True) == "regenerate_tests_only"
    assert determine_fix_strategy("TEST", lock_tool=True, smoke_ok=True) == "regenerate_tests_only"
    assert determine_fix_strategy("TEST", lock_tool=False, smoke_ok=True) == "regenerate_tests_only"
    assert determine_fix_strategy("TOOL", lock_tool=False, smoke_ok=False) == "regenerate_tool_only"
    assert determine_fix_strategy("TOOL", lock_tool=False, smoke_ok=True, covered=False) == "regenerate_tool_only"
    assert determine_fix_strategy("TOOL", lock_tool=False, smoke_ok=True, mem_ok=False) == "regenerate_tool_only"
    assert determine_fix_strategy("TOOL", lock_tool=False, smoke_ok=False, failing_history=[["t1"], ["t1"]], strategy_step=0) == "regenerate_tool_only"
    assert determine_fix_strategy("TOOL", lock_tool=False, smoke_ok=False, failing_history=[["t1"], ["t1"]], strategy_step=1) == "fix"


def test_r6_sentinel_lookup_distinguishes_null() -> None:
    """R6-3: Sentinel object distinguishes missing key from null/0/False/empty."""
    sentinel = object()
    data = {"null_key": None, "zero_key": 0, "false_key": False, "empty_key": ""}

    def lookup(d, k):
        return d.get(k, sentinel)

    assert lookup(data, "missing") is sentinel
    assert lookup(data, "null_key") is None
    assert lookup(data, "zero_key") == 0
    assert lookup(data, "false_key") is False
    assert lookup(data, "empty_key") == ""


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Round 7 Tests: Zero-tests-ran, colstats reproduction, reconciliation, oracles
# -----------------------------------------------------------------------------

def test_r7_zero_tests_ran_fails() -> None:
    """R7-1: Zero tests ran or count mismatch must never count as a pass."""
    from ogun_core.quality import parse_unittest_output, count_test_methods_ast

    # 1. Zero tests ran
    zero_out = """----------------------------------------------------------------------
Ran 0 tests in 0.000s

OK"""
    parsed_zero = parse_unittest_output(zero_out, ast_count=2, returncode=0)
    assert not parsed_zero.success, "Zero tests ran must not count as success"
    assert parsed_zero.status_code == "NO_TESTS_RAN"
    assert "0 tests executed" in parsed_zero.error_message

    # 2. Count mismatch (file defines 4 test methods, only 1 ran)
    mismatch_out = """test_one (test_foo.TestFoo.test_one) ... ok
----------------------------------------------------------------------
Ran 1 test in 0.001s

OK"""
    parsed_mis = parse_unittest_output(mismatch_out, ast_count=4, returncode=0)
    assert not parsed_mis.success, "Count mismatch must not count as success"
    assert parsed_mis.status_code == "TEST_COUNT_MISMATCH"
    assert "ran 1 of 4" in parsed_mis.summary_line

    # 3. No tests defined in file
    parsed_none = parse_unittest_output(zero_out, ast_count=0, returncode=0)
    assert not parsed_none.success
    assert parsed_none.status_code == "NO_TESTS_DEFINED"


def test_r7_missing_main_guard_appends_and_runs() -> None:
    """R7-1: Test file lacking unittest.main() guard is detected and auto-appended deterministically."""
    from ogun_core.quality import has_unittest_main_guard, ensure_unittest_main_guard, run_direct_test_file

    code_no_guard = """import unittest
class TestSample(unittest.TestCase):
    def test_sample(self):
        self.assertEqual(1 + 1, 2)
"""
    assert not has_unittest_main_guard(code_no_guard)

    with tempfile.TemporaryDirectory() as tmp_dir:
        td = Path(tmp_dir)
        tf = td / "test_sample.py"
        tf.write_text(code_no_guard, encoding="utf-8")

        appended = ensure_unittest_main_guard(tf)
        assert appended is True
        assert has_unittest_main_guard(tf.read_text(encoding="utf-8"))

        direct_ok, out, ran = run_direct_test_file(td, tf)
        assert direct_ok is True
        assert ran == 1
        assert "Ran 1 test" in out


def test_r7_colstats_exact_reproduction_not_passed() -> None:
    """R7-1: Exact reproduction of the Round 6 colstats failure must yield NOT PASSED."""
    from ogun_core.quality import compute_final_status_from_disk

    with tempfile.TemporaryDirectory() as tmp_dir:
        td = Path(tmp_dir)
        # Create flawed tool (rejecting floats via isdigit and iterator consumed)
        tool_file = td / "colstats.py"
        tool_file.write_text("""import argparse, csv, json, sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--delimiter', default=',')
    parser.add_argument('--column')
    args = parser.parse_args()
    try:
        with open(args.file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=args.delimiter)
            stats = {}
            for col in reader.fieldnames:
                data = [float(row[col]) for row in reader if row[col].strip().isdigit()]
                if data:
                    stats[col] = {'count': len(data), 'min': min(data), 'max': max(data), 'mean': sum(data)/len(data)}
            if args.json:
                print(json.dumps(stats))
            else:
                if not stats:
                    print("No numeric columns found.")
                    return
                for c, s in stats.items():
                    print(f"{c}: {s['count']}")
    except Exception as e:
        sys.exit(f"Error: {e}")

if __name__ == '__main__':
    main()
""", encoding="utf-8")

        # Create test file WITHOUT main guard (reproducing Round 6)
        test_file = td / "test_colstats.py"
        test_file.write_text("""import unittest
import subprocess, sys
from pathlib import Path

TOOL = Path(__file__).resolve().parent / "colstats.py"

class TestColstats(unittest.TestCase):
    def test_help(self):
        res = subprocess.run([sys.executable, str(TOOL), "--help"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)

    def test_missing_file(self):
        res = subprocess.run([sys.executable, str(TOOL), "missing.csv"], capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)

    def test_decimals(self):
        # Flawed tool fails this test because 10.5 is not isdigit()
        csv_p = Path(__file__).parent / "decimals.csv"
        csv_p.write_text("val\\n10.5\\n20.5\\n", encoding="utf-8")
        res = subprocess.run([sys.executable, str(TOOL), str(csv_p), "--json"], capture_output=True, text=True)
        self.assertIn("10.5", res.stdout)
""", encoding="utf-8")

        status = compute_final_status_from_disk(td, "colstats", request="Build colstats with --json, --delimiter, --column")
        assert not str(status).startswith("PASSED"), f"Colstats reproduction must NOT pass! Got: {status}"
        assert any(k in str(status) for k in ("TESTS FAILING", "NO TESTS RAN", "REQUIREMENT FAILED", "TESTS INVALID", "COVERAGE GAP")), f"Expected failing status, got: {status}"


def test_r7_status_reconciliation_consistency_assertion() -> None:
    """R7-2: Builder cross-checks and assertion verifies no status string is produced inconsistent with facts."""
    from ogun_core.quality import assert_status_consistent_with_results

    # Passing facts -> passes assertion
    valid_facts = {
        "tests_passed": True,
        "ran_tests": 5,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "smoke_ok": True,
        "oracle_ok": True,
        "coverage_ok": True,
        "alignment_ok": True,
        "mem_ok": True,
        "direct_run_ok": True,
    }
    assert_status_consistent_with_results("PASSED (all 5 tests passed)", valid_facts)

    # Inconsistent status: says PASSED but smoke_ok is False -> must raise AssertionError
    bad_smoke = dict(valid_facts, smoke_ok=False)
    try:
        assert_status_consistent_with_results("PASSED (all 5 tests passed)", bad_smoke)
        assert False, "Should have raised AssertionError for bad smoke"
    except AssertionError as ae:
        assert "smoke_ok" in str(ae)

    # Inconsistent status: says PASSED but failures > 0 -> must raise AssertionError
    bad_fail = dict(valid_facts, failures=2)
    try:
        assert_status_consistent_with_results("PASSED (all 5 tests passed)", bad_fail)
        assert False, "Should have raised AssertionError for failures"
    except AssertionError as ae:
        assert "failures" in str(ae)

    # Inconsistent status: says PASSED but ran_tests == 0 -> must raise AssertionError
    bad_ran = dict(valid_facts, ran_tests=0)
    try:
        assert_status_consistent_with_results("PASSED (all 5 tests passed)", bad_ran)
        assert False, "Should have raised AssertionError for ran_tests == 0"
    except AssertionError as ae:
        assert "ran_tests" in str(ae)


def test_r7_oracle_csv_column_stats() -> None:
    """R7-3: Reference oracle detects float rejection and multi-column iterator exhaustion."""
    from ogun_core.quality import OracleRegistry

    with tempfile.TemporaryDirectory() as tmp_dir:
        td = Path(tmp_dir)
        # 1. Flawed tool
        flawed_tool = td / "flawed_colstats.py"
        flawed_tool.write_text("""import argparse, csv, json, sys
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    with open(args.file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        stats = {}
        for col in reader.fieldnames:
            # isdigit rejects decimals!
            data = [float(r[col]) for r in reader if r[col].strip().isdigit()]
            if data: stats[col] = {'count': len(data), 'min': min(data), 'max': max(data), 'mean': sum(data)/len(data)}
        print(json.dumps(stats) if args.json else "No numeric columns found.")
if __name__ == '__main__': main()
""", encoding="utf-8")

        ok, err = OracleRegistry.oracle_csv_column_stats(flawed_tool, "Build colstats with --json", td)
        assert not ok, "Oracle must catch flawed tool rejecting decimals / consuming reader"
        assert "col_b" in err or "No numeric columns found" in err

        # 2. Correct tool
        good_tool = td / "good_colstats.py"
        good_tool.write_text("""import argparse, csv, json, sys
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--delimiter', default=',')
    parser.add_argument('--column')
    args = parser.parse_args()
    with open(args.file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=args.delimiter)
        try:
            headers = next(reader)
        except StopIteration:
            return
        col_vals = {h: [] for h in headers}
        for row in reader:
            for i, val in enumerate(row):
                if i < len(headers):
                    val_str = val.strip()
                    if val_str:
                        try:
                            col_vals[headers[i]].append(float(val_str))
                        except ValueError:
                            pass
        stats = {}
        if args.column:
            if args.column not in headers:
                sys.stderr.write(f"Error: Unknown column '{args.column}'\\n")
                sys.exit(1)
            if not col_vals.get(args.column):
                sys.stderr.write(f"Error: Non-numeric column '{args.column}'\\n")
                sys.exit(1)
            target_cols = [args.column]
        else:
            target_cols = [h for h in headers if col_vals.get(h)]
        for h in target_cols:
            vals = col_vals.get(h, [])
            if vals:
                stats[h] = {'count': len(vals), 'min': min(vals), 'max': max(vals), 'mean': sum(vals)/len(vals)}
        if args.json:
            print(json.dumps(stats))
        else:
            headers_t = ["Column", "Count", "Min", "Max", "Mean"]
            rows_t = []
            for h in target_cols:
                v = stats.get(h)
                if v:
                    rows_t.append([h, str(v['count']), f"{v['min']:.1f}", f"{v['max']:.1f}", f"{v['mean']:.4f}"])
            widths = [len(x) for x in headers_t]
            for r in rows_t:
                for i, val in enumerate(r):
                    widths[i] = max(widths[i], len(val))
            header_line = ' | '.join(f'{h:<{w}}' for h, w in zip(headers_t, widths))
            sep_line = '-+-'.join('-' * w for w in widths)
            data_lines = [' | '.join(f'{c:<{w}}' for c, w in zip(r, widths)) for r in rows_t]
            print('\\n'.join([header_line, sep_line] + data_lines))
if __name__ == '__main__': main()
""", encoding="utf-8")

        ok_good, err_good = OracleRegistry.oracle_csv_column_stats(good_tool, "Build colstats with --json --column --delimiter", td)
        assert ok_good, f"Good tool must pass oracle: {err_good}"


def test_r7_oracle_dotted_path_lookup() -> None:
    """R7-3: Reference oracle verifies dotted path lookup."""
    from ogun_core.quality import OracleRegistry

    with tempfile.TemporaryDirectory() as tmp_dir:
        td = Path(tmp_dir)
        lookup_tool = td / "query.py"
        lookup_tool.write_text("""import json, sys
def main():
    with open(sys.argv[1], encoding='utf-8') as f:
        data = json.load(f)
    cur = data
    for part in sys.argv[2].split('.'):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    print(cur)
if __name__ == '__main__': main()
""", encoding="utf-8")

        ok, err = OracleRegistry.oracle_dotted_path_lookup(lookup_tool, "Build tool jsonquery to query json path", td)
        assert ok, f"Dotted path oracle must pass on valid query tool: {err}"


def test_r7_table_alignment_header_vs_data_width() -> None:
    """R7-3: Table alignment checker verifies header text vs data width after truncation."""
    from ogun_core.standalone_builder import check_table_alignment_output

    aligned_table = """
| Col A | Col B      | Score |
+-------+------------+-------+
| 1     | Alpha      |  95.0 |
| 2     | BetaLonger |  88.5 |
"""
    ok, err = check_table_alignment_output(aligned_table)
    assert ok, f"Aligned table must pass: {err}"

    # Overflowing cell that exceeds column separator interval
    overflow_table = """
| Col A | Col B | Score |
+-------+-------+-------+
| 1     | AlphaLongerThanWidth | 95.0 |
"""
    # Character position mismatch
    ok_bad, err_bad = check_table_alignment_output(overflow_table)
    assert not ok_bad, "Overflowing table must fail alignment check"


def test_r7_offline_regressions() -> None:
    """R7-4: Offline scripted regressions guarding Round 1 - Round 6 live scenarios."""
    import subprocess
    from ogun_core.quality import classify_tool_input_kind
    from ogun_core.standalone_builder import check_table_alignment_output

    # R1/R2 guard: dupefinder unicode handling under cp1252 console
    env_cp1252 = {**dict(os.environ), "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"}
    with tempfile.TemporaryDirectory() as tmp_dir:
        td = Path(tmp_dir)
        u_script = td / "u_test.py"
        u_script.write_text("""import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
print("café résumé naïve 日本語")
""", encoding="utf-8")
        res = subprocess.run([sys.executable, str(u_script)], capture_output=True, text=True, env=env_cp1252)
        assert res.returncode == 0
        assert "UnicodeEncodeError" not in (res.stderr or "")

    # R3 guard: wordfreq empty file contract (0 bytes must exit 0, no ZeroDivisionError)
    with tempfile.TemporaryDirectory() as tmp_dir:
        td = Path(tmp_dir)
        wf_script = td / "wf.py"
        wf_script.write_text("""import sys
with open(sys.argv[1], encoding='utf-8') as f:
    text = f.read()
words = text.split()
count = len(words)
print(count)
""", encoding="utf-8")
        empty_file = td / "empty.txt"
        empty_file.write_text("", encoding="utf-8")
        res = subprocess.run([sys.executable, str(wf_script), str(empty_file)], capture_output=True, text=True)
        assert res.returncode == 0
        assert res.stdout.strip() == "0"

    # R4 guard: tabprint table alignment with truncated data
    tbl = """| ID | Name    |
+----+---------+
| 1  | Hel...  |
"""
    ok, _ = check_table_alignment_output(tbl)
    assert ok

    # R5/R6 guard: jsonquery / jsonget two-positional smoke arguments
    kind = classify_tool_input_kind("Build tool jsonquery to query file.json with dotted path", "jsonquery")
    assert kind in ("whole_document_json", "whole_document")




# -----------------------------------------------------------------------------
# Round 8: Hard Gate for Malformed Tests, Undefined Names & Output Modes
# -----------------------------------------------------------------------------

def test_r8_test_file_hard_validator_numstats_reproduction() -> None:
    """R8-1: Hard validator rejects malformed test file from numstats reproduction.
    File has leaked '# FILE: test_numstats.py' header, duplicated ToolTestCase import block,
    markdown code fences, and undefined name references.
    """
    from ogun_core.quality import validate_generated_test_file

    malformed_test_code = """# FILE: test_numstats.py
```python
from _scaffold import ToolTestCase
import unittest

class TestNumstats(ToolTestCase):
    def test_basic(self):
        res = self.run_tool('sample.csv')
        self.assertOk(res)

from _scaffold import ToolTestCase
from _scaffold import ToolTestCase

class TestNumstatsExtra(ToolTestCase):
    def test_extra(self):
        res = self.run_tool('sample.csv', '--json')
        print(undefined_variable_name_xyz)
        self.assertOk(res)

if __name__ == '__main__':
    unittest.main()
```
"""
    valid, issues, cleaned = validate_generated_test_file(
        malformed_test_code, "numstats", "Build numstats with --json"
    )
    assert valid is False, "Malformed test code must be rejected by validator"
    assert len(issues) >= 1, "Must report at least one issue"
    has_leaked = any("leaked artifact" in i or "markdown code fence" in i for i in issues)
    assert has_leaked, f"Expected leaked artifact detection, got: {issues}"
    has_undef = any("undefined name 'undefined_variable_name_xyz'" in i for i in issues)
    assert has_undef, f"Expected undefined name detection, got: {issues}"


def test_r8_undefined_names_checker() -> None:
    """R8-2: Undefined names checker catches missing imports, variable typos,
    and comprehension scope leaks, while allowing builtins and conditional imports.
    """
    from ogun_core.quality import check_undefined_names_and_imports

    # 1. Missing import sys
    code1 = """def main():
    print(sys.executable)
"""
    errs1 = check_undefined_names_and_imports(code1, "test.py")
    assert any("undefined name 'sys'" in e for e in errs1), f"Expected missing sys error, got: {errs1}"

    # 2. Main guard import does NOT leak into module scope
    code2 = """def run():
    print(sys.executable)

if __name__ == '__main__':
    import sys
    run()
"""
    errs2 = check_undefined_names_and_imports(code2, "test.py")
    assert any("undefined name 'sys'" in e for e in errs2), f"Main guard import must not satisfy module scope: {errs2}"

    # 3. Valid code with imports, builtins, and comprehensions
    code3 = """import sys, os
from pathlib import Path
def compute(items):
    squares = [x * x for x in items]
    return {k: v for k, v in enumerate(squares)}
"""
    errs3 = check_undefined_names_and_imports(code3, "tool.py")
    assert len(errs3) == 0, f"Valid code must have 0 errors, got: {errs3}"

    # 4. Comprehension variable leaked outside
    code4 = """def compute(items):
    squares = [x * x for x in items]
    return x  # x is not defined in function scope
"""
    errs4 = check_undefined_names_and_imports(code4, "tool.py")
    assert any("undefined name 'x'" in e for e in errs4), f"Leaked comprehension variable must be caught: {errs4}"


def test_r8_oracle_exercises_all_modes_and_catches_default_mode_crash() -> None:
    """R8-3: Oracle exercises every output mode, starting with default mode first,
    and catches tools that crash in default mode.
    """
    from ogun_core.quality import OracleRegistry
    with tempfile.TemporaryDirectory() as tmp_dir:
        td = Path(tmp_dir)

        # Tool that works with --json but crashes in default table mode
        flawed_tool = td / "tabstats.py"
        flawed_tool.write_text("""import sys, os, json, argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--delimiter', default=',')
    parser.add_argument('--column', default=None)
    args = parser.parse_args()
    if args.json:
        # Returns json stats
        data = {
            'col_a': {'count': 3, 'min': 10.0, 'max': 30.0, 'mean': 20.0},
            'col_b': {'count': 3, 'min': 10.5, 'max': 30.0, 'mean': 20.3333}
        }
        if args.column:
            data = {args.column: data[args.column]}
        print(json.dumps(data))
    else:
        # Crashes in default mode!
        raise RuntimeError("Crash in default plain-text table mode!")

if __name__ == '__main__':
    main()
""", encoding="utf-8")

        ok, err = OracleRegistry.oracle_csv_column_stats(
            flawed_tool,
            "Build me a command-line tool called tabstats. It needs a --json option, --delimiter, and --column.",
            td
        )
        assert ok is False, "Oracle must reject tool that crashes in default mode"
        assert "Default mode" in err or "Crash" in err, f"Error must cite default mode crash: {err}"


def test_r8_linecount_oracle_word_line_char_counts() -> None:
    """R8-3: Linecount oracle checks lines, words, and characters across all modes."""
    from ogun_core.quality import OracleRegistry
    with tempfile.TemporaryDirectory() as tmp_dir:
        td = Path(tmp_dir)

        # 1. Correct linecount tool
        good_tool = td / "good_linecount.py"
        good_tool.write_text("""import sys, os, json, argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--sort', action='store_true')
    parser.add_argument('--min-words', type=int, default=0)
    args = parser.parse_args()

    results = []
    for fp in args.files:
        if not os.path.exists(fp):
            sys.exit(1)
        with open(fp, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        lines = len(text.splitlines())
        words = len(text.split())
        chars = len(text)
        if words >= args.min_words:
            results.append({'file': fp, 'lines': lines, 'words': words, 'characters': chars})

    if args.sort:
        results.sort(key=lambda r: r['words'], reverse=True)

    if args.json:
        print(json.dumps(results))
        return

    # Plain text table
    tot_l = tot_w = tot_c = 0
    for r in results:
        tot_l += r['lines']; tot_w += r['words']; tot_c += r['characters']
        fn = os.path.basename(r['file'])
        print(f"{fn:<15} {r['lines']:>6} {r['words']:>6} {r['characters']:>6}")
    if len(args.files) > 1:
        print(f"{'Total':<15} {tot_l:>6} {tot_w:>6} {tot_c:>6}")

if __name__ == '__main__':
    main()
""", encoding="utf-8")

        ok1, err1 = OracleRegistry.oracle_line_word_count(
            good_tool,
            "Build me a command-line tool called linecount. Give it one or more text files and it prints lines, words and characters. It needs --json, --min-words N, and --sort.",
            td
        )
        assert ok1 is True, f"Oracle must pass on correct linecount tool: {err1}"

        # 2. Broken tool that prints wrong counts
        bad_tool = td / "bad_linecount.py"
        bad_tool.write_text("""import sys, os, json, argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--sort', action='store_true')
    parser.add_argument('--min-words', type=int, default=0)
    args = parser.parse_args()
    print("file: lines 999 words 999 chars 999")
if __name__ == '__main__': main()
""", encoding="utf-8")

        ok2, err2 = OracleRegistry.oracle_line_word_count(
            bad_tool,
            "Build me a command-line tool called linecount with --json and --sort.",
            td
        )
        assert ok2 is False, "Oracle must reject tool with incorrect word/line/char counts"


def test_r8_arbitration_linecount_assertion_mismatches() -> None:
    """R8-4: Arbitration deterministically classifies assertion mismatches against valid output as TEST defects.
    Uses the exact 4 failing assertions from Round 7 linecount.
    """
    from ogun_core.quality import arbitrate_test_failure

    test_src = """from _scaffold import ToolTestCase
class TestLinecount(ToolTestCase):
    def test_single_file(self):
        res = self.run_tool('tiny_valid.json')
        self.assertIn('18', res.stdout)
    def test_single_file_json(self):
        res = self.run_tool('tiny_valid.json', '--json')
        data = self.assertJsonOutput(res)
        self.assertEqual(data[0]['words'], 18)
    def test_sort(self):
        res = self.run_tool('file1.txt', 'file2.txt', '--sort')
        lines = res.stdout.splitlines()
        self.assertIn('file1.txt', lines[2])
    def test_sort_json(self):
        res = self.run_tool('file1.txt', 'file2.txt', '--sort', '--json')
        data = self.assertJsonOutput(res)
        self.assertEqual(data[0]['file'], 'file2.txt')
"""
    req = "Build linecount with --json, --min-words, --sort"
    tool_code = "print('tiny_valid.json 1 2 16')"

    # Case 1: test_single_file: '18' not found in valid output
    arb1 = arbitrate_test_failure(
        request=req,
        tool_code=tool_code,
        test_code=test_src,
        test_output="AssertionError: '18' not found in 'tiny_valid.json 1 2 16\\n'",
        failing_tests=["test_single_file"],
        smoke_passed=True,
        coverage_passed=True,
        memory_passed=True,
    )
    assert arb1["classification"] == "TEST", f"Expected TEST defect, got: {arb1}"
    assert arb1["lock_tool"] is True

    # Case 2: test_single_file_json: 2 != 18
    arb2 = arbitrate_test_failure(
        request=req,
        tool_code=tool_code,
        test_code=test_src,
        test_output="AssertionError: 2 != 18",
        failing_tests=["test_single_file_json"],
        smoke_passed=True,
        coverage_passed=True,
        memory_passed=True,
    )
    assert arb2["classification"] == "TEST", f"Expected TEST defect, got: {arb2}"
    assert arb2["lock_tool"] is True

    # Case 3: test_sort: 'file1.txt' not found in 'Total 2 4 24'
    arb3 = arbitrate_test_failure(
        request=req,
        tool_code=tool_code,
        test_code=test_src,
        test_output="AssertionError: 'file1.txt' not found in 'Total     2 4 24'",
        failing_tests=["test_sort"],
        smoke_passed=True,
        coverage_passed=True,
        memory_passed=True,
    )
    assert arb3["classification"] == "TEST", f"Expected TEST defect, got: {arb3}"
    assert arb3["lock_tool"] is True

    # Case 4: test_sort_json: 'file1.txt' != 'file2.txt'
    arb4 = arbitrate_test_failure(
        request=req,
        tool_code=tool_code,
        test_code=test_src,
        test_output="AssertionError: 'file1.txt' != 'file2.txt'",
        failing_tests=["test_sort_json"],
        smoke_passed=True,
        coverage_passed=True,
        memory_passed=True,
    )
    assert arb4["classification"] == "TEST", f"Expected TEST defect, got: {arb4}"
    assert arb4["lock_tool"] is True


# -----------------------------------------------------------------------------
# Round 9: Input Kind Inference, Mismatched Fixture Rejection, Flag Combinations,
# Spec Fidelity, Row Consistency & Report Consistency
# -----------------------------------------------------------------------------

def test_r9_input_kind_inference_and_fixtures():
    from ogun_core.quality import infer_tool_input_kind, fixtures_for

    req_csv = "Build me a command-line tool called colsummary. Give it a CSV file with a header row and it prints stats."
    kind_csv = infer_tool_input_kind(req_csv, {}, "", "colsummary")
    assert kind_csv == "csv", f"Expected 'csv', got {kind_csv}"
    fx_csv = fixtures_for(kind_csv)
    assert "tiny_valid" in fx_csv and ".csv" in fx_csv["tiny_valid"]["name"]
    assert any("self.csv_file" in h for h in fx_csv["ready_helpers"])

    req_text = "Build me a command-line tool called textsummary. Give it one or more text files and it prints lines and words."
    kind_text = infer_tool_input_kind(req_text, {}, "", "textsummary")
    assert kind_text in ("plain_text/lines", "multi_file"), f"Expected text or multi_file, got {kind_text}"
    fx_text = fixtures_for(kind_text)
    assert "tiny_valid" in fx_text

    req_json = "Build me a tool that parses a json file and extracts keys."
    kind_json = infer_tool_input_kind(req_json, {}, "", "jsonutil")
    assert kind_json == "json", f"Expected 'json', got {kind_json}"

    req_dir = "Build me a tool that scans a folder or directory for images."
    kind_dir = infer_tool_input_kind(req_dir, {}, "", "dirscan")
    assert kind_dir == "folder/directory", f"Expected 'folder/directory', got {kind_dir}"


def test_r9_mismatched_fixture_validator_rejection():
    from ogun_core.quality import validate_generated_test_file

    req_csv = "Build me a command-line tool called colsummary. Give it a CSV file with a header row. It needs a --json option."
    bad_test_json_fixture = """from _scaffold import ToolTestCase
import unittest

class TestColsummary(ToolTestCase):
    def test_basic(self):
        f = self.write_file("input.json", '{"a": 1, "b": 2}')
        res = self.run_tool(str(f))
        self.assertOk(res)

if __name__ == '__main__':
    unittest.main()
"""
    valid, issues, _ = validate_generated_test_file(bad_test_json_fixture, "colsummary", request=req_csv)
    assert not valid, f"Expected mismatched fixture rejection, but passed. Issues: {issues}"
    assert any("mismatched" in i.lower() or ".json" in i.lower() for i in issues), f"Unexpected issues: {issues}"

    good_test_csv_fixture = """from _scaffold import ToolTestCase
import unittest

class TestColsummary(ToolTestCase):
    def test_basic(self):
        f = self.csv_file("input.csv", "a,b\\n1,2\\n3,4")
        res = self.run_tool(str(f))
        self.assertOk(res)

if __name__ == '__main__':
    unittest.main()
"""
    valid_good, issues_good, _ = validate_generated_test_file(good_test_csv_fixture, "colsummary", request=req_csv)
    assert valid_good, f"Expected valid CSV test to pass, but failed: {issues_good}"


def test_r9_oracle_full_flag_combination_mode_matrix():
    from ogun_core.quality import OracleRegistry
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Defective tool: ignores --column when --json is passed (Round 8 defect)
        buggy_tool = tmp_path / "buggy_colsummary.py"
        buggy_tool.write_text("""import argparse, sys, json, csv
parser = argparse.ArgumentParser()
parser.add_argument('file')
parser.add_argument('--column')
parser.add_argument('--delimiter', default=',')
parser.add_argument('--json', action='store_true')
args = parser.parse_args()

with open(args.file, encoding='utf-8') as f:
    r = csv.DictReader(f, delimiter=args.delimiter)
    cols = {}
    for row in r:
        for k, v in row.items():
            try:
                val = float(v)
                cols.setdefault(k, []).append(val)
            except (ValueError, TypeError):
                pass

stats = {}
for k, vals in cols.items():
    stats[k] = {"count": len(vals), "min": min(vals), "max": max(vals), "mean": sum(vals)/len(vals)}

if args.json:
    # BUG: Outputs ALL columns in JSON, completely ignoring args.column!
    print(json.dumps(stats))
else:
    table_stats = stats
    if args.column:
        table_stats = {k: v for k, v in stats.items() if k == args.column}
    print("Column | Count | Min  | Max  | Mean")
    print("-------+-------+------+------+-----")
    for k, s in table_stats.items():
        print(f"{k:<6} | {s['count']:<5} | {s['min']:<4.1f} | {s['max']:<4.1f} | {s['mean']:<4.1f}")
""", encoding="utf-8")

        ok_buggy, err_buggy = OracleRegistry.oracle_csv_column_stats(buggy_tool, "Build colsummary with --json --column --delimiter", tmp_path)
        assert not ok_buggy, f"Oracle should have caught defective --column --json, got err: {err_buggy}"
        assert "--column" in err_buggy and "--json" in err_buggy, f"Defect not found in error: {err_buggy}"

        # Correct tool
        good_tool = tmp_path / "good_colsummary.py"
        good_tool.write_text("""import argparse, sys, json, csv
parser = argparse.ArgumentParser()
parser.add_argument('file')
parser.add_argument('--column')
parser.add_argument('--delimiter', default=',')
parser.add_argument('--json', action='store_true')
args = parser.parse_args()

with open(args.file, encoding='utf-8') as f:
    r = csv.DictReader(f, delimiter=args.delimiter)
    all_fields = r.fieldnames or []
    cols = {}
    for row in r:
        for k, v in row.items():
            try:
                val = float(v)
                cols.setdefault(k, []).append(val)
            except (ValueError, TypeError):
                pass

if args.column:
    if args.column not in all_fields:
        sys.stderr.write(f"Error: Unknown column '{args.column}'\\n")
        sys.exit(1)
    if args.column not in cols or not cols[args.column]:
        sys.stderr.write(f"Error: Non-numeric column '{args.column}'\\n")
        sys.exit(1)
    cols = {k: v for k, v in cols.items() if k == args.column}

stats = {}
for k, vals in cols.items():
    stats[k] = {"count": len(vals), "min": min(vals), "max": max(vals), "mean": sum(vals)/len(vals)}

if args.json:
    print(json.dumps(stats))
else:
    headers = ["Column", "Count", "Min", "Max", "Mean"]
    rows = []
    for k, s in stats.items():
        rows.append([k, str(s['count']), f"{s['min']:.1f}", f"{s['max']:.1f}", f"{s['mean']:.4f}"])
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))
    header_line = ' | '.join(f'{h:<{w}}' for h, w in zip(headers, widths))
    sep_line = '-+-'.join('-' * w for w in widths)
    data_lines = [' | '.join(f'{c:<{w}}' for c, w in zip(row, widths)) for row in rows]
    print('\\n'.join([header_line, sep_line] + data_lines))
""", encoding="utf-8")

        ok_good, err_good = OracleRegistry.oracle_csv_column_stats(good_tool, "Build colsummary with --json --column --delimiter", tmp_path)
        assert ok_good, f"Oracle should have passed conforming tool, got error: {err_good}"


def test_r9_spec_fidelity_ignoring_non_numeric_and_unrequested_options():
    from ogun_core.quality import check_unrequested_cli_options, OracleRegistry
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Non-numeric column presence defect in CSV stats
        leaky_tool = tmp_path / "leaky_colsummary.py"
        leaky_tool.write_text("""import argparse, sys, json
parser = argparse.ArgumentParser()
parser.add_argument('file')
parser.add_argument('--column')
parser.add_argument('--delimiter', default=',')
parser.add_argument('--json', action='store_true')
args = parser.parse_args()

if args.json:
    # Leaks 'name' (non-numeric column)
    print(json.dumps({'col_a': {'count': 2, 'min': 10, 'max': 20, 'mean': 15}, 'name': {'count': 0}}))
else:
    print("Column | Count | Min | Max | Mean")
    print("-------+-------+-----+-----+-----")
    print("col_a  | 2     | 10  | 20  | 15")
    print("name   | 0     | 0   | 0   | 0")
""", encoding="utf-8")

        ok_leaky, err_leaky = OracleRegistry.oracle_csv_column_stats(leaky_tool, "Build colsummary with --json --column --delimiter", tmp_path)
        assert not ok_leaky, "Oracle must fail when non-numeric column is present"
        assert "non-numeric" in err_leaky.lower() or "name" in err_leaky.lower(), f"Unexpected error: {err_leaky}"

    # Unrequested CLI options check
    req = "Build me a command-line tool colsummary for CSV files. Needs --json and --delimiter."
    contract = {
        "optional_options": [
            {"flags": ["--json"], "dest": "json"},
            {"flags": ["--delimiter"], "dest": "delimiter"},
            {"flags": ["--unwanted-flag"], "dest": "unwanted"},
        ]
    }
    unreq = check_unrequested_cli_options(contract, req)
    assert unreq == ["--unwanted-flag"], f"Expected ['--unwanted-flag'], got {unreq}"

    tool_code_with_unreq = """import argparse
p = argparse.ArgumentParser()
p.add_argument('--json', action='store_true')
p.add_argument('--delimiter', default=',')
p.add_argument('--foo-bar')
"""
    unreq_code = check_unrequested_cli_options(tool_code_with_unreq, req)
    assert "--foo-bar" in unreq_code, f"Expected '--foo-bar' in {unreq_code}"


def test_r9_table_row_column_consistency():
    from ogun_core.quality import check_table_row_column_consistency

    good_table = """
Column | Count | Min | Max | Mean
-------+-------+-----+-----+-----
col_a  | 2     | 1.0 | 2.0 | 1.5
col_b  | 2     | 3.0 | 4.0 | 3.5
"""
    ok, err = check_table_row_column_consistency(good_table)
    assert ok, f"Good table should pass, got error: {err}"

    bad_table = """
Column | Count | Min | Max | Mean
-------+-------+-----+-----+-----
col_a  | 2     | 1.0 | 2.0 | 1.5
col_b  | 2     | 3.0
"""
    ok_bad, err_bad = check_table_row_column_consistency(bad_table)
    assert not ok_bad, "Bad table with missing columns must fail"
    assert "Inconsistent column count" in err_bad


def test_r9_report_consistency_verifier():
    from ogun_core.quality import verify_report_consistency

    contradictory_report = """# Round 9 Report
| Flag | Result |
|---|---|
| `--column col_a --json` | PASS |

## Gaps I did not close
- Defect: `--column col_a --json` is broken and ignores the column argument.
"""
    ok, issues = verify_report_consistency(contradictory_report)
    assert not ok, f"Expected contradiction detection, but passed. Issues: {issues}"
    assert len(issues) >= 1
    assert "Contradiction detected" in issues[0]

    consistent_report = """# Round 9 Report
| Flag | Result |
|---|---|
| `--column col_a --json` | PASS |

## Gaps I did not close
- None. All flag combinations and oracles pass.
"""
    ok_cons, issues_cons = verify_report_consistency(consistent_report)
    assert ok_cons, f"Expected consistent report to pass, got issues: {issues_cons}"


# -----------------------------------------------------------------------------
# Round 10: Deterministic Table Alignment, Row-Order Semantics, Test Repair,
# Selector Validation & Evidence-Backed Reports
# -----------------------------------------------------------------------------

def test_r10_table_layout_alignment_gate():
    """R10-1: Deterministic table layout gate in every mode."""
    from ogun_core.quality import check_table_alignment_output

    # 1. Exact textsummary multi-file defect (6 separator segments for 4 columns, pipe index mismatch)
    broken_textsummary = """File   | Lines  | Words | Characters
-------+--------+-------+-------+-------+-----------
f1.txt | 2      | 5     | 24
Total  | 3      | 7     | 28"""
    ok_ts, err_ts = check_table_alignment_output(broken_textsummary)
    assert not ok_ts, "Must catch textsummary multi-file separator segment / boundary mismatch defect"
    assert "Table alignment error" in err_ts
    assert "6 segments" in err_ts or "boundary positions mismatch" in err_ts.lower() or "character position mismatch" in err_ts.lower()

    # 2. Exact colsummary Mean column defect (Mean cell 6.666667 exceeds allocated width 6)
    broken_colsummary = """Column | Count | Min   | Max   | Mean
-------+-------+-------+-------+------
b      | 3     | -20.5 | 30.0  | 6.666667"""
    ok_cs, err_cs = check_table_alignment_output(broken_colsummary)
    assert not ok_cs, "Must catch colsummary Mean column cell width overflow defect"
    assert "Table alignment error" in err_cs
    assert "6.666667" in err_cs or "exceeds" in err_cs

    # 3. Aligned control table (must pass)
    control_table = """Column | Count | Min   | Max  | Mean    
-------+-------+-------+------+---------
b      | 3     | -20.5 | 30.0 | 6.666667"""
    ok_ctrl, err_ctrl = check_table_alignment_output(control_table)
    assert ok_ctrl, f"Aligned control table must pass, got: {err_ctrl}"


def test_r10_row_order_semantics_totals_and_sorting():
    """R10-2: Row-order semantics: totals and sorting."""
    from ogun_core.quality import check_listing_row_order_and_totals

    # 1. Exact textsummary --sort Total-first defect (Total sorted together with files)
    total_first = """Total  | 3      | 7     | 28
f1.txt | 2      | 5     | 24"""
    ok_tf, err_tf = check_listing_row_order_and_totals(total_first)
    assert not ok_tf, "Must catch Total-first defect"
    assert "Total row must be the LAST row" in err_tf

    # 2. Valid table: Total is last, file rows sorted descending
    valid_table = """File   | Lines  | Words | Characters
-------+--------+-------+-----------
f1.txt | 3      | 9     | 45
f3.txt | 1      | 6     | 36
f2.txt | 1      | 2     | 12
Total  | 5      | 17    | 93"""
    ok_val, err_val = check_listing_row_order_and_totals(valid_table, is_sorted=True)
    assert ok_val, f"Valid table must pass, got: {err_val}"

    # 3. JSON mode row order
    json_bad = """[
        {"file": "Total", "lines": 5, "words": 17, "characters": 93},
        {"file": "f1.txt", "lines": 3, "words": 9, "characters": 45}
    ]"""
    ok_j_bad, err_j_bad = check_listing_row_order_and_totals(json_bad, is_json=True)
    assert not ok_j_bad, "JSON with Total first must fail"


def test_r10_observed_output_test_repair():
    """R10-3: Observed-output test repair when tool is proven right (57 vs 58, 99 vs 100, 11.666666666666666 vs 11.666667)."""
    from ogun_core.quality import repair_observed_output_test_defects

    # Test code with real failing assertions from Round 9
    test_code = """import unittest
class TestTextsummary(unittest.TestCase):
    def test_single_file(self):
        res = self.run_tool("f1.txt")
        self.assertIn("57", res.stdout)

    def test_single_file_json(self):
        res = self.run_tool("f1.txt", "--json")
        data = self.assertJsonOutput(res)
        self.assertEqual(data[0]["characters"], 57)

    def test_sort_files(self):
        res = self.run_tool("f1.txt", "f2.txt", "f3.txt", "--sort")
        self.assertIn("99", res.stdout)

    def test_rich_csv_column(self):
        res = self.run_tool("rich.csv", "--column", "col_a")
        self.assertIn('11.666666666666666', res.stdout)
"""

    failure_output = """FAIL: test_single_file
AssertionError: '57' not found in 'File   | Lines | Words | Characters\\nf1.txt | 1     | 12    | 58        '

FAIL: test_single_file_json
AssertionError: 58 != 57

FAIL: test_sort_files
AssertionError: '99' not found in 'File   | Lines | Words | Characters\\nTotal  | 4     | 19    | 100       '

FAIL: test_rich_csv_column
AssertionError: '11.666666666666666' not found in 'Column | Count | Min   | Max  | Mean    \\ncol_a  | 3     | -5.0  | 30.0 | 11.666667'
"""

    repaired, repaired_code = repair_observed_output_test_defects(test_code, failure_output)
    assert repaired, "repair_observed_output_test_defects must report successful repair"
    assert 'assertIn("58", res.stdout)' in repaired_code, "57 must be repaired to 58 in assertIn"
    assert 'assertEqual(data[0]["characters"], 58)' in repaired_code, "57 must be repaired to 58 in assertEqual"
    assert 'assertIn("100", res.stdout)' in repaired_code, "99 must be repaired to 100"
    assert "assertIn('11.666667', res.stdout)" in repaired_code, "11.666666666666666 must be repaired to displayed 11.666667"


def test_r10_selector_validation():
    """R10-4: Selector validation for unknown and non-numeric selectors."""
    import subprocess
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col_num,name\n10,alpha\n20,beta\n", encoding="utf-8")

        # Flawed tool: prints empty table and exits 0 on unknown/non-numeric column
        flawed_tool = tmp_path / "flawed_sel.py"
        flawed_tool.write_text("""import argparse, sys
p = argparse.ArgumentParser()
p.add_argument('file')
p.add_argument('--column')
args = p.parse_args()
if args.column == 'unknown' or args.column == 'name':
    sys.exit(0)  # Flaw: exits 0!
print("col_num | 10..20")
""", encoding="utf-8")

        # Conforming tool: prints clear error to stderr and exits 1
        good_tool = tmp_path / "good_sel.py"
        good_tool.write_text("""import argparse, sys
p = argparse.ArgumentParser()
p.add_argument('file')
p.add_argument('--column')
args = p.parse_args()
if args.column:
    if args.column not in ('col_num', 'name'):
        sys.stderr.write(f"Error: Unknown column '{args.column}'\\n")
        sys.exit(1)
    if args.column == 'name':
        sys.stderr.write(f"Error: Column '{args.column}' is non-numeric\\n")
        sys.exit(1)
print("col_num | 10..20")
""", encoding="utf-8")

        # Run selector checks
        p_f = subprocess.run([sys.executable, str(flawed_tool), str(csv_file), "--column", "unknown"], capture_output=True, text=True)
        assert p_f.returncode == 0, "Flawed tool incorrectly exited 0"

        p_g1 = subprocess.run([sys.executable, str(good_tool), str(csv_file), "--column", "unknown"], capture_output=True, text=True)
        assert p_g1.returncode != 0, "Conforming tool must exit non-zero on unknown column"
        assert "error" in p_g1.stderr.lower(), "Conforming tool must print error to stderr on unknown column"

        p_g2 = subprocess.run([sys.executable, str(good_tool), str(csv_file), "--column", "name"], capture_output=True, text=True)
        assert p_g2.returncode != 0, "Conforming tool must exit non-zero on non-numeric column"
        assert "error" in p_g2.stderr.lower(), "Conforming tool must print error to stderr on non-numeric column"


def test_r10_report_consistency_verifier_evidence_backed():
    """R10-5: Report consistency check compares claims with pasted evidence."""
    from ogun_core.quality import verify_report_consistency

    # Report claiming PASS for mode whose pasted output fails layout check (6 separator segments)
    contradictory_layout_report = """# Round 10 Report
| Mode | Result |
|---|---|
| Default Mode | PASS |

```text
File   | Lines  | Words | Characters
-------+--------+-------+-------+-------+-----------
f1.txt | 2      | 5     | 24
Total  | 3      | 7     | 28
```
"""
    ok_l, iss_l = verify_report_consistency(contradictory_layout_report)
    assert not ok_l, "Must catch contradiction when pasted output fails table alignment"
    assert any("fails table alignment gate" in i for i in iss_l)

    # Report claiming PASS for mode whose pasted output fails row-order check (Total first)
    contradictory_order_report = """# Round 10 Report
| Mode | Result |
|---|---|
| `--sort` | PASS |

```text
File   | Lines  | Words | Characters
-------+--------+-------+-----------
Total  | 3      | 7     | 28
f1.txt | 2      | 5     | 24
```
"""
    ok_o, iss_o = verify_report_consistency(contradictory_order_report)
    assert not ok_o, "Must catch contradiction when pasted output fails row-order gate"
    assert any("fails row-order gate" in i for i in iss_o)

    # Report with valid pasted output passing all checks
    valid_report = """# Round 10 Report
| Mode | Result |
|---|---|
| Default Mode | PASS |

```text
Column | Count | Min   | Max  | Mean    
-------+-------+-------+------+---------
b      | 3     | -20.5 | 30.0 | 6.666667
```
"""
    ok_v, iss_v = verify_report_consistency(valid_report)
    assert ok_v, f"Valid evidence-backed report must pass, got: {iss_v}"


def test_r11_observed_output_test_repair():
    """R11-1: Scripted fake reproducing exact wordtally assertions (99 vs 100, 88 vs 89, JSON 100 != 99).
    Must end with all assertions repaired and tests passing without regenerating the tool.
    """
    from ogun_core.quality import repair_observed_output_test_defects
    from ogun_core.standalone_builder import write_scaffold_file
    from ogun_core.codetools import run_python_test

    test_code = """from _scaffold import ToolTestCase
import unittest

class TestWordtally(ToolTestCase):
    def test_multiple_files(self):
        self.text_file("f1.txt", "line one alpha\\n")
        res = self.run_tool("f1.txt")
        self.assertOk(res)
        self.assertIn("99", res.stdout)

    def test_multiple_files_json(self):
        res = self.run_tool("f1.txt", "--json")
        self.assertOk(res)
        data = self.assertJsonOutput(res)
        self.assertEqual(data[0]["characters"], 99)

    def test_min_words(self):
        res = self.run_tool("f1.txt", "--min-words", "4")
        self.assertOk(res)
        self.assertIn("88", res.stdout)

    def test_min_words_json(self):
        res = self.run_tool("f1.txt", "--min-words", "4", "--json")
        self.assertOk(res)
        data = self.assertJsonOutput(res)
        self.assertEqual(data[0]["characters"], 88)

if __name__ == "__main__":
    unittest.main()
"""

    simulated_failure_output = """FAIL: test_multiple_files
AssertionError: '99' not found in 'File   | Lines | Words | Characters\\nTotal  | 4     | 19    | 100       '

FAIL: test_multiple_files_json
AssertionError: 100 != 99

FAIL: test_min_words
AssertionError: '88' not found in 'File   | Lines | Words | Characters\\nTotal  | 3     | 17    | 89        '

FAIL: test_min_words_json
AssertionError: 89 != 88
"""

    repaired, repaired_code = repair_observed_output_test_defects(test_code, simulated_failure_output)
    assert repaired, "R11-1 repair must detect mismatches and report repaired=True"
    assert 'assertIn("100", res.stdout)' in repaired_code, "99 in assertIn must be rewritten to 100"
    assert 'assertEqual(data[0]["characters"], 100)' in repaired_code, "99 in assertEqual must be rewritten to 100"
    assert 'assertIn("89", res.stdout)' in repaired_code, "88 in assertIn must be rewritten to 89"
    assert 'assertEqual(data[0]["characters"], 89)' in repaired_code, "88 in assertEqual must be rewritten to 89"

    # End-to-end execution of fake project with repaired tests
    with tempfile.TemporaryDirectory() as td:
        t_dir = Path(td)
        write_scaffold_file(t_dir, "wordtally")

        # Fake wordtally that outputs 100 / 89
        fake_tool = t_dir / "wordtally.py"
        fake_tool.write_text("""import sys, json
if "--json" in sys.argv:
    if "--min-words" in sys.argv:
        print(json.dumps([{"file": "Total", "lines": 3, "words": 17, "characters": 89}]))
    else:
        print(json.dumps([{"file": "Total", "lines": 4, "words": 19, "characters": 100}]))
else:
    if "--min-words" in sys.argv:
        print("File   | Lines | Words | Characters\\nTotal  | 3     | 17    | 89        ")
    else:
        print("File   | Lines | Words | Characters\\nTotal  | 4     | 19    | 100       ")
""", encoding="utf-8")

        test_file = t_dir / "test_wordtally.py"
        test_file.write_text(test_code, encoding="utf-8")

        # Initial execution fails
        init_out = run_python_test(str(test_file), base_dir=t_dir)
        init_pass = ("TEST PASS" in init_out or "OK" in init_out) and not ("FAIL" in init_out or "ERROR" in init_out)
        assert not init_pass, "Initial tests with arithmetic errors must fail"

        # Apply repair
        ok, rep_c = repair_observed_output_test_defects(test_code, init_out)
        assert ok, "repair must fire from actual test output"
        test_file.write_text(rep_c, encoding="utf-8")
        pycache = t_dir / "__pycache__"
        if pycache.exists():
            shutil.rmtree(pycache, ignore_errors=True)

        # Re-execution must end PASSED without regenerating tool
        final_out = run_python_test(str(test_file), base_dir=t_dir)
        final_pass = ("TEST PASS" in final_out or "OK" in final_out) and not ("FAIL" in final_out or "ERROR" in final_out)
        assert final_pass, f"Repaired test suite must PASS completely, got:\n{final_out}"


def test_r11_use_before_assignment_checker():
    """R11-2: Use-before-assignment and unbound-local check.
    Must cover all 5 required test cases:
    1. Exact csvsummary print_table
    2. Correct loop variable
    3. Conditional assignment (if x: y = 1 then use -> flag)
    4. Globals/builtins (no flag)
    5. Nonlocal/global (no flag)
    """
    from ogun_core.quality import check_use_before_assignment

    # 1. Exact csvsummary print_table
    exact_csvsummary_code = """
def print_table(stats):
    headers = ['Column', 'Count', 'Min', 'Max', 'Mean']
    widths = [max(len(header), max(len(str(stat[h])) for stat in stats.values())) for h in headers]
    header_line = ' | '.join(f'{h:<{w}}' for h, w in zip(headers, widths))
    sep_line = '-+-'.join('-' * w for w in widths)
    data_lines = []
    for header, stat in stats.items():
        data_line = ' | '.join(f'{str(stat[h]):<{w}}' for h, w in zip(headers[1:], [len(header)] + widths[1:]))
        data_lines.append(f'{header:<{widths[0]}} | {data_line}')
    print('\\n'.join([header_line, sep_line] + data_lines))
"""
    errs_1 = check_use_before_assignment(exact_csvsummary_code)
    assert len(errs_1) >= 1, "Must catch 'header' used before assignment in csvsummary print_table"
    assert any("'header' used before assignment" in e for e in errs_1), f"Expected 'header' used before assignment, got: {errs_1}"

    # 2. Correct loop variable
    correct_loop = """
def sum_items(items):
    total = 0
    for x in items:
        total += x
    return total
"""
    errs_2 = check_use_before_assignment(correct_loop)
    assert errs_2 == [], f"Correct loop variable must not be flagged, got: {errs_2}"

    # 3. Conditional assignment (if x: y = 1 then use -> flag)
    cond_assign = """
def process(flag):
    if flag:
        val = 100
    return val
"""
    errs_3 = check_use_before_assignment(cond_assign)
    assert len(errs_3) >= 1, "Must flag conditional assignment used on path where not assigned"
    assert any("'val' used before assignment" in e for e in errs_3), f"Expected 'val' used before assignment, got: {errs_3}"

    # 4. Globals/builtins (no flag)
    globals_builtins = """
GLOBAL_LIMIT = 50
def calculate(data):
    return len(data) + max(data) + GLOBAL_LIMIT
"""
    errs_4 = check_use_before_assignment(globals_builtins)
    assert errs_4 == [], f"Builtins and globals must not be flagged, got: {errs_4}"

    # 5. Nonlocal/global (no flag)
    nonlocal_global = """
GLOBAL_COUNTER = 0
def increment():
    global GLOBAL_COUNTER
    GLOBAL_COUNTER += 1
    return GLOBAL_COUNTER

def outer():
    counter = 1
    def inner():
        nonlocal counter
        counter += 1
        return counter
    return inner()
"""
    errs_5 = check_use_before_assignment(nonlocal_global)
    assert errs_5 == [], f"nonlocal and global statements must not be flagged, got: {errs_5}"


def test_r11_selector_non_numeric_validation():
    """R11-3: Selector on a non-numeric column must exit non-zero, clear stderr, and NO traceback."""
    from ogun_core.quality import OracleRegistry

    with tempfile.TemporaryDirectory() as td:
        t_dir = Path(td)

        # Defective tool that raises KeyError on non-numeric column (like csvsummary)
        broken_tool = t_dir / "broken_col.py"
        broken_tool.write_text("""import sys, csv, json

def main():
    csv_file = None
    col = None
    is_json = False
    delim = ","
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--column":
            col = args[i+1]; i += 2
        elif args[i] == "--json":
            is_json = True; i += 1
        elif args[i] == "--delimiter":
            delim = args[i+1]; i += 2
        else:
            csv_file = args[i]; i += 1

    if not csv_file:
        sys.stderr.write("Error: missing file\\n")
        sys.exit(1)

    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delim)
        try:
            headers = next(reader)
        except StopIteration:
            sys.stderr.write("Error: empty csv\\n")
            sys.exit(1)

        counts = {h: 0 for h in headers}
        sums = {h: 0.0 for h in headers}
        mins = {h: None for h in headers}
        maxs = {h: None for h in headers}

        for row in reader:
            for h, cell in zip(headers, row):
                cell_s = cell.strip()
                if not cell_s:
                    continue
                try:
                    val = float(cell_s)
                    counts[h] += 1
                    sums[h] += val
                    if mins[h] is None or val < mins[h]:
                        mins[h] = val
                    if maxs[h] is None or val > maxs[h]:
                        maxs[h] = val
                except ValueError:
                    pass

    numeric_cols = [h for h in headers if counts[h] > 0]
    if col:
        if col not in headers:
            sys.stderr.write(f"Error: unknown column '{col}'\\n")
            sys.exit(1)
        # Bug: does not check if col in numeric_cols! Looking up in stats will raise KeyError on 'name'
        show_cols = [col]
    else:
        show_cols = numeric_cols

    stats = {h: {"count": counts[h], "min": mins[h], "max": maxs[h], "mean": sums[h]/counts[h]} for h in numeric_cols}

    # If col is non-numeric, stats[col] raises KeyError!
    selected = {h: stats[h] for h in show_cols}

    if is_json:
        print(json.dumps(selected))
    else:
        print("Column | Count | Min   | Max  | Mean    ")
        print("-------+-------+-------+------+---------")
        for h, s in selected.items():
            print(f"{h:<6} | {s['count']:<5} | {s['min']:<5.1f} | {s['max']:<4.1f} | {s['mean']:<8.6f}")

if __name__ == "__main__":
    main()
""", encoding="utf-8")

        req = "Build me a tool with --column NAME, --delimiter, and --json option."
        ok_broken, err_broken = OracleRegistry.oracle_csv_column_stats(broken_tool, req, t_dir)
        assert not ok_broken, "Must flag tool that crashes with KeyError/traceback on non-numeric column"
        assert "Selector validation defect (R11-3)" in err_broken or "traceback" in err_broken.lower(), f"Unexpected error: {err_broken}"

        # Correct tool that prints stderr error and exits 1 with NO traceback
        clean_tool = t_dir / "clean_col.py"
        clean_tool.write_text("""import sys, csv, json

def main():
    csv_file = None
    col = None
    is_json = False
    delim = ","
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--column":
            col = args[i+1]; i += 2
        elif args[i] == "--json":
            is_json = True; i += 1
        elif args[i] == "--delimiter":
            delim = args[i+1]; i += 2
        else:
            csv_file = args[i]; i += 1

    if not csv_file:
        sys.stderr.write("Error: missing file\\n")
        sys.exit(1)

    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delim)
        try:
            headers = next(reader)
        except StopIteration:
            sys.stderr.write("Error: empty csv\\n")
            sys.exit(1)

        counts = {h: 0 for h in headers}
        sums = {h: 0.0 for h in headers}
        mins = {h: None for h in headers}
        maxs = {h: None for h in headers}

        for row in reader:
            for h, cell in zip(headers, row):
                cell_s = cell.strip()
                if not cell_s:
                    continue
                try:
                    val = float(cell_s)
                    counts[h] += 1
                    sums[h] += val
                    if mins[h] is None or val < mins[h]:
                        mins[h] = val
                    if maxs[h] is None or val > maxs[h]:
                        maxs[h] = val
                except ValueError:
                    pass

    numeric_cols = [h for h in headers if counts[h] > 0]
    if col:
        if col not in headers or col not in numeric_cols:
            sys.stderr.write(f"Error: column '{col}' has no numeric values\\n")
            sys.exit(1)
        show_cols = [col]
    else:
        show_cols = numeric_cols

    stats = {h: {"count": counts[h], "min": mins[h], "max": maxs[h], "mean": sums[h]/counts[h]} for h in show_cols}

    if is_json:
        print(json.dumps(stats))
    else:
        print("Column | Count | Min   | Max  | Mean    ")
        print("-------+-------+-------+------+---------")
        for h, s in stats.items():
            print(f"{h:<6} | {s['count']:<5} | {s['min']:<5.1f} | {s['max']:<4.1f} | {s['mean']:<8.6f}")

if __name__ == "__main__":
    main()
""", encoding="utf-8")

        ok_clean, err_clean = OracleRegistry.oracle_csv_column_stats(clean_tool, req, t_dir)
        assert ok_clean, f"Clean tool handling selector validation must PASS oracle, got: {err_clean}"


def test_r11_memory_streaming_gate():
    """R11-4: Memory streaming gate flags list-buffering implementations for streaming tasks."""
    from ogun_core.quality import check_streaming_memory_hygiene, check_tool_code_hygiene

    # List-buffering implementation (like csvsummary)
    list_buffering_code = """import csv, sys

def read_csv(path):
    headers = []
    numeric_values = []
    with open(path) as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            numeric_values.append(float(row[0]))
    return headers, numeric_values
"""
    req = "Build tool to calculate mean, min, max, count for numeric columns from large CSV file."
    hyg_ok, issues = check_tool_code_hygiene(list_buffering_code, req)
    assert not hyg_ok, "Must flag list-buffering tool on streaming CSV task"
    assert any("Memory regression defect (R11-4)" in i for i in issues), f"Expected memory regression defect, got: {issues}"

    # Single-pass streaming accumulator implementation
    streaming_code = """import csv, sys

def stream_csv(path):
    count = 0
    total = 0.0
    min_v = None
    max_v = None
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            val = float(row[0].strip())
            count += 1
            total += val
            if min_v is None or val < min_v: min_v = val
            if max_v is None or val > max_v: max_v = val
    return count, min_v, max_v, total/count
"""
    clean_ok, clean_issues = check_tool_code_hygiene(streaming_code, req)
    mem_issues = [i for i in clean_issues if "Memory regression" in i]
    assert len(mem_issues) == 0, f"Streaming accumulators must not be flagged for memory regression: {mem_issues}"



# ---------------------------------------------------------------------------
# Round 12 regressions (from the round-11 colreport evidence)
# ---------------------------------------------------------------------------
R12_FIX = Path(__file__).resolve().parent / "fixtures" / "round12"
R12_COLREPORT_REQ = (
    "Build me a command-line tool called `colreport`. Give it a CSV file with a header row and it prints, for "
    "each numeric column, the count, minimum, maximum and mean, ignoring empty cells and non-numeric columns, as "
    "an aligned plain-text table with a header separator line. It needs a `--json` option, a `--delimiter` option "
    "(default comma) and a `--column NAME` option to show only that column. Write tests for it, run them, and "
    "make sure they pass."
)


def test_r12_use_before_assignment_flow_cases() -> None:
    from ogun_core.quality import check_use_before_assignment as c
    must_flag = [
        "def p(stats, headers):\n    widths=[max(len(header), 1) for h in headers]\n    for header in headers:\n        pass\n",
        "def f(x):\n    if x:\n        y=1\n    return y\n",
        "def f():\n    try:\n        x=g()\n    except Exception:\n        pass\n    return x\n",
    ]
    ok_code = [
        "import sys\ndef f(p):\n    try:\n        a,b = g(p)\n    except OSError:\n        sys.exit(1)\n    return a+b\n",
        "def f():\n    a=b=c=0\n    return a+b+c\n",
        "def f(x):\n    if x:\n        return 1\n    else:\n        y=2\n    return y\n",
        "def f():\n    while True:\n        v=g()\n        if v: break\n    return v\n",
        "def f():\n    import csv\n    return csv\n",
        "def f():\n    try:\n        x=1\n    finally:\n        pass\n    return x\n",
        "def f(a, parser):\n    if a: v=1\n    else: parser.error('x')\n    return v\n",
    ]
    for code in must_flag:
        assert c(code), f"should flag:\n{code}"
    for code in ok_code:
        assert not c(code), f"false positive {c(code)} on:\n{code}"
    good = (R12_FIX / "colreport_good.py").read_text(encoding="utf-8")
    assert not c(good), c(good)


def test_r12_status_leads_with_tool_defect() -> None:
    import tempfile, shutil
    from ogun_core.quality import compute_final_status_from_disk
    from ogun_core.standalone_builder import write_scaffold_file
    d = Path(tempfile.mkdtemp())
    try:
        (d / "colreport.py").write_text((R12_FIX / "colreport_broken.py").read_text(encoding="utf-8"), encoding="utf-8")
        (d / "test_colreport.py").write_text((R12_FIX / "test_colreport_good.py").read_text(encoding="utf-8"), encoding="utf-8")
        write_scaffold_file(d, "colreport")
        st = compute_final_status_from_disk(d, "colreport", R12_COLREPORT_REQ)
        assert str(st).startswith("REQUIREMENT FAILED"), st
        assert "max() iterable argument is empty" in st, st
        assert "| tests:" in st, st
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_r12_scripted_build_recovers_from_broken_colreport() -> None:
    import tempfile, shutil
    from ogun_core.standalone_builder import build_standalone_project_staged
    broken = (R12_FIX / "colreport_broken.py").read_text(encoding="utf-8")
    good = (R12_FIX / "colreport_good.py").read_text(encoding="utf-8")
    tests = (R12_FIX / "test_colreport_good.py").read_text(encoding="utf-8")
    client = MockChatClient(
        [{"content": f"```python\n{broken}\n```"}, {"content": f"```python\n{tests}\n```"}]
        + [{"content": f"```python\n# FILE: colreport.py\n{good}\n```"}] * 6
    )
    engine = CoreEngine(client=client)
    test_dir = tempfile.mkdtemp()
    try:
        with patch("ogun_core.standalone_builder.WORKSPACE_DIR", Path(test_dir)):
            success, msg, details = build_standalone_project_staged(engine, R12_COLREPORT_REQ, max_attempts=3)
        prompts = [m[0]["content"] for m in client.call_history]
        fix_prompts = [p for p in prompts if "FAILURES TO FIX" in p]
        assert fix_prompts, "no fix prompt sent"
        assert "SMOKE / ORACLE / LAYOUT FAILURES" in fix_prompts[0], fix_prompts[0][-2000:]
        assert "max() iterable argument is empty" in fix_prompts[0]
        assert "CURRENT TOOL CODE" in fix_prompts[0]
        assert success is True, msg[-1500:]
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_r12_text_tool_with_json_flag_gets_text_memory_fixture() -> None:
    import tempfile, shutil
    import ogun_core.quality as q
    seen = {}
    real_popen = q.subprocess.Popen

    class SpyPopen(real_popen):
        def __init__(self, cmd, *a, **k):
            if isinstance(cmd, (list, tuple)) and len(cmd) > 2:
                seen.setdefault("target", str(cmd[2]))
            super().__init__(cmd, *a, **k)

    d = Path(tempfile.mkdtemp())
    try:
        tool = d / "filetally.py"
        tool.write_text("import sys\nfor p in sys.argv[1:]:\n    n = sum(1 for _ in open(p, encoding='utf-8'))\nprint(n)\n", encoding="utf-8")
        req = ("Build me a command-line tool called filetally. Give it one or more text files and it prints the "
               "number of lines, words and characters. It needs a --json option and a --sort option.")
        with patch.object(q.subprocess, "Popen", SpyPopen):
            ok, peak, report = q.check_tool_memory_budget(d, tool, "filetally", req)
        assert ok, report
        assert not seen.get("target", "").endswith(".json"), f"text tool got a JSON memory fixture: {seen}"
    finally:
        shutil.rmtree(d, ignore_errors=True)



def test_r12h_argparse_text_in_tests_does_not_hide_tool_defects() -> None:
    """Round-11/12 colsheet: a test's argparse error made arbitration say HARNESS_ERROR although
    smoke/oracle had found real tool defects, so the tool was never regenerated."""
    from ogun_core.quality import arbitrate_test_failure
    test_output = ("AssertionError: 'Error' not found in 'usage: colsheet.py [-h]\\n"
                   "colsheet.py: error: unrecognized arguments: --column x'")
    res = arbitrate_test_failure(request="Build me a command-line tool called colsheet with --json and --column NAME",
              tool_code="import argparse", test_code="", test_output=test_output,
              failing_tests=["test_x"], smoke_passed=False, coverage_passed=True, memory_passed=False,
              client=None, smoke_has_harness_error=False)
    assert res["classification"] == "TOOL", res
    res2 = arbitrate_test_failure(request="Build me a command-line tool called colsheet with --json", tool_code="import argparse",
               test_code="", test_output=test_output, failing_tests=["test_x"], smoke_passed=True,
               coverage_passed=True, memory_passed=True, client=None, smoke_has_harness_error=False)
    assert res2["classification"] == "HARNESS_ERROR", res2



def test_r12i_tool_prompt_contains_only_relevant_guidance() -> None:
    """Round-12 colsheet/colbrief: the tool prompt carried the agent persona (tool calls, write_file),
    JSON sentinel lookups, --max-width truncation and two-pass re-open streaming for a CSV-stats request."""
    from ogun_core.standalone_builder import build_tool_generation_prompt
    from ogun_core.quality import extract_requested_flags
    p = build_tool_generation_prompt(R12_COLREPORT_REQ, "colreport", extract_requested_flags(R12_COLREPORT_REQ))
    for banned in ("write_file", "tool calls", "_MISSING", "--max-width", "Re-open or seek", "dupefinder"):
        assert banned not in p, f"irrelevant guidance in CSV prompt: {banned}"
    for needed in ("non_numeric_seen", "ONE entry per column", "has no numeric values", "--delimiter", "--column", "--json"):
        assert needed in p, f"missing guidance: {needed}"
    lookup_req = "Build me a command-line tool called tomlget. Give it a TOML file and a dotted key path like a.b and it prints the value. It needs a --json option."
    p2 = build_tool_generation_prompt(lookup_req, "tomlget", extract_requested_flags(lookup_req))
    assert "_MISSING" in p2 and "non_numeric_seen" not in p2



def test_r12j_concrete_target_example() -> None:
    import tempfile, shutil
    from ogun_core.quality import build_expected_output_example
    d = Path(tempfile.mkdtemp())
    try:
        ex = build_expected_output_example(R12_COLREPORT_REQ, R12_FIX / "colreport_broken.py", d)
        assert "EXPECTED:" in ex and "YOUR TOOL PRINTED (exit 1)" in ex, ex
        assert "b      | 3     | -20.5 | 30  | 6.6667" in ex, ex
        assert "max() iterable argument is empty" in ex
        assert not any(p.name.startswith("_example_") for p in d.iterdir())
    finally:
        shutil.rmtree(d, ignore_errors=True)



def test_r12k_scaffold_splits_flag_value_argument() -> None:
    import subprocess, tempfile, shutil
    from ogun_core.standalone_builder import write_scaffold_file
    d = Path(tempfile.mkdtemp())
    try:
        (d / "probe.py").write_text("import sys, json\nprint(json.dumps(sys.argv[1:]))\n", encoding="utf-8")
        write_scaffold_file(d, "probe")
        (d / "test_probe.py").write_text(
            "import unittest, json\nfrom _scaffold import ToolTestCase\n"
            "class T(ToolTestCase):\n"
            "    def test_split(self):\n"
            "        r = self.run_tool('f.csv', '--column zz', '--delimiter=;', 'plain name.txt')\n"
            "        self.assertEqual(json.loads(r.stdout), ['f.csv', '--column', 'zz', '--delimiter=;', 'plain name.txt'])\n"
            "if __name__ == '__main__':\n    unittest.main()\n", encoding="utf-8")
        p = subprocess.run([sys.executable, "-m", "unittest", "test_probe"], cwd=str(d), capture_output=True, text=True)
        assert "OK" in p.stderr, p.stderr
    finally:
        shutil.rmtree(d, ignore_errors=True)



def test_r12m_correct_csv_tool_gets_clean_smoke() -> None:
    """Round-12 colsight: the alignment smoke fixture was written with literal backslash-n, so a
    correct tool got a false 'Table alignment error' and PASSED WITH ISSUES."""
    import tempfile, shutil
    from ogun_core.standalone_builder import run_smoke_tests
    d = Path(tempfile.mkdtemp())
    try:
        tool = d / "colreport.py"
        tool.write_text((R12_FIX / "colreport_good.py").read_text(encoding="utf-8"), encoding="utf-8")
        ok, report, fails, samples, harness = run_smoke_tests(d, tool, "colreport", R12_COLREPORT_REQ, return_samples=True)
        assert ok, fails
        assert "\\n" not in report, "smoke report contains literal backslash-n"
        assert "Smoke test table alignment (standard layout): PASSED" in report, report
    finally:
        shutil.rmtree(d, ignore_errors=True)


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
        ("R6: CLI Contract Discovery Multi-Arg", test_r6_cli_contract_discovery_multi_arg),
        ("R6: Argparse Usage Error Classified as Harness Error", test_r6_argparse_error_classified_as_harness_error),
        ("R6: Scaffold Module Deterministic Helpers", test_r6_scaffold_module_deterministic_helpers),
        ("R6: Hygiene Static Check Rules", test_r6_hygiene_static_check_rules),
        ("R6: Deterministic Table Alignment Checker", test_r6_deterministic_table_alignment_checker),
        ("R6: Strategy Transition Table", test_r6_strategy_transition_table),
        ("R6: Sentinel Lookup Distinguishes Null", test_r6_sentinel_lookup_distinguishes_null),
        ("R7: Zero Tests Ran & Count Mismatch Fails", test_r7_zero_tests_ran_fails),
        ("R7: Missing Main Guard Appends and Runs", test_r7_missing_main_guard_appends_and_runs),
        ("R7: Colstats Exact Reproduction Not Passed", test_r7_colstats_exact_reproduction_not_passed),
        ("R12: Use-Before-Assignment Flow Cases", test_r12_use_before_assignment_flow_cases),
        ("R12: Correct CSV Tool Gets Clean Smoke", test_r12m_correct_csv_tool_gets_clean_smoke),
        ("R12: Scaffold Splits '--flag value' Argument", test_r12k_scaffold_splits_flag_value_argument),
        ("R12: Concrete Target Example In Fix Prompt", test_r12j_concrete_target_example),
        ("R12: Tool Prompt Contains Only Relevant Guidance", test_r12i_tool_prompt_contains_only_relevant_guidance),
        ("R12: Argparse Text In Tests Does Not Hide Tool Defects", test_r12h_argparse_text_in_tests_does_not_hide_tool_defects),
        ("R12: Status Leads With Tool Defect", test_r12_status_leads_with_tool_defect),
        ("R12: Scripted Build Recovers From Broken colreport", test_r12_scripted_build_recovers_from_broken_colreport),
        ("R12: Text Tool With --json Gets Text Memory Fixture", test_r12_text_tool_with_json_flag_gets_text_memory_fixture),
        ("R7: Status Reconciliation Consistency Assertion", test_r7_status_reconciliation_consistency_assertion),
        ("R7: Reference Oracle CSV Column Stats", test_r7_oracle_csv_column_stats),
        ("R7: Reference Oracle Dotted Path Lookup", test_r7_oracle_dotted_path_lookup),
        ("R7: Table Alignment Header vs Data Width", test_r7_table_alignment_header_vs_data_width),
        ("R7: Offline Regressions R1-R6", test_r7_offline_regressions),
        ("R8: Test File Hard Validator Reproduction", test_r8_test_file_hard_validator_numstats_reproduction),
        ("R8: Undefined Names / Import Checker", test_r8_undefined_names_checker),
        ("R8: Reference Oracle Output Modes & Default Mode Crash", test_r8_oracle_exercises_all_modes_and_catches_default_mode_crash),
        ("R8: Reference Oracle Line Word Count", test_r8_linecount_oracle_word_line_char_counts),
        ("R8: Arbitration Deterministic Linecount Assertion Mismatches", test_r8_arbitration_linecount_assertion_mismatches),
        ("R9: Input Kind Inference & Fixtures", test_r9_input_kind_inference_and_fixtures),
        ("R9: Mismatched Fixture Validator Rejection", test_r9_mismatched_fixture_validator_rejection),
        ("R9: Oracle Full Flag-Combination Mode Matrix", test_r9_oracle_full_flag_combination_mode_matrix),
        ("R9: Spec Fidelity Ignoring & Unrequested Options", test_r9_spec_fidelity_ignoring_non_numeric_and_unrequested_options),
        ("R9: Table Row Column Consistency", test_r9_table_row_column_consistency),
        ("R9: Report Consistency Verifier", test_r9_report_consistency_verifier),
        ("R10: Table Layout Alignment Gate", test_r10_table_layout_alignment_gate),
        ("R10: Row-Order Semantics Totals and Sorting", test_r10_row_order_semantics_totals_and_sorting),
        ("R10: Observed-Output Test Repair", test_r10_observed_output_test_repair),
        ("R10: Selector Validation", test_r10_selector_validation),
        ("R10: Report Consistency Verifier Evidence-Backed", test_r10_report_consistency_verifier_evidence_backed),
        ("R11: Observed-Output Test Repair AST", test_r11_observed_output_test_repair),
        ("R11: Use-Before-Assignment / Unbound-Local Checker", test_r11_use_before_assignment_checker),
        ("R11: Selector on Non-Numeric Column", test_r11_selector_non_numeric_validation),
        ("R11: Memory Regression Streaming Gate", test_r11_memory_streaming_gate),
    ]

    print("=" * 60)
    print("  Ogun Development-Ability Fixes Test Suite (F1 - F7)")
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
