"""Comprehensive offline unit and integration test suite for Ogun Core.

Runs completely offline: zero network calls, zero model inference, zero pytest dependencies.
All filesystem mutations execute inside isolated tempfile.TemporaryDirectory() environments.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List

# Ensure Ogun root is on sys.path
OGUN_ROOT = Path(__file__).resolve().parents[1]
if str(OGUN_ROOT) not in sys.path:
    sys.path.insert(0, str(OGUN_ROOT))

from ogun_core.base_tool import BaseTool, ToolResult
from ogun_core.codetools import (
    check_syntax,
    read_file,
    replace_in_file,
    run_python_test,
    write_file,
)
from ogun_core.config import NUM_CTX
from ogun_core.engine import ContextManager, CoreEngine
from ogun_core.plugin_manager import PluginManager, normalize_plugin_name
from ogun_core.security import PathSecurityGuard
from ogun_core.toolcalls import parse_tool_calls
from test_harness import extract_python_blocks


def test_parser_corpus() -> None:
    """Test all edge cases in the text tool call parser."""
    registered = {"read_file", "write_file", "search_code"}

    # 1. Fenced JSON block
    txt1 = '```json\n{"name": "read_file", "arguments": {"filepath": "foo.py"}}\n```'
    res1 = parse_tool_calls(txt1, registered)
    assert len(res1) == 1, f"Expected 1 call, got {res1}"
    assert res1[0]["name"] == "read_file"
    assert res1[0]["arguments"] == {"filepath": "foo.py"}

    # 2. Bare JSON inside prose
    txt2 = 'I will now inspect the file: {"name": "read_file", "arguments": {"filepath": "bar.py"}} let me know.'
    res2 = parse_tool_calls(txt2, registered)
    assert len(res2) == 1
    assert res2[0]["name"] == "read_file"
    assert res2[0]["arguments"] == {"filepath": "bar.py"}

    # 3. Two calls in one message
    txt3 = (
        '```json\n{"name": "read_file", "arguments": {"filepath": "a.py"}}\n```\n'
        'and then:\n```json\n{"name": "write_file", "arguments": {"filepath": "b.py", "content": "x"}}\n```'
    )
    res3 = parse_tool_calls(txt3, registered)
    assert len(res3) == 2
    assert res3[0]["name"] == "read_file"
    assert res3[1]["name"] == "write_file"

    # 4. Stringified JSON argument
    txt4 = '```json\n{"name": "read_file", "arguments": "{\\"filepath\\": \\"escaped.py\\"}"}\n```'
    res4 = parse_tool_calls(txt4, registered)
    assert len(res4) == 1
    assert res4[0]["arguments"] == {"filepath": "escaped.py"}

    # 5. Python-repr dict with single quotes
    txt5 = "{'name': 'read_file', 'arguments': {'filepath': 'py_dict.py'}}"
    res5 = parse_tool_calls(txt5, registered)
    assert len(res5) == 1
    assert res5[0]["arguments"] == {"filepath": "py_dict.py"}

    # 6. 'parameters' key instead of 'arguments'
    txt6 = '{"name": "read_file", "parameters": {"filepath": "param.py"}}'
    res6 = parse_tool_calls(txt6, registered)
    assert len(res6) == 1
    assert res6[0]["arguments"] == {"filepath": "param.py"}

    # 7. Qwen <tool_call> tags
    txt7 = '<tool_call>\n{"name": "read_file", "arguments": {"filepath": "qwen.py"}}\n</tool_call>'
    res7 = parse_tool_calls(txt7, registered)
    assert len(res7) == 1
    assert res7[0]["name"] == "read_file"

    # 8. CRLF line endings
    txt8 = '```json\r\n{\r\n  "name": "read_file",\r\n  "arguments": {\r\n    "filepath": "crlf.py"\r\n  }\r\n}\r\n```'
    res8 = parse_tool_calls(txt8, registered)
    assert len(res8) == 1
    assert res8[0]["arguments"] == {"filepath": "crlf.py"}

    # 9. Unterminated JSON (must not crash, returns empty)
    txt9 = '{"name": "read_file", "arguments": {"filepath":'
    res9 = parse_tool_calls(txt9, registered)
    assert len(res9) == 0

    # 10. Unregistered tool name must be filtered out
    txt10 = '{"name": "unregistered_malicious_tool", "arguments": {"cmd": "rm -rf"}}'
    res10 = parse_tool_calls(txt10, registered)
    assert len(res10) == 0

    # 11. Nested braces inside code strings must survive intact
    code_content = "def test():\n    d = {'key': 123}\n    return d\n"
    txt11 = json.dumps({"name": "write_file", "arguments": {"filepath": "code.py", "content": code_content}})
    res11 = parse_tool_calls(txt11, registered)
    assert len(res11) == 1
    assert res11[0]["arguments"]["content"] == code_content


def test_security_guard() -> None:
    """Test PathSecurityGuard against traversal, UNC, sibling prefix, and core immutability."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir).resolve()
        guard = PathSecurityGuard(base_dir=root)

        # 1. Allowed directories
        allowed1 = root / "plugins" / "custom_tools" / "my_tool.py"
        allowed2 = root / "plugins" / "tests" / "test_my_tool.py"
        allowed3 = root / "workspace" / "scratch.txt"
        assert guard.validate_write(allowed1) == allowed1
        assert guard.validate_write(allowed2) == allowed2
        assert guard.validate_write(allowed3) == allowed3

        # 2. Path traversal attempts
        try:
            guard.validate_write(root / "plugins" / "custom_tools" / ".." / ".." / "ogun_core" / "evil.py")
            assert False, "Should have raised PermissionError for traversal to core"
        except PermissionError:
            pass

        # 3. Windows mixed case normalization
        try:
            guard.validate_write(root / "OGUN_CORE" / "evil.py")
            assert False, "Should have raised PermissionError for mixed case core"
        except PermissionError:
            pass

        # 4. UNC path rejection
        try:
            guard.validate_write("\\\\server\\share\\evil.py")
            assert False, "Should have raised PermissionError for UNC path"
        except PermissionError:
            pass

        # 5. Drive-relative path rejection
        try:
            guard.validate_write("C:evil.py")
            assert False, "Should have raised PermissionError for drive-relative path"
        except PermissionError:
            pass

        # 6. Sibling prefix trick
        try:
            guard.validate_write(root / "plugins" / "custom_tools_evil" / "tool.py")
            assert False, "Should have raised PermissionError for sibling prefix trick"
        except PermissionError:
            pass

        # 7. Core files, main.py, registry.json, and batch scripts immutability
        blocked_targets = [
            root / "main.py",
            root / "run_ogun.bat",
            root / "test_plugin_builder.bat",
            root / "plugins" / "registry.json",
            root / "training" / "prepare_training_data.py",
            root / "ogun_core" / "paths.py",
        ]
        for target in blocked_targets:
            try:
                guard.validate_write(target)
                assert False, f"Target '{target}' should have been blocked"
            except PermissionError:
                pass


def test_basetool_contracts() -> None:
    """Test ToolResult and BaseTool contract interfaces."""
    # ToolResult success
    r_ok = ToolResult.success(output={"value": 42})
    assert r_ok.status == "success"
    assert r_ok.output == {"value": 42}
    assert r_ok.error_trace is None

    # ToolResult error
    r_err = ToolResult.error("Failure reason")
    assert r_err.status == "error"
    assert r_err.error_trace == "Failure reason"

    # Invalid status
    try:
        ToolResult(status="pending")
        assert False, "Should raise ValueError for invalid status"
    except ValueError:
        pass

    # BaseTool implementation
    class DummyTool(BaseTool):
        name: str = "dummy_tool"
        version: str = "1.0.0"
        description: str = "A dummy tool."
        parameters_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {"val": {"type": "integer"}},
        }

        def execute(self, **kwargs: Any) -> ToolResult:
            return ToolResult.success(output=kwargs.get("val", 0) * 2)

    tool = DummyTool()
    schema = tool.to_tool_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "dummy_tool"
    res = tool.execute(val=21)
    assert res.status == "success"
    assert res.output == 42


def test_ast_validation() -> None:
    """Test PluginManager AST checks for stubs, comments, and placeholders."""
    # 1. Valid code
    valid_code = """
from ogun_core.base_tool import BaseTool, ToolResult

class GoodTool(BaseTool):
    name = "good_tool"
    def execute(self, **kwargs):
        val = kwargs.get("x", 0)
        return ToolResult.success(output=val + 1)
"""
    ok, err = PluginManager.validate_plugin_code(valid_code)
    assert ok, f"Expected valid code to pass, got: {err}"

    # 2. Placeholder comment
    todo_code = valid_code + "\n# TODO: implement real logic later"
    ok, err = PluginManager.validate_plugin_code(todo_code)
    assert not ok
    assert "Mock/placeholder comment" in err

    # 3. Placeholder string
    mock_str_code = """
from ogun_core.base_tool import BaseTool, ToolResult
class MockTool(BaseTool):
    name = "mock_tool"
    def execute(self, **kwargs):
        return ToolResult.success(output="mock implementation")
"""
    ok, err = PluginManager.validate_plugin_code(mock_str_code)
    assert not ok
    assert "Mock/placeholder phrase" in err

    # 4. Pass-only method
    pass_code = """
from ogun_core.base_tool import BaseTool, ToolResult
class StubTool(BaseTool):
    name = "stub_tool"
    def execute(self, **kwargs):
        pass
"""
    ok, err = PluginManager.validate_plugin_code(pass_code)
    assert not ok
    assert "contains only 'pass'" in err

    # 5. NotImplementedError
    nie_code = """
from ogun_core.base_tool import BaseTool, ToolResult
class NieTool(BaseTool):
    name = "nie_tool"
    def execute(self, **kwargs):
        raise NotImplementedError("not done")
"""
    ok, err = PluginManager.validate_plugin_code(nie_code)
    assert not ok
    assert "raises NotImplementedError" in err

    # 6. Missing BaseTool inheritance
    no_tool_code = """
def simple_function(x):
    return x * 2
"""
    ok, err = PluginManager.validate_plugin_code(no_tool_code)
    assert not ok
    assert "must define a class subclassing BaseTool" in err


def test_plugin_lifecycle_and_registry_safety() -> None:
    """Test plugin creation, isolated test subprocess, hot-reload, UTF-8-BOM registry, and corrupt recovery."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir).resolve()
        tools_dir = root / "plugins" / "custom_tools"
        tests_dir = root / "plugins" / "tests"
        quarantine_dir = root / "plugins" / "quarantine"
        reg_file = root / "plugins" / "registry.json"

        # Initialize manager with NO directories or registry existing yet
        pm = PluginManager(
            custom_tools_dir=tools_dir,
            tests_dir=tests_dir,
            quarantine_dir=quarantine_dir,
            registry_path=reg_file,
            base_dir=root,
        )

        # Prove registry.json is NOT written on init or plain discovery
        assert not reg_file.exists(), "registry.json should not exist at import or init time"
        pm.discover_tools()
        assert not reg_file.exists(), "discover_tools() must not create or write registry.json"

        # Create valid plugin
        tool_code = f"""
from __future__ import annotations
import sys
from pathlib import Path
from ogun_core.base_tool import BaseTool, ToolResult

class AdderTool(BaseTool):
    name: str = "adder_tool"
    version: str = "1.0.0"
    description: str = "Adds two numbers."
    parameters_schema = {{"type": "object", "properties": {{"a": {{"type": "number"}}, "b": {{"type": "number"}}}}}}

    def execute(self, **kwargs) -> ToolResult:
        a = kwargs.get("a", 0)
        b = kwargs.get("b", 0)
        return ToolResult.success(output={{"sum": a + b}})
"""
        test_code = f"""
from __future__ import annotations
import sys
from pathlib import Path

# Add Ogun root to sys.path
OGUN_ROOT = Path(r"{OGUN_ROOT}")
if str(OGUN_ROOT) not in sys.path:
    sys.path.insert(0, str(OGUN_ROOT))

# Add temporary custom_tools to sys.path
TOOLS_DIR = Path(r"{tools_dir}")
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from adder_tool import AdderTool

tool = AdderTool()
res = tool.execute(a=5, b=7)
assert res.status == "success", f"Expected success, got {{res.status}}"
assert res.output == {{"sum": 12}}, f"Unexpected output: {{res.output}}"
print("PASS")
sys.exit(0)
"""

        res = pm.create_or_update_plugin("adder_tool", tool_code, test_code)
        assert res.status == "success", f"Plugin creation failed: {res.error_trace}"
        assert reg_file.exists(), "registry.json should now be written"

        # Execute plugin
        exec_res = pm.execute_plugin("adder_tool", a=10, b=20)
        assert exec_res.status == "success"
        assert exec_res.output == {"sum": 30}

        # Verify registry was written with normalized name and active status
        reg_data = pm.load_registry()
        assert "adder_tool" in reg_data["tools"]
        assert reg_data["tools"]["adder_tool"]["status"] == "active"

        # Test UTF-8-BOM tolerance in load_registry
        reg_file.write_bytes(b"\xef\xbb\xbf" + json.dumps(reg_data).encode("utf-8"))
        bom_data = pm.load_registry()
        assert "adder_tool" in bom_data["tools"], "Failed to load registry containing UTF-8-BOM"

        # Test corrupt registry recovery
        reg_file.write_text("CORRUPTED NOT JSON {{{", encoding="utf-8")
        recovered = pm.load_registry()
        assert recovered == {"tools": {}}
        # Verify corrupt backup was created
        corrupt_files = list((root / "plugins").glob("registry.json.corrupt-*"))
        assert len(corrupt_files) == 1, "Expected corrupt registry backup file"


def test_engine_turn_loop_with_mock_client() -> None:
    """Test CoreEngine tool execution, text fallback, and history pruning."""
    class FakeLLMClient:
        def __init__(self, responses: List[Dict[str, Any]]) -> None:
            self.responses = list(responses)
            self.calls: List[Any] = []

        def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
            self.calls.append({"messages": list(messages), "tools": tools})
            if self.responses:
                return self.responses.pop(0)
            return {"content": "Final reply with no tools."}

    # Case 1: Native tool call followed by completion
    resp1 = {
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "read_file",
                    "arguments": {"filepath": "sample.txt"},
                }
            }
        ],
    }
    resp2 = {"content": "The file contains sample text."}

    client = FakeLLMClient([resp1, resp2])
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir).resolve()
        (root / "sample.txt").write_text("hello world from sample", encoding="utf-8")

        engine = CoreEngine(client=client, base_dir=root)
        out = engine.run_turn("Please read sample.txt")
        assert "The file contains sample text." in out
        assert len(client.calls) == 2

    # Case 2: History pruning under token budget
    cm = ContextManager(max_tokens=100, budget_ratio=1.0)
    msgs = [{"role": "system", "content": "system instruction"}]
    for i in range(20):
        msgs.append({"role": "user", "content": f"Turn {i} with lots of filler text to fill the budget."})
    pruned = cm.prune_history(msgs)
    assert pruned[0]["role"] == "system"
    assert len(pruned) < len(msgs)


def test_codetools_operations() -> None:
    """Test codetools: read_file, write_file, replace_in_file, and check_syntax."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir).resolve()
        guard = PathSecurityGuard(base_dir=root)

        # 1. write_file in workspace
        ws_file = "workspace/test_file.py"
        res = write_file(ws_file, "def foo():\n    return 1\n", security_guard=guard)
        assert "Successfully created" in res

        # 2. write_file refuses overwrite
        res_dup = write_file(ws_file, "def bar():\n    return 2\n", security_guard=guard)
        assert "Error: File" in res_dup and "already exists" in res_dup

        # 3. read_file with line numbers
        content = read_file(ws_file, start_line=1, end_line=2, base_dir=root)
        assert "    1| def foo():" in content

        # 4. replace_in_file with CRLF and prefix handling
        pasted_old = "    1| def foo():\r\n    2|     return 1"
        res_rep = replace_in_file(ws_file, pasted_old, "def foo():\n    return 42", count=1, security_guard=guard)
        assert "Successfully updated" in res_rep
        updated_content = read_file(ws_file, base_dir=root)
        assert "return 42" in updated_content

        # 5. replace_in_file with missing text provides nearest-line hint
        res_miss = replace_in_file(ws_file, "def nonexistent():", "def new():", count=1, security_guard=guard)
        assert "not found" in res_miss

        # 6. check_syntax
        syn_ok = check_syntax(ws_file, base_dir=root)
        assert syn_ok == "Syntax OK"
        syn_bad = check_syntax("def broken(:\n    pass")
        assert "SyntaxError" in syn_bad


def test_utf8_subprocess_output() -> None:
    """Test that subprocess runner correctly decodes non-ASCII UTF-8 output under Windows cp1252."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir).resolve()
        script = root / "utf8_test.py"
        script.write_text(
            'print("BOX: ┌───┐ │ → │ └───┘ EMOJI: ⚡✨")\n',
            encoding="utf-8",
        )
        out = run_python_test(str(script), base_dir=root)
        assert "TEST PASS" in out
        assert "┌───┐" in out
        assert "⚡✨" in out


def test_harness_code_extraction() -> None:
    """Test extract_python_blocks on model replies with markdown fences."""
    reply = """
Here is the tool:
```python
class ToolA:
    pass
```
And here is the unit test:
```python
def test_a():
    assert True
```
"""
    blocks = extract_python_blocks(reply)
    assert len(blocks) == 2
    assert "class ToolA:" in blocks[0]
    assert "def test_a():" in blocks[1]


def test_vram_config_and_options() -> None:
    import importlib
    import ogun_core.config as cfg

    keys = ("OGUN_KEEP_ALIVE", "OGUN_NUM_GPU", "OGUN_VRAM_PROFILE")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        importlib.reload(cfg)
        # the owner's profile table: 48 layers shown for max (99 is SENT so the output layer is offloaded too), 24, 0
        assert cfg.VRAM_PROFILES["MAX_PERFORMANCE"]["layers"] == 48 and cfg.VRAM_PROFILES["MAX_PERFORMANCE"]["num_gpu"] >= 49
        assert cfg.VRAM_PROFILES["SHARED_VRAM"]["num_gpu"] == 24 and cfg.VRAM_PROFILES["CPU_ONLY"]["num_gpu"] == 0
        assert cfg.get_vram_profile() == "SHARED_VRAM" and cfg.get_num_gpu() == 24 and cfg.KEEP_ALIVE == "2m"
        opts = cfg.llm_options()
        assert opts["num_gpu"] == 24 and opts["num_ctx"] == 8192 and opts["temperature"] == 0.1 and opts["top_p"] == 0.95
        assert opts["stop"] == ["<|im_end|>"] and cfg.llm_options(num_predict=5)["num_predict"] == 5

        # aliases
        for alias, key in (("max", "MAX_PERFORMANCE"), ("MAX-PERFORMANCE", "MAX_PERFORMANCE"), ("shared", "SHARED_VRAM"),
                           ("Shared VRAM", "SHARED_VRAM"), ("cpu", "CPU_ONLY"), ("cpu_only", "CPU_ONLY")):
            assert cfg.resolve_vram_profile(alias) == key, alias
        assert cfg.resolve_vram_profile("free") is None and cfg.resolve_vram_profile("banana") is None

        # runtime switching: read at CALL TIME, 0 is SENT (not dropped), same profile is not a change, bad name is not applied
        ok, changed, prof, n = cfg.set_vram_profile("max")
        assert ok and changed and prof == "MAX_PERFORMANCE" and n >= 49 and cfg.llm_options()["num_gpu"] == n
        assert cfg.set_vram_profile("max")[:2] == (True, False)
        assert cfg.set_vram_profile("banana")[0] is False and cfg.get_vram_profile() == "MAX_PERFORMANCE"
        assert cfg.set_vram_profile("cpu")[3] == 0 and cfg.llm_options()["num_gpu"] == 0 and "num_gpu" in cfg.llm_options()

        # environment: profile name, custom layer count, 'auto' omits, garbage falls back (never raises)
        os.environ["OGUN_VRAM_PROFILE"] = "max"
        importlib.reload(cfg)
        assert cfg.get_vram_profile() == "MAX_PERFORMANCE"
        os.environ["OGUN_VRAM_PROFILE"] = "banana"
        importlib.reload(cfg)
        assert cfg.get_vram_profile() == "SHARED_VRAM"
        os.environ.pop("OGUN_VRAM_PROFILE", None)
        os.environ["OGUN_NUM_GPU"] = "12"
        importlib.reload(cfg)
        assert cfg.get_vram_profile() == "CUSTOM(12)" and cfg.llm_options()["num_gpu"] == 12
        os.environ["OGUN_NUM_GPU"] = "auto"
        os.environ["OGUN_KEEP_ALIVE"] = "10m"
        importlib.reload(cfg)
        assert cfg.get_num_gpu() is None and "num_gpu" not in cfg.llm_options() and cfg.KEEP_ALIVE == "10m"
        assert cfg._parse_num_gpu("banana") == 24 and cfg._parse_num_gpu("-5") == 24 and cfg._parse_num_gpu("") == 24
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(cfg)


def _fake_ollama_module(seen):
    import types

    class FakeClient:
        def __init__(self, host=None, timeout=None):
            pass

        def chat(self, **kwargs):
            seen.append(kwargs)
            return {"message": {"content": "ok", "tool_calls": []}}

    mod = types.ModuleType("ollama")
    mod.Client = FakeClient
    return mod


def test_engine_sends_identical_load_options_and_unloads() -> None:
    import ogun_core.config as cfg
    import ogun_core.engine as eng

    cfg.set_vram_profile("shared")
    seen = []
    saved_mod = sys.modules.get("ollama")
    sys.modules["ollama"] = _fake_ollama_module(seen)
    try:
        eng.OllamaClient(model="m").chat([{"role": "user", "content": "hi"}])
    finally:
        if saved_mod is None:
            sys.modules.pop("ollama", None)
        else:
            sys.modules["ollama"] = saved_mod
    assert seen[0]["keep_alive"] == cfg.KEEP_ALIVE == "2m"
    assert seen[0]["options"]["num_gpu"] == 24 and seen[0]["options"]["num_ctx"] == 8192

    # unload(): sends {"model": ..., "keep_alive": 0} to /api/generate, and never raises
    import urllib.request

    captured = {}
    orig = urllib.request.urlopen

    class Resp:
        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def ok_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return Resp()

    def bad_urlopen(req, timeout=None):
        raise OSError("connection refused")

    try:
        urllib.request.urlopen = ok_urlopen
        res = eng.unload_model(model="qwen2.5-coder:14b", host="http://127.0.0.1:11434")
        assert res["unloaded"] is True
        assert captured["url"].endswith("/api/generate") and captured["body"]["model"] == "qwen2.5-coder:14b"
        assert captured["body"]["keep_alive"] == 0 and captured["timeout"] <= 10
        urllib.request.urlopen = bad_urlopen
        res2 = eng.unload_model()
        assert res2["unloaded"] is False and "Ollama" in res2["detail"]          # a down daemon is not an error
        assert eng.unload_model()["unloaded"] is False                            # idempotent
    finally:
        urllib.request.urlopen = orig


def test_vram_toggle_reaches_every_chat_call() -> None:
    """After /vram changes the profile, the VERY NEXT chat() payload carries the new num_gpu (nothing caches the old one)."""
    import ogun_core.config as cfg
    import ogun_core.engine as eng

    seen = []
    saved_mod = sys.modules.get("ollama")
    sys.modules["ollama"] = _fake_ollama_module(seen)
    try:
        client = eng.OllamaClient(model="m")
        for profile, expected in (("shared", 24), ("max", 99), ("cpu", 0), ("shared", 24)):
            cfg.set_vram_profile(profile)
            client.chat([{"role": "user", "content": "hi"}])
            assert seen[-1]["options"]["num_gpu"] == expected, (profile, seen[-1]["options"])
            assert seen[-1]["keep_alive"] == "2m"
    finally:
        cfg.set_vram_profile("shared")
        if saved_mod is None:
            sys.modules.pop("ollama", None)
        else:
            sys.modules["ollama"] = saved_mod


def test_vram_commands() -> None:
    import urllib.request
    import ogun_core.config as cfg
    import ogun_core.engine as eng

    for text in ("/vram", "/vram shared", "/VRAM Max", "  /vram free ", "/vram status", "/vram cpu", "/vram banana"):
        assert eng.is_vram_command(text), text
    for text in ("vram shared", "what is /vram", "/vramx", "/vram shared please", "share the vram budget", "", "/vram a b"):
        assert not eng.is_vram_command(text), text
    assert eng.handle_vram_command("hello there") is None

    unloads = []
    state = {"ps": {"models": []}, "down": False}
    orig = urllib.request.urlopen

    class Resp:
        def __init__(self, data):
            self.data = data

        def read(self):
            return json.dumps(self.data).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        if state["down"]:
            raise OSError("connection refused")
        url = req if isinstance(req, str) else req.full_url
        if url.endswith("/api/ps"):
            return Resp(state["ps"])
        unloads.append(json.loads(req.data.decode("utf-8")))
        return Resp({})

    urllib.request.urlopen = fake_urlopen
    try:
        cfg.set_vram_profile("shared")
        # already shared -> no change, no unload
        reply = eng.handle_vram_command("/vram shared", model="m")
        assert "Already using SHARED_VRAM" in reply and unloads == []
        # switching -> profile changes, the model is unloaded once, the owner is warned about the reload
        reply = eng.handle_vram_command("/vram max", model="m")
        assert cfg.get_vram_profile() == "MAX_PERFORMANCE" and cfg.llm_options()["num_gpu"] >= 49
        assert len(unloads) == 1 and unloads[0] == {"model": "m", "prompt": "", "keep_alive": 0, "stream": False}
        assert "MAX_PERFORMANCE" in reply and "40 seconds" in reply
        reply = eng.handle_vram_command("/vram cpu", model="m")
        assert cfg.llm_options()["num_gpu"] == 0 and len(unloads) == 2 and "CPU_ONLY" in reply
        # /vram free -> unload, profile untouched
        reply = eng.handle_vram_command("/vram free", model="m")
        assert len(unloads) == 3 and cfg.get_vram_profile() == "CPU_ONLY" and "unloaded" in reply.lower()
        # status shows the profile, keep_alive and what Ollama holds
        state["ps"] = {"models": [{"name": "m", "size": 10 * 2**30, "size_vram": 5 * 2**30}]}
        reply = eng.handle_vram_command("/vram status", model="m")
        assert "CPU_ONLY" in reply and "keep_alive=2m" in reply and "50% on the GPU" in reply
        assert "Loaded: m" in eng.handle_vram_command("/vram", model="m")
        # unknown option -> help, nothing changes
        before = len(unloads)
        reply = eng.handle_vram_command("/vram banana", model="m")
        assert "Unknown /vram option" in reply and "/vram shared" in reply and len(unloads) == before
        # Ollama down: status and free degrade to plain messages, never raise
        state["down"] = True
        assert "No model is loaded" in eng.handle_vram_command("/vram status", model="m")
        assert "Could not unload" in eng.handle_vram_command("/vram free", model="m")
        assert "could not be reached" in eng.handle_vram_command("/vram shared", model="m")
    finally:
        urllib.request.urlopen = orig
        cfg.set_vram_profile("shared")


def test_main_unload_command_cli_and_no_stale_keepalive() -> None:
    import main as ogun_main

    for phrase in ("unload the model", "Free the GPU", "free up VRAM?", "/unload", "release gpu memory"):
        assert ogun_main.is_unload_command(phrase), phrase
    for phrase in ("unload the dishwasher", "what is vram", "please free some memory in my script", "", "unload model foo.py"):
        assert not ogun_main.is_unload_command(phrase), phrase

    # --unload is handled before the connection check, workspace setup or any engine construction
    calls = {"unload": 0, "check": 0}
    orig_unload, orig_check, orig_argv = ogun_main.unload_model, ogun_main.OllamaClient.check_connection, sys.argv
    ogun_main.unload_model = lambda model=None: calls.__setitem__("unload", calls["unload"] + 1) or {"unloaded": True, "detail": "done"}
    ogun_main.OllamaClient.check_connection = lambda self: calls.__setitem__("check", calls["check"] + 1) or (False, "down")
    sys.argv = ["main.py", "--unload"]
    try:
        try:
            ogun_main.main()
            raise AssertionError("main() should exit")
        except SystemExit as exit_:
            assert exit_.code == 0
    finally:
        ogun_main.unload_model, ogun_main.OllamaClient.check_connection, sys.argv = orig_unload, orig_check, orig_argv
    assert calls == {"unload": 1, "check": 0}, calls

    # nothing in Ogun may still carry the old 30-minute keep-alive
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for py in root.rglob("*.py"):
        if ".venv" in py.parts or ".venv-train" in py.parts or "__pycache__" in py.parts or py.name == "test_ogun_core.py":
            continue
        text = py.read_text(encoding="utf-8", errors="replace")
        if '"30m"' in text or "'30m'" in text:
            offenders.append(str(py.relative_to(root)))
    assert not offenders, offenders


def run_all_tests() -> bool:
    suites = [
        ("1. Parser Corpus (Fences, Bare, CRLF, Nested Braces, etc.)", test_parser_corpus),
        ("2. Security Guard (Allowed Dirs, Traversal, UNC, Sibling Prefix)", test_security_guard),
        ("3. BaseTool & ToolResult Contracts", test_basetool_contracts),
        ("4. AST Pre-Flight & Placeholder Checks", test_ast_validation),
        ("5. Dynamic Plugin Lifecycle & Registry Safety", test_plugin_lifecycle_and_registry_safety),
        ("6. Engine Turn Loop & Context Budget (Mock Client)", test_engine_turn_loop_with_mock_client),
        ("7. Codetools Operations (Write, Replace, Read, Syntax)", test_codetools_operations),
        ("8. UTF-8 Non-ASCII Subprocess Output (Charmap Trap)", test_utf8_subprocess_output),
        ("9. Test Harness Two-Block Code Extraction", test_harness_code_extraction),
        ("10. VRAM profiles: table, aliases, runtime switching, env overrides", test_vram_config_and_options),
        ("11. Engine sends the active load options; unload_model", test_engine_sends_identical_load_options_and_unloads),
        ("12. unload command/CLI handling; no stale 30m", test_main_unload_command_cli_and_no_stale_keepalive),
        ("13. /vram toggle reaches the very next chat call", test_vram_toggle_reaches_every_chat_call),
        ("14. /vram commands (shared, max, cpu, free, status, help, Ollama down)", test_vram_commands),
    ]

    print("=" * 60)
    print("  Ogun Core Offline Test Suite")
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
    success = run_all_tests()
    sys.exit(0 if success else 1)
