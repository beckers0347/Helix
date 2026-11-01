"""Helix Core Engine.

Coordinates the headless chat loop, Ollama API interactions, bounded context window,
native and text-based tool execution, and the dynamic plugin registry.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import urllib.error
import urllib.request

from helix_core.base_tool import BaseTool, ToolResult
from helix_core.codetools import (
    check_syntax,
    create_plugin,
    list_plugins,
    read_file,
    replace_in_file,
    run_python_test,
    write_file,
)
from helix_core.config import (
    KEEP_ALIVE,
    MODEL_NAME,
    NUM_CTX,
    NUM_PREDICT,
    OLLAMA_HOST,
    REQUEST_TIMEOUT,
    STOP_TOKENS,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
    VRAM_PROFILES,
    get_num_gpu,
    get_vram_profile,
    llm_options,
    get_num_ctx,
    get_hardware_profile,
    apply_hardware_profile,
    resolve_vram_profile,
    set_vram_profile,
)
from helix_core.paths import HELIX_ROOT
from helix_core.plugin_manager import PluginManager
from helix_core.security import PathSecurityGuard
from helix_core.toolcalls import parse_tool_calls

BUILTIN_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file, optionally bounded by line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to the file to read."},
                    "start_line": {"type": "integer", "description": "Optional starting line (1-indexed)."},
                    "end_line": {"type": "integer", "description": "Optional ending line (1-indexed)."},
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create and write a NEW file. Refuses to overwrite existing files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to the new file."},
                    "content": {"type": "string", "description": "Full file content to write."},
                },
                "required": ["filepath", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": "Replace an exact block of text in an existing file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to the existing file."},
                    "old_text": {"type": "string", "description": "Exact text block to replace."},
                    "new_text": {"type": "string", "description": "New replacement text block."},
                    "count": {"type": "integer", "description": "Expected number of occurrences (default 1)."},
                },
                "required": ["filepath", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_syntax",
            "description": "Verify Python syntax via AST parser on a file or code string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path_or_code": {"type": "string", "description": "File path or Python source code string."},
                },
                "required": ["path_or_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python_test",
            "description": "Execute a Python unit test in an isolated child process.",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_path": {"type": "string", "description": "Path to the test file to run."},
                },
                "required": ["test_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_plugin",
            "description": "Create, test, hot-load, and mount a new dynamic plugin.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the plugin (snake_case)."},
                    "code": {"type": "string", "description": "Complete Python source code for the tool class."},
                    "test_code": {"type": "string", "description": "Complete Python source code for the unit test."},
                },
                "required": ["name", "code", "test_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_plugins",
            "description": "List all currently active and mounted dynamic plugins.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class OllamaClient:
    """Client for Ollama API with fallback to standard library urllib."""

    def __init__(
        self,
        host: str = OLLAMA_HOST,
        model: str = MODEL_NAME,
        timeout: float = REQUEST_TIMEOUT,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def check_connection(self) -> Tuple[bool, str]:
        """Verify Ollama daemon is reachable."""
        url = f"{self.host}/api/tags"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if resp.status == 200:
                    return True, "Connected to Ollama."
                return False, f"Ollama returned HTTP status {resp.status}."
        except Exception as exc:
            return (
                False,
                f"Cannot connect to Ollama daemon at {self.host} ({exc}). "
                "Ensure Ollama is running ('ollama serve').",
            )

    def unload(self) -> Dict[str, Any]:
        """Tell Ollama to drop the model from memory NOW (`{"model": ..., "keep_alive": 0}`), freeing its GPU VRAM.

        Never raises: Ollama being down or the model not being loaded is not an error here.
        """
        payload = {"model": self.model, "prompt": "", "keep_alive": 0, "stream": False}
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                resp.read()
            return {"unloaded": True, "model": self.model, "detail": "Model unloaded; GPU memory released."}
        except Exception as exc:
            return {"unloaded": False, "model": self.model, "detail": f"Could not unload ({exc}). Ollama may not be running."}

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Send chat request to Ollama."""
        options = llm_options()

        # Try official ollama package first
        try:
            import ollama

            client = ollama.Client(host=self.host, timeout=self.timeout)
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "options": options,
                "keep_alive": KEEP_ALIVE,
            }
            if tools:
                kwargs["tools"] = tools

            resp = client.chat(**kwargs)
            msg = resp.get("message", {})
            return {
                "content": msg.get("content", ""),
                "tool_calls": msg.get("tool_calls", []),
            }
        except ImportError:
            pass
        except Exception as exc:
            # Re-raise timeout specifically
            if "timeout" in str(exc).lower():
                raise TimeoutError(str(exc))
            raise

        # Fallback to stdlib urllib
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "options": options,
            "keep_alive": KEEP_ALIVE,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                msg = data.get("message", {})
                return {
                    "content": msg.get("content", ""),
                    "tool_calls": msg.get("tool_calls", []),
                }
        except urllib.error.URLError as ue:
            if "timed out" in str(ue).lower():
                raise TimeoutError(str(ue))
            raise


class ContextManager:
    """Manages conversational history within context token boundaries."""

    def __init__(self, max_tokens: Optional[int] = None, budget_ratio: float = 0.85) -> None:
        if max_tokens is None:
            max_tokens = get_num_ctx()      # live: follows the auto-tuned context window
        self.max_tokens = int(max_tokens * budget_ratio)

    def estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        total_chars = 0
        for m in messages:
            content = m.get("content") or ""
            total_chars += len(content)
            for tc in m.get("tool_calls", []):
                total_chars += len(json.dumps(tc))
        return total_chars // 4

    def prune_history(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep system message and drop oldest non-system turns if over budget."""
        if not messages:
            return messages

        system_msg = messages[0] if messages[0].get("role") == "system" else None
        turns = messages[1:] if system_msg else messages[:]

        while turns and self.estimate_tokens(([system_msg] if system_msg else []) + turns) > self.max_tokens:
            turns.pop(0)

        return ([system_msg] if system_msg else []) + turns


class CoreEngine:
    """Helix Core Engine handling execution turns and tool dispatches."""

    def __init__(
        self,
        client: Optional[Any] = None,
        plugin_manager: Optional[PluginManager] = None,
        security_guard: Optional[PathSecurityGuard] = None,
        base_dir: Optional[Path] = None,
    ) -> None:
        self.base_dir = (base_dir or HELIX_ROOT).resolve()
        self.client = client or OllamaClient()
        self.plugin_manager = plugin_manager or PluginManager(base_dir=self.base_dir)
        self.security_guard = security_guard or PathSecurityGuard(base_dir=self.base_dir)
        self.context_manager = ContextManager()

        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def _get_active_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas = list(BUILTIN_TOOL_SCHEMAS)
        if self.plugin_manager:
            schemas.extend(self.plugin_manager.list_tools())
        return schemas

    def _get_registered_tool_names(self) -> Set[str]:
        names = {s["function"]["name"] for s in BUILTIN_TOOL_SCHEMAS}
        if self.plugin_manager:
            names.update(self.plugin_manager._active_tools.keys())
        return names

    def dispatch_tool(self, name: str, args: Dict[str, Any]) -> str:
        """Dispatch a tool call to built-in code tools or dynamic plugins."""
        try:
            if name == "read_file":
                return read_file(
                    filepath=args.get("filepath", ""),
                    start_line=args.get("start_line"),
                    end_line=args.get("end_line"),
                    base_dir=self.base_dir,
                )
            elif name == "write_file":
                return write_file(
                    filepath=args.get("filepath", ""),
                    content=args.get("content", ""),
                    security_guard=self.security_guard,
                )
            elif name == "replace_in_file":
                return replace_in_file(
                    filepath=args.get("filepath", ""),
                    old_text=args.get("old_text", ""),
                    new_text=args.get("new_text", ""),
                    count=int(args.get("count", 1)),
                    security_guard=self.security_guard,
                )
            elif name == "check_syntax":
                return check_syntax(
                    path_or_code=args.get("path_or_code", ""),
                    base_dir=self.base_dir,
                )
            elif name == "run_python_test":
                return run_python_test(
                    test_path=args.get("test_path", ""),
                    base_dir=self.base_dir,
                )
            elif name == "create_plugin":
                res = create_plugin(
                    name=args.get("name", ""),
                    code=args.get("code", ""),
                    test_code=args.get("test_code", ""),
                    plugin_manager=self.plugin_manager,
                )
                return json.dumps(res)
            elif name == "list_plugins":
                return json.dumps(list_plugins(self.plugin_manager))
            else:
                # Dynamic plugin execution
                res = self.plugin_manager.execute_plugin(name=name, **args)
                return json.dumps(res)
        except Exception as exc:
            return f"Error executing tool '{name}': {exc}"

    def run_turn(self, user_text: str, max_rounds: int = 10) -> str:
        """Run an agentic conversational turn with tool execution loop."""
        self.messages.append({"role": "user", "content": user_text})
        self.messages = self.context_manager.prune_history(self.messages)

        tool_schemas = self._get_active_tool_schemas()
        registered_names = self._get_registered_tool_names()

        for round_idx in range(max_rounds):
            try:
                response = self.client.chat(self.messages, tools=tool_schemas)
            except TimeoutError:
                err_msg = "Model request timed out after 300 seconds. Ollama may be loading weights or overloaded."
                self.messages.append({"role": "assistant", "content": err_msg})
                return err_msg
            except Exception as exc:
                err_msg = f"Inference error: {exc}"
                self.messages.append({"role": "assistant", "content": err_msg})
                return err_msg

            content = response.get("content", "")
            raw_tool_calls = response.get("tool_calls", [])

            # Extract tool calls: native first, then text parser
            tool_calls: List[Dict[str, Any]] = []
            if raw_tool_calls:
                for tc in raw_tool_calls:
                    fn = tc.get("function", {})
                    tname = fn.get("name")
                    if tname and tname in registered_names:
                        targs = fn.get("arguments", {})
                        if isinstance(targs, str):
                            try:
                                targs = json.loads(targs)
                            except Exception:
                                targs = {"raw": targs}
                        tool_calls.append({"name": tname, "arguments": targs})

            if not tool_calls and content:
                tool_calls = parse_tool_calls(content, registered_names)

            # If no tool calls, return visible text
            if not tool_calls:
                visible_text = content.strip() if content else "(No response generated)"
                self.messages.append({"role": "assistant", "content": visible_text})
                return visible_text

            # Execute tool calls
            self.messages.append({"role": "assistant", "content": content or ""})
            for tc in tool_calls:
                name = tc["name"]
                args = tc.get("arguments", {})
                result = self.dispatch_tool(name, args)
                self.messages.append(
                    {
                        "role": "tool",
                        "content": str(result),
                    }
                )

            # Prune context before next round
            self.messages = self.context_manager.prune_history(self.messages)

        final_msg = "Reached maximum tool execution rounds."
        self.messages.append({"role": "assistant", "content": final_msg})
        return final_msg


def unload_model(model: str = MODEL_NAME, host: str = OLLAMA_HOST) -> Dict[str, Any]:
    """Free the GPU: send {"model": MODEL_NAME, "keep_alive": 0} to Ollama. Safe to call any time, any number of times."""
    return OllamaClient(host=host, model=model).unload()


# ---------------------------------------------------------------------------------------------------------------------
# /vram commands (interactive chat)
# ---------------------------------------------------------------------------------------------------------------------
import re as _re

_VRAM_CMD = _re.compile(r"^\s*/vram(?:\s+(\S+))?\s*$", _re.IGNORECASE)
VRAM_HELP = (
    "Usage: /vram shared (half the layers on the GPU, leaves ~5-6 GB VRAM for Unreal / Fusion 360) | "
    "/vram max (all layers on the GPU, fastest) | /vram cpu (no GPU) | /vram free (unload the model now) | /vram status"
)


def is_vram_command(text: str) -> bool:
    """True only for '/vram' optionally followed by ONE word, so ordinary sentences are never hijacked."""
    return bool(_VRAM_CMD.match(text or ""))


def ollama_ps(host: str = OLLAMA_HOST) -> List[Dict[str, Any]]:
    """Models Ollama currently holds in memory (GET /api/ps). Never raises: returns [] when Ollama is unreachable."""
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/ps", timeout=5.0) as resp:
            return list(json.loads(resp.read().decode("utf-8")).get("models", []))
    except Exception:
        return []


def _describe_loaded(models: List[Dict[str, Any]]) -> str:
    if not models:
        return "No model is loaded right now (GPU memory is free)."
    parts = []
    for m in models:
        size, in_vram = float(m.get("size") or 0), float(m.get("size_vram") or 0)
        gpu = round(100 * in_vram / size) if size else 0
        parts.append(f"{m.get('name', '?')}: {gpu}% on the GPU ({in_vram / 2**30:.1f} of {size / 2**30:.1f} GB in VRAM)")
    return "Loaded: " + "; ".join(parts) + "."


def _vram_auto(model: str, host: str) -> str:
    """/vram auto: probe this machine again, tune against what is free RIGHT NOW (memory held by the loaded model counts as free), apply."""
    from helix_core import hardware
    try:
        hw = hardware.probe_hardware()
        oll = hardware.query_ollama(host, family=str(model).split(":")[0])
        hw = hardware.adjust_for_resident_models(hw, oll["resident_vram_gb"], oll["resident_ram_gb"])
        prof = hardware.auto_configure(hw, installed=oll["installed"] or None, overrides={"configured_model": model})
    except Exception as exc:  # never break the chat loop over a probe problem
        return f"Hardware auto-tune failed ({exc}); settings unchanged."
    before = (get_vram_profile(), get_num_gpu(), get_num_ctx())
    set_active = apply_hardware_profile(hardware.profile_to_dict(prof))
    if not set_active:
        return "Hardware auto-tune produced no usable profile; settings unchanged."
    after = (get_vram_profile(), get_num_gpu(), get_num_ctx())
    notes = " ".join(prof.warnings[:2])
    if before == after:
        return f"Already using the auto-tuned settings: {prof.model_tag}, mode {prof.mode}, num_gpu={prof.num_gpu}, num_ctx={prof.num_ctx}. {notes}".strip()
    result = unload_model(model=model, host=host)
    note = "The model was unloaded" if result["unloaded"] else "Ollama could not be reached to unload it"
    return (f"Auto-tuned: {prof.model_tag} [{prof.quant}], mode {prof.mode}, {prof.gpu_units}/{prof.total_units} layers on GPU "
            f"(num_gpu={prof.num_gpu}), num_ctx={prof.num_ctx}, num_thread={prof.num_thread}. {note}; the next reply reloads "
            f"the model with these settings. {notes}").strip()


def handle_vram_command(text: str, model: str = MODEL_NAME, host: str = OLLAMA_HOST) -> Optional[str]:
    """Run a /vram command. Returns the reply to print, or None when `text` is not a /vram command (never raises)."""
    match = _VRAM_CMD.match(text or "")
    if not match:
        return None
    arg = (match.group(1) or "status").strip().lower()

    if arg == "free":
        result = unload_model(model=model, host=host)
        if result["unloaded"]:
            return "Model unloaded: its GPU memory is free now. The next reply reloads it (about 40 seconds)."
        return f"Could not unload the model: {result['detail']}"

    if arg == "auto":
        return _vram_auto(model, host)

    if arg == "status":
        profile, num_gpu = get_vram_profile(), get_num_gpu()
        shown = "automatic" if num_gpu is None else f"num_gpu={num_gpu}"
        info = VRAM_PROFILES.get(profile, {}).get("summary", "")
        hw_prof = get_hardware_profile() if profile == "AUTO" else None
        if hw_prof:
            info = (f"auto-tuned: {hw_prof.get('model_tag')} [{hw_prof.get('mode')}], "
                    f"{hw_prof.get('gpu_units')}/{hw_prof.get('total_units')} layers on GPU, "
                    f"num_ctx={hw_prof.get('num_ctx')}, num_thread={hw_prof.get('num_thread')}")
        return (f"VRAM profile: {profile} ({shown}){' - ' + info if info else ''}. keep_alive={KEEP_ALIVE}. "
                f"{_describe_loaded(ollama_ps(host))}")

    key = resolve_vram_profile(arg)
    if key is None:
        return f"Unknown /vram option '{arg}'. {VRAM_HELP}"
    ok, changed, profile, num_gpu = set_vram_profile(key)
    summary = VRAM_PROFILES[profile]["summary"]
    if not changed:
        return f"Already using {profile}: {summary}."
    result = unload_model(model=model, host=host)       # a changed split forces a reload anyway: free the VRAM right now
    note = "The model was unloaded" if result["unloaded"] else "Ollama could not be reached to unload it"
    return (f"Switched to {profile} (num_gpu={num_gpu}): {summary}. {note}; your next reply takes about 40 seconds "
            "while the model reloads with the new split.")
