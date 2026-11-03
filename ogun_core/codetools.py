"""Ogun code-writing and file inspection tools.

Provides safe file operations, exact-text edits, AST syntax checks,
and subprocess unit-test runners guarded by PathSecurityGuard.
"""
from __future__ import annotations

import ast
import difflib
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Union

from ogun_core.base_tool import ToolResult
from ogun_core.paths import OGUN_ROOT
from ogun_core.security import PathSecurityGuard

_LINE_PREFIX_RE = re.compile(r"^\s*\d+\| ?")


def _strip_prefixes(text: str) -> str:
    """Strip line number prefixes like 'NNNNN| ' if present on lines."""
    if not text:
        return ""
    lines = text.split("\n")
    # Check if lines have prefixes
    prefixed = any(_LINE_PREFIX_RE.match(line) for line in lines)
    if not prefixed:
        return text
    stripped = [_LINE_PREFIX_RE.sub("", line) for line in lines]
    return "\n".join(stripped)


def read_file(
    filepath: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    base_dir: Optional[Path] = None,
) -> str:
    """Read contents of a file, optionally bounded by line numbers (1-indexed)."""
    root = (base_dir or OGUN_ROOT).resolve()
    raw = str(filepath).strip().strip('"').strip("'")
    p = Path(raw)
    if not p.is_absolute():
        p = (root / p).resolve()
    else:
        p = p.resolve()

    if not p.exists():
        return f"Error: File '{filepath}' does not exist."
    if not p.is_file():
        return f"Error: '{filepath}' is a directory, not a file."

    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Error reading file '{filepath}': {exc}"

    lines = content.splitlines()
    total_lines = len(lines)

    if start_line is not None or end_line is not None:
        start = max(1, start_line or 1)
        end = min(total_lines, end_line or total_lines)
        if start > total_lines:
            return f"Error: start_line {start} exceeds total file lines ({total_lines})."
        selected = lines[start - 1 : end]
        output_lines = [f"{start + i:5d}| {line}" for i, line in enumerate(selected)]
        return "\n".join(output_lines)

    return content


def write_file(
    filepath: str,
    content: str,
    security_guard: Optional[PathSecurityGuard] = None,
) -> str:
    """Write content to a NEW file. Refuses to overwrite existing files."""
    guard = security_guard or PathSecurityGuard()
    try:
        target = guard.validate_write(filepath)
    except PermissionError as pe:
        return str(pe)

    if target.exists():
        return f"Error: File '{filepath}' already exists. Use replace_in_file to modify existing files."

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Normalize CRLF
        normalized = content.replace("\r\n", "\n")
        target.write_text(normalized, encoding="utf-8")
        line_count = len(normalized.splitlines())
        return f"Successfully created '{filepath}' ({len(normalized)} bytes, {line_count} lines)."
    except Exception as exc:
        return f"Error writing file '{filepath}': {exc}"


def replace_in_file(
    filepath: str,
    old_text: str,
    new_text: str,
    count: int = 1,
    security_guard: Optional[PathSecurityGuard] = None,
) -> str:
    """Replace an exact block of text in an existing file."""
    guard = security_guard or PathSecurityGuard()
    try:
        target = guard.validate_write(filepath)
    except PermissionError as pe:
        return str(pe)

    if not target.exists():
        return f"Error: File '{filepath}' does not exist. Use write_file to create a new file."

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Error reading file '{filepath}': {exc}"

    # Normalize CRLF
    content = content.replace("\r\n", "\n")
    clean_old = _strip_prefixes(old_text.replace("\r\n", "\n"))
    clean_new = _strip_prefixes(new_text.replace("\r\n", "\n"))

    if not clean_old:
        return "Error: old_text is empty. Please provide the exact text to replace."

    occurrences = content.count(clean_old)
    if occurrences == 0:
        # Generate nearest-line hint
        lines = content.splitlines()
        old_lines = clean_old.splitlines()
        target_first_line = old_lines[0].strip() if old_lines else ""
        close_lines = []
        if target_first_line:
            for idx, line in enumerate(lines, 1):
                if difflib.SequenceMatcher(None, target_first_line, line.strip()).ratio() > 0.6:
                    close_lines.append(f"line {idx}: {line.strip()}")
        hint = f" Nearest candidate matches:\n  " + "\n  ".join(close_lines[:3]) if close_lines else ""
        return f"Error: old_text not found in '{filepath}'.{hint}"

    if occurrences > count:
        return (
            f"Error: old_text found {occurrences} times in '{filepath}' (expected {count}). "
            "Please include more surrounding context to disambiguate."
        )

    updated = content.replace(clean_old, clean_new, count)
    try:
        target.write_text(updated, encoding="utf-8")
        return f"Successfully updated '{filepath}' ({occurrences} replacement(s) made)."
    except Exception as exc:
        return f"Error saving changes to '{filepath}': {exc}"


def check_syntax(path_or_code: str, base_dir: Optional[Path] = None) -> str:
    """Verify Python syntax via ast.parse. Accepts a file path or raw code string."""
    root = (base_dir or OGUN_ROOT).resolve()
    code = path_or_code
    p = Path(path_or_code.strip().strip('"').strip("'"))
    if not p.is_absolute():
        cand = (root / p).resolve()
    else:
        cand = p.resolve()

    if cand.is_file():
        try:
            code = cand.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"Error reading file for syntax check: {exc}"

    try:
        ast.parse(code)
        return "Syntax OK"
    except SyntaxError as se:
        return f"SyntaxError at line {se.lineno}, col {se.offset}: {se.msg}"
    except Exception as exc:
        return f"AST parse error: {exc}"


def run_python_test(
    test_path: Union[str, Path],
    timeout: float = 30.0,
    base_dir: Optional[Path] = None,
) -> str:
    """Execute a Python test via module discovery and direct execution with summary verification."""
    root = (base_dir or OGUN_ROOT).resolve()
    raw = str(test_path).strip().strip('"').strip("'")
    p = Path(raw)
    if not p.is_absolute():
        p = (root / p).resolve()
    else:
        p = p.resolve()

    if not p.exists():
        return f"Error: Test file '{test_path}' does not exist."

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    sep = ";" if sys.platform == "win32" else ":"
    paths = [str(root), str(p.parent)]
    if str(OGUN_ROOT) != str(root):
        paths.append(str(OGUN_ROOT))
    existing_pp = env.get("PYTHONPATH")
    if existing_pp:
        paths.append(existing_pp)
    env["PYTHONPATH"] = sep.join(paths)

    has_unittest = False
    if p.is_file():
        try:
            c = p.read_text(encoding="utf-8")
            has_unittest = "unittest" in c or "ToolTestCase" in c or "def test_" in c
        except Exception:
            pass

    is_test_file = (p.name.startswith("test_") or p.name.endswith("_test.py")) and has_unittest

    if is_test_file:
        from ogun_core.quality import ensure_unittest_main_guard, run_and_parse_unittest, run_direct_test_file
        ensure_unittest_main_guard(p)

        summary = run_and_parse_unittest(p.parent, p.stem, timeout=timeout)
        direct_ok, direct_out, direct_ran = run_direct_test_file(p.parent, p, timeout=timeout)

        if not summary.success or summary.ran_count == 0 or (summary.ast_count > 0 and summary.ran_count < summary.ast_count) or not direct_ok:
            fail_reason = summary.error_message or summary.summary_line or "Test execution failed"
            if not direct_ok and summary.success:
                fail_reason = f"direct execution failed (ran {direct_ran} tests)"
            return (
                f"TEST FAIL ({fail_reason})\n"
                f"Stdout:\n{summary.raw_output}\n"
                + (f"Direct Run Output:\n{direct_out}\n" if not direct_ok else "")
            )

        return f"TEST PASS\n{summary.raw_output}"

    # Non-test plain script fallback
    try:
        proc = subprocess.run(
            [sys.executable, str(p)],
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
        )
        if proc.returncode == 0:
            return f"TEST PASS\n{proc.stdout}"
        else:
            return f"TEST FAIL (exit code {proc.returncode})\nStdout:\n{proc.stdout}\nStderr:\n{proc.stderr}"
    except subprocess.TimeoutExpired:
        return f"TEST TIMEOUT (exceeded {timeout}s)"
    except Exception as exc:
        return f"TEST EXECUTION ERROR: {exc}"


def list_plugins(plugin_manager: Any) -> List[Dict[str, Any]]:
    """Return active tools registered in PluginManager."""
    if hasattr(plugin_manager, "list_tools"):
        return plugin_manager.list_tools()
    return []


def create_plugin(
    name: str,
    code: str,
    test_code: str,
    plugin_manager: Any,
) -> ToolResult:
    """Create, test, and mount a dynamic plugin via PluginManager."""
    if hasattr(plugin_manager, "create_or_update_plugin"):
        return plugin_manager.create_or_update_plugin(name=name, code=code, test_code=test_code)
    return ToolResult.error("PluginManager not available.")
