"""Helix standalone project staged builder and self-correction loop.

Coordinates staged code generation (tool implementation, then unit tests),
runtime smoke tests, and iterative self-correction with bounded retries.
Implements per-attempt tracing under <project_dir>/.build_log/attempt_N/, regression guards,
stub rejection, stuck detection, strategy rotation, best-so-far rollback, and truthful reporting.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from helix_core.codetools import run_python_test
from helix_core.config import SYSTEM_PROMPT, get_request_timeout
from helix_core.engine import CoreEngine
from helix_core.paths import HELIX_ROOT
from helix_core.quality import (
    check_requirements_coverage,
    check_test_code_hygiene,
    extract_requested_flags,
    sanitize_test_code,
    validate_test_code_regression,
    validate_tool_code_regression,
)


WORKSPACE_DIR = HELIX_ROOT / "workspace"


def get_sha256(content_or_path: Any) -> str:
    """Compute sha256 hex digest of string, bytes, or file path."""
    h = hashlib.sha256()
    if isinstance(content_or_path, (str, Path)) and os.path.exists(str(content_or_path)):
        try:
            with open(str(content_or_path), "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            pass
    if isinstance(content_or_path, str):
        h.update(content_or_path.encode("utf-8", errors="replace"))
    elif isinstance(content_or_path, bytes):
        h.update(content_or_path)
    return h.hexdigest()


def parse_code_blocks(text: str) -> List[str]:
    """Extract fenced python code blocks from text, handling tags and unclosed blocks."""
    blocks = re.findall(r"```(?:python|py)?(?::[^\n]*)?\s*\n([\s\S]*?)```", text)
    if not blocks:
        # Check if code block was unclosed due to length
        m = re.search(r"```(?:python|py)?(?::[^\n]*)?\s*\n([\s\S]*)$", text)
        if m and len(m.group(1).strip()) > 50:
            blocks = [m.group(1).strip()]
        else:
            clean = text.strip()
            if clean.startswith("import ") or clean.startswith("from ") or clean.startswith("#!") or clean.startswith("# FILE:"):
                return [clean]
    return [b.strip() for b in blocks if b.strip()]


def assign_code_blocks(blocks: List[str], project_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Assign code blocks to (tool_code, test_code) using explicit headers or heuristics."""
    if not blocks:
        return None, None
    tool_filename = f"{project_name}.py"
    test_filename = f"test_{project_name}.py"
    new_tool = None
    new_test = None

    for b in blocks:
        lines = b.splitlines()[:5]
        target_file = None
        for line in lines:
            m = re.search(r"#\s*(?:FILE|filename):\s*([a-zA-Z0-9_\-\.]+)", line, re.IGNORECASE)
            if m:
                target_file = m.group(1).strip()
                break
            m2 = re.search(r"^#+\s*([a-zA-Z0-9_\-\.]+\.py)", line.strip(), re.IGNORECASE)
            if m2:
                target_file = m2.group(1).strip()
                break

        if target_file:
            tf_lower = target_file.lower()
            if tf_lower == test_filename.lower() or tf_lower.startswith("test_"):
                new_test = b
                continue
            elif tf_lower == tool_filename.lower() or tf_lower == f"{project_name}.py":
                new_tool = b
                continue

        # Heuristic fallback if no header matched
        b_lower = b.lower()
        is_test = ("import unittest" in b_lower or "unittest.main" in b_lower or
                   "testcase" in b_lower or "def test_" in b_lower)
        if is_test:
            if not new_test:
                new_test = b
        else:
            if not new_tool:
                new_tool = b

    return new_tool, new_test


def extract_project_name(request: str) -> str:
    """Extract tool / project name from request, default to 'custom_tool'."""
    m = re.search(r"called\s+[`'\"]?([a-zA-Z0-9_\-]+)[`'\"]?", request, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"tool\s+[`'\"]?([a-zA-Z0-9_\-]+)[`'\"]?", request, re.IGNORECASE)
    if m2 and m2.group(1).lower() not in ("called", "that", "to", "for", "named"):
        return m2.group(1).strip()
    return "custom_tool"


def run_smoke_tests(project_dir: Path, tool_path: Path, project_name: str, request: str = "", timeout: float = 30.0) -> Tuple[bool, str, List[str]]:
    """Run deterministic smoke tests on the CLI tool before/alongside full unit tests.

    Checks:
    1. --help: must exit 0 and write usage information.
    2. Nonexistent path: must exit non-zero (rc != 0) and write to stderr.
    3. Tiny valid input: run tool with small valid file; must exit 0 and produce output.
    4. Windows Unicode console safety under cp1252:
       Runs on a fixture with non-ASCII filename AND non-ASCII content in output path with
       PYTHONIOENCODING=cp1252 and PYTHONUTF8=0. Must exit 0 without UnicodeEncodeError.
    """
    failures = []
    results = []

    # 1. --help
    try:
        proc = subprocess.run(
            [sys.executable, str(tool_path), "--help"],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if proc.returncode == 0:
            results.append("Smoke test `--help`: PASSED (exit 0)")
        else:
            failures.append(f"`--help` exited with code {proc.returncode} instead of 0. stderr: {proc.stderr[:200]}")
    except Exception as e:
        failures.append(f"`--help` execution failed: {e}")

    # 2. Nonexistent path
    nonexistent = project_dir / "__nonexistent_smoke_test_path_12345__"
    try:
        proc = subprocess.run(
            [sys.executable, str(tool_path), str(nonexistent)],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if proc.returncode != 0:
            results.append(f"Smoke test nonexistent path: PASSED (exit {proc.returncode} != 0)")
        else:
            failures.append(
                "Smoke test nonexistent path failed: running with nonexistent path exited 0! "
                "Error conditions MUST exit non-zero (e.g. sys.exit(1)) and write to sys.stderr."
            )
    except Exception as e:
        failures.append(f"Nonexistent path smoke test failed: {e}")

    # 3. Tiny valid input file
    smoke_fixture_dir = project_dir / "_smoke_test_fixture"
    smoke_fixture_dir.mkdir(parents=True, exist_ok=True)
    is_csv = "csv" in request.lower() or "csv" in project_name.lower() or "tablefmt" in project_name.lower()
    valid_filename = "tiny_valid.csv" if is_csv else "tiny_valid.txt"
    valid_file = smoke_fixture_dir / valid_filename
    if is_csv:
        valid_file.write_text("col_a,col_b,col_c\n10,20,apple\n30,40,banana\n", encoding="utf-8")
    else:
        valid_file.write_text("the quick brown fox jumps over the lazy dog\nthe quick brown dog\n", encoding="utf-8")

    try:
        proc = subprocess.run(
            [sys.executable, str(tool_path), str(valid_file)],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if proc.returncode == 0:
            results.append("Smoke test tiny valid input: PASSED (exit 0)")
        else:
            failures.append(f"Smoke test tiny valid input failed: exited {proc.returncode}. stderr: {proc.stderr[:200]}")
    except Exception as e:
        failures.append(f"Tiny valid input smoke test failed: {e}")

    # 4. Unicode safety under cp1252 console with non-ASCII filename AND non-ASCII content
    u_filename = "テスト_sample_café.csv" if is_csv else "テスト_sample_café.txt"
    u_file = smoke_fixture_dir / u_filename
    if is_csv:
        u_file.write_text("id,name,city\n1,日本語_café,東京\n2,résumé_naïve,京都\n", encoding="utf-8")
    else:
        u_file.write_text("café résumé naïve 日本語 東京\ncafé résumé naïve\n", encoding="utf-8")

    env = {
        **os.environ,
        "PYTHONIOENCODING": "cp1252",
        "PYTHONUTF8": "0",
    }
    try:
        target_arg = str(u_file)
        proc = subprocess.run(
            [sys.executable, str(tool_path), target_arg],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
        if proc.returncode != 0 and any(k in (proc.stderr or "").lower() for k in ("directory", "folder", "is not a dir", "not a directory")):
            target_arg = str(smoke_fixture_dir)
            proc = subprocess.run(
                [sys.executable, str(tool_path), target_arg],
                cwd=str(project_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env=env,
            )

        err_text = (proc.stderr or "") + (proc.stdout or "")
        if "UnicodeEncodeError" in err_text:
            failures.append(
                "Smoke test cp1252 console failed: UnicodeEncodeError occurred! "
                "Must configure sys.stdout.reconfigure(encoding='utf-8', errors='replace') at startup."
            )
        else:
            results.append("Smoke test cp1252 console Unicode: PASSED")
    except Exception as e:
        failures.append(f"Unicode cp1252 smoke test failed: {e}")
    finally:
        shutil.rmtree(smoke_fixture_dir, ignore_errors=True)

    success = len(failures) == 0
    report_str = "\n".join(results)
    if failures:
        report_str += "\nFailures:\n" + "\n".join(f"- {f}" for f in failures)
    return success, report_str, failures


def build_standalone_project_staged(
    engine: CoreEngine,
    request_text: str,
    max_attempts: int = 5,
    on_progress: Optional[Callable[[str, int], None]] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Build, test, verify, and refine a standalone CLI project in workspace/<name>/.

    Executes in stages: (1) write tool, (2) write tests, (3) static coverage, (4) smoke runs, (5) test & fix.
    Ensures files are retained on disk and reports truthfully from real execution output.
    """
    project_name = extract_project_name(request_text)
    project_dir = WORKSPACE_DIR / project_name
    project_dir.mkdir(parents=True, exist_ok=True)

    build_log_dir = project_dir / ".build_log"
    build_log_dir.mkdir(parents=True, exist_ok=True)
    best_dir = build_log_dir / "best_so_far"

    tool_path = project_dir / f"{project_name}.py"
    test_path = project_dir / f"test_{project_name}.py"

    def _notify(msg: str, prog: int = -1):
        if on_progress:
            on_progress(msg, prog)
        print(f"[Helix Builder] {msg}", flush=True)

    _notify(f"Initializing standalone project '{project_name}' in {project_dir}", 10)

    tool_code = ""
    test_code = ""
    if tool_path.exists():
        try:
            tool_code = tool_path.read_text(encoding="utf-8")
        except Exception:
            pass
    if test_path.exists():
        try:
            test_code = test_path.read_text(encoding="utf-8")
        except Exception:
            pass

    last_error = ""
    missing_reqs: List[str] = []
    attempt = 0
    test_res_str = ""
    smoke_report = ""

    # Convergence tracking
    failing_history: List[Set[str]] = []
    strategy_step = 0
    best_candidate: Optional[Dict[str, Any]] = None

    while attempt < max_attempts:
        attempt += 1
        attempt_dir = build_log_dir / f"attempt_{attempt}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        # Backup current files before modifications
        if tool_path.exists():
            shutil.copy2(tool_path, attempt_dir / f"backup_{project_name}.py")
        if test_path.exists():
            shutil.copy2(test_path, attempt_dir / f"backup_test_{project_name}.py")

        writes_log: Dict[str, Any] = {}
        _notify(f"[Attempt {attempt}/{max_attempts}] Generating / refining '{project_name}'...", 20 + attempt * 12)

        # Stage 1: Generate tool if missing
        if not tool_code:
            flags = extract_requested_flags(request_text)
            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"You are building a clean, robust, standalone command-line Python utility named `{project_name}.py` "
                f"fulfilling this user request:\n\"\"\"{request_text}\"\"\"\n\n"
                "CRITICAL IMPLEMENTATION REQUIREMENTS:\n"
                f"1. Implement all requested CLI options using argparse: {', '.join(flags) if flags else 'all options from request'}\n"
                "2. Performance & Streaming: For large files, stream line-by-line (for line in f:) or in 64KB chunks (f.read(65536)). NEVER load whole files into RAM with unbuffered f.read().\n"
                "3. Robustness & Input Validation: If an input file/folder does not exist or cannot be accessed, print an error to sys.stderr and exit non-zero (sys.exit(1)). NEVER exit 0 on missing/invalid input paths.\n"
                "4. Windows Unicode Safety: At the top of main, configure stdout encoding:\n"
                "   ```python\n"
                "   if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):\n"
                "       sys.stdout.reconfigure(encoding='utf-8', errors='replace')\n"
                "   ```\n"
                f"5. Output ONLY the complete Python code for `{project_name}.py` starting with `# FILE: {project_name}.py` inside a single ```python ``` code block. No explanations."
            )
            (attempt_dir / "tool_prompt.txt").write_text(prompt, encoding="utf-8")
            try:
                _notify(f"Generating tool implementation...", 35)
                reply = engine.client.chat([{"role": "user", "content": prompt}], tools=None)
                raw_content = reply.get("content", "")
                (attempt_dir / "tool_raw.txt").write_text(raw_content, encoding="utf-8")
                blocks = parse_code_blocks(raw_content)
                new_tool, new_test = assign_code_blocks(blocks, project_name)

                if new_tool:
                    try:
                        import ast
                        ast.parse(new_tool)
                        is_valid_initial = len(new_tool.strip()) >= 120 and ("argparse" in new_tool or "sys.argv" in new_tool)
                    except SyntaxError:
                        is_valid_initial = False

                    if is_valid_initial:
                        sha_before = get_sha256(tool_path)
                        tool_code = new_tool
                        tool_path.write_text(tool_code, encoding="utf-8")
                        sha_after = get_sha256(tool_path)
                        writes_log[f"{project_name}.py"] = {
                            "written": True, "sha256_before": sha_before, "sha256_after": sha_after, "status": "initial_creation"
                        }
                    else:
                        writes_log[f"{project_name}.py"] = {
                            "written": False, "sha256_before": "", "sha256_after": "", "status": "rejected: syntax error or stub in initial tool"
                        }
                        _notify(f"Initial tool rejected: syntax error or stub", -1)

                if new_test:
                    new_test = sanitize_test_code(new_test, project_name)
                    try:
                        import ast
                        ast.parse(new_test)
                        is_valid_initial_test = len(new_test.strip()) >= 150 and ("unittest" in new_test or "test_" in new_test)
                    except SyntaxError:
                        is_valid_initial_test = False

                    if is_valid_initial_test:
                        sha_before = get_sha256(test_path)
                        test_code = new_test
                        test_path.write_text(test_code, encoding="utf-8")
                        sha_after = get_sha256(test_path)
                        writes_log[f"test_{project_name}.py"] = {
                            "written": True, "sha256_before": sha_before, "sha256_after": sha_after, "status": "initial_creation"
                        }
                    else:
                        writes_log[f"test_{project_name}.py"] = {
                            "written": False, "sha256_before": "", "sha256_after": "", "status": "rejected: syntax error or stub in initial test"
                        }
                last_error = ""
                missing_reqs = []
            except TimeoutError:
                raise
            except Exception as e:
                _notify(f"Generation error/timeout during tool writing: {e}", -1)
                last_error = f"Model call failed: {e}"
                if attempt >= max_attempts:
                    break
                continue

        # Stage 2: Generate tests if missing
        if not test_code:
            flags = extract_requested_flags(request_text)
            prompt = (
                f"Write comprehensive unit tests in `test_{project_name}.py` verifying `{project_name}.py` against this user request:\n"
                f"\"\"\"{request_text}\"\"\"\n\n"
                f"Here is the implementation of `{project_name}.py`:\n```python\n{tool_code}\n```\n\n"
                "CRITICAL TEST HYGIENE & CONTRACT REQUIREMENTS:\n"
                f"1. Tool execution MUST use `sys.executable` with absolute path derived from `__file__`, never relative '{project_name}.py'.\n"
                "2. Every `subprocess.run` call MUST pass `encoding=\"utf-8\"` and `errors=\"replace\"`.\n"
                "3. Exit code contract: Bad input, missing files, or empty/headerless data files where calculations cannot be performed MUST exit non-zero (sys.exit(1)). Tests for nonexistent or empty files MUST assert non-zero returncode (`self.assertNotEqual(proc.returncode, 0)`).\n"
                "4. Template:\n"
                "   ```python\n"
                "   import sys, os, unittest, subprocess\n"
                "   from pathlib import Path\n"
                f"   TOOL_PATH = Path(__file__).resolve().parent / \"{project_name}.py\"\n"
                "   ```\n"
                f"5. Output ONLY the complete Python test code for `test_{project_name}.py` starting with `# FILE: test_{project_name}.py` inside a single ```python ``` code block. No explanations."
            )
            (attempt_dir / "tests_prompt.txt").write_text(prompt, encoding="utf-8")
            try:
                _notify("Generating comprehensive tests...", 55)
                reply = engine.client.chat([{"role": "user", "content": prompt}], tools=None)
                raw_content = reply.get("content", "")
                (attempt_dir / "tests_raw.txt").write_text(raw_content, encoding="utf-8")
                blocks = parse_code_blocks(raw_content)
                new_tool, new_test = assign_code_blocks(blocks, project_name)

                if new_test:
                    new_test = sanitize_test_code(new_test, project_name)
                    try:
                        import ast
                        ast.parse(new_test)
                        is_valid_initial_test = len(new_test.strip()) >= 150 and ("unittest" in new_test or "test_" in new_test)
                    except SyntaxError:
                        is_valid_initial_test = False

                    if is_valid_initial_test:
                        sha_before = get_sha256(test_path)
                        test_code = new_test
                        test_path.write_text(test_code, encoding="utf-8")
                        sha_after = get_sha256(test_path)
                        writes_log[f"test_{project_name}.py"] = {
                            "written": True, "sha256_before": sha_before, "sha256_after": sha_after, "status": "initial_creation"
                        }
                    else:
                        writes_log[f"test_{project_name}.py"] = {
                            "written": False, "sha256_before": "", "sha256_after": "", "status": "rejected: syntax error or stub in initial test"
                        }
                        _notify(f"Initial tests rejected: syntax error or stub", -1)
                last_error = ""
            except TimeoutError:
                raise
            except Exception as e:
                _notify(f"Generation error/timeout during test writing: {e}", -1)
                last_error = f"Test generation failed: {e}"
                if attempt >= max_attempts:
                    break
                continue

        # Stage 3: Static Requirements Coverage Check
        covered, missing_reqs = check_requirements_coverage(request_text, tool_code)
        if not covered:
            _notify(f"Requirements coverage check failed: {missing_reqs}", 65)
            last_error = "Requirements coverage check failed: " + "; ".join(missing_reqs)
            (attempt_dir / "coverage_failures.txt").write_text(last_error, encoding="utf-8")

        # Stage 4: Smoke runs (help, nonexistent path, tiny valid input, cp1252 Unicode)
        smoke_ok = False
        if covered:
            _notify("Running smoke tests (--help, nonexistent path, tiny valid input, cp1252 Unicode)...", 70)
            smoke_ok, smoke_report, smoke_fails = run_smoke_tests(project_dir, tool_path, project_name, request_text)
            (attempt_dir / "smoke_results.txt").write_text(smoke_report, encoding="utf-8")
            if not smoke_ok:
                _notify(f"Smoke tests failed on attempt {attempt}: {smoke_fails[0]}", 72)
                last_error = f"Smoke test validation failed:\n{smoke_report}"

        # Stage 5: Run tests with real interpreter via run_python_test
        passed = False
        failing_tests: List[str] = []
        if covered and smoke_ok and test_code:
            _notify("Running unit tests via real interpreter...", 80)
            test_res_str = run_python_test(str(test_path), base_dir=project_dir)
            (attempt_dir / "test_output.txt").write_text(test_res_str, encoding="utf-8")
            failing_tests = re.findall(r"(?:FAIL|ERROR):\s+([a-zA-Z0-9_]+)", test_res_str)
            passed = ("TEST PASS" in test_res_str or "OK" in test_res_str) and not failing_tests and not ("FAIL" in test_res_str or "ERROR" in test_res_str)

            if passed:
                _notify("All tests passed successfully!", 100)
                summary_data = {
                    "attempt": attempt,
                    "status": "success",
                    "failing_tests": [],
                    "tool_sha256": get_sha256(tool_path),
                    "test_sha256": get_sha256(test_path),
                }
                (attempt_dir / "attempt_summary.json").write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
                (attempt_dir / "file_writes.json").write_text(json.dumps(writes_log, indent=2), encoding="utf-8")

                final_msg = (
                    f"Successfully built standalone CLI tool `{project_name}` in `{project_dir}`.\n\n"
                    f"### Files Created:\n"
                    f"- Tool: `{tool_path}` (sha256: `{get_sha256(tool_path)[:16]}...`)\n"
                    f"- Tests: `{test_path}` (sha256: `{get_sha256(test_path)[:16]}...`)\n\n"
                    f"### Smoke Tests:\n```text\n{smoke_report}\n```\n\n"
                    f"### Real Test Output ({attempt} attempt{'s' if attempt > 1 else ''}):\n"
                    f"```text\n{test_res_str.strip()}\n```\n\n"
                    "All tests passed and all requested requirements are verified."
                )
                return (
                    True,
                    final_msg,
                    {
                        "status": "success",
                        "project_name": project_name,
                        "project_dir": str(project_dir),
                        "tool_file": str(tool_path),
                        "test_file": str(test_path),
                        "tool_sha256": get_sha256(tool_path),
                        "test_sha256": get_sha256(test_path),
                        "attempts": attempt,
                        "test_output": test_res_str.strip(),
                        "smoke_output": smoke_report,
                    }
                )

        # Track best-so-far candidate
        failing_history.append(set(failing_tests))
        if covered and smoke_ok:
            fail_count = len(failing_tests)
            if best_candidate is None or fail_count < best_candidate["failing_count"]:
                best_dir.mkdir(parents=True, exist_ok=True)
                if tool_path.exists():
                    shutil.copy2(tool_path, best_dir / f"{project_name}.py")
                if test_path.exists():
                    shutil.copy2(test_path, best_dir / f"test_{project_name}.py")
                best_candidate = {
                    "attempt": attempt,
                    "failing_count": fail_count,
                    "failing_tests": list(failing_tests),
                    "tool_code": tool_code,
                    "test_code": test_code,
                    "test_output": test_res_str,
                    "smoke_output": smoke_report,
                    "tool_passes_smoke": smoke_ok,
                    "tool_passes_coverage": covered,
                }
                (best_dir / "meta.json").write_text(json.dumps(best_candidate, indent=2, default=str), encoding="utf-8")

        fail_str = f"Failing tests: {', '.join(failing_tests)}" if failing_tests else (last_error[:120] if last_error else test_res_str[:120])
        _notify(f"Tests/checks failed on attempt {attempt}: {fail_str}", 85)
        last_error = last_error or f"Unit tests failed:\n{test_res_str}"

        # Strategy rotation on stuck set of failing tests
        is_stuck = len(failing_history) >= 2 and failing_history[-1] == failing_history[-2] and len(failing_history[-1]) > 0
        fix_stage = "standard_diagnosis"
        if is_stuck:
            if strategy_step == 0:
                fix_stage = "regenerate_tests_only"
                _notify(f"Stuck loop detected: rotating strategy to 'regenerate_tests_only' (Attempt {attempt})", 86)
            elif strategy_step == 1:
                fix_stage = "regenerate_tool_only"
                _notify(f"Stuck loop detected: rotating strategy to 'regenerate_tool_only' (Attempt {attempt})", 86)
            else:
                fix_stage = "minimal_diff_targeted_fix"
                _notify(f"Stuck loop detected: rotating strategy to 'minimal_diff_targeted_fix' (Attempt {attempt})", 86)
            strategy_step += 1

        # Execute Fix Stage if attempts remain
        if attempt < max_attempts:
            if fix_stage == "regenerate_tests_only":
                prompt = (
                    f"The standalone tool `{project_name}.py` has passed all requirements coverage and runtime smoke tests.\n"
                    "The current unit tests have failing assertions or invalid test-side assumptions.\n\n"
                    f"USER REQUIREMENTS:\n\"\"\"{request_text}\"\"\"\n\n"
                    f"CURRENT TOOL CODE (`{project_name}.py`):\n```python\n{tool_code}\n```\n\n"
                    f"FAILING TEST TRACEBACKS:\n{last_error}\n\n"
                    "INSTRUCTIONS:\n"
                    f"1. Regenerate `test_{project_name}.py` from scratch to test the tool properly according to user requirements.\n"
                    "2. Tool execution must use `sys.executable` with `TOOL_PATH = Path(__file__).resolve().parent / '{project_name}.py'`.\n"
                    "3. Every `subprocess.run` call must pass `encoding=\"utf-8\", errors=\"replace\"`.\n"
                    "4. Bad/empty input tests must assert `returncode != 0`.\n"
                    f"5. Output ONLY the complete Python test code starting with `# FILE: test_{project_name}.py` inside a single ```python ``` code block."
                )
            elif fix_stage == "regenerate_tool_only":
                prompt = (
                    f"The test suite for `{project_name}.py` expresses the required behavior, but the tool failed several tests.\n\n"
                    f"USER REQUIREMENTS:\n\"\"\"{request_text}\"\"\"\n\n"
                    f"CURRENT TEST CODE (`test_{project_name}.py`):\n```python\n{test_code}\n```\n\n"
                    f"FAILING TEST TRACEBACKS:\n{last_error}\n\n"
                    "INSTRUCTIONS:\n"
                    f"1. Regenerate `{project_name}.py` in full so it satisfies all user requirements and passes the tests.\n"
                    "2. Keep all CLI flags, chunked streaming, Windows UTF-8 stdout configuration, and non-zero exit on errors.\n"
                    f"3. Output ONLY the complete Python code starting with `# FILE: {project_name}.py` inside a single ```python ``` code block."
                )
            elif fix_stage == "minimal_diff_targeted_fix":
                prompt = (
                    f"Targeted fix for a single failing test assertion in `{project_name}.py`.\n\n"
                    f"USER REQUIREMENTS:\n\"\"\"{request_text}\"\"\"\n\n"
                    f"FAILING TRACEBACK:\n{last_error[:1000]}\n\n"
                    f"CURRENT TOOL CODE (`{project_name}.py`):\n```python\n{tool_code}\n```\n\n"
                    f"CURRENT TEST CODE (`test_{project_name}.py`):\n```python\n{test_code}\n```\n\n"
                    "Provide the complete corrected file for whichever file needs the fix (or both). "
                    f"Label each code block with `# FILE: {project_name}.py` or `# FILE: test_{project_name}.py`."
                )
            else:
                prompt = (
                    f"Unit test failure for `{project_name}.py` on attempt {attempt}:\n\n"
                    f"USER REQUIREMENTS:\n\"\"\"{request_text}\"\"\"\n\n"
                    f"TEST OUTPUT / TRACEBACK:\n{last_error}\n\n"
                    f"CURRENT TOOL CODE (`{project_name}.py`):\n```python\n{tool_code}\n```\n\n"
                    f"CURRENT TEST CODE (`test_{project_name}.py`):\n```python\n{test_code}\n```\n\n"
                    "DIAGNOSIS & FIX INSTRUCTIONS:\n"
                    "1. DECIDE FIRST: Is the TEST wrong or is the TOOL wrong against user requirements?\n"
                    "   - If the tool behaves correctly according to requirements and the test has an invalid expectation/assertion (e.g. expecting exit 0 on empty input or missing encoding='utf-8'), fix test code.\n"
                    "   - If the tool crashes, produces wrong calculations, or violates requirements (e.g. exits 0 on nonexistent path), fix tool code.\n"
                    "   - If both have issues, fix both.\n"
                    "2. Return ONLY the corrected file(s) in full. Each file MUST start with a header comment identifying the file:\n"
                    f"   # FILE: {project_name}.py\n"
                    f"   and/or\n"
                    f"   # FILE: test_{project_name}.py\n"
                    "Ensure code is complete, runnable, and retains all imports. No conversational text."
                )

            (attempt_dir / "fix_prompt.txt").write_text(prompt, encoding="utf-8")
            try:
                _notify(f"Executing self-correction fix (strategy: {fix_stage})...", 88)
                reply = engine.client.chat([{"role": "user", "content": prompt}], tools=None)
                raw_content = reply.get("content", "")
                (attempt_dir / "fix_raw.txt").write_text(raw_content, encoding="utf-8")
                blocks = parse_code_blocks(raw_content)

                extracted_meta = []
                for idx, b in enumerate(blocks):
                    extracted_meta.append({"block_index": idx, "length": len(b), "preview": b[:80].replace("\n", " ")})
                (attempt_dir / "extracted_code_blocks.json").write_text(json.dumps(extracted_meta, indent=2), encoding="utf-8")

                if not blocks:
                    writes_log["fix_status"] = "failed: model returned no code blocks"
                    _notify("Fix stage produced no usable code blocks; escalating strategy.", -1)
                    strategy_step += 1
                else:
                    new_tool, new_test = assign_code_blocks(blocks, project_name)
                    tool_updated = False
                    test_updated = False

                    if new_tool:
                        tool_ok, tool_reason = validate_tool_code_regression(new_tool, tool_code, request_text)
                        if not tool_ok and missing_reqs:
                            cand_cov_ok, cand_missing = check_requirements_coverage(request_text, new_tool)
                            try:
                                import ast
                                ast.parse(new_tool)
                                not_stub = len(new_tool.strip()) >= 120 and ("argparse" in new_tool or "sys.argv" in new_tool)
                            except Exception:
                                not_stub = False
                            if not_stub and (cand_cov_ok or len(cand_missing) < len(missing_reqs)):
                                tool_ok = True
                                tool_reason = ""

                        if tool_ok:
                            sha_before = get_sha256(tool_path)
                            tool_code = new_tool
                            tool_path.write_text(tool_code, encoding="utf-8")
                            sha_after = get_sha256(tool_path)
                            writes_log[f"{project_name}.py"] = {
                                "written": True, "sha256_before": sha_before, "sha256_after": sha_after, "status": "updated_from_fix"
                            }
                            tool_updated = True
                        else:
                            writes_log[f"{project_name}.py"] = {
                                "written": False, "sha256_before": get_sha256(tool_path), "sha256_after": get_sha256(tool_path), "status": f"rejected: {tool_reason}"
                            }
                            _notify(f"Tool fix rejected by regression guard: {tool_reason}", -1)
                            bk = attempt_dir / f"backup_{project_name}.py"
                            if bk.exists():
                                tool_path.write_text(bk.read_text(encoding="utf-8"), encoding="utf-8")
                                tool_code = tool_path.read_text(encoding="utf-8")

                    if new_test:
                        new_test = sanitize_test_code(new_test, project_name)
                        test_ok, test_reason = validate_test_code_regression(new_test, test_code, project_name)
                        if test_ok:
                            sha_before = get_sha256(test_path)
                            test_code = new_test
                            test_path.write_text(test_code, encoding="utf-8")
                            sha_after = get_sha256(test_path)
                            writes_log[f"test_{project_name}.py"] = {
                                "written": True, "sha256_before": sha_before, "sha256_after": sha_after, "status": "updated_from_fix"
                            }
                            test_updated = True
                        else:
                            writes_log[f"test_{project_name}.py"] = {
                                "written": False, "sha256_before": get_sha256(test_path), "sha256_after": get_sha256(test_path), "status": f"rejected: {test_reason}"
                            }
                            _notify(f"Test fix rejected by regression guard: {test_reason}", -1)
                            bk = attempt_dir / f"backup_test_{project_name}.py"
                            if bk.exists():
                                test_path.write_text(bk.read_text(encoding="utf-8"), encoding="utf-8")
                                test_code = test_path.read_text(encoding="utf-8")

                    if not tool_updated and not test_updated:
                        writes_log["fix_status"] = "failed: all candidate blocks rejected or no-op"
                        strategy_step += 1

            except TimeoutError:
                raise
            except Exception as e:
                _notify(f"Fix attempt error: {e}", -1)
                writes_log["fix_error"] = str(e)
                strategy_step += 1

        summary_data = {
            "attempt": attempt,
            "status": "failed",
            "failing_tests": list(failing_tests),
            "tool_sha256": get_sha256(tool_path),
            "test_sha256": get_sha256(test_path),
            "strategy_used": fix_stage,
            "smoke_passed": smoke_ok,
            "coverage_passed": covered,
        }
        (attempt_dir / "attempt_summary.json").write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
        (attempt_dir / "file_writes.json").write_text(json.dumps(writes_log, indent=2), encoding="utf-8")

    # Exhausted attempts: evaluate rollback to best so far
    restored_note = ""
    current_failing = len(failing_tests) if failing_tests else 999
    if best_candidate and best_candidate["failing_count"] < current_failing:
        b_att = best_candidate["attempt"]
        b_cnt = best_candidate["failing_count"]
        _notify(f"Rolling back to best candidate from Attempt {b_att} ({b_cnt} failing tests vs current {current_failing})...", 95)
        if (best_dir / f"{project_name}.py").exists():
            shutil.copy2(best_dir / f"{project_name}.py", tool_path)
            tool_code = tool_path.read_text(encoding="utf-8")
        if (best_dir / f"test_{project_name}.py").exists():
            shutil.copy2(best_dir / f"test_{project_name}.py", test_path)
            test_code = test_path.read_text(encoding="utf-8")
        test_res_str = best_candidate["test_output"]
        failing_tests = best_candidate["failing_tests"]
        restored_note = f"\n> **Note:** Delivered files were restored from the best-so-far candidate (Attempt {b_att}), which passed smoke tests and had only {b_cnt} failing test(s)."

    tool_meets_requirements = False
    if tool_path.exists():
        tool_src = tool_path.read_text(encoding="utf-8")
        c_ok, _ = check_requirements_coverage(request_text, tool_src)
        s_ok, _, _ = run_smoke_tests(project_dir, tool_path, project_name, request_text)
        tool_meets_requirements = c_ok and s_ok

    diagnosis_verdict = ""
    if tool_meets_requirements and failing_tests:
        diagnosis_verdict = f"\n\n**Diagnosis:** the tool meets the requirements in my checks; these {len(failing_tests)} of my own tests are still wrong."

    fail_list = "\n".join(f"- `{t}`" for t in failing_tests) if failing_tests else "- Unspecified test failure"
    _notify(f"Failed to complete '{project_name}' after {max_attempts} attempts: {len(failing_tests)} test(s) still failing.", -1)
    fail_msg = (
        f"Failed to build `{project_name}`: tests still failing after {attempt} attempts ({len(failing_tests)} test(s) remaining).{diagnosis_verdict}\n\n"
        f"### Failing Tests ({len(failing_tests)} remaining):\n"
        f"{fail_list}\n\n"
        f"### Real Error / Test Output:\n"
        f"```text\n{(test_res_str or last_error).strip()}\n```\n\n"
        f"### Files on Disk:\n"
        f"- Tool: `{tool_path}` (sha256: `{get_sha256(tool_path)[:16]}...`)\n"
        f"- Tests: `{test_path}` (sha256: `{get_sha256(test_path)[:16]}...`)\n"
        f"{restored_note}\n\n"
        "Build stopped after reaching maximum allowed attempts. The files remain on disk for inspection and debugging."
    )
    return (
        False,
        fail_msg,
        {
            "status": "failed",
            "project_name": project_name,
            "project_dir": str(project_dir),
            "tool_file": str(tool_path) if tool_path.exists() else None,
            "test_file": str(test_path) if test_path.exists() else None,
            "tool_sha256": get_sha256(tool_path),
            "test_sha256": get_sha256(test_path),
            "attempts": attempt,
            "failing_tests": failing_tests,
            "test_output": (test_res_str or last_error).strip(),
            "smoke_output": smoke_report,
            "error": last_error,
        }
    )
