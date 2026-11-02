"""Helix software quality and static analysis utilities.

Provides static requirements coverage verification and project classification.
"""
from __future__ import annotations

import ast
import re
from typing import List, Set, Tuple


_FLAG_PATTERN = re.compile(r"--([a-zA-Z0-9_\-]+)")
_PLUGIN_KEYWORDS = {"plugin", "dynamic tool", "helix tool", "add tool to helix", "basetool"}
_STANDALONE_PATTERNS = [
    re.compile(r"\bcommand-line tool\b", re.IGNORECASE),
    re.compile(r"\bcommand line tool\b", re.IGNORECASE),
    re.compile(r"\bcli tool\b", re.IGNORECASE),
    re.compile(r"\bcli utility\b", re.IGNORECASE),
    re.compile(r"\bstandalone tool\b", re.IGNORECASE),
    re.compile(r"\bstandalone script\b", re.IGNORECASE),
    re.compile(r"\bstandalone project\b", re.IGNORECASE),
    re.compile(r"\bstandalone utility\b", re.IGNORECASE),
    re.compile(r"\b(?:build|write|create|develop)\s+(?:me\s+)?(?:a\s+)?(?:python\s+)?program\b", re.IGNORECASE),
    re.compile(r"\b(?:build|write|create|develop)\s+(?:me\s+)?(?:a\s+)?tool\s+called\s+[`'\"]?([a-zA-Z0-9_\-]+)", re.IGNORECASE),
]


def is_standalone_project_intent(prompt: str) -> bool:
    """True if request is specifically asking to build a standalone CLI utility or script."""
    if not prompt or not isinstance(prompt, str):
        return False
    lower = prompt.lower()
    if any(k in lower for k in ("plugin", "dynamic tool", "helix tool", "basetool")):
        return False
    return any(p.search(prompt) for p in _STANDALONE_PATTERNS)


def classify_project_request(prompt: str) -> str:
    """Classify a user request as either a 'plugin', 'standalone', or 'general' project."""
    lower = prompt.lower()
    for kw in _PLUGIN_KEYWORDS:
        if kw in lower:
            return "plugin"
    if is_standalone_project_intent(prompt):
        return "standalone"
    return "general"


def extract_requested_flags(request_text: str) -> Set[str]:
    """Extract CLI options requested in text (e.g. --json, --min-size, --delimiter)."""
    flags: Set[str] = set()
    for match in _FLAG_PATTERN.finditer(request_text):
        flags.add(match.group(0))
    return flags


def check_requirements_coverage(request_text: str, code: str) -> Tuple[bool, List[str]]:
    """Verify that requested command-line flags and core features appear in generated code.

    Returns:
        (is_covered: bool, missing_requirements: list[str])
    """
    missing: List[str] = []
    code_lower = code.lower()

    # 1. Check CLI flags
    requested_flags = extract_requested_flags(request_text)
    for flag in requested_flags:
        flag_clean = flag.lstrip("-")
        flag_norm = flag_clean.replace("-", "_")
        if flag not in code and flag_clean not in code_lower and flag_norm not in code_lower:
            missing.append(f"Missing requested flag/option: {flag}")

    # 2. Check for wasted space if requested
    req_lower = request_text.lower()
    if ("wasted" in req_lower or "waste" in req_lower) and "space" in req_lower:
        wasted_terms = ["waste", "wasted", "disk_space", "space_wasted", "bytes_wasted"]
        if not any(term in code_lower for term in wasted_terms):
            missing.append("Missing duplicate disk space waste calculation/reporting.")

    # 3. Check for grouping if requested
    if "group" in req_lower:
        group_terms = ["group", "groups", "grouped", "file_hashes", "duplicates_dict", "hash_map", "by_hash"]
        if not any(term in code_lower for term in group_terms):
            missing.append("Missing duplicate grouping by content logic.")

    # 4. Check chunked streaming (no whole-file buffering `f.read()`)
    raw_read_matches = re.findall(r"\.read\(\s*\)", code)
    if raw_read_matches:
        missing.append("Unsafe full-file read detected: use chunked streaming (e.g. while chunk := f.read(65536)).")

    # 5. For standalone CLI tools, verify Windows UTF-8 stdout safety and error exit handling
    if is_standalone_project_intent(request_text):
        has_encoding_guard = any(k in code for k in ("reconfigure", "encoding='utf-8'", 'encoding="utf-8"', "errors='replace'", 'errors="replace"', "surrogateescape", "PYTHONUTF8"))
        if not has_encoding_guard and ("print(" in code or "sys.stdout" in code):
            missing.append("Missing Windows console UTF-8 safety: CLI source must configure stdout encoding (e.g. sys.stdout.reconfigure(encoding='utf-8', errors='replace')).")

        has_error_exit = any(k in code for k in ("sys.exit", "exit(", "parser.error", "SystemExit", "raise FileNotFoundError", "raise ValueError"))
        if not has_error_exit:
            missing.append("Missing non-zero exit on error conditions (e.g. sys.exit(1) on nonexistent or invalid input path).")

    return len(missing) == 0, missing


def check_test_code_hygiene(code: str, project_name: str) -> Tuple[bool, List[str]]:
    """Static check on test code ensuring hygiene requirements are met:
    - subprocess calls must pass encoding='utf-8' and errors='replace'
    - tool execution must use sys.executable with absolute path derived from __file__, never a relative name
    - tests must not assume cwd
    - tests for bad/empty input must not assert exit code 0
    """
    issues = []

    # 1. subprocess calls must pass encoding='utf-8'
    if "subprocess." in code:
        calls = re.findall(r"subprocess\.(?:run|Popen|check_output|check_call)\((.*?)\)", code, re.DOTALL)
        for call in calls:
            has_enc = ("encoding='utf-8'" in call or 'encoding="utf-8"' in call or
                       "encoding= 'utf-8'" in call or 'encoding= "utf-8"' in call)
            if not has_enc and any(k in call for k in ("text=True", "universal_newlines=True", "capture_output=True")):
                issues.append("subprocess execution must specify encoding='utf-8' (and errors='replace')")
                break

    # 2. Must run tool via sys.executable with absolute path derived from __file__, never a relative name
    rel_tool_pattern = rf"\[\s*sys\.executable\s*,\s*['\"]{re.escape(project_name)}\.py['\"]"
    if re.search(rel_tool_pattern, code):
        issues.append(f"tests must run tool via sys.executable with absolute path derived from __file__, not relative '{project_name}.py'")

    # 3. Bad/empty input must not assert exit 0
    bad_zero_patterns = [
        re.compile(r"def\s+test_[a-zA-Z0-9_]*(?:empty|missing|nonexist|invalid|bad)[a-zA-Z0-9_]*\(.*?\):.*?(?:assertEqual\(\s*(?:proc|res|result)\.returncode\s*,\s*0\s*\)|assertEqual\(\s*0\s*,\s*(?:proc|res|result)\.returncode\s*\))", re.DOTALL | re.IGNORECASE),
    ]
    for p in bad_zero_patterns:
        if p.search(code):
            issues.append("tests for empty or invalid input must assert non-zero returncode per error contract, not exit code 0")

    return len(issues) == 0, issues


def validate_tool_code_regression(candidate_code: str, current_code: str, request: str) -> Tuple[bool, str]:
    """Validate candidate tool code before accepting rewritten tool.

    Guards:
    1. Must parse (ast.parse)
    2. Must keep all requested CLI options (coverage check)
    3. Must not shrink drastically (< 40% of previous length)
    4. Must not be a stub (trivially short / prints of constants like 'Done')
    5. Must not be a no-op (identical to current file)
    """
    if not candidate_code or not candidate_code.strip():
        return False, "Candidate tool code is empty"

    clean_cand = candidate_code.strip()
    clean_curr = current_code.strip() if current_code else ""

    try:
        tree = ast.parse(clean_cand)
    except SyntaxError as e:
        return False, f"SyntaxError in rewritten tool: {e}"

    if clean_curr and clean_cand == clean_curr:
        return False, "No-op: candidate tool code is identical to current tool file"

    if len(clean_curr) >= 150:
        if len(clean_cand) < 0.40 * len(clean_curr):
            return False, f"Drastic code shrinkage: candidate length ({len(clean_cand)}) is < 40% of previous length ({len(clean_curr)})"

    if len(clean_cand) < 120:
        return False, f"Tool code rejected: trivially short ({len(clean_cand)} chars), likely a stub"

    if "argparse" not in clean_cand and "sys.argv" not in clean_cand:
        return False, "Tool code rejected: missing CLI argument parsing (argparse)"

    if re.search(r"print\s*\(\s*['\"]Done['\"]\s*\)", clean_cand) and "def " not in clean_cand and len(tree.body) < 10:
        has_real_logic = any(isinstance(node, (ast.For, ast.While, ast.With, ast.FunctionDef)) for node in tree.body)
        if not has_real_logic:
            return False, "Tool code rejected: detected degenerate stub printing 'Done'"

    cov_ok, missing = check_requirements_coverage(request, clean_cand)
    if not cov_ok:
        return False, f"Regression in requirements coverage: missing {', '.join(missing)}"

    return True, ""


def validate_test_code_regression(candidate_code: str, current_code: str, project_name: str) -> Tuple[bool, str]:
    """Validate candidate test code before accepting rewritten tests."""
    if not candidate_code or not candidate_code.strip():
        return False, "Candidate test code is empty"

    clean_cand = candidate_code.strip()
    clean_curr = current_code.strip() if current_code else ""

    try:
        tree = ast.parse(clean_cand)
    except SyntaxError as e:
        return False, f"SyntaxError in rewritten tests: {e}"

    if clean_curr and clean_cand == clean_curr:
        return False, "No-op: candidate test code is identical to current test file"

    if len(clean_cand) < 150:
        return False, f"Test code rejected: trivially short ({len(clean_cand)} chars), likely a stub"

    if "unittest" not in clean_cand and "test_" not in clean_cand:
        return False, "Test code rejected: missing unittest TestCase or test methods"

    hyg_ok, hyg_issues = check_test_code_hygiene(clean_cand, project_name)
    if not hyg_ok:
        return False, f"Test hygiene violation: {'; '.join(hyg_issues)}"

    return True, ""


def sanitize_test_code(code: str, project_name: str) -> str:
    """Auto-sanitize minor test hygiene defects:
    - ensure absolute TOOL_PATH derived from __file__ is used instead of relative name
    - ensure encoding='utf-8' and errors='replace' on subprocess.run
    """
    if not code:
        return code

    tool_filename = f"{project_name}.py"
    if "TOOL_PATH" not in code:
        header_addition = (
            "from pathlib import Path\n"
            f"TOOL_PATH = Path(__file__).resolve().parent / '{tool_filename}'\n"
        )
        if "import unittest" in code:
            code = code.replace("import unittest", f"import unittest\n{header_addition}", 1)
        else:
            code = f"{header_addition}\n{code}"

    # Replace relative command invocations
    code = re.sub(
        rf"\[\s*['\"]python[0-9\.]*['\"]\s*,\s*['\"]{re.escape(tool_filename)}['\"]",
        "[sys.executable, str(TOOL_PATH)",
        code
    )
    code = re.sub(
        rf"\[\s*sys\.executable\s*,\s*['\"]{re.escape(tool_filename)}['\"]",
        "[sys.executable, str(TOOL_PATH)",
        code
    )

    # Ensure encoding='utf-8', errors='replace'
    if 'encoding="utf-8"' not in code and "encoding='utf-8'" not in code:
        if "text=True" in code:
            code = code.replace("text=True", 'text=True, encoding="utf-8", errors="replace"')
        elif "capture_output=True" in code:
            code = code.replace("capture_output=True", 'capture_output=True, encoding="utf-8", errors="replace"')

    return code
