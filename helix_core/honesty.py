"""Helix honesty guard.

Validates assistant responses against real tool execution logs.
Prevents hallucinated claims of file creation or test execution when
no corresponding tool actions were executed or succeeded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class TurnEvents:
    """Records real operations executed during a conversational turn."""
    files_written: List[Path] = field(default_factory=list)
    tests_executed: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    def add_file_written(self, path: Path) -> None:
        self.files_written.append(path)

    def add_test_execution(self, test_path: str, passed: bool, output: str) -> None:
        self.tests_executed.append({
            "test_path": test_path,
            "passed": passed,
            "output": output,
        })

    def has_successful_write(self, filepath_substring: str = "") -> bool:
        for p in self.files_written:
            if p.exists() and (not filepath_substring or filepath_substring.lower() in str(p).lower()):
                return True
        return False

    def has_successful_test(self, test_name_substring: str = "") -> bool:
        for t in self.tests_executed:
            if t.get("passed"):
                if not test_name_substring or test_name_substring.lower() in str(t.get("test_path", "")).lower():
                    return True
        return False


_TEST_OUTPUT_BLOCK_RE = re.compile(
    r"```(?:text|sh|bash)?\s*(?:-+\s*\n)?Ran\s+\d+\s+tests?\s+in\s+[\d\.]+s\s*\n\s*(?:OK|FAILED[^\n]*)\s*```",
    re.IGNORECASE | re.MULTILINE,
)

_BARE_TEST_OUTPUT_RE = re.compile(
    r"-{5,}\s*\n\s*Ran\s+\d+\s+tests?\s+in\s+[\d\.]+s\s*\n\s*OK",
    re.IGNORECASE | re.MULTILINE,
)

_CONFIRM_PHRASES_RE = re.compile(
    r"(?:This confirms that (?:the )?[^\n]+ (?:tool and its tests are working correctly|tests? are passing|tests? pass)\.?|"
    r"\ball green\b|"
    r"✅|"
    r"\btests are passing\b|"
    r"\ball tests are passing\b|"
    r"\ball tests pass\b|"
    r"\beverything (?:is )?passing\b)",
    re.IGNORECASE,
)

_IF_PASS_PHRASE_RE = re.compile(
    r"If all tests pass, you should see the following output:\s*",
    re.IGNORECASE,
)

_FILE_CLAIM_PATTERNS = [
    re.compile(r"\b(?:I have|I've|I)\s+(?:saved|created|written)\s+(?:the\s+)?(?:tool|file|code|script)?\s*(?:to|in|at)\s*[`'\"]?([^`'\"\s]+)[`'\"]?", re.IGNORECASE),
    re.compile(r"\b(?:Saved|Created|Written)\s+(?:the\s+)?(?:tool|file|code|script)\s+(?:to|in|at)\s*[`'\"]?([^`'\"\s]+)[`'\"]?", re.IGNORECASE),
    re.compile(r"\bCreate a new file named\s+[`'\"]?([^`'\"\s]+)[`'\"]?", re.IGNORECASE),
    re.compile(r"\bCreated file\s+[`'\"]?([^`'\"\s]+)[`'\"]?", re.IGNORECASE),
]


def extract_claimed_files(text: str) -> Set[str]:
    """Extract file names or paths claimed to be created or saved in the text."""
    claimed: Set[str] = set()
    for pat in _FILE_CLAIM_PATTERNS:
        for match in pat.finditer(text):
            val = match.group(1).strip("`'\"").strip()
            if "." in val and not val.endswith("."):
                name = Path(val).name
                if name:
                    claimed.add(name.lower())
    return claimed


def has_fabricated_test_claims(text: str) -> bool:
    """Check if text contains fabricated test execution output or confirmation claims."""
    if _TEST_OUTPUT_BLOCK_RE.search(text) or _BARE_TEST_OUTPUT_RE.search(text):
        return True
    if _CONFIRM_PHRASES_RE.search(text):
        return True
    return False


def sanitize_reply(
    reply: str,
    turn_events: TurnEvents,
    base_dir: Optional[Path] = None,
) -> Tuple[str, bool]:
    """Inspect reply for claims not supported by real tool calls in this turn.

    Returns:
        (sanitized_text, intercepted)
        Where intercepted is True if unsupported claims were rewritten/sanitized.
    """
    clean_reply = reply
    intercepted = False
    notes: List[str] = []

    # Preserve honest failure reports
    is_honest_failure = (
        "tests still failing" in reply.lower()
        or "failed to build" in reply.lower()
        or "failing tests" in reply.lower()
        or "of my own tests are still wrong" in reply.lower()
        or ("tests" in reply.lower() and "failed" in reply.lower() and not _CONFIRM_PHRASES_RE.search(reply))
    )
    if is_honest_failure:
        # Preserve genuine test failure tracebacks; only strip contradictory confirmation claims
        clean_reply = _CONFIRM_PHRASES_RE.sub("", clean_reply)
        clean_reply = _IF_PASS_PHRASE_RE.sub("", clean_reply)
        return clean_reply.strip(), False

    # 1. Check for fabricated test output / claims
    fab_test = has_fabricated_test_claims(reply)
    real_test_passed = turn_events.has_successful_test()

    if fab_test and not real_test_passed:
        intercepted = True
        # Strip out fabricated test output blocks
        clean_reply = _TEST_OUTPUT_BLOCK_RE.sub("", clean_reply)
        clean_reply = _BARE_TEST_OUTPUT_RE.sub("", clean_reply)
        clean_reply = _CONFIRM_PHRASES_RE.sub("", clean_reply)
        clean_reply = _IF_PASS_PHRASE_RE.sub("", clean_reply)
        # Clean up empty code blocks or leftover whitespace
        clean_reply = re.sub(r"```\s*```", "", clean_reply)
        clean_reply = re.sub(r"\n{3,}", "\n\n", clean_reply).strip()

        if turn_events.tests_executed:
            last_test = turn_events.tests_executed[-1]
            notes.append(
                f"**Real Test Status:** Test execution failed (output: {last_test.get('output', '').strip()[:200]}). "
                "Invented test pass claims were removed."
            )
        else:
            notes.append(
                "**Real Test Status:** No tests were executed by Helix during this turn. "
                "Invented test output was removed."
            )

    # 2. Check for claimed file creations
    claimed_files = extract_claimed_files(clean_reply)
    if claimed_files:
        real_written_names = {p.name.lower() for p in turn_events.files_written if p.exists()}
        if base_dir:
            ws = base_dir / "workspace"
            if ws.exists():
                for cf in list(claimed_files):
                    for p in ws.rglob(cf):
                        if p.exists():
                            real_written_names.add(cf)
        unsupported_files = [f for f in claimed_files if f not in real_written_names]

        # If files were claimed but nothing was written
        if unsupported_files and not real_written_names:
            intercepted = True
            notes.append(
                "**File Persistence Status:** No files were written to disk during this turn. "
                "The code above is provided as reference text only."
            )

    if intercepted and notes:
        disclaimer = "\n\n" + "\n\n".join(notes)
        return clean_reply.strip() + disclaimer, True

    return reply, False
