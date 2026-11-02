"""Helix dynamic plugin staged builder and self-correction loop.

Coordinates staged code generation (tool implementation, then unit tests)
and iterative self-correction with bounded retries.
"""
from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

from helix_core.config import PLUGIN_BUILDER_PROMPT
from helix_core.engine import CoreEngine
from helix_core.plugin_manager import PluginManager
from helix_core.quality import check_requirements_coverage


def parse_code_blocks(text: str) -> List[str]:
    """Extract fenced python code blocks from text."""
    blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)\s*```", text)
    return [b.strip() for b in blocks if b.strip()]


def build_plugin_staged(
    engine: CoreEngine,
    plugin_manager: PluginManager,
    request_text: str,
    max_attempts: int = 3,
) -> Tuple[bool, str, Optional[str]]:
    """Build, test, verify, and register a dynamic plugin in stages with self-correction.

    Returns:
        (success: bool, final_message: str, plugin_name: Optional[str])
    """
    max_attempts = max(1, max_attempts)
    last_error = "Unknown error"
    tool_code = ""
    test_code = ""
    plugin_name = None

    for attempt in range(1, max_attempts + 1):
        if attempt == 1:
            # Stage 1: Request tool implementation
            stage1_prompt = (
                f"{PLUGIN_BUILDER_PROMPT}\n\n"
                f"Stage 1/2: Author the complete Python tool class for this plugin request:\n{request_text}\n"
                "Provide the tool code in a fenced ```python code block."
            )
            reply1 = engine.run_turn(stage1_prompt)
            blocks1 = parse_code_blocks(reply1)

            if len(blocks1) >= 2:
                # Model provided both tool and test in stage 1
                tool_code, test_code = blocks1[0], blocks1[1]
            elif len(blocks1) == 1:
                tool_code = blocks1[0]
                # Stage 2: Request unit tests
                stage2_prompt = (
                    "Stage 2/2: Author the standalone unit test for the tool class above.\n"
                    "Requirements: Standalone runnable via `python plugins/tests/test_<name>.py` "
                    "(imports the tool, asserts real behavior, exits 0 on success).\n"
                    "Provide the test code in a single fenced ```python code block."
                )
                reply2 = engine.run_turn(stage2_prompt)
                blocks2 = parse_code_blocks(reply2)
                if blocks2:
                    test_code = blocks2[0]
            else:
                last_error = "Model generated no fenced python code blocks in Stage 1."
                continue
        else:
            # Self-correction: Feed real failure/error output back into next attempt
            correction_prompt = (
                f"Attempt {attempt}/{max_attempts} - Plugin verification failed with the following error:\n"
                f"{last_error}\n\n"
                "Please fix all identified errors. Output exactly two fenced ```python code blocks:\n"
                "1. The corrected tool implementation.\n"
                "2. The corrected unit test."
            )
            fix_reply = engine.run_turn(correction_prompt)
            blocks = parse_code_blocks(fix_reply)
            if len(blocks) >= 2:
                tool_code, test_code = blocks[0], blocks[1]
            elif len(blocks) == 1:
                tool_code = blocks[0]

        if not tool_code or not test_code:
            last_error = (
                f"Expected 2 code blocks (tool and test), got tool={bool(tool_code)}, test={bool(test_code)}."
            )
            continue

        # Static requirements-coverage check
        covered, missing = check_requirements_coverage(request_text, tool_code)
        if not covered:
            last_error = f"Static requirements coverage check failed: {'; '.join(missing)}"
            continue

        # Extract plugin name
        name_match = re.search(r'name\s*:\s*str\s*=\s*["\']([^"\']+)["\']', tool_code)
        plugin_name = name_match.group(1) if name_match else "custom_tool"

        # Attempt creation and isolated subprocess testing
        res = plugin_manager.create_or_update_plugin(plugin_name, tool_code, test_code)
        if res.status == "success":
            return (
                True,
                f"[PASS] Plugin '{plugin_name}' successfully built, tested, and mounted on attempt {attempt}.",
                plugin_name,
            )
        else:
            last_error = res.error_trace or "Subprocess unit test execution failed."

    return (
        False,
        f"[FAIL] Plugin creation failed after {max_attempts} attempts. Last error: {last_error}",
        plugin_name,
    )
