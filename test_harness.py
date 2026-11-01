"""Helix automated end-to-end test harness for plugin builder.

Prompts local Ollama (qwen2.5-coder:14b) to build StringHasherTool, runs isolated
subprocess unit tests, hot-loads the plugin, and performs independent verification
against hashlib.sha256 for standard, empty, and non-ASCII string inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from helix_core.config import MODEL_NAME, PLUGIN_BUILDER_PROMPT
from helix_core.engine import OllamaClient
from helix_core.paths import HELIX_ROOT, init_user_workspace
from helix_core.plugin_manager import PluginManager


def extract_python_blocks(text: str) -> List[str]:
    """Extract fenced python code blocks from model response."""
    blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)\s*```", text)
    return [b.strip() for b in blocks if b.strip()]


def run_harness(model: str = MODEL_NAME, keep: bool = False) -> bool:
    print("=" * 60)
    print(f"  Helix Dynamic Plugin Builder Test Harness")
    print(f"  Target Model: {model}")
    print("=" * 60)

    client = OllamaClient(model=model)
    connected, conn_msg = client.check_connection()
    if not connected:
        print(f"[FAIL] Ollama connection check failed: {conn_msg}", file=sys.stderr)
        return False

    init_user_workspace()
    plugin_manager = PluginManager(base_dir=HELIX_ROOT)

    tool_name = "string_hasher_tool"
    tool_file = HELIX_ROOT / "plugins" / "custom_tools" / f"{tool_name}.py"
    test_file = HELIX_ROOT / "plugins" / "tests" / f"test_{tool_name}.py"

    prompt = (
        f"{PLUGIN_BUILDER_PROMPT}\n\n"
        "Generate a dynamic plugin named StringHasherTool with name='string_hasher_tool'.\n"
        "Parameters schema must define a required string parameter 'input_string'.\n"
        "The tool computes the SHA-256 hash of 'input_string' and returns the hex digest in the output dict as {'hash': '<hex_digest>'}.\n"
        "CRITICAL REQUIREMENT: Treat an empty string ('') as a VALID input! Test with 'if input_string is None:', NOT 'if not input_string:'.\n"
        "Provide exactly two fenced python code blocks: the tool implementation, then the unit test."
    )

    messages = [
        {"role": "system", "content": "You are Helix, an autonomous coding engine specializing in writing dynamic plugins."},
        {"role": "user", "content": prompt},
    ]

    max_attempts = 3
    success = False
    start_time = time.time()
    last_error = ""
    attempt_used = 0

    for attempt in range(1, max_attempts + 1):
        attempt_used = attempt
        print(f"\n[*] Attempt {attempt}/{max_attempts}: Querying model {model}...")
        t0 = time.time()
        try:
            resp = client.chat(messages)
            content = resp.get("content", "")
        except Exception as exc:
            last_error = f"Inference call error: {exc}"
            print(f"[!] {last_error}", file=sys.stderr)
            messages.append({"role": "user", "content": f"The inference call failed with error: {exc}. Please try again."})
            continue

        elapsed = time.time() - t0
        print(f"[*] Response received in {elapsed:.2f}s ({len(content)} chars).")

        blocks = extract_python_blocks(content)
        if len(blocks) < 2:
            last_error = f"Expected at least 2 fenced python blocks, got {len(blocks)}."
            print(f"[!] {last_error}")
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": f"Error: {last_error}. Please provide exactly two fenced python code blocks: tool code first, test code second.",
            })
            continue

        tool_code, test_code = blocks[0], blocks[1]
        print(f"[*] Extracted tool code ({len(tool_code)} bytes) and test code ({len(test_code)} bytes).")

        res = plugin_manager.create_or_update_plugin(tool_name, tool_code, test_code)
        if res.status != "success":
            last_error = res.error_trace or "Unknown plugin creation error"
            print(f"[!] Plugin verification failed: {last_error}")
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": (
                    f"The plugin failed unit test execution or AST verification:\n{last_error}\n"
                    "Please fix all errors and output the corrected tool and unit test code blocks."
                ),
            })
            continue

        print(f"[+] Plugin creation and isolated subprocess test PASSED on Attempt {attempt}.")
        success = True
        break

    total_time = time.time() - start_time

    if not success:
        print(f"\n[FAIL] Test harness failed after {max_attempts} attempts. Last error: {last_error}", file=sys.stderr)
        return False

    # Independent Verification by the harness itself
    print("\n[*] Running independent verification suite by the test harness...")
    loaded_tool = plugin_manager.get_tool(tool_name)
    if not loaded_tool:
        print(f"[FAIL] Could not find loaded tool '{tool_name}' in active tools.", file=sys.stderr)
        return False

    test_cases: List[Tuple[str, str, str]] = [
        ("Standard string", "hello", "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"),
        ("Empty string", "", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        (
            "Non-ASCII Unicode",
            "Helix Coder こんにちは世界 ⚡",
            hashlib.sha256("Helix Coder こんにちは世界 ⚡".encode("utf-8")).hexdigest(),
        ),
    ]

    harness_all_passed = True
    for label, inp, expected_hash in test_cases:
        res = loaded_tool.execute(input_string=inp)
        if res.get("status") != "success":
            print(f"[FAIL] Case '{label}' execution failed: {res.get('error_trace')}")
            harness_all_passed = False
            continue

        output = res.get("output", {})
        actual_hash = output.get("hash") if isinstance(output, dict) else None
        if actual_hash == expected_hash:
            print(f"  [PASS] {label}: digest matched ({actual_hash})")
        else:
            print(f"  [FAIL] {label}: expected {expected_hash}, got {actual_hash}")
            harness_all_passed = False

    # Cleanup if not --keep
    if not keep:
        print("\n[*] Cleaning up generated test plugin to preserve clean shipping state...")
        try:
            if tool_file.exists():
                tool_file.unlink()
            if test_file.exists():
                test_file.unlink()
            # Reset registry
            reg_file = HELIX_ROOT / "plugins" / "registry.json"
            reg_file.write_bytes(b"{}\n")
            print("[+] Plugins directory and registry.json successfully reset.")
        except Exception as exc:
            print(f"[!] Warning during cleanup: {exc}", file=sys.stderr)
    else:
        print(f"\n[*] --keep specified: Preserved {tool_file} and {test_file}.")

    print("=" * 60)
    print(f"  Harness Result: {'PASS' if harness_all_passed else 'FAIL'}")
    print(f"  Attempts Used: {attempt_used}/{max_attempts}")
    print(f"  Total Duration: {total_time:.2f}s")
    print("=" * 60)

    return harness_all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Helix Plugin Builder Test Harness")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name")
    parser.add_argument("--keep", action="store_true", help="Keep generated plugin files")
    args = parser.parse_args()

    passed = run_harness(model=args.model, keep=args.keep)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
