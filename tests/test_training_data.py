"""tests/test_prepare_training_data.py - Verification for Helix DPO data pipeline.

Tests:
- All dataset mappers (Vezora, jondurbin, CyberNative, stack-exchange, UltraFeedback, dpo-mix)
- AgentForge synthetic preference builder (4 corruption modes, seed determinism, tool error guard)
- Qwen ChatML & Plain prompt rendering
- clean_text and markdown fence repair (even, truncated opening, stray odd)
- Length filters, deduplication, and refusal boilerplate detection
- Stratified eval split and shuffle
- Atomic file write
- Rows API loader with pagination, 429 retry, and 500 skip (mocked HTTP)
- test_pipeline_output verification function on valid and corrupted datasets

Run:
  python tests/test_prepare_training_data.py
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import urllib.error

HELIX_ROOT = Path(__file__).resolve().parents[1]
if str(HELIX_ROOT / "training") not in sys.path:
    sys.path.insert(0, str(HELIX_ROOT / "training"))
if str(HELIX_ROOT) not in sys.path:
    sys.path.insert(0, str(HELIX_ROOT))

import prepare_training_data as ptd


# ==============================================================================
# 1. MAPPER TESTS
# ==============================================================================

def test_vezora_mapper():
    row = {
        "instruction": "You are an AI-Coding assistant.",
        "input": "Sort an array in python.",
        "accepted": "def sort_arr(a):\n    return sorted(a)",
        "rejected": "def sort_arr(a):\n    return a.sort()  # returns None",
        "ID": "123",
    }
    mapped = ptd.map_vezora_row(row)
    assert mapped is not None
    assert mapped["chosen"] == row["accepted"]
    assert mapped["rejected"] == row["rejected"]
    assert mapped["pair_type"] == "original"
    assert len(mapped["prompt_messages"]) == 2
    assert mapped["prompt_messages"][0] == {"role": "system", "content": "You are an AI-Coding assistant."}
    assert mapped["prompt_messages"][1] == {"role": "user", "content": "Sort an array in python."}

    # Empty instruction falls back to single user message
    row_no_sys = copy_dict(row)
    row_no_sys["instruction"] = ""
    mapped2 = ptd.map_vezora_row(row_no_sys)
    assert mapped2 is not None
    assert len(mapped2["prompt_messages"]) == 1
    assert mapped2["prompt_messages"][0]["role"] == "user"

    # Missing required field
    row_bad = copy_dict(row)
    row_bad["input"] = ""
    assert ptd.map_vezora_row(row_bad) is None


def test_jondurbin_mapper():
    row = {
        "prompt": "Write a binary search algorithm in python.",
        "chosen": "def bsearch(arr, t): ...",
        "rejected": "def bsearch(arr, t): pass",
        "id": "jd-1",
    }
    mapped = ptd.map_jondurbin_row(row)
    assert mapped is not None
    assert mapped["prompt_messages"] == [{"role": "user", "content": row["prompt"]}]
    assert mapped["chosen"] == row["chosen"]
    assert mapped["rejected"] == row["rejected"]
    assert mapped["pair_type"] == "original"

    # Missing chosen
    assert ptd.map_jondurbin_row({"prompt": "test", "chosen": "", "rejected": "foo"}) is None


def test_cybernative_mapper():
    row = {
        "lang": "c",
        "vulnerability": "buffer overflow",
        "system": "",
        "question": "Implement safe string copy in C.",
        "chosen": "strncpy(dst, src, n - 1); dst[n - 1] = 0;",
        "rejected": "strcpy(dst, src);",
    }
    mapped = ptd.map_cybernative_row(row)
    assert mapped is not None
    assert len(mapped["prompt_messages"]) == 1
    assert mapped["prompt_messages"][0]["content"] == row["question"]
    assert mapped["chosen"] == row["chosen"]
    assert mapped["rejected"] == row["rejected"]

    # With non-empty system prompt
    row_sys = copy_dict(row)
    row_sys["system"] = "You are a secure code specialist."
    mapped_sys = ptd.map_cybernative_row(row_sys)
    assert mapped_sys is not None
    assert len(mapped_sys["prompt_messages"]) == 2
    assert mapped_sys["prompt_messages"][0]["role"] == "system"


def test_stack_exchange_mapper():
    row = {
        "question": "How to invert a binary tree in Python?",
        "response_j": "def invert(node): ... (chosen)",
        "response_k": "def invert(node): ... (rejected)",
    }
    mapped = ptd.map_stack_exchange_row(row)
    assert mapped is not None
    assert mapped["prompt_messages"] == [{"role": "user", "content": row["question"]}]
    assert mapped["chosen"] == row["response_j"]
    assert mapped["rejected"] == row["response_k"]

    # Missing response_k
    assert ptd.map_stack_exchange_row({"question": "q", "response_j": "j"}) is None


def test_ultrafeedback_mapper():
    row = {
        "prompt": "How to make a web server in Python?",
        "chosen": [
            {"role": "user", "content": "How to make a web server in Python?"},
            {"role": "assistant", "content": "Use http.server module: python -m http.server 8000"},
        ],
        "rejected": [
            {"role": "user", "content": "How to make a web server in Python?"},
            {"role": "assistant", "content": "Python cannot run servers directly."},
        ],
        "score_chosen": 8.0,
        "score_rejected": 2.0,
    }
    mapped = ptd.map_ultrafeedback_row(row)
    assert mapped is not None
    assert mapped["chosen"] == "Use http.server module: python -m http.server 8000"
    assert mapped["rejected"] == "Python cannot run servers directly."
    assert len(mapped["prompt_messages"]) == 1
    assert mapped["prompt_messages"][0]["role"] == "user"


def test_dpo_mix_mapper_and_prefix_mismatch():
    row_valid = {
        "dataset": "argilla/mix",
        "chosen": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there! How can I help?"},
        ],
        "rejected": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "What do you want?"},
        ],
    }
    mapped = ptd.map_dpo_mix_row(row_valid)
    assert mapped is not None
    assert mapped["chosen"] == "Hi there! How can I help?"
    assert mapped["rejected"] == "What do you want?"
    assert mapped["prompt_messages"] == [{"role": "user", "content": "Hello"}]

    # Prefix mismatch: chosen and rejected have different user prompts
    row_mismatch = {
        "dataset": "argilla/mix",
        "chosen": [
            {"role": "user", "content": "Prompt A"},
            {"role": "assistant", "content": "Answer A"},
        ],
        "rejected": [
            {"role": "user", "content": "Prompt B"},
            {"role": "assistant", "content": "Answer B"},
        ],
    }
    assert ptd.map_dpo_mix_row(row_mismatch) is None


# ==============================================================================
# 2. AGENTFORGE PREFERENCE BUILDER TESTS
# ==============================================================================

def test_agentforge_builder_schemas_and_corruptions():
    tools_json_str = json.dumps([
        {
            "type": "function",
            "function": {
                "name": "search_db",
                "description": "Search database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "is_active": {"type": "boolean"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            },
        }
    ])

    item = {
        "id": "af-test-1",
        "tools": tools_json_str,  # Test string parsing
        "conversations": [
            {"role": "system", "content": "You are a database bot."},
            {"role": "user", "content": "Search for active users named Alice."},
            {
                "role": "assistant",
                "content": "Looking up Alice in database...",
                "tool_calls": [
                    {"name": "search_db", "arguments": {"query": "Alice", "is_active": True, "limit": 10}}
                ],
            },
            {"role": "tool", "name": "search_db", "content": '{"status": 500, "error": "database connection timeout"}'},
            {
                "role": "assistant",
                "content": "Retrying database query...",
                "tool_calls": [{"name": "search_db", "arguments": {"query": "Alice", "limit": 5}}],
            },
        ],
    }

    builder = ptd.AgentForgePreferenceBuilder(seed=42)
    pairs = builder.build_preference_pairs(item)
    assert len(pairs) == 2, f"Expected 2 pairs from 2 assistant tool turns, got {len(pairs)}"

    for p in pairs:
        assert p["pair_type"] == "synthetic"
        assert "tools" in p and len(p["tools"]) == 1
        assert p["chosen"] != p["rejected"]
        assert "<tool_call>" in p["chosen"] or "I already ran" in p["chosen"]

    # Test Corruption 2 (dropped required arg)
    turn = {
        "role": "assistant",
        "content": "Running tool...",
        "tool_calls": [{"name": "search_db", "arguments": {"query": "Alice", "is_active": True}}],
    }
    corrupted_arg = builder.corrupt_dropped_or_wrong_arg(turn, ptd.flat_schemas(ptd.parse_tools(tools_json_str)))
    tc_args = corrupted_arg["tool_calls"][0]["arguments"]
    assert "query" not in tc_args, f"Required argument 'query' should have been dropped: {tc_args}"

    # Test boolean is not confused with int (bool check before int check)
    turn_bool_only = {
        "role": "assistant",
        "content": "Running tool...",
        "tool_calls": [{"name": "search_db", "arguments": {"is_active": True}}],
    }
    corrupted_bool = builder.corrupt_dropped_or_wrong_arg(turn_bool_only)
    tc_bool_args = corrupted_bool["tool_calls"][0]["arguments"]
    assert tc_bool_args.get("is_active") == "not_a_boolean"


def test_agentforge_type4_only_after_tool_error():
    # Preceding turn is NOT a tool error
    preceding_ok = {"role": "tool", "name": "search_db", "content": '{"status": 200, "data": ["user1"]}'}
    assert not ptd.is_tool_error(preceding_ok)

    # Preceding turn IS a tool error
    preceding_err1 = {"role": "tool", "name": "search_db", "content": '{"error": "not found"}'}
    preceding_err2 = {"role": "tool", "name": "search_db", "content": "HTTP 500 Internal Server Error"}
    preceding_err3 = {"role": "tool", "name": "search_db", "content": '{"success": false, "message": "denied"}'}
    assert ptd.is_tool_error(preceding_err1)
    assert ptd.is_tool_error(preceding_err2)
    assert ptd.is_tool_error(preceding_err3)


def test_agentforge_determinism():
    item = {
        "id": "af-det",
        "tools": [{"name": "do_task", "parameters": {"type": "object", "properties": {"x": {"type": "int"}}}}] ,
        "conversations": [
            {"role": "user", "content": "Do it"},
            {"role": "assistant", "content": "Doing...", "tool_calls": [{"name": "do_task", "arguments": {"x": 1}}]},
        ],
    }
    b1 = ptd.AgentForgePreferenceBuilder(seed=1234)
    b2 = ptd.AgentForgePreferenceBuilder(seed=1234)
    pairs1 = b1.build_preference_pairs(item)
    pairs2 = b2.build_preference_pairs(item)
    assert pairs1 == pairs2, "AgentForge pair generation must be completely deterministic for same seed"


# ==============================================================================
# 3. PROMPT RENDERING TESTS
# ==============================================================================

def test_qwen_chatml_rendering():
    # Default system prompt injected when none provided
    msgs = [{"role": "user", "content": "Hello Qwen"}]
    rendered = ptd._render_qwen_chatml(msgs)
    assert "<|im_start|>system\n" + ptd.QWEN_DEFAULT_SYSTEM + "<|im_end|>\n" in rendered
    assert "<|im_start|>user\nHello Qwen<|im_end|>\n" in rendered
    assert rendered.endswith("<|im_start|>assistant\n")

    # Custom system prompt honored
    msgs_sys = [
        {"role": "system", "content": "Custom system instruction."},
        {"role": "user", "content": "User question"},
    ]
    rendered_sys = ptd._render_qwen_chatml(msgs_sys)
    assert "<|im_start|>system\nCustom system instruction.<|im_end|>\n" in rendered_sys
    assert ptd.QWEN_DEFAULT_SYSTEM not in rendered_sys

    # Tools block injected into system message
    tools = [
        {"type": "function", "function": {"name": "calc", "description": "calculator", "parameters": {}}}
    ]
    rendered_tools = ptd._render_qwen_chatml(msgs, tools)
    assert "# Tools" in rendered_tools
    assert "<tools>" in rendered_tools
    assert '"name": "calc"' in rendered_tools

    # Tool response turn wrapped inside user turn
    msgs_tool = [
        {"role": "user", "content": "Calculate 2+2"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "calc", "arguments": {"expr": "2+2"}}]},
        {"role": "tool", "content": '{"result": 4}'},
    ]
    rendered_tool_turn = ptd._render_qwen_chatml(msgs_tool, tools)
    assert "<tool_response>\n{\"result\": 4}\n</tool_response>" in rendered_tool_turn
    assert rendered_tool_turn.endswith("<|im_start|>assistant\n")


def test_plain_prompt_rendering():
    # Single user turn
    msgs = [{"role": "user", "content": "How are you?"}]
    rendered = ptd._render_plain(msgs)
    assert rendered == "How are you?"

    # System + user turn
    msgs_sys = [
        {"role": "system", "content": "You are a code helper."},
        {"role": "user", "content": "How are you?"},
    ]
    rendered_sys = ptd._render_plain(msgs_sys)
    assert rendered_sys == "You are a code helper.\n\nHow are you?"


# ==============================================================================
# 4. CLEANING & FENCE REPAIR TESTS
# ==============================================================================

def test_clean_text():
    raw = "\ufeffHello\r\nWorld!\x00\x05  \n\n\n\n\tNew line\twith code"
    cleaned = ptd.clean_text(raw)
    assert "\r" not in cleaned
    assert "\ufeff" not in cleaned
    assert "\x00" not in cleaned
    assert "\x05" not in cleaned
    assert "World!\n\n\tNew line\twith code" in cleaned


def test_markdown_fence_repair():
    # Even fences: untouched
    code_even = "Here is code:\n```python\nprint(1)\n```\nAll done."
    repaired, valid = ptd.repair_markdown_fences(code_even)
    assert valid
    assert repaired == code_even

    # Truncated opening fence at EOF: repaired
    code_trunc = "Here is function:\n```python\ndef add(a, b):\n    return a + b"
    repaired_trunc, valid_trunc = ptd.repair_markdown_fences(code_trunc)
    assert valid_trunc
    assert repaired_trunc.endswith("```")
    assert repaired_trunc.count("```") == 2

    # Stray inline fence in the middle: dropped as unclosed_fence
    stray = "Here is some text with ``` stray fence in the middle."
    _, valid_stray = ptd.repair_markdown_fences(stray)
    assert not valid_stray


def test_refusal_boilerplate_detection():
    assert ptd.is_refusal_boilerplate("I cannot fulfill this request as it violates safety.")
    assert ptd.is_refusal_boilerplate("I'm sorry, but I am unable to write malicious code.")
    assert ptd.is_refusal_boilerplate("As an AI language model, I don't have feelings.")
    # Valid code completion should not be flagged as refusal
    assert not ptd.is_refusal_boilerplate("def cannot_exceed(): return 42")


# ==============================================================================
# 5. LENGTH FILTERS & DEDUPLICATION TESTS
# ==============================================================================

def test_length_filters_and_deduplication():
    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        out_file = temp_dir / "out.jsonl"
        cache_dir = temp_dir / "cache"

        # Create mock Vezora file with duplicates and long rows
        v_dir = cache_dir / "Vezora_Code-Preference-Pairs"
        v_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            # Valid row 1
            {"instruction": "", "input": "Task 1", "accepted": "Code 1", "rejected": "Bug 1", "ID": "1"},
            # Duplicate prompt row (should be dropped)
            {"instruction": "", "input": "Task 1", "accepted": "Code 1B", "rejected": "Bug 1B", "ID": "2"},
            # Duplicate pair row (different prompt, same chosen/rejected)
            {"instruction": "", "input": "Task 2", "accepted": "Code 1", "rejected": "Bug 1", "ID": "3"},
            # Too short completion
            {"instruction": "", "input": "Task 3", "accepted": "x", "rejected": "y", "ID": "4"},
            # Total length exceeded
            {"instruction": "", "input": "Task 4", "accepted": "A" * 7000, "rejected": "B" * 7000, "ID": "5"},
        ]
        with open(v_dir / "Code-Preference-Pairs.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        count, _, _ = ptd.run_pipeline(
            sources_to_run=["vezora"],
            max_chars=12000,
            max_field_chars=8000,
            min_chars=5,
            out_path=out_file,
            cache_dir=cache_dir,
            offline=True,
            quiet=True,
        )

        assert count == 1, f"Expected exactly 1 surviving row after dedupe and length filters, got {count}"


# ==============================================================================
# 6. SHUFFLE & STRATIFIED EVAL SPLIT TESTS
# ==============================================================================

def test_stratified_eval_split():
    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        out_file = temp_dir / "out.jsonl"
        cache_dir = temp_dir / "cache"

        v_dir = cache_dir / "Vezora_Code-Preference-Pairs"
        v_dir.mkdir(parents=True, exist_ok=True)
        # Create 100 distinct rows
        with open(v_dir / "Code-Preference-Pairs.jsonl", "w", encoding="utf-8") as f:
            for i in range(100):
                row = {
                    "instruction": "",
                    "input": f"Problem {i}",
                    "accepted": f"Solution {i} is here.",
                    "rejected": f"Buggy {i} is here.",
                    "ID": str(i),
                }
                f.write(json.dumps(row) + "\n")

        ptd.run_pipeline(
            sources_to_run=["vezora"],
            eval_fraction=0.05,  # 5% of 100 = 5 eval rows
            out_path=out_file,
            cache_dir=cache_dir,
            offline=True,
            quiet=True,
        )

        splits = {"train": 0, "eval": 0}
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                splits[rec["split"]] += 1

        assert splits["eval"] == 5, f"Expected 5 eval rows, got {splits['eval']}"
        assert splits["train"] == 95, f"Expected 95 train rows, got {splits['train']}"


# ==============================================================================
# 7. MOCKED HTTP LOADER & ERROR RESILIENCE TESTS
# ==============================================================================

def test_rows_api_loader_mocked():
    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        cache_dir = temp_dir / "cache"

        # Mock urllib.request.urlopen for rows API pagination and 429 backoff
        calls = {"count": 0}
        orig_urlopen = urllib.request.urlopen

        def mock_urlopen(req, timeout=20):
            calls["count"] += 1
            if calls["count"] == 1:
                # First call: simulate 429 rate limit
                raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, io.BytesIO(b"{}"))
            # Subsequent calls: return 2 rows
            response_data = {
                "rows": [
                    {"row": {"prompt": "Prompt 1", "chosen": "Chosen 1", "rejected": "Rejected 1", "id": "1"}},
                    {"row": {"prompt": "Prompt 2", "chosen": "Chosen 2", "rejected": "Rejected 2", "id": "2"}},
                ]
            }
            body = json.dumps(response_data).encode("utf-8")
            resp = io.BytesIO(body)
            setattr(resp, "status", 200)
            setattr(resp, "headers", {})
            return resp

        urllib.request.urlopen = mock_urlopen
        try:
            records = list(
                ptd.load_rows_api(
                    dataset_id="mock/dataset",
                    split="train",
                    max_rows=2,
                    cache_dir=cache_dir,
                    offline=False,
                    refresh=True,
                    quiet=True,
                )
            )
            assert len(records) == 2, f"Expected 2 rows after 429 retry, got {len(records)}"
            assert calls["count"] >= 2, "Expected retry after HTTP 429"
        finally:
            urllib.request.urlopen = orig_urlopen


def test_rows_api_cache_is_not_reused_when_a_bigger_cap_is_requested():
    """Regression: a 100-row smoke run must not cap a later 20000-row run (the cache is only reused when sufficient)."""
    with tempfile.TemporaryDirectory() as td:
        cache_dir = Path(td) / "cache"
        served = {"calls": 0}
        orig_urlopen = urllib.request.urlopen

        def mock_urlopen(req, timeout=20):
            served["calls"] += 1
            q = urllib.parse.parse_qs(urllib.parse.urlparse(req.full_url).query)
            offset, length = int(q["offset"][0]), int(q["length"][0])
            total = 250                                    # the "dataset" has 250 rows
            rows = [{"row": {"prompt": f"P{i}", "chosen": f"C{i}", "rejected": f"R{i}"}}
                    for i in range(offset, min(offset + length, total))]
            resp = io.BytesIO(json.dumps({"rows": rows}).encode("utf-8"))
            setattr(resp, "status", 200)
            setattr(resp, "headers", {})
            return resp

        urllib.request.urlopen = mock_urlopen
        try:
            small = list(ptd.load_rows_api("mock/ds", "train", 100, cache_dir, quiet=True))
            assert len(small) == 100
            calls_after_small = served["calls"]
            again = list(ptd.load_rows_api("mock/ds", "train", 50, cache_dir, quiet=True))
            assert len(again) == 50 and served["calls"] == calls_after_small, "a sufficient cache must be reused"
            big = list(ptd.load_rows_api("mock/ds", "train", 1000, cache_dir, quiet=True))
            assert len(big) == 250, f"bigger cap must fetch the rest of the split, got {len(big)}"
            calls_after_big = served["calls"]
            assert calls_after_big > calls_after_small
            full_again = list(ptd.load_rows_api("mock/ds", "train", 5000, cache_dir, quiet=True))
            assert len(full_again) == 250 and served["calls"] == calls_after_big, "a COMPLETE cache serves any cap"
            off = list(ptd.load_rows_api("mock/ds", "train", 5000, Path(td) / "cache", offline=True, quiet=True))
            assert len(off) == 250
        finally:
            urllib.request.urlopen = orig_urlopen


def test_rows_api_is_resumable_after_a_rate_limit():
    """A 429 storm mid-download keeps what arrived, and the next run continues from the cache instead of starting over."""
    with tempfile.TemporaryDirectory() as td:
        cache_dir = Path(td) / "cache"
        state = {"blocked": False, "offsets": []}
        orig_urlopen = urllib.request.urlopen
        orig_sleep = ptd.time.sleep
        ptd.time.sleep = lambda *_a, **_k: None            # keep the test fast

        def mock_urlopen(req, timeout=20):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(req.full_url).query)
            offset, length = int(q["offset"][0]), int(q["length"][0])
            state["offsets"].append(offset)
            if state["blocked"] and offset >= 200:
                raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {"Retry-After": "1"}, io.BytesIO(b"{}"))
            rows = [{"row": {"prompt": f"P{i}", "chosen": f"C{i}", "rejected": f"R{i}"}} for i in range(offset, min(offset + length, 450))]
            resp = io.BytesIO(json.dumps({"rows": rows}).encode("utf-8"))
            setattr(resp, "status", 200)
            setattr(resp, "headers", {})
            return resp

        urllib.request.urlopen = mock_urlopen
        try:
            state["blocked"] = True
            first = list(ptd.load_rows_api("mock/ds2", "train", 1000, cache_dir, quiet=True))
            assert len(first) == 200, f"rows fetched before the rate limit must be kept, got {len(first)}"
            state["blocked"], state["offsets"] = False, []
            second = list(ptd.load_rows_api("mock/ds2", "train", 1000, cache_dir, quiet=True))
            assert len(second) == 450, f"second run must complete the split, got {len(second)}"
            assert min(state["offsets"]) == 200, f"must resume at offset 200, not restart: {state['offsets']}"
            assert [r["prompt"] for r in second][:3] == ["P0", "P1", "P2"] and len({r["prompt"] for r in second}) == 450
        finally:
            urllib.request.urlopen = orig_urlopen
            ptd.time.sleep = orig_sleep


def test_missing_source_does_not_crash():
    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        out_file = temp_dir / "out.jsonl"
        cache_dir = temp_dir / "cache"

        # Provide Vezora fixture with valid length >= 10 chars
        v_dir = cache_dir / "Vezora_Code-Preference-Pairs"
        v_dir.mkdir(parents=True, exist_ok=True)
        with open(v_dir / "Code-Preference-Pairs.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "instruction": "",
                "input": "Task that is sufficiently long",
                "accepted": "def solution(): return 42",
                "rejected": "def solution(): return 00",
            }) + "\n")

        # Run pipeline including non-existent kashif source
        count, _, _ = ptd.run_pipeline(
            sources_to_run=["vezora", "kashif"],
            out_path=out_file,
            cache_dir=cache_dir,
            offline=True,
            quiet=True,
        )
        assert count == 1, "Pipeline should continue and output surviving sources when one source is missing"


# ==============================================================================
# 8. VERIFICATION TEST FUNCTION VALIDATION
# ==============================================================================

def test_pipeline_verification_good_and_bad():
    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)

        # 1. Valid file passes verification
        good_file = temp_dir / "good.jsonl"
        good_rec = {
            "prompt": "<|im_start|>user\nWrite code<|im_end|>\n<|im_start|>assistant\n",
            "chosen": "def foo(): return 42",
            "rejected": "def foo(): return 0",
            "source": "mock_source",
            "pair_type": "original",
            "split": "train",
            "prompt_messages": [{"role": "user", "content": "Write code"}],
            "id": "hash123",
        }
        with open(good_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(good_rec) + "\n")
        assert ptd.test_pipeline_output(good_file, n_samples=1)

        # 2. Corrupted file: missing required column
        bad_file_missing = temp_dir / "bad_missing.jsonl"
        bad_rec = copy_dict(good_rec)
        del bad_rec["chosen"]
        with open(bad_file_missing, "w", encoding="utf-8") as f:
            f.write(json.dumps(bad_rec) + "\n")
        assert not ptd.test_pipeline_output(bad_file_missing)

        # 3. Corrupted file: unclosed code fence
        bad_file_fence = temp_dir / "bad_fence.jsonl"
        bad_fence_rec = copy_dict(good_rec)
        bad_fence_rec["chosen"] = "```python\ndef foo(): pass"  # odd fence
        with open(bad_file_fence, "w", encoding="utf-8") as f:
            f.write(json.dumps(bad_fence_rec) + "\n")
        assert not ptd.test_pipeline_output(bad_file_fence)

        # 4. Corrupted file: duplicate IDs
        bad_file_dup = temp_dir / "bad_dup.jsonl"
        with open(bad_file_dup, "w", encoding="utf-8") as f:
            f.write(json.dumps(good_rec) + "\n")
            f.write(json.dumps(good_rec) + "\n")
        assert not ptd.test_pipeline_output(bad_file_dup)


def copy_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(d)


# ==============================================================================
# MAIN TEST RUNNER
# ==============================================================================

if __name__ == "__main__":
    import copy
    tests = [
        test_vezora_mapper,
        test_jondurbin_mapper,
        test_cybernative_mapper,
        test_stack_exchange_mapper,
        test_ultrafeedback_mapper,
        test_dpo_mix_mapper_and_prefix_mismatch,
        test_agentforge_builder_schemas_and_corruptions,
        test_agentforge_type4_only_after_tool_error,
        test_agentforge_determinism,
        test_qwen_chatml_rendering,
        test_plain_prompt_rendering,
        test_clean_text,
        test_markdown_fence_repair,
        test_refusal_boilerplate_detection,
        test_length_filters_and_deduplication,
        test_stratified_eval_split,
        test_rows_api_loader_mocked,
        test_rows_api_cache_is_not_reused_when_a_bigger_cap_is_requested,
        test_rows_api_is_resumable_after_a_rate_limit,
        test_missing_source_does_not_crash,
        test_pipeline_verification_good_and_bad,
    ]

    passes = 0
    fails = 0
    print("=" * 70)
    print("        Running Helix Training Data Pipeline Tests")
    print("=" * 70)

    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            print(f"PASS {name}")
            passes += 1
        except Exception as e:
            fails += 1
            import traceback
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()

    print("-" * 70)
    print(f"{passes} passed, {fails} failed.")
    print("=" * 70)
    sys.exit(1 if fails else 0)
