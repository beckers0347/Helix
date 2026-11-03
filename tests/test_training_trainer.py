"""Unit and integration tests for Ogun Unsloth Trainer (ogun_unsloth_trainer.py).

Runs completely offline: zero GPU access, zero model weight downloads, zero pytest dependencies.
All tests run via standard assertions under the plain test runner.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import types
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

# Ensure Ogun root and training are on sys.path
OGUN_ROOT = Path(__file__).resolve().parents[1]
if str(OGUN_ROOT / "training") not in sys.path:
    sys.path.insert(0, str(OGUN_ROOT / "training"))
if str(OGUN_ROOT) not in sys.path:
    sys.path.insert(0, str(OGUN_ROOT))

import ogun_unsloth_trainer as trainer


# ==============================================================================
# MOCKS AND FIXTURES
# ==============================================================================

class FakeTokenizer:
    """Deterministic mock tokenizer for token budget tests."""

    def __init__(self, tokens_per_word: int = 1):
        self.tokens_per_word = tokens_per_word

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        words = text.strip().split()
        tokens = []
        for i, _ in enumerate(words):
            for _ in range(self.tokens_per_word):
                tokens.append(100 + (i % 500))
        return tokens


SAMPLE_TOOL_SCHEMA = [
    {
        "name": "generate_code",
        "required": ["language", "prompt"],
        "params": {
            "language": {"type": "string"},
            "prompt": {"type": "string"},
            "optimize": {"type": "boolean"},
        },
    }
]

SAMPLE_CONVERSATION = {
    "messages": [
        {"role": "system", "content": "You are Ogun, an AI coding assistant."},
        {"role": "user", "content": "Write quicksort in python."},
        {
            "role": "assistant",
            "content": "Generating quicksort implementation.",
            "tool_calls": [
                {
                    "name": "generate_code",
                    "arguments": json.dumps({"language": "python", "prompt": "quicksort"}),
                }
            ],
        },
        {
            "role": "tool",
            "name": "generate_code",
            "content": json.dumps({"status": "SUCCESS", "code": "def qs(): pass"}),
        },
        {
            "role": "assistant",
            "content": "Quicksort generated successfully.",
        },
    ]
}


# ==============================================================================
# 1. HARDWARE GUARDS TESTS
# ==============================================================================

def test_vram_guard_below_11gb_refuses() -> None:
    fake_mem = lambda: (int(10.5 * 1024**3), int(12.0 * 1024**3))
    passed, msg = trainer.check_vram_preflight(
        model_size="14b",
        force=False,
        mem_get_info_fn=fake_mem,
        is_available_fn=lambda: True,
    )
    assert not passed
    assert "Refusing to proceed: Insufficient free VRAM" in msg


def test_vram_guard_below_11gb_with_force_proceeds() -> None:
    fake_mem = lambda: (int(8.0 * 1024**3), int(12.0 * 1024**3))
    passed, msg = trainer.check_vram_preflight(
        model_size="14b",
        force=True,
        mem_get_info_fn=fake_mem,
        is_available_fn=lambda: True,
    )
    assert passed
    assert "VRAM Warning bypassed via --force" in msg


def test_vram_guard_above_11gb_passes() -> None:
    fake_mem = lambda: (int(11.5 * 1024**3), int(12.0 * 1024**3))
    passed, msg = trainer.check_vram_preflight(
        model_size="14b",
        force=False,
        mem_get_info_fn=fake_mem,
        is_available_fn=lambda: True,
    )
    assert passed
    assert "VRAM Pre-Flight Passed" in msg


def test_vram_guard_32b_refuses() -> None:
    fake_mem = lambda: (int(12.0 * 1024**3), int(12.0 * 1024**3))
    passed, msg = trainer.check_vram_preflight(
        model_size="32b",
        force=False,
        mem_get_info_fn=fake_mem,
        is_available_fn=lambda: True,
    )
    assert not passed
    assert "Refusing to proceed: 32B model requested on 12GB VRAM card" in msg


def test_vram_guard_32b_with_force_proceeds() -> None:
    fake_mem = lambda: (int(11.5 * 1024**3), int(12.0 * 1024**3))
    passed, msg = trainer.check_vram_preflight(
        model_size="32b",
        force=True,
        mem_get_info_fn=fake_mem,
        is_available_fn=lambda: True,
    )
    assert passed


def test_vram_guard_cuda_unavailable() -> None:
    passed, msg = trainer.check_vram_preflight(
        model_size="14b",
        force=False,
        is_available_fn=lambda: False,
    )
    assert not passed
    assert "CUDA is not available" in msg


def test_disk_guard_sufficient_space() -> None:
    with tempfile.TemporaryDirectory() as td:
        with patch("shutil.disk_usage") as mock_usage:
            mock_usage.return_value = MagicMock(free=int(40 * 1024**3))
            passed, free_gb, msg = trainer.check_disk_space(Path(td), required_gb=30.0)
            assert passed
            assert free_gb >= 30.0
            assert "Disk check passed" in msg


def test_disk_guard_insufficient_space() -> None:
    with tempfile.TemporaryDirectory() as td:
        with patch("shutil.disk_usage") as mock_usage:
            mock_usage.return_value = MagicMock(free=int(15 * 1024**3))
            passed, free_gb, msg = trainer.check_disk_space(Path(td), required_gb=30.0)
            assert not passed
            assert free_gb < 30.0
            assert "Insufficient disk space" in msg


def test_disk_guard_handles_winerror() -> None:
    with tempfile.TemporaryDirectory() as td:
        with patch("shutil.disk_usage", side_effect=OSError("Drive access denied")):
            passed, free_gb, msg = trainer.check_disk_space(Path(td), required_gb=30.0)
            assert not passed
            assert free_gb == 0.0
            assert "Failed to check disk usage" in msg


# ==============================================================================
# 2. AGENTFORGE SYNTHETIC BUILDER TESTS
# ==============================================================================

def test_corruption_type_1_wrong_tool_name() -> None:
    builder = trainer.AgentForgePreferenceBuilder(seed=0)
    builder.counter = -1  # force mode 0
    pair = builder.build_pair(
        SAMPLE_CONVERSATION["messages"],
        assistant_idx=2,
        tools=SAMPLE_TOOL_SCHEMA,
    )
    assert pair is not None
    assert "invalid_generate_code_operation" in pair["rejected"]


def test_corruption_type_2_dropped_or_wrong_argument() -> None:
    builder = trainer.AgentForgePreferenceBuilder(seed=0)
    builder.counter = 0  # force mode 1
    pair = builder.build_pair(
        SAMPLE_CONVERSATION["messages"],
        assistant_idx=2,
        tools=SAMPLE_TOOL_SCHEMA,
    )
    assert pair is not None
    assert "language" not in pair["rejected"] or "prompt" not in pair["rejected"] or "invalid_param" in pair["rejected"]


def test_corruption_type_3_prose_only_no_tool() -> None:
    builder = trainer.AgentForgePreferenceBuilder(seed=0)
    builder.counter = 1  # force mode 2
    pair = builder.build_pair(
        SAMPLE_CONVERSATION["messages"],
        assistant_idx=2,
        tools=SAMPLE_TOOL_SCHEMA,
    )
    assert pair is not None
    assert "<tool_call>" not in pair["rejected"]
    assert "successfully executed" in pair["rejected"]


def test_corruption_type_4_ignore_tool_error() -> None:
    err_conv = [
        {"role": "user", "content": "run task"},
        {"role": "assistant", "content": "calling", "tool_calls": [{"name": "tool1", "arguments": {}}]},
        {"role": "tool", "name": "tool1", "content": json.dumps({"status": "failed", "error": "500 Error"})},
        {"role": "assistant", "content": "handling", "tool_calls": [{"name": "tool1", "arguments": {}}]},
    ]
    builder = trainer.AgentForgePreferenceBuilder(seed=0)
    builder.counter = 2  # mode 3
    pair = builder.build_pair(err_conv, assistant_idx=3, tools=[{"name": "tool1"}])
    assert pair is not None
    assert "Everything succeeded without any errors" in pair["rejected"]


def test_agentforge_preference_builder_rotation() -> None:
    builder = trainer.AgentForgePreferenceBuilder(seed=3407)
    modes = [builder._next_mode() for _ in range(8)]
    assert len(set(modes)) == 4


def test_ignore_tool_error_corruption_only_follows_a_real_tool_error() -> None:
    # Preceding turn is success -> mode 3 should fallback to mode 0
    ok_conv = [
        {"role": "user", "content": "run task"},
        {"role": "assistant", "content": "calling", "tool_calls": [{"name": "tool1", "arguments": {}}]},
        {"role": "tool", "name": "tool1", "content": json.dumps({"status": "success"})},
        {"role": "assistant", "content": "next", "tool_calls": [{"name": "tool1", "arguments": {}}]},
    ]
    builder = trainer.AgentForgePreferenceBuilder(seed=0)
    builder.counter = 2  # mode 3
    pair = builder.build_pair(ok_conv, assistant_idx=3, tools=[{"name": "tool1"}])
    assert pair is not None
    assert "invalid_tool1_operation" in pair["rejected"]


# ==============================================================================
# 3. FILTERING TESTS
# ==============================================================================

def test_filter_drops_identical_pairs() -> None:
    tok = FakeTokenizer()
    pairs = [{"prompt": "p", "chosen": "c", "rejected": "c"}]
    res = trainer.filter_dpo_dataset(pairs, tokenizer=tok)
    assert len(res) == 0


def test_filter_drops_exceeding_prompt_tokens() -> None:
    tok = FakeTokenizer()
    long_p = "word " * 600
    pairs = [{"prompt": long_p, "chosen": "c", "rejected": "r"}]
    res = trainer.filter_dpo_dataset(pairs, tokenizer=tok, max_prompt_length=512)
    assert len(res) == 0


def test_filter_drops_exceeding_completion_tokens() -> None:
    tok = FakeTokenizer()
    long_c = "word " * 600
    pairs = [{"prompt": "p", "chosen": long_c, "rejected": "r"}]
    res = trainer.filter_dpo_dataset(pairs, tokenizer=tok, max_prompt_length=512)
    assert len(res) == 0


def test_filter_drops_exceeding_total_tokens() -> None:
    tok = FakeTokenizer()
    p = "word " * 400
    c = "word " * 400
    r = "word " * 100
    pairs = [{"prompt": p, "chosen": c, "rejected": r}]
    res = trainer.filter_dpo_dataset(pairs, tokenizer=tok, max_prompt_length=512, max_seq_length=600)
    assert len(res) == 0


def test_filter_deduplicates_by_prompt_hash() -> None:
    tok = FakeTokenizer()
    pairs = [
        {"prompt": "same prompt", "chosen": "c1", "rejected": "r1"},
        {"prompt": "same prompt", "chosen": "c2", "rejected": "r2"},
    ]
    res = trainer.filter_dpo_dataset(pairs, tokenizer=tok)
    assert len(res) == 1


# ==============================================================================
# 4. CHAT-TEMPLATE RENDERING & TOOL PARSING
# ==============================================================================

def test_parse_tools_accepts_json_string_nested_and_flat() -> None:
    nested = [{"type": "function", "function": {"name": "test_f", "parameters": {}}}]
    assert trainer.parse_tools(json.dumps(nested))[0]["function"]["name"] == "test_f"
    assert trainer.parse_tools(nested)[0]["type"] == "function"
    assert trainer.parse_tools("bad json") == []


def test_render_prompt_uses_qwen_template_with_tools_block_and_tool_responses() -> None:
    msgs = [
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": "hello"},
        {"role": "tool", "content": "tool output"},
    ]
    tools = [{"name": "my_tool", "parameters": {}}]
    rendered = trainer.render_prompt(msgs, tools=tools)
    assert "<|im_start|>system\nYou are a test assistant." in rendered
    assert "# Tools" in rendered
    assert "<tool_response>\ntool output\n</tool_response>" in rendered
    assert rendered.endswith("<|im_start|>assistant\n")


def test_render_prompt_prefers_the_real_tokenizer_template_and_survives_its_failure() -> None:
    class Tok:
        def __init__(self, boom: bool = False):
            self.boom = boom

        def apply_chat_template(self, messages: Any, tools: Any = None, add_generation_prompt: bool = False, tokenize: bool = True) -> str:
            if self.boom:
                raise ValueError("template exploded")
            return "<<TOKENIZER_TEMPLATE>>"

    try:
        trainer.set_prompt_tokenizer(Tok())
        out = trainer.render_prompt([{"role": "user", "content": "test"}])
        assert out == "<<TOKENIZER_TEMPLATE>>"

        trainer.set_prompt_tokenizer(Tok(boom=True))
        fallback = trainer.render_prompt([{"role": "user", "content": "test"}])
        assert fallback.endswith("<|im_start|>assistant\n")
    finally:
        trainer.set_prompt_tokenizer(None)


# ==============================================================================
# 5. EXPORT FALLBACK & MODELFILE
# ==============================================================================

def test_export_fallback_on_gguf_failure() -> None:
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        mock_model = MagicMock()
        mock_model.save_pretrained_gguf.side_effect = RuntimeError("CMake not found")
        mock_tokenizer = MagicMock()

        merged_dir = trainer.export_model(
            model=mock_model,
            tokenizer=mock_tokenizer,
            output_dir=out_dir,
            model_name="test_model",
            quantization_method="f16",
            required_disk_gb=0.001,
            force=True,
        )
        assert mock_model.save_pretrained_merged.called
        assert (out_dir / "Modelfile").exists()
        assert "test_model_merged_16bit" in str(merged_dir)


def test_write_ollama_modelfile() -> None:
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        gguf_path = out_dir / "qwen2.5-coder-14b-f16.gguf"
        modelfile = trainer.write_ollama_modelfile(out_dir, gguf_path)
        assert modelfile.exists()
        content = modelfile.read_text(encoding="utf-8")
        assert f"FROM {gguf_path.name}" in content
        assert "PARAMETER stop \"<|im_end|>\"" in content


# ==============================================================================
# 6. TRL COMPATIBILITY MOCK TESTS
# ==============================================================================

def _install_fake_stack(new_api: bool):
    created: Dict[str, Any] = {}

    class DPOConfig:
        if new_api:
            def __init__(self, beta=0.1, learning_rate=5e-5, per_device_train_batch_size=8,
                         gradient_accumulation_steps=1, optim="adamw", warmup_ratio=0.0,
                         max_grad_norm=1.0, bf16=False, fp16=False, max_length=1024,
                         max_prompt_length=512, max_completion_length=None, num_train_epochs=1,
                         max_steps=-1, seed=42, output_dir="x", logging_steps=10, save_steps=500,
                         save_total_limit=None, **extra):
                self.__dict__.update(locals())
        else:
            def __init__(self, learning_rate=5e-5, per_device_train_batch_size=8,
                         gradient_accumulation_steps=1, optim="adamw", warmup_ratio=0.0,
                         max_grad_norm=1.0, bf16=False, fp16=False, max_length=1024,
                         max_prompt_length=512, max_target_length=None, num_train_epochs=1,
                         max_steps=-1, seed=42, output_dir="x", logging_steps=10, save_steps=500,
                         save_total_limit=None, **extra):
                self.__dict__.update(locals())

    class DPOTrainer:
        if new_api:
            def __init__(self, model=None, ref_model=None, args=None, train_dataset=None, processing_class=None, **extra):
                created.update(model=model, ref_model=ref_model, args=args, processing_class=processing_class, style="new")
        else:
            def __init__(self, model=None, ref_model=None, args=None, train_dataset=None, tokenizer=None, beta=0.1, **extra):
                created.update(model=model, ref_model=ref_model, args=args, tokenizer=tokenizer, beta=beta, style="old")

    trl_mod = types.ModuleType("trl")
    trl_mod.DPOConfig = DPOConfig
    trl_mod.DPOTrainer = DPOTrainer
    unsloth_mod = types.ModuleType("unsloth")
    unsloth_mod.PatchDPOTrainer = lambda: None
    return {"trl": trl_mod, "unsloth": unsloth_mod}, created


def test_setup_dpo_trainer_adapts_to_new_trl_api() -> None:
    mods, created = _install_fake_stack(new_api=True)
    with patch.dict(sys.modules, mods):
        tok = object()
        with tempfile.TemporaryDirectory() as td:
            args = trainer.parse_args(["--max-steps", "20", "--seed", "7"])
            trainer.setup_dpo_trainer(
                model="M",
                tokenizer=tok,
                train_dataset=[1],
                eval_dataset=None,
                output_dir=Path(td),
                args=args,
            )
            assert created["style"] == "new"
            assert created["processing_class"] is tok


def test_setup_dpo_trainer_adapts_to_old_trl_api() -> None:
    mods, created = _install_fake_stack(new_api=False)
    with patch.dict(sys.modules, mods):
        tok = object()
        with tempfile.TemporaryDirectory() as td:
            args = trainer.parse_args(["--max-steps", "20", "--seed", "7"])
            trainer.setup_dpo_trainer(
                model="M",
                tokenizer=tok,
                train_dataset=[1],
                eval_dataset=None,
                output_dir=Path(td),
                args=args,
            )
            assert created["style"] == "old"
            assert created["tokenizer"] is tok


def test_supported_kwargs_drops_unknown_options_and_keeps_var_kwargs_targets() -> None:
    def fn1(a: int, b: int = 1) -> int:
        return a

    res1 = trainer._supported_kwargs(fn1, {"a": 1, "b": 2, "unknown": 3})
    assert res1 == {"a": 1, "b": 2}

    def fn2(a: int, **kwargs: Any) -> int:
        return a

    res2 = trainer._supported_kwargs(fn2, {"a": 1, "unknown": 3})
    assert res2 == {"a": 1, "unknown": 3}


def test_cli_exposes_the_documented_flags() -> None:
    args = trainer.parse_args([
        "--model-size", "32b",
        "--dataset-file", "my_data.jsonl",
        "--max-steps", "50",
        "--lora-r", "64",
        "--lora-alpha", "128",
        "--gguf-quant", "q4_k_m",
        "--force",
        "--dry-run",
    ])
    assert args.model_size == "32b"
    assert str(args.dataset_file) == "my_data.jsonl"
    assert args.max_steps == 50
    assert args.lora_r == 64
    assert args.lora_alpha == 128
    assert args.gguf_quant == "q4_k_m"
    assert args.force is True
    assert args.dry_run is True


# ==============================================================================
# 7. DATASET FILE LOADING & DRY RUN
# ==============================================================================

def test_load_dataset_file_success() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "data.jsonl"
        rows = [
            {"prompt": "<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\n", "chosen": "b", "rejected": "c", "split": "train"},
            {"prompt": "<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n", "chosen": "y", "rejected": "z", "split": "eval"},
        ]
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        train, ev = trainer.load_dataset_file(p)
        assert len(train) == 1
        assert len(ev) == 1


def test_load_dataset_file_missing_refusal() -> None:
    try:
        trainer.load_dataset_file("nonexistent_dpo_file.jsonl")
        assert False, "Should raise FileNotFoundError"
    except FileNotFoundError as fnf:
        assert "run training\\run_data_prep.bat first" in str(fnf)


def test_load_dataset_file_missing_columns_refusal() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "bad.jsonl"
        p.write_text('{"prompt": "hello", "chosen": "world"}\n', encoding="utf-8")
        try:
            trainer.load_dataset_file(p)
            assert False, "Should raise ValueError for missing rejected column"
        except ValueError as ve:
            assert "missing required columns" in str(ve)


def test_load_dataset_file_plain_format_warning() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "plain.jsonl"
        p.write_text('{"prompt": "Plain prompt without chatml", "chosen": "a", "rejected": "b"}\n', encoding="utf-8")
        train, ev = trainer.load_dataset_file(p)
        assert len(train) == 1


def test_load_dataset_file_max_train_rows() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "data.jsonl"
        rows = [
            {"prompt": "<|im_start|>user\na<|im_end|>", "chosen": "b", "rejected": "c", "split": "train"}
            for _ in range(10)
        ]
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        train, _ = trainer.load_dataset_file(p, max_train_rows=3)
        assert len(train) == 3


def test_dry_run_execution() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "data.jsonl"
        rows = [
            {"prompt": "<|im_start|>user\ncode<|im_end|>\n<|im_start|>assistant\n", "chosen": "c1", "rejected": "r1", "split": "train"},
            {"prompt": "<|im_start|>user\ncode2<|im_end|>\n<|im_start|>assistant\n", "chosen": "c2", "rejected": "r2", "split": "train"},
        ]
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        args = trainer.parse_args(["--dataset-file", str(p), "--dry-run", "--output-dir", td])
        ret = trainer.run_dry_run(args)
        assert ret == 0


def test_dry_run_with_missing_dataset_file() -> None:
    args = trainer.parse_args(["--dataset-file", "missing_path.jsonl", "--dry-run"])
    ret = trainer.run_dry_run(args)
    assert ret == 1


def run_all_tests() -> bool:
    tests = [
        test_vram_guard_below_11gb_refuses,
        test_vram_guard_below_11gb_with_force_proceeds,
        test_vram_guard_above_11gb_passes,
        test_vram_guard_32b_refuses,
        test_vram_guard_32b_with_force_proceeds,
        test_vram_guard_cuda_unavailable,
        test_disk_guard_sufficient_space,
        test_disk_guard_insufficient_space,
        test_disk_guard_handles_winerror,
        test_corruption_type_1_wrong_tool_name,
        test_corruption_type_2_dropped_or_wrong_argument,
        test_corruption_type_3_prose_only_no_tool,
        test_corruption_type_4_ignore_tool_error,
        test_agentforge_preference_builder_rotation,
        test_ignore_tool_error_corruption_only_follows_a_real_tool_error,
        test_filter_drops_identical_pairs,
        test_filter_drops_exceeding_prompt_tokens,
        test_filter_drops_exceeding_completion_tokens,
        test_filter_drops_exceeding_total_tokens,
        test_filter_deduplicates_by_prompt_hash,
        test_parse_tools_accepts_json_string_nested_and_flat,
        test_render_prompt_uses_qwen_template_with_tools_block_and_tool_responses,
        test_render_prompt_prefers_the_real_tokenizer_template_and_survives_its_failure,
        test_export_fallback_on_gguf_failure,
        test_setup_dpo_trainer_adapts_to_new_trl_api,
        test_setup_dpo_trainer_adapts_to_old_trl_api,
        test_supported_kwargs_drops_unknown_options_and_keeps_var_kwargs_targets,
        test_cli_exposes_the_documented_flags,
        test_load_dataset_file_success,
        test_load_dataset_file_missing_refusal,
        test_load_dataset_file_missing_columns_refusal,
        test_load_dataset_file_plain_format_warning,
        test_load_dataset_file_max_train_rows,
        test_write_ollama_modelfile,
        test_dry_run_execution,
        test_dry_run_with_missing_dataset_file,
    ]

    print("=" * 60)
    print("  Ogun Unsloth Trainer Offline Test Suite")
    print("=" * 60)

    passes = 0
    fails = 0

    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            print(f"PASS {name}")
            passes += 1
        except Exception as exc:
            import traceback

            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
            fails += 1

    print("-" * 60)
    print(f"Summary: {passes} PASSED, {fails} FAILED (Total: {len(tests)})")
    print("=" * 60)
    return fails == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
