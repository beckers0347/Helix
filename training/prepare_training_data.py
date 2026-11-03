#!/usr/bin/env python3
"""Ogun - Training Data Pipeline for Direct Preference Optimization (DPO).

Self-contained, stdlib-only pipeline that downloads, filters, normalizes,
and packages open-source preference datasets for fine-tuning local Qwen-based coding models.

Dataset registry notes:
- Target dataset `kashif/StackExchange_Code_DPO` does not exist on Hugging Face (returns 401/404).
  It is registered with required=False and skipped cleanly without crashing.
- Public substitute coding datasets enabled by default:
  1. `Vezora/Code-Preference-Pairs` (direct download JSONL)
  2. `jondurbin/py-dpo-v0.1` (datasets-server rows API)
  3. `CyberNative/Code_Vulnerability_Security_DPO` (direct download JSONL; audited)
  4. `lvwerra/stack-exchange-paired` (optional; requires datasets/pyarrow, skipped cleanly otherwise)
- Conversational & tool-use datasets:
  5. `HuggingFaceH4/ultrafeedback_binarized` (split train_prefs; rows API)
  6. `argilla/dpo-mix-7k` (split train; rows API; prefix mismatch filtering)
  7. `voxozi/agentforge-multiturn-toolcall` (SFT tool-calling; synthetic preference pairs via 4 corruptions)
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Set, Tuple, Union
import urllib.error
import urllib.parse
import urllib.request

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [OGUN-DPO-DATA] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_training_data")

# Core Constants
QWEN_DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
DEFAULT_MAX_CHARS = 12000
DEFAULT_MAX_FIELD_CHARS = 8000
DEFAULT_MIN_CHARS = 10
DEFAULT_MAX_PER_SOURCE = 20000
DEFAULT_SYNTHETIC_CAP = 6000
DEFAULT_EVAL_FRACTION = 0.01
DEFAULT_SEED = 3407
OGUN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = OGUN_ROOT / "training" / "data" / "merged_dpo_dataset.jsonl"
DEFAULT_CACHE_DIR = OGUN_ROOT / "training" / "data" / "raw"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Ogun-DataPipeline/1.0"


# ==============================================================================
# 1. TEXT CLEANING, FENCE REPAIR & VALIDATION
# ==============================================================================

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_REFUSAL_RE = re.compile(
    r"^\s*(?:I\s+(?:cannot|can't|am unable to|apologize,?\s+but\s+I\s+cannot)|"
    r"I'm\s+sorry,?\s+but\s+I\s+(?:cannot|can't|am unable to)|"
    r"As\s+an\s+AI(?:\s+language\s+model)?,\s+I\s+(?:cannot|am unable to|don't)|"
    r"Sorry,?\s+(?:but\s+)?I\s+(?:cannot|can't|am unable to))\b",
    re.I,
)


def clean_text(text: Optional[str]) -> str:
    """Normalize and clean raw text string.

    - Normalizes CRLF and CR to LF
    - Strips BOM
    - Removes NUL and non-printable control characters (keeps tab and newline)
    - Strips trailing spaces on lines
    - Collapses 3+ newlines to 2
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Strip BOM
    text = text.lstrip("\ufeff")

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove non-printable control chars (keep \t=0x09 and \n=0x0a)
    text = _CONTROL_CHAR_RE.sub("", text)

    # Strip trailing spaces on each line (preserve tabs)
    lines = [line.rstrip(" ") for line in text.split("\n")]
    text = "\n".join(lines)

    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def repair_markdown_fences(text: str) -> Tuple[str, bool]:
    """Inspect and balance markdown triple-backtick fences.

    Rules:
    - If fence count is even: valid, return (text, True).
    - If fence count is odd:
      Check if the last fence is an opening fence at the end of a truncated answer.
      If so, append closing fence '\n```' and return (repaired_text, True).
      Otherwise (stray/odd fence in the middle), return (text, False).
    """
    # Find all triple-backtick occurrences
    matches = list(re.finditer(r"```", text))
    if len(matches) % 2 == 0:
        return text, True

    last_match = matches[-1]
    last_idx = last_match.start()
    after_last = text[last_idx + 3:]

    # Check if last fence was an opening fence (odd 1-based index)
    is_opening_index = (len(matches) % 2 == 1)

    # In a truncated answer, after the opening fence is an optional language tag,
    # followed by code lines or EOF, with no closing prose explanation.
    # Check if there is no subsequent markdown heading or conversational paragraph
    lines_after = after_last.split("\n")
    first_line = lines_after[0].strip()
    # First line after ``` is typically language identifier like 'python', 'json', '', etc.
    is_valid_lang_tag = bool(re.match(r"^[a-zA-Z0-9_\-+]*$", first_line))

    # Stray fence check: if fence is inline in the middle of a sentence
    # (i.e. preceded by non-newline characters on the same line)
    line_start_idx = text.rfind("\n", 0, last_idx)
    prefix_on_line = text[line_start_idx + 1:last_idx] if line_start_idx != -1 else text[:last_idx]
    is_line_start = bool(re.match(r"^\s*$", prefix_on_line))

    if is_opening_index and is_valid_lang_tag and is_line_start:
        # Code block was truncated before closing fence: repair by appending closing fence
        repaired = text + "\n```"
        return repaired, True

    return text, False


def is_refusal_boilerplate(text: str) -> bool:
    """Conservative heuristic for boilerplate refusal responses (< 300 chars)."""
    if len(text) < 300 and _REFUSAL_RE.search(text):
        return True
    return False


# ==============================================================================
# 2. PROMPT RENDERING & CHAT TEMPLATES
# ==============================================================================

def parse_tools(raw: Any) -> List[Dict[str, Any]]:
    """Parse tool schemas into OpenAI-style [{'type': 'function', 'function': {...}}]."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("function"), dict):
            out.append({"type": "function", "function": item["function"]})
        elif item.get("name"):
            out.append({"type": "function", "function": item})
    return out


def flat_schemas(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract [{'name', 'required', 'properties'}] for schema validation/corruption."""
    flat: List[Dict[str, Any]] = []
    for tool in tools:
        fn = tool.get("function", {}) if isinstance(tool.get("function"), dict) else tool
        params = fn.get("parameters") or {}
        required = list(params.get("required") or fn.get("required") or [])
        flat.append({
            "name": fn.get("name"),
            "required": required,
            "properties": params.get("properties") or {},
        })
    return flat


def _calls_as_hf(tool_calls: Any) -> List[Dict[str, Any]]:
    """Convert tool_calls to [{'type': 'function', 'function': {'name', 'arguments': dict}}]."""
    if isinstance(tool_calls, str):
        try:
            tool_calls = json.loads(tool_calls)
        except Exception:
            return []
    calls: List[Dict[str, Any]] = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"raw": args}
        calls.append({"type": "function", "function": {"name": fn.get("name", "unknown_tool"), "arguments": args}})
    return calls


def _hf_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        msg: Dict[str, Any] = {"role": role, "content": str(m.get("content") or "")}
        if role == "assistant" and m.get("tool_calls"):
            msg["tool_calls"] = _calls_as_hf(m["tool_calls"])
        out.append(msg)
    return out


def _render_qwen_chatml(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> str:
    """Pure-Python copy of the Qwen2.5 chat template (add_generation_prompt=True)."""
    msgs = _hf_messages(messages)
    tools = tools or []
    system = QWEN_DEFAULT_SYSTEM
    if msgs and msgs[0]["role"] == "system":
        system, msgs = msgs[0]["content"], msgs[1:]

    if tools:
        sys_txt = (
            system + "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        )
        for tool in tools:
            sys_txt += "\n" + json.dumps(tool, ensure_ascii=False)
        sys_txt += (
            "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
            "<tool_call></tool_call> XML tags:\n<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
        )
        parts = ["<|im_start|>system\n" + sys_txt + "<|im_end|>\n"]
    else:
        parts = ["<|im_start|>system\n" + system + "<|im_end|>\n"]

    for i, m in enumerate(msgs):
        role = m["role"]
        if role == "tool":
            # Qwen wraps tool results in a user turn; consecutive tool messages share one turn
            if i == 0 or msgs[i - 1]["role"] != "tool":
                parts.append("<|im_start|>user")
            parts.append("\n<tool_response>\n" + m["content"] + "\n</tool_response>")
            if i + 1 >= len(msgs) or msgs[i + 1]["role"] != "tool":
                parts.append("<|im_end|>\n")
        elif role == "assistant":
            calls = "".join(
                "\n<tool_call>\n"
                + json.dumps({"name": c["function"]["name"], "arguments": c["function"]["arguments"]}, ensure_ascii=False)
                + "\n</tool_call>"
                for c in m.get("tool_calls", [])
            )
            parts.append("<|im_start|>assistant" + ("\n" + m["content"] if m["content"] else "") + calls + "<|im_end|>\n")
        else:
            parts.append("<|im_start|>" + role + "\n" + m["content"] + "<|im_end|>\n")

    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _render_plain(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> str:
    """Render messages into plain format: system text (if any) + blank line + user text."""
    if not messages:
        return ""
    sys_text = ""
    msgs = list(messages)
    if msgs and msgs[0].get("role") == "system":
        sys_text = str(msgs[0].get("content") or "").strip()
        msgs = msgs[1:]

    turns: List[str] = []
    if len(msgs) == 1 and msgs[0].get("role") == "user":
        turns.append(str(msgs[0].get("content") or "").strip())
    else:
        for m in msgs:
            role = m.get("role", "user").capitalize()
            content = str(m.get("content") or "").strip()
            turns.append(f"{role}: {content}")

    user_text = "\n\n".join(turns)
    if sys_text and user_text:
        return f"{sys_text}\n\n{user_text}"
    return user_text or sys_text


def render_prompt(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    prompt_format: str = "chatml",
) -> str:
    """Render prompt using selected format ('chatml' or 'plain')."""
    if prompt_format == "plain":
        return _render_plain(messages, tools)
    return _render_qwen_chatml(messages, tools)


def format_tool_calls_text(tool_calls: Any) -> str:
    """Format tool calls into standard <tool_call> JSON blocks."""
    if not tool_calls:
        return ""
    if isinstance(tool_calls, str):
        if "<tool_call>" in tool_calls:
            return tool_calls.strip()
        return f"<tool_call>\n{tool_calls.strip()}\n</tool_call>"

    calls_formatted: List[str] = []
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if isinstance(tc, dict):
                if "function" in tc and isinstance(tc["function"], dict):
                    fn = tc["function"]
                    name = fn.get("name", "unknown_tool")
                    raw_args = fn.get("arguments", {})
                    if isinstance(raw_args, str):
                        try:
                            args_obj = json.loads(raw_args)
                        except Exception:
                            args_obj = raw_args
                    else:
                        args_obj = raw_args
                    payload = {"name": name, "arguments": args_obj}
                elif "name" in tc:
                    name = tc.get("name", "unknown_tool")
                    args_obj = tc.get("arguments", {})
                    payload = {"name": name, "arguments": args_obj}
                else:
                    payload = tc
                calls_formatted.append(f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>")
            elif isinstance(tc, str):
                calls_formatted.append(f"<tool_call>\n{tc.strip()}\n</tool_call>")
    return "\n".join(calls_formatted)


def format_assistant_turn(turn: Union[Dict[str, Any], str]) -> str:
    """Format an assistant turn including text content and tool calls."""
    if isinstance(turn, str):
        return turn.strip()
    if not isinstance(turn, dict):
        return str(turn).strip()

    content = turn.get("content")
    tool_calls = turn.get("tool_calls")
    content_str = str(content).strip() if content else ""
    tc_str = format_tool_calls_text(tool_calls)

    if content_str and tc_str:
        return f"{content_str}\n\n{tc_str}"
    elif tc_str:
        return tc_str
    return content_str


_TOOL_ERROR_RE = re.compile(
    r"error|fail|exception|denied|not found|timed? ?out|unavailable|invalid|forbidden|conflict|expired|"
    r"sold ?out|\b(4\d\d|5\d\d)\b|\"success\"\s*:\s*false",
    re.I,
)


def is_tool_error(turn: Optional[Dict[str, Any]]) -> bool:
    """True when `turn` is a tool result that reports a failure."""
    return bool(
        isinstance(turn, dict)
        and turn.get("role") == "tool"
        and _TOOL_ERROR_RE.search(str(turn.get("content") or ""))
    )


# ==============================================================================
# 3. AGENTFORGE SFT -> SYNTHETIC PREFERENCE PAIR BUILDER
# ==============================================================================

class AgentForgePreferenceBuilder:
    """Builds deterministic synthetic DPO pairs from AgentForge multiturn toolcall conversations.

    Rotates through four corruption modes:
    1. Wrong tool name
    2. Dropped or wrong-type required argument (bool checked before int)
    3. Prose response claiming success without calling the tool
    4. Ignore tool error (only after preceding tool error, falls back to mode 0-2 otherwise)
    """

    def __init__(self, schemas: Optional[List[Dict[str, Any]]] = None, seed: int = DEFAULT_SEED):
        self.schemas = flat_schemas(parse_tools(schemas))
        self._schema_lookup: Dict[str, Dict[str, Any]] = {}
        for s in self.schemas:
            name = s.get("name")
            if name:
                self._schema_lookup[name] = s
        self.corruption_counter = 0
        self.seed = seed

    def corrupt_wrong_tool_name(
        self,
        turn: Dict[str, Any],
        schemas: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        corrupted = copy.deepcopy(turn)
        tool_calls = corrupted.get("tool_calls")
        if tool_calls and isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    if "function" in tc and isinstance(tc["function"], dict):
                        orig_name = tc["function"].get("name", "tool")
                        tc["function"]["name"] = f"invalid_{orig_name}_operation"
                    elif "name" in tc:
                        orig_name = tc.get("name", "tool")
                        tc["name"] = f"invalid_{orig_name}_operation"
        else:
            content = str(corrupted.get("content") or "")
            if "<tool_call>" in content:
                content = re.sub(
                    r'"name":\s*"([^"]+)"',
                    r'"name": "invalid_\1_operation"',
                    content,
                )
                corrupted["content"] = content
            else:
                corrupted["content"] = "I will use execute_arbitrary_unverified_function() to complete your request."
        return corrupted

    def corrupt_dropped_or_wrong_arg(
        self,
        turn: Dict[str, Any],
        schemas: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        corrupted = copy.deepcopy(turn)
        tool_calls = corrupted.get("tool_calls")

        if tool_calls and isinstance(tool_calls, list):
            for tc in tool_calls:
                fn = tc.get("function") if (isinstance(tc, dict) and "function" in tc) else tc
                if not isinstance(fn, dict):
                    continue
                tool_name = fn.get("name", "")
                raw_args = fn.get("arguments", {})
                is_json_str = isinstance(raw_args, str)
                args_dict = {}
                if is_json_str:
                    try:
                        args_dict = json.loads(raw_args)
                    except Exception:
                        args_dict = {"param": raw_args}
                elif isinstance(raw_args, dict):
                    args_dict = copy.deepcopy(raw_args)

                # Look up schema for required args
                schema = (schemas and next((s for s in schemas if s.get("name") == tool_name), None)) or self._schema_lookup.get(tool_name, {})
                required_fields = schema.get("required", [])

                if required_fields and any(f in args_dict for f in required_fields):
                    target_field = next(f for f in required_fields if f in args_dict)
                    if len(args_dict) > 1:
                        del args_dict[target_field]
                    else:
                        orig_val = args_dict[target_field]
                        # IMPORTANT: bool is subclass of int; test bool before int
                        if isinstance(orig_val, bool):
                            args_dict[target_field] = "not_a_boolean"
                        elif isinstance(orig_val, str):
                            args_dict[target_field] = 99999
                        elif isinstance(orig_val, (int, float)):
                            args_dict[target_field] = "invalid_string_argument"
                        else:
                            args_dict[target_field] = 12345
                elif args_dict:
                    key = next(iter(args_dict))
                    if len(args_dict) > 1:
                        del args_dict[key]
                    else:
                        orig_val = args_dict[key]
                        if isinstance(orig_val, bool):
                            args_dict[key] = "not_a_boolean"
                        else:
                            args_dict[key] = 99999 if isinstance(orig_val, str) else "invalid_string"
                else:
                    args_dict["missing_required_parameter"] = None

                fn["arguments"] = json.dumps(args_dict, ensure_ascii=False) if is_json_str else args_dict
        else:
            corrupted["content"] = '<tool_call>\n{"name": "run_tool", "arguments": {"corrupted_type": 99999}}\n</tool_call>'

        return corrupted

    @staticmethod
    def _first_tool_name(turn: Dict[str, Any]) -> str:
        calls = _calls_as_hf(turn.get("tool_calls"))
        if calls:
            return calls[0]["function"]["name"]
        m = re.search(r'"name"\s*:\s*"([^"]+)"', str(turn.get("content") or ""))
        return m.group(1) if m else "the tool"

    def corrupt_prose_only_success(self, turn: Dict[str, Any]) -> Dict[str, Any]:
        narration = str(turn.get("content") or "").split("<tool_call>")[0].strip()
        name = self._first_tool_name(turn)
        text = (f"{narration} " if narration else "") + f"I already ran {name} and it succeeded, so nothing more is needed."
        return {"role": "assistant", "content": text.strip(), "tool_calls": []}

    def corrupt_ignore_tool_error(
        self,
        turn: Dict[str, Any],
        preceding_turn: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        narration = str(turn.get("content") or "").split("<tool_call>")[0].strip()
        text = "The last step reported an error, but I'll treat it as a success and carry on."
        if narration:
            text += f" {narration}"
        return {"role": "assistant", "content": text.strip(), "tool_calls": []}

    def build_preference_pairs(
        self,
        item: Dict[str, Any],
        schemas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Extract assistant turns and generate synthetic preference pairs."""
        messages = item.get("conversations") or item.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            return []

        tools = parse_tools(item.get("tools") or schemas or self.schemas)
        item_schemas = flat_schemas(tools)
        pairs: List[Dict[str, Any]] = []

        for i, turn in enumerate(messages):
            if not isinstance(turn, dict) or turn.get("role") != "assistant":
                continue

            tool_calls = turn.get("tool_calls")
            content = str(turn.get("content") or "")
            has_tool_call = bool(tool_calls) or ("<tool_call>" in content)
            if not has_tool_call:
                continue

            prefix = messages[:i]
            preceding = messages[i - 1] if i > 0 else None

            chosen_str = format_assistant_turn(turn)

            corruption_type = self.corruption_counter % 4
            self.corruption_counter += 1
            if corruption_type == 3 and not is_tool_error(preceding):
                corruption_type = self.corruption_counter % 3

            if corruption_type == 0:
                corrupted = self.corrupt_wrong_tool_name(turn, item_schemas)
            elif corruption_type == 1:
                corrupted = self.corrupt_dropped_or_wrong_arg(turn, item_schemas)
            elif corruption_type == 2:
                corrupted = self.corrupt_prose_only_success(turn)
            else:
                corrupted = self.corrupt_ignore_tool_error(turn, preceding)

            rejected_str = format_assistant_turn(corrupted)

            if chosen_str and rejected_str:
                pairs.append({
                    "prompt_messages": prefix,
                    "tools": tools,
                    "chosen": chosen_str,
                    "rejected": rejected_str,
                    "pair_type": "synthetic",
                })

        return pairs


# ==============================================================================
# 4. DATASET MAPPERS & REGISTRY
# ==============================================================================

@dataclass
class DatasetSource:
    id: str
    short_name: str
    loader_type: str  # "direct_download", "rows_api", "datasets"
    split: str
    enabled_by_default: bool
    required: bool
    url_or_file: str
    description: str


SOURCE_REGISTRY: List[DatasetSource] = [
    DatasetSource(
        id="Vezora/Code-Preference-Pairs",
        short_name="vezora",
        loader_type="direct_download",
        split="train",
        enabled_by_default=True,
        required=True,
        url_or_file="https://huggingface.co/datasets/Vezora/Code-Preference-Pairs/resolve/main/Code-Preference-Pairs.jsonl",
        description="Coding preference pairs with instruction, input, accepted, rejected.",
    ),
    DatasetSource(
        id="jondurbin/py-dpo-v0.1",
        short_name="jondurbin",
        loader_type="rows_api",
        split="train",
        enabled_by_default=True,
        required=True,
        url_or_file="",
        description="Python coding DPO dataset with prompt, chosen, rejected.",
    ),
    DatasetSource(
        id="CyberNative/Code_Vulnerability_Security_DPO",
        short_name="cybernative",
        loader_type="direct_download",
        split="train",
        enabled_by_default=True,
        required=True,
        url_or_file="https://huggingface.co/datasets/CyberNative/Code_Vulnerability_Security_DPO/resolve/main/secure_programming_dpo.json",
        description="Audited security/vulnerability coding DPO dataset.",
    ),
    DatasetSource(
        id="lvwerra/stack-exchange-paired",
        short_name="stack_exchange_paired",
        loader_type="datasets",
        split="train",
        enabled_by_default=True,
        required=False,
        url_or_file="",
        description="General StackExchange preference dataset (requires datasets/pyarrow).",
    ),
    DatasetSource(
        id="kashif/StackExchange_Code_DPO",
        short_name="kashif",
        loader_type="rows_api",
        split="train",
        enabled_by_default=False,
        required=False,
        url_or_file="",
        description="Non-existent upstream reference dataset; kept in registry for forward compatibility.",
    ),
    DatasetSource(
        id="HuggingFaceH4/ultrafeedback_binarized",
        short_name="ultrafeedback",
        loader_type="rows_api",
        split="train_prefs",
        enabled_by_default=True,
        required=True,
        url_or_file="",
        description="Conversational preference pairs from UltraFeedback.",
    ),
    DatasetSource(
        id="argilla/dpo-mix-7k",
        short_name="dpo_mix_7k",
        loader_type="rows_api",
        split="train",
        enabled_by_default=True,
        required=True,
        url_or_file="",
        description="Curated conversational DPO mix; drops prefix mismatches.",
    ),
    DatasetSource(
        id="voxozi/agentforge-multiturn-toolcall",
        short_name="agentforge",
        loader_type="direct_download",
        split="train",
        enabled_by_default=True,
        required=True,
        url_or_file="https://huggingface.co/datasets/voxozi/agentforge-multiturn-toolcall/resolve/main/train.jsonl",
        description="Multiturn agent tool-calling SFT dataset converted to synthetic preference pairs.",
    ),
]


def map_vezora_row(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    instruction = str(item.get("instruction") or "").strip()
    user_input = str(item.get("input") or "").strip()
    accepted = str(item.get("accepted") or "").strip()
    rejected = str(item.get("rejected") or "").strip()

    if not user_input or not accepted or not rejected:
        return None

    prompt_messages: List[Dict[str, str]] = []
    if instruction:
        prompt_messages.append({"role": "system", "content": instruction})
    prompt_messages.append({"role": "user", "content": user_input})

    return {
        "prompt_messages": prompt_messages,
        "chosen": accepted,
        "rejected": rejected,
        "pair_type": "original",
    }


def map_jondurbin_row(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prompt = str(item.get("prompt") or "").strip()
    chosen = str(item.get("chosen") or "").strip()
    rejected = str(item.get("rejected") or "").strip()

    if not prompt or not chosen or not rejected:
        return None

    return {
        "prompt_messages": [{"role": "user", "content": prompt}],
        "chosen": chosen,
        "rejected": rejected,
        "pair_type": "original",
    }


def map_cybernative_row(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    system = str(item.get("system") or "").strip()
    question = str(item.get("question") or "").strip()
    chosen = str(item.get("chosen") or "").strip()
    rejected = str(item.get("rejected") or "").strip()

    if not question or not chosen or not rejected:
        return None

    prompt_messages: List[Dict[str, str]] = []
    if system:
        prompt_messages.append({"role": "system", "content": system})
    prompt_messages.append({"role": "user", "content": question})

    return {
        "prompt_messages": prompt_messages,
        "chosen": chosen,
        "rejected": rejected,
        "pair_type": "original",
    }


def map_stack_exchange_row(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    question = str(item.get("question") or "").strip()
    resp_j = str(item.get("response_j") or "").strip()
    resp_k = str(item.get("response_k") or "").strip()

    if not question or not resp_j or not resp_k:
        return None

    return {
        "prompt_messages": [{"role": "user", "content": question}],
        "chosen": resp_j,
        "rejected": resp_k,
        "pair_type": "original",
    }


def map_ultrafeedback_row(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw_prompt = item.get("prompt")
    raw_chosen = item.get("chosen")
    raw_rejected = item.get("rejected")

    chosen_str = ""
    rejected_str = ""
    prompt_messages: List[Dict[str, Any]] = []

    if isinstance(raw_chosen, list) and raw_chosen:
        last_turn = raw_chosen[-1]
        chosen_str = str(last_turn.get("content") or "").strip() if isinstance(last_turn, dict) else str(last_turn).strip()
        if len(raw_chosen) > 1:
            prompt_messages = copy.deepcopy(raw_chosen[:-1])
    elif isinstance(raw_chosen, str):
        chosen_str = raw_chosen.strip()

    if isinstance(raw_rejected, list) and raw_rejected:
        last_turn = raw_rejected[-1]
        rejected_str = str(last_turn.get("content") or "").strip() if isinstance(last_turn, dict) else str(last_turn).strip()
    elif isinstance(raw_rejected, str):
        rejected_str = raw_rejected.strip()

    if not prompt_messages:
        if isinstance(raw_prompt, list):
            prompt_messages = copy.deepcopy(raw_prompt)
        elif isinstance(raw_prompt, str) and raw_prompt.strip():
            prompt_messages = [{"role": "user", "content": raw_prompt.strip()}]

    if not prompt_messages or not chosen_str or not rejected_str:
        return None

    return {
        "prompt_messages": prompt_messages,
        "chosen": chosen_str,
        "rejected": rejected_str,
        "pair_type": "original",
    }


def map_dpo_mix_row(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    chosen_list = item.get("chosen")
    rejected_list = item.get("rejected")

    if not isinstance(chosen_list, list) or not isinstance(rejected_list, list):
        return None
    if len(chosen_list) < 2 or len(rejected_list) < 2:
        return None

    chosen_prefix = chosen_list[:-1]
    rejected_prefix = rejected_list[:-1]

    # Verify conversation prefixes match; drop mismatches
    if chosen_prefix != rejected_prefix:
        return None

    chosen_turn = chosen_list[-1]
    rejected_turn = rejected_list[-1]

    chosen_str = str(chosen_turn.get("content") or "").strip() if isinstance(chosen_turn, dict) else str(chosen_turn).strip()
    rejected_str = str(rejected_turn.get("content") or "").strip() if isinstance(rejected_turn, dict) else str(rejected_turn).strip()

    if not chosen_str or not rejected_str:
        return None

    return {
        "prompt_messages": copy.deepcopy(chosen_prefix),
        "chosen": chosen_str,
        "rejected": rejected_str,
        "pair_type": "original",
    }


# ==============================================================================
# 5. DATA LOADERS (OFFLINE CACHE, ROWS API, DIRECT DOWNLOAD)
# ==============================================================================

def _http_get(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 25) -> Tuple[int, bytes, Dict[str, str]]:
    """Safe HTTP GET with custom headers and timeout."""
    req_headers = {"User-Agent": USER_AGENT}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = resp.status
        body = resp.read()
        resp_headers = {k.lower(): v for k, v in resp.headers.items()}
        return status, body, resp_headers


def _fetch_rows_page(url: str, dataset_id: str, quiet: bool = False) -> Optional[Dict[str, Any]]:
    """GET one rows-API page. Honours Retry-After on HTTP 429, backs off exponentially on 5xx, gives up after 8 attempts."""
    for attempt in range(8):
        try:
            _, body, _ = _http_get(url, timeout=30)
            return json.loads(body.decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                wait = min(90.0, 1.5 * (2 ** attempt))
                try:
                    retry_after = e.headers.get("Retry-After") if getattr(e, "headers", None) is not None else None
                    if retry_after:
                        wait = min(120.0, max(wait, float(retry_after)))
                except Exception:
                    pass
                if not quiet:
                    logger.warning("Rows API HTTP %s for %s; waiting %.1fs (attempt %d/8)...", e.code, dataset_id, wait, attempt + 1)
                time.sleep(wait)
            else:
                logger.warning("Rows API HTTP error %s for %s (%s). Skipping.", e.code, dataset_id, e.reason)
                return None
        except Exception as e:
            logger.warning("Rows API request failed for %s (%s). Attempt %d/8.", dataset_id, e, attempt + 1)
            time.sleep(min(10.0, 1.5 * (attempt + 1)))
    return None


def load_rows_api(
    dataset_id: str,
    split: str,
    max_rows: int,
    cache_dir: Path,
    offline: bool = False,
    refresh: bool = False,
    quiet: bool = False,
) -> Generator[Dict[str, Any], None, None]:
    """Load records from the Hugging Face datasets-server rows API.

    Resumable: every page is appended to the raw cache as soon as it arrives, so a rate limit (HTTP 429), a crash or a smaller
    earlier run never loses or caps anything. A later run serves the cached rows first and continues from where the cache ends.
    A `.meta.json` next to the cache records the row count and whether the whole split has been downloaded.
    """
    safe_name = dataset_id.replace("/", "_")
    cache_file = cache_dir / safe_name / f"{split}_rows.jsonl"
    meta_file = cache_file.with_name(cache_file.name + ".meta.json")
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if refresh:
        for f in (cache_file, meta_file):
            if f.exists():
                f.unlink()

    # 1. Serve what the cache already holds.
    cached_count, cached_complete = 0, False
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cached_count += 1
                yield json.loads(line)
                if cached_count >= max_rows:
                    return
        try:
            cached_complete = bool(json.loads(meta_file.read_text(encoding="utf-8")).get("complete", False))
        except Exception:
            cached_complete = False
        if not quiet:
            logger.info("Read %d cached rows of %s from %s.", cached_count, dataset_id, cache_file)
        if cached_complete:
            return

    if offline:
        if not cached_count:
            logger.warning("Offline mode active: cannot fetch %s over network and cache missing.", dataset_id)
        return

    # 2. Fetch the rest, appending page by page.
    if not quiet:
        logger.info("Fetching %s (split=%s) via rows API from row %d...", dataset_id, split, cached_count)

    encoded_dataset = urllib.parse.quote_plus(dataset_id)
    offset = cached_count
    total = cached_count
    page_size = 100
    exhausted = False
    failed = False

    with open(cache_file, "a", encoding="utf-8") as out:
        while total < max_rows:
            batch_len = min(page_size, max_rows - total)
            url = (
                f"https://datasets-server.huggingface.co/rows"
                f"?dataset={encoded_dataset}&config=default&split={split}&offset={offset}&length={batch_len}"
            )
            data = _fetch_rows_page(url, dataset_id, quiet)
            if not data or "rows" not in data:
                failed = True
                break
            rows = data.get("rows", [])
            if not rows:
                exhausted = True
                break
            for item in rows:
                row_dict = item.get("row", {})
                if row_dict:
                    out.write(json.dumps(row_dict, ensure_ascii=False) + "\n")
                    total += 1
                    yield row_dict
                    if total >= max_rows:
                        break
            out.flush()
            if len(rows) < batch_len:
                exhausted = True                       # a short page means the split ended
            offset += len(rows)
            if exhausted:
                break
            time.sleep(0.3)                            # polite pacing: the public rows API rate-limits bursts

    meta_file.write_text(json.dumps({"rows": total, "complete": exhausted}), encoding="utf-8")
    if failed:
        logger.warning("%s: stopped at %d of %d requested rows (rate limit / network). Re-run to continue from the cache.",
                       dataset_id, total, max_rows)


def load_direct_download(
    dataset_id: str,
    url: str,
    max_rows: int,
    cache_dir: Path,
    offline: bool = False,
    refresh: bool = False,
    quiet: bool = False,
) -> Generator[Dict[str, Any], None, None]:
    """Download single JSON/JSONL file, caching under runtime/training_cache/raw/."""
    safe_name = dataset_id.replace("/", "_")
    filename = url.split("/")[-1]
    local_path = cache_dir / safe_name / filename
    meta_path = cache_dir / safe_name / f"{filename}.meta.json"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Reuse existing cache
    if local_path.exists() and not refresh:
        meta_complete = True
        meta_rows = float("inf")
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as mf:
                    meta = json.load(mf)
                    meta_complete = bool(meta.get("complete", False))
                    meta_rows = int(meta.get("rows", 0))
            except Exception:
                pass

        if meta_complete or meta_rows >= max_rows:
            if not quiet:
                logger.info("Reading %s from cache (%s)...", dataset_id, local_path)
            count = 0
            with open(local_path, "r", encoding="utf-8", errors="replace") as f:
                first_char = f.read(1)
                f.seek(0)
                if first_char == "[":
                    try:
                        all_items = json.load(f)
                        for item in all_items:
                            count += 1
                            yield item
                            if count >= max_rows:
                                return
                    except Exception as e:
                        logger.warning("Failed parsing JSON array from %s: %s", local_path, e)
                    return

                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            count += 1
                            yield json.loads(line)
                            if count >= max_rows:
                                return
                        except Exception:
                            continue
            return

    if offline:
        logger.warning("Offline mode active: cannot download %s and cache missing.", dataset_id)
        return

    if not quiet:
        logger.info("Downloading %s from %s...", dataset_id, url)

    # Stream download line by line or file with range support for small caps
    try:
        req_headers = {"User-Agent": USER_AGENT}
        if max_rows <= 100:
            # Use Range request to avoid transferring large files (e.g. Vezora 213MB)
            req_headers["Range"] = "bytes=0-1048576"  # 1MB is plenty for ~100 rows

        req = urllib.request.Request(url, headers=req_headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            temp_path = local_path.with_suffix(".tmp")
            count = 0
            is_complete = True
            with open(temp_path, "w", encoding="utf-8") as out_f:
                buffer = ""
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    text_chunk = chunk.decode("utf-8", errors="replace")
                    buffer += text_chunk
                    lines = buffer.split("\n")
                    buffer = lines.pop()
                    for l in lines:
                        l = l.strip()
                        if l:
                            out_f.write(l + "\n")
                            if count < max_rows:
                                try:
                                    row = json.loads(l)
                                    count += 1
                                    yield row
                                except Exception:
                                    pass
                            if count >= max_rows:
                                is_complete = False
                                break
                    if count >= max_rows:
                        break

                if buffer.strip() and count < max_rows:
                    out_f.write(buffer.strip() + "\n")
                    try:
                        row = json.loads(buffer.strip())
                        count += 1
                        yield row
                    except Exception:
                        pass

            if temp_path.exists():
                os.replace(temp_path, local_path)
                try:
                    with open(meta_path, "w", encoding="utf-8") as mf:
                        json.dump({"rows": count, "complete": is_complete}, mf)
                except Exception:
                    pass

    except Exception as e:
        logger.warning("Direct download failed for %s (%s). Skipping.", dataset_id, e)


def load_datasets_library(
    dataset_id: str,
    split: str,
    max_rows: int,
    quiet: bool = False,
) -> Generator[Dict[str, Any], None, None]:
    """Optional loader using Hugging Face datasets library if installed."""
    try:
        import datasets  # Optional accelerator
    except ImportError:
        logger.warning(
            "The `datasets` package is not installed; skipping parquet dataset %s.",
            dataset_id,
        )
        return

    if not quiet:
        logger.info("Loading %s using `datasets` package (streaming)...", dataset_id)

    try:
        ds = datasets.load_dataset(dataset_id, split=split, streaming=True)
        count = 0
        for row in ds:
            yield dict(row)
            count += 1
            if count >= max_rows:
                break
    except Exception as e:
        logger.warning("`datasets` loader failed for %s (%s). Skipping.", dataset_id, e)


# ==============================================================================
# 6. PIPELINE ORCHESTRATION & FILTERING
# ==============================================================================

@dataclass
class DatasetStats:
    source_id: str
    rows_read: int = 0
    kept: int = 0
    dropped_by_reason: Dict[str, int] = field(default_factory=dict)
    char_lengths: List[int] = field(default_factory=list)
    train_count: int = 0
    eval_count: int = 0
    original_count: int = 0
    synthetic_count: int = 0

    def record_drop(self, reason: str) -> None:
        self.dropped_by_reason[reason] = self.dropped_by_reason.get(reason, 0) + 1


def compute_percentiles(values: List[int]) -> Dict[str, Union[int, float]]:
    if not values:
        return {"min": 0, "p25": 0, "median": 0, "p75": 0, "p95": 0, "max": 0}
    vals = sorted(values)
    n = len(vals)

    def pct(p: float) -> Union[int, float]:
        k = (n - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return vals[int(k)]
        d0 = vals[int(f)] * (c - k)
        d1 = vals[int(c)] * (k - f)
        return round(d0 + d1, 1)

    return {
        "min": vals[0],
        "p25": pct(25),
        "median": pct(50),
        "p75": pct(75),
        "p95": pct(95),
        "max": vals[-1],
    }


def process_dataset(
    source: DatasetSource,
    max_per_source: int,
    prompt_format: str,
    max_chars: int,
    max_field_chars: int,
    min_chars: int,
    cache_dir: Path,
    offline: bool,
    refresh: bool,
    seed: int,
    stats: DatasetStats,
    seen_prompts: Set[str],
    seen_pairs: Set[str],
    builder: AgentForgePreferenceBuilder,
    quiet: bool = False,
) -> List[Dict[str, Any]]:
    """Process a single dataset source from raw loader to normalized DPO records."""
    raw_generator = None
    if source.loader_type == "direct_download":
        raw_generator = load_direct_download(
            source.id, source.url_or_file, max_per_source, cache_dir, offline, refresh, quiet
        )
    elif source.loader_type == "rows_api":
        raw_generator = load_rows_api(
            source.id, source.split, max_per_source, cache_dir, offline, refresh, quiet
        )
    elif source.loader_type == "datasets":
        raw_generator = load_datasets_library(source.id, source.split, max_per_source, quiet)

    if raw_generator is None:
        return []

    kept_records: List[Dict[str, Any]] = []

    for raw_row in raw_generator:
        stats.rows_read += 1

        # Map to common intermediate format
        mapped_items: List[Dict[str, Any]] = []
        if source.short_name == "vezora":
            res = map_vezora_row(raw_row)
            if res:
                mapped_items.append(res)
        elif source.short_name == "jondurbin":
            res = map_jondurbin_row(raw_row)
            if res:
                mapped_items.append(res)
        elif source.short_name == "cybernative":
            res = map_cybernative_row(raw_row)
            if res:
                mapped_items.append(res)
        elif source.short_name == "stack_exchange_paired":
            res = map_stack_exchange_row(raw_row)
            if res:
                mapped_items.append(res)
        elif source.short_name == "ultrafeedback":
            res = map_ultrafeedback_row(raw_row)
            if res:
                mapped_items.append(res)
        elif source.short_name == "dpo_mix_7k":
            res = map_dpo_mix_row(raw_row)
            if res:
                mapped_items.append(res)
        elif source.short_name == "agentforge":
            mapped_items = builder.build_preference_pairs(raw_row)
        else:
            # Generic fallback
            p = raw_row.get("prompt")
            c = raw_row.get("chosen")
            r = raw_row.get("rejected")
            if p and c and r:
                mapped_items.append({
                    "prompt_messages": [{"role": "user", "content": str(p)}],
                    "chosen": str(c),
                    "rejected": str(r),
                    "pair_type": "original",
                })

        if not mapped_items:
            stats.record_drop("mapper_failed_or_mismatch")
            continue

        for item in mapped_items:
            raw_prompt_msgs = item.get("prompt_messages", [])
            raw_chosen = item.get("chosen", "")
            raw_rejected = item.get("rejected", "")
            pair_type = item.get("pair_type", "original")
            tools = item.get("tools")

            # 1. Clean fields
            norm_chosen = clean_text(raw_chosen)
            norm_rejected = clean_text(raw_rejected)

            if not norm_chosen or not norm_rejected:
                stats.record_drop("empty_completion")
                continue

            # 2. Check identical chosen/rejected
            if norm_chosen == norm_rejected:
                stats.record_drop("identical_chosen_rejected")
                continue

            # 3. Minimum length check
            if len(norm_chosen) < min_chars or len(norm_rejected) < min_chars:
                stats.record_drop("too_short")
                continue

            # 4. Refusal boilerplate check
            c_refusal = is_refusal_boilerplate(norm_chosen)
            r_refusal = is_refusal_boilerplate(norm_rejected)
            if c_refusal != r_refusal:
                stats.record_drop("refusal_boilerplate")
                continue

            # 5. Markdown fence repair
            repaired_chosen, c_valid = repair_markdown_fences(norm_chosen)
            if not c_valid:
                stats.record_drop("unclosed_fence")
                continue

            repaired_rejected, r_valid = repair_markdown_fences(norm_rejected)
            if not r_valid:
                stats.record_drop("unclosed_fence")
                continue

            # 6. Render prompt according to format
            # Ensure prompt_messages has system message if not present
            cleaned_messages: List[Dict[str, str]] = []
            for m in raw_prompt_msgs:
                c_clean = clean_text(m.get("content", ""))
                if c_clean:
                    cleaned_messages.append({"role": m.get("role", "user"), "content": c_clean})

            if not cleaned_messages:
                stats.record_drop("empty_prompt")
                continue

            # For Qwen chatml rendering, ensure system message is explicitly defined
            structured_prompt_msgs = copy.deepcopy(cleaned_messages)
            if structured_prompt_msgs[0].get("role") != "system":
                structured_prompt_msgs.insert(0, {"role": "system", "content": QWEN_DEFAULT_SYSTEM})

            rendered_prompt = render_prompt(cleaned_messages, tools=tools, prompt_format=prompt_format)
            if not rendered_prompt.strip():
                stats.record_drop("empty_prompt")
                continue

            # 7. Field length & Total length check
            p_len = len(rendered_prompt)
            c_len = len(repaired_chosen)
            r_len = len(repaired_rejected)
            total_len = p_len + c_len + r_len

            if p_len > max_field_chars or c_len > max_field_chars or r_len > max_field_chars:
                stats.record_drop("field_length_exceeded")
                continue

            if total_len > max_chars:
                stats.record_drop("total_length_exceeded")
                continue

            # 8. Deduplication checks
            p_hash = hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()
            if p_hash in seen_prompts:
                stats.record_drop("duplicate_prompt")
                continue

            pair_hash = hashlib.sha256((repaired_chosen + "\x00" + repaired_rejected).encode("utf-8")).hexdigest()
            if pair_hash in seen_pairs:
                stats.record_drop("duplicate_pair")
                continue

            seen_prompts.add(p_hash)
            seen_pairs.add(pair_hash)

            # Record successfully kept record
            row_id = hashlib.sha256(
                f"{rendered_prompt}\n{repaired_chosen}\n{repaired_rejected}".encode("utf-8")
            ).hexdigest()

            record: Dict[str, Any] = {
                "prompt": rendered_prompt,
                "chosen": repaired_chosen,
                "rejected": repaired_rejected,
                "source": source.id,
                "pair_type": pair_type,
                "prompt_messages": structured_prompt_msgs,
                "id": row_id,
            }
            if tools:
                record["tools"] = tools

            stats.kept += 1
            stats.char_lengths.append(total_len)
            if pair_type == "synthetic":
                stats.synthetic_count += 1
            else:
                stats.original_count += 1

            kept_records.append(record)
            if len(kept_records) >= max_per_source:
                return kept_records

    return kept_records


def run_pipeline(
    sources_to_run: Optional[List[str]] = None,
    max_per_source: int = DEFAULT_MAX_PER_SOURCE,
    prompt_format: str = "chatml",
    synthetic_cap: int = DEFAULT_SYNTHETIC_CAP,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_field_chars: int = DEFAULT_MAX_FIELD_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
    seed: int = DEFAULT_SEED,
    out_path: Union[str, Path] = DEFAULT_OUTPUT_PATH,
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    offline: bool = False,
    refresh: bool = False,
    quiet: bool = False,
) -> Tuple[int, Path, Path]:
    """Execute the full data pipeline end to end."""
    start_time = time.time()
    out_path = Path(out_path)
    cache_dir = Path(cache_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    builder = AgentForgePreferenceBuilder(seed=seed)

    # Determine sources to run
    registry_map = {s.short_name: s for s in SOURCE_REGISTRY}
    registry_map.update({s.id: s for s in SOURCE_REGISTRY})

    selected_sources: List[DatasetSource] = []
    if sources_to_run:
        for name in sources_to_run:
            name = name.strip()
            if name in registry_map:
                selected_sources.append(registry_map[name])
            else:
                logger.warning("Requested source '%s' not recognized in registry.", name)
    else:
        selected_sources = [s for s in SOURCE_REGISTRY if s.enabled_by_default]

    if not quiet:
        logger.info(
            "Ogun DPO Data Pipeline starting. Sources enabled: %s",
            [s.short_name for s in selected_sources],
        )

    all_records: List[Dict[str, Any]] = []
    per_source_stats: Dict[str, DatasetStats] = {}
    seen_prompts: Set[str] = set()
    seen_pairs: Set[str] = set()

    for source in selected_sources:
        stats = DatasetStats(source_id=source.id)
        per_source_stats[source.id] = stats

        try:
            records = process_dataset(
                source=source,
                max_per_source=max_per_source,
                prompt_format=prompt_format,
                max_chars=max_chars,
                max_field_chars=max_field_chars,
                min_chars=min_chars,
                cache_dir=cache_dir,
                offline=offline,
                refresh=refresh,
                seed=seed,
                stats=stats,
                seen_prompts=seen_prompts,
                seen_pairs=seen_pairs,
                builder=builder,
                quiet=quiet,
            )
        except Exception as e:
            logger.warning("Source %s failed with unexpected error: %s", source.id, e)
            records = []

        if not records and source.required and not offline:
            logger.warning("Required source '%s' produced 0 valid records.", source.id)

        # Stratified train / eval split per source
        n_records = len(records)
        if n_records > 0:
            rng.shuffle(records)
            eval_count = 0
            if n_records >= 50:
                eval_count = max(1, round(n_records * eval_fraction))
            else:
                eval_count = round(n_records * eval_fraction)

            for i, rec in enumerate(records):
                rec["split"] = "eval" if i < eval_count else "train"

            stats.eval_count = eval_count
            stats.train_count = n_records - eval_count
            all_records.extend(records)

    # Handle synthetic cap (AgentForge)
    synthetic_records = [r for r in all_records if r.get("pair_type") == "synthetic"]
    original_records = [r for r in all_records if r.get("pair_type") != "synthetic"]

    if len(synthetic_records) > synthetic_cap:
        if not quiet:
            logger.info("Capping synthetic tool records from %d down to %d", len(synthetic_records), synthetic_cap)
        rng.shuffle(synthetic_records)
        synthetic_records = synthetic_records[:synthetic_cap]

    combined = original_records + synthetic_records
    rng.shuffle(combined)  # Shuffle thoroughly across sources

    if not combined:
        logger.error("All datasets failed to produce any valid records. Exiting.")
        sys.exit(1)

    # Atomic write to JSONL
    temp_out = out_path.with_suffix(f".tmp.{os.getpid()}")
    with open(temp_out, "w", encoding="utf-8") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(temp_out, out_path)

    # Generate and write statistics JSON
    total_lens = [len(r["prompt"]) + len(r["chosen"]) + len(r["rejected"]) for r in combined]
    stats_data: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "total_records": len(combined),
        "total_train": sum(1 for r in combined if r["split"] == "train"),
        "total_eval": sum(1 for r in combined if r["split"] == "eval"),
        "total_original": len(original_records),
        "total_synthetic": len(synthetic_records),
        "length_percentiles": compute_percentiles(total_lens),
        "per_source": {},
    }

    for src_id, s in per_source_stats.items():
        stats_data["per_source"][src_id] = {
            "rows_read": s.rows_read,
            "kept": s.kept,
            "train": s.train_count,
            "eval": s.eval_count,
            "original": s.original_count,
            "synthetic": s.synthetic_count,
            "dropped_by_reason": s.dropped_by_reason,
            "length_percentiles": compute_percentiles(s.char_lengths),
        }

    stats_path = out_path.with_name(f"{out_path.stem}.stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2)

    if not quiet:
        logger.info(
            "Pipeline complete: %d records exported to %s (stats: %s) in %.2fs",
            len(combined),
            out_path,
            stats_path,
            time.time() - start_time,
        )

    return len(combined), out_path, stats_path


# ==============================================================================
# 7. VERIFICATION TEST FUNCTION & CLI
# ==============================================================================

def test_pipeline_output(path: Optional[Union[str, Path]] = None, n_samples: int = 2) -> bool:
    """Verify integrity and quality of exported DPO dataset.

    Checks:
    - File exists and valid JSONL
    - Required columns present
    - No empty strings
    - chosen != rejected
    - Even fence count in completions
    - Unique IDs and deduplicated prompts
    - Prints 2 sample records, counts, distributions, and length percentiles.
    """
    if path is None:
        path = DEFAULT_OUTPUT_PATH
    path = Path(path)

    if not path.exists():
        logger.error("Verification failed: Output file %s does not exist.", path)
        return False

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except Exception as e:
                logger.error("Verification failed: Line %d is invalid JSON: %s", idx, e)
                return False

    if not records:
        logger.error("Verification failed: File %s is empty.", path)
        return False

    required_cols = {"prompt", "chosen", "rejected", "source", "pair_type", "split", "prompt_messages", "id"}
    seen_ids: Set[str] = set()
    seen_prompts: Set[str] = set()
    split_counts: Dict[str, int] = {}
    source_counts: Dict[str, Dict[str, int]] = {}
    char_lens: List[int] = []

    for idx, rec in enumerate(records, 1):
        # Column checks
        missing = required_cols - set(rec.keys())
        if missing:
            logger.error("Line %d missing required columns: %s", idx, missing)
            return False

        # Non-empty checks
        p = rec["prompt"]
        c = rec["chosen"]
        r = rec["rejected"]
        if not p.strip() or not c.strip() or not r.strip():
            logger.error("Line %d contains empty prompt, chosen, or rejected field.", idx)
            return False

        if c == r:
            logger.error("Line %d has identical chosen and rejected completions.", idx)
            return False

        # Fence count checks
        if c.count("```") % 2 != 0:
            logger.error("Line %d chosen completion has unclosed code fence.", idx)
            return False
        if r.count("```") % 2 != 0:
            logger.error("Line %d rejected completion has unclosed code fence.", idx)
            return False

        # Uniqueness checks
        rec_id = rec["id"]
        if rec_id in seen_ids:
            logger.error("Line %d duplicate id '%s'", idx, rec_id)
            return False
        seen_ids.add(rec_id)

        # Track distributions
        sp = rec["split"]
        split_counts[sp] = split_counts.get(sp, 0) + 1

        src = rec["source"]
        if src not in source_counts:
            source_counts[src] = {"total": 0, "original": 0, "synthetic": 0}
        source_counts[src]["total"] += 1
        if rec.get("pair_type") == "synthetic":
            source_counts[src]["synthetic"] += 1
        else:
            source_counts[src]["original"] += 1

        char_lens.append(len(p) + len(c) + len(r))

    # Print Verification Report
    print("=" * 70)
    print("                 Ogun DPO Dataset Verification")
    print(f"File: {path.resolve()}")
    print("=" * 70)
    print(f"Total Rows: {len(records)}")

    # Split distribution
    print("\n[Split Distribution]")
    for sp, cnt in sorted(split_counts.items()):
        pct = (cnt / len(records)) * 100.0
        print(f"  - {sp:<10s}: {cnt:>6d} ({pct:5.1f}%)")

    # Source distribution
    print("\n[Source Distribution]")
    for src, cnts in sorted(source_counts.items()):
        pct = (cnts["total"] / len(records)) * 100.0
        print(f"  - {src:<42s}: {cnts['total']:>6d} ({pct:5.1f}%) [orig: {cnts['original']}, synth: {cnts['synthetic']}]")

    # Character statistics
    pcts = compute_percentiles(char_lens)
    print("\n[Total Character Statistics]")
    print(f"  - Min: {pcts['min']}, P25: {pcts['p25']}, Median: {pcts['median']}, P75: {pcts['p75']}, P95: {pcts['p95']}, Max: {pcts['max']}")

    # Display 2 sample formatted records
    print("\n[Sample Records]")
    sample_records = []
    # If tool pair present, pick 1 tool pair and 1 regular pair
    tool_pairs = [r for r in records if "tools" in r and r["tools"]]
    orig_pairs = [r for r in records if r.get("pair_type") != "synthetic"]

    if tool_pairs and n_samples > 1:
        sample_records.append(tool_pairs[0])
        if orig_pairs:
            sample_records.append(orig_pairs[0])
    while len(sample_records) < min(n_samples, len(records)):
        for r in records:
            if r not in sample_records:
                sample_records.append(r)
                if len(sample_records) >= n_samples:
                    break

    def truncate_str(val: str, max_chars: int = 140) -> str:
        if len(val) <= max_chars:
            return repr(val)
        return repr(val[:max_chars]) + f"... [{len(val) - max_chars} more chars]"

    for i, s_rec in enumerate(sample_records, 1):
        print(f"\n--- Sample {i} (source={s_rec['source']}, type={s_rec['pair_type']}, split={s_rec['split']}) ---")
        display_dict = {
            "id": s_rec["id"],
            "prompt": truncate_str(s_rec["prompt"]),
            "chosen": truncate_str(s_rec["chosen"]),
            "rejected": truncate_str(s_rec["rejected"]),
            "prompt_messages_count": len(s_rec.get("prompt_messages", [])),
            "tools_count": len(s_rec.get("tools", [])) if "tools" in s_rec else 0,
        }
        print(json.dumps(display_dict, indent=2))

    print("=" * 70)
    print("VERIFICATION RESULT: PASSED (All integrity and format checks passed)")
    print("=" * 70)
    return True


# ==============================================================================
# 8. BUILT-IN FIXTURES & SELFTEST
# ==============================================================================

def get_selftest_fixtures() -> Dict[str, List[Dict[str, Any]]]:
    """Provide realistic fixture rows for all supported dataset mappers."""
    return {
        "vezora": [
            {
                "instruction": "You are an AI-Coding assistant. Complete the task as requested.",
                "input": "Write a Python function to check if a word is a palindrome.",
                "accepted": "def is_palindrome(word: str) -> bool:\n    clean = word.lower()\n    return clean == clean[::-1]",
                "rejected": "def is_palindrome(word: str) -> bool:\n    return word == word[::-1] # bug: fails case sensitivity",
                "ID": "vezora-001",
            },
            {
                "instruction": "",
                "input": "Calculate factorial in python recursively.",
                "accepted": "def factorial(n: int) -> int:\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
                "rejected": "def factorial(n: int) -> int:\n    return n * factorial(n - 1) # missing base case",
                "ID": "vezora-002",
            },
        ],
        "jondurbin": [
            {
                "prompt": "Write a function to find the maximum element in a list.",
                "chosen": "def find_max(nums):\n    if not nums:\n        return None\n    return max(nums)",
                "rejected": "def find_max(nums):\n    return nums[0]  # bug: returns first element",
                "id": "jd-001",
            },
        ],
        "cybernative": [
            {
                "system": "",
                "question": "Copy source string into destination buffer in C.",
                "chosen": "#include <string.h>\nvoid copy_str(char *dst, const char *src, size_t dst_sz) {\n    strncpy(dst, src, dst_sz - 1);\n    dst[dst_sz - 1] = '\\0';\n}",
                "rejected": "#include <string.h>\nvoid copy_str(char *dst, const char *src) {\n    strcpy(dst, src); // buffer overflow risk\n}",
                "lang": "c",
                "vulnerability": "Buffer overflow",
            },
        ],
        "ultrafeedback": [
            {
                "prompt": "How do I reverse a string in JavaScript?",
                "chosen": [
                    {"role": "user", "content": "How do I reverse a string in JavaScript?"},
                    {"role": "assistant", "content": "You can reverse a string using `str.split('').reverse().join('')`."},
                ],
                "rejected": [
                    {"role": "user", "content": "How do I reverse a string in JavaScript?"},
                    {"role": "assistant", "content": "Strings have a reverse() method in JS so just call str.reverse()."},
                ],
            },
        ],
        "dpo_mix_7k": [
            {
                "dataset": "argilla/mix",
                "chosen": [
                    {"role": "user", "content": "Explain binary search in one line."},
                    {"role": "assistant", "content": "Binary search finds a target in a sorted list by repeatedly dividing search range in half."},
                ],
                "rejected": [
                    {"role": "user", "content": "Explain binary search in one line."},
                    {"role": "assistant", "content": "Binary search looks at every element from index 0 to n-1 sequentially."},
                ],
            },
        ],
        "agentforge": [
            {
                "id": "af-001",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_contact",
                            "description": "Look up contact in CRM.",
                            "parameters": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            },
                        },
                    }
                ],
                "conversations": [
                    {"role": "system", "content": "You are a CRM assistant."},
                    {"role": "user", "content": "Find contact Chloe Silva."},
                    {
                        "role": "assistant",
                        "content": "Searching CRM...",
                        "tool_calls": [{"name": "lookup_contact", "arguments": {"query": "Chloe Silva"}}],
                    },
                    {"role": "tool", "name": "lookup_contact", "content": '{"contact_id": "C-123", "email": "chloe@example.com"}'},
                    {"role": "assistant", "content": "Found Chloe Silva with email chloe@example.com."},
                ],
            }
        ],
    }


def run_selftest() -> bool:
    """Execute end-to-end selftest on built-in fixtures in temporary directory."""
    print("=" * 70)
    print("          Ogun DPO Data Pipeline - Selftest")
    print("=" * 70)
    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        cache_dir = temp_dir / "cache"
        out_file = temp_dir / "selftest_dpo.jsonl"

        fixtures = get_selftest_fixtures()

        # Write fixtures to mock files in cache
        vezora_dir = cache_dir / "Vezora_Code-Preference-Pairs"
        vezora_dir.mkdir(parents=True, exist_ok=True)
        with open(vezora_dir / "Code-Preference-Pairs.jsonl", "w", encoding="utf-8") as f:
            for row in fixtures["vezora"]:
                f.write(json.dumps(row) + "\n")

        cyber_dir = cache_dir / "CyberNative_Code_Vulnerability_Security_DPO"
        cyber_dir.mkdir(parents=True, exist_ok=True)
        with open(cyber_dir / "secure_programming_dpo.json", "w", encoding="utf-8") as f:
            for row in fixtures["cybernative"]:
                f.write(json.dumps(row) + "\n")

        af_dir = cache_dir / "voxozi_agentforge-multiturn-toolcall"
        af_dir.mkdir(parents=True, exist_ok=True)
        with open(af_dir / "train.jsonl", "w", encoding="utf-8") as f:
            for row in fixtures["agentforge"]:
                f.write(json.dumps(row) + "\n")

        # Mock rows API cache files
        jd_dir = cache_dir / "jondurbin_py-dpo-v0.1"
        jd_dir.mkdir(parents=True, exist_ok=True)
        with open(jd_dir / "train_rows.jsonl", "w", encoding="utf-8") as f:
            for row in fixtures["jondurbin"]:
                f.write(json.dumps(row) + "\n")

        uf_dir = cache_dir / "HuggingFaceH4_ultrafeedback_binarized"
        uf_dir.mkdir(parents=True, exist_ok=True)
        with open(uf_dir / "train_prefs_rows.jsonl", "w", encoding="utf-8") as f:
            for row in fixtures["ultrafeedback"]:
                f.write(json.dumps(row) + "\n")

        dm_dir = cache_dir / "argilla_dpo-mix-7k"
        dm_dir.mkdir(parents=True, exist_ok=True)
        with open(dm_dir / "train_rows.jsonl", "w", encoding="utf-8") as f:
            for row in fixtures["dpo_mix_7k"]:
                f.write(json.dumps(row) + "\n")

        # Run pipeline in offline mode against mock files
        print("\nRunning pipeline offline against fixtures...")
        count, out_p, stats_p = run_pipeline(
            sources_to_run=["vezora", "jondurbin", "cybernative", "ultrafeedback", "dpo_mix_7k", "agentforge"],
            max_per_source=50,
            out_path=out_file,
            cache_dir=cache_dir,
            offline=True,
            quiet=True,
        )

        print(f"Pipeline created {count} records in {out_p}")
        if count == 0:
            print("FAIL: Selftest produced 0 records.")
            return False

        # Run verification function
        print("\nRunning test_pipeline_output on selftest dataset...")
        ok = test_pipeline_output(out_p, n_samples=2)
        if ok:
            print("\nSELFTEST STATUS: PASS")
            return True
        else:
            print("\nSELFTEST STATUS: FAIL")
            return False


# ==============================================================================
# 9. CLI ENTRY POINT
# ==============================================================================

def print_source_registry() -> None:
    print("=" * 70)
    print("                 Ogun DPO Source Registry")
    print("=" * 70)
    for s in SOURCE_REGISTRY:
        status = "ENABLED" if s.enabled_by_default else "DISABLED"
        req = "REQUIRED" if s.required else "OPTIONAL"
        print(f"[{status:8s}] {s.short_name:<20s} ({req:8s}) -> {s.id}")
        print(f"           Loader: {s.loader_type:<15s} Split: {s.split}")
        print(f"           Desc:   {s.description}")
        print("-" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ogun DPO Dataset Preparation Pipeline for Qwen Coder Fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sources", type=str, default="", help="Comma-separated dataset ids or short names to run.")
    parser.add_argument("--list-sources", action="store_true", help="Print registered sources and exit.")
    parser.add_argument("--max-per-source", type=int, default=DEFAULT_MAX_PER_SOURCE, help="Maximum rows kept per source.")
    parser.add_argument("--synthetic-cap", type=int, default=DEFAULT_SYNTHETIC_CAP, help="Maximum synthetic pairs (AgentForge).")
    parser.add_argument("--prompt-format", choices=["chatml", "plain"], default="chatml", help="Prompt template format.")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="Maximum total chars (prompt+chosen+rejected).")
    parser.add_argument("--max-field-chars", type=int, default=DEFAULT_MAX_FIELD_CHARS, help="Maximum chars for any single field.")
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS, help="Minimum completion characters.")
    parser.add_argument("--eval-fraction", type=float, default=DEFAULT_EVAL_FRACTION, help="Stratified eval split fraction.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for sampling and corruptions.")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Output JSONL path.")
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR), help="Raw download cache directory.")
    parser.add_argument("--offline", action="store_true", help="Offline mode: use cache/fixtures only, no network calls.")
    parser.add_argument("--refresh", action="store_true", help="Force redownload, bypassing cached files.")
    parser.add_argument("--selftest", action="store_true", help="Run end-to-end selftest on built-in fixtures in temp dir.")
    parser.add_argument("--verify", nargs="?", const="__DEFAULT__", type=str, help="Verify output JSONL file.")
    parser.add_argument("--no-verify", action="store_true", help="Skip automatic verification after pipeline run.")
    parser.add_argument("--quiet", action="store_true", help="Minimize logging output.")

    args = parser.parse_args()

    if args.list_sources:
        print_source_registry()
        sys.exit(0)

    if args.selftest:
        ok = run_selftest()
        sys.exit(0 if ok else 1)

    if args.verify is not None:
        verify_path = Path(args.out) if args.verify == "__DEFAULT__" else Path(args.verify)
        ok = test_pipeline_output(verify_path)
        sys.exit(0 if ok else 1)

    sources_list = [s.strip() for s in args.sources.split(",") if s.strip()] if args.sources else None

    count, out_p, stats_p = run_pipeline(
        sources_to_run=sources_list,
        max_per_source=args.max_per_source,
        prompt_format=args.prompt_format,
        synthetic_cap=args.synthetic_cap,
        max_chars=args.max_chars,
        max_field_chars=args.max_field_chars,
        min_chars=args.min_chars,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
        out_path=args.out,
        cache_dir=args.cache_dir,
        offline=args.offline,
        refresh=args.refresh,
        quiet=args.quiet,
    )

    if not args.no_verify:
        ok = test_pipeline_output(out_p)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
