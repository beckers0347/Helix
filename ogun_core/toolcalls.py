"""Ogun pure text tool-call parser.

Extracts structured tool calls from model text outputs:
- Markdown fenced JSON blocks (non-greedy)
- Qwen <tool_call>...</tool_call> tags
- Bare JSON objects via json.JSONDecoder().raw_decode
- tool_calls list wrappers
- Python literal evaluation fallback for single-quoted dicts
- Stringified JSON argument unpacking
- Strictly filters out unregistered tool names
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any, Collection, Dict, List, Optional, Set, Tuple, Union

_FENCE_RE = re.compile(r"```(?:json)?\s*?(\{[\s\S]*?\})\s*?```")
_QWEN_TOOL_CALL_RE = re.compile(r"<tool_call>([\s\S]*?)</tool_call>")


def _unpack_arguments(raw_args: Any) -> Dict[str, Any]:
    """Unpack arguments if provided as a stringified JSON or python literal."""
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        raw_str = raw_args.strip()
        if not raw_str:
            return {}
        try:
            val = json.loads(raw_str)
            if isinstance(val, dict):
                return val
        except Exception:
            pass
        try:
            val = ast.literal_eval(raw_str)
            if isinstance(val, dict):
                return val
        except Exception:
            pass
        return {"input": raw_args}
    return {}


def _parse_candidate(raw: str) -> Optional[Any]:
    """Parse candidate string using json.loads then ast.literal_eval fallback."""
    cleaned = raw.strip()
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    try:
        val = ast.literal_eval(cleaned)
        if isinstance(val, (dict, list)):
            return val
    except Exception:
        pass
    return None


def parse_tool_calls(
    text: str,
    registered_tools: Union[Collection[str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Parse tool calls from LLM response text against registered tool names.

    Pure, importable, with no side effects.
    """
    if not text or not isinstance(text, str):
        return []

    if isinstance(registered_tools, dict):
        reg_names: Set[str] = set(registered_tools.keys())
    else:
        reg_names = set(registered_tools)

    normalized_text = text.replace("\r\n", "\n")
    candidates: List[Any] = []
    seen_spans: Set[Tuple[int, int]] = set()

    # 1. Qwen <tool_call> tags
    for match in _QWEN_TOOL_CALL_RE.finditer(normalized_text):
        parsed = _parse_candidate(match.group(1))
        if parsed is not None:
            candidates.append(parsed)
            seen_spans.add(match.span())

    # 3. Markdown fenced code blocks (non-greedy)
    for match in _FENCE_RE.finditer(normalized_text):
        parsed = _parse_candidate(match.group(1))
        if parsed is not None:
            candidates.append(parsed)
            seen_spans.add(match.span())

    # 4. Bare JSON decoder scan for objects inside prose, with ast.literal_eval fallback
    decoder = json.JSONDecoder()
    pos = 0
    length = len(normalized_text)
    while pos < length:
        idx = normalized_text.find("{", pos)
        if idx == -1:
            break
        # Skip if within an already-parsed fence or tag span
        in_seen = any(start <= idx < end for start, end in seen_spans)
        if in_seen:
            pos = idx + 1
            continue

        # Try JSONDecoder first
        try:
            obj, end_idx = decoder.raw_decode(normalized_text, idx)
            candidates.append(obj)
            seen_spans.add((idx, end_idx))
            pos = end_idx
            continue
        except Exception:
            pass

        # Fallback: scan candidate Python dict literal ending with '}'
        matched_literal = False
        end_search = idx + 1
        while end_search < length:
            brace_idx = normalized_text.find("}", end_search)
            if brace_idx == -1:
                break
            candidate_chunk = normalized_text[idx : brace_idx + 1].strip()
            try:
                lit_obj = ast.literal_eval(candidate_chunk)
                if isinstance(lit_obj, (dict, list)):
                    candidates.append(lit_obj)
                    seen_spans.add((idx, brace_idx + 1))
                    pos = brace_idx + 1
                    matched_literal = True
                    break
            except Exception:
                pass
            end_search = brace_idx + 1

        if not matched_literal:
            pos = idx + 1

    # Extract tool call items from candidates
    results: List[Dict[str, Any]] = []

    def process_item(item: Any) -> None:
        if not isinstance(item, dict):
            return

        # Check for tool_calls list wrapper
        if "tool_calls" in item and isinstance(item["tool_calls"], list):
            for tc in item["tool_calls"]:
                process_item(tc)
            return

        # Extract name
        name = item.get("name")
        if not name and isinstance(item.get("function"), dict):
            name = item["function"].get("name")
        if not name and "tool" in item:
            name = item.get("tool")

        if not name or not isinstance(name, str):
            return

        name = name.strip()
        if name not in reg_names:
            return

        # Extract arguments / parameters
        args = item.get("arguments")
        if args is None:
            args = item.get("parameters")
        if args is None and isinstance(item.get("function"), dict):
            args = item["function"].get("arguments")
        if args is None:
            args = item.get("args")

        args = _unpack_arguments(args)
        results.append({"name": name, "arguments": args})

    for cand in candidates:
        if isinstance(cand, list):
            for sub in cand:
                process_item(sub)
        else:
            process_item(cand)

    return results
