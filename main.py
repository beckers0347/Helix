"""Helix standalone autonomous coding assistant CLI entrypoint."""
from __future__ import annotations

import argparse
import atexit
import os
import re
import sys

from helix_core.config import MODEL_NAME, PLUGIN_BUILDER_PROMPT
from helix_core.config import set_vram_profile
from helix_core.engine import CoreEngine, OllamaClient, handle_vram_command, unload_model
from helix_core.paths import init_user_workspace
from helix_core.plugin_manager import PluginManager


UNLOAD_PHRASES = {
    "/unload", "unload", "unload model", "unload the model", "free gpu", "free the gpu", "free up gpu",
    "free vram", "free up vram", "release gpu memory", "release vram",
}


def is_unload_command(text: str) -> bool:
    """True only for an exact unload phrase, so ordinary sentences ('unload the dishwasher') are never hijacked."""
    return " ".join((text or "").lower().replace("?", " ").replace(".", " ").split()) in UNLOAD_PHRASES


def parse_code_blocks(text: str) -> list[str]:
    """Extract fenced python code blocks from text."""
    blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)\s*```", text)
    return [b.strip() for b in blocks if b.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Helix Autonomous Coding Assistant")
    parser.add_argument("--once", type=str, help="Execute a single turn and exit.")
    parser.add_argument("--build-plugin", type=str, help="Prompt the model to build, verify, and mount a plugin.")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Ollama model to use.")
    parser.add_argument("--no-tools", action="store_true", help="Disable tool execution.")
    parser.add_argument("--unload", action="store_true",
                        help="Unload the model from Ollama (free its GPU memory) and exit.")
    parser.add_argument("--vram", type=str, default=None, metavar="PROFILE",
                        help="Start with a VRAM profile: max | shared | cpu (default: shared). Change it in chat with /vram.")

    args = parser.parse_args()

    # --unload must work even when nothing else can: handle it before any connection check or workspace setup.
    if args.unload:
        result = unload_model(model=args.model)
        print(f"[{'OK' if result['unloaded'] else 'WARN'}] {result['detail']}")
        sys.exit(0)

    if args.vram:
        ok, _changed, profile, num_gpu = set_vram_profile(args.vram)
        if not ok:
            print(f"[ERROR] Unknown --vram profile '{args.vram}'. Use max, shared or cpu.", file=sys.stderr)
            sys.exit(2)
        print(f"[*] VRAM profile: {profile} (num_gpu={num_gpu})")

    # Pre-flight check on Ollama connection
    client = OllamaClient(model=args.model)
    connected, conn_msg = client.check_connection()
    if not connected:
        print(f"[ERROR] {conn_msg}", file=sys.stderr)
        sys.exit(1)

    # Free the GPU when Helix exits (normal exit, Ctrl+C, --once, --build-plugin). Opt out: HELIX_UNLOAD_ON_EXIT=0.
    if os.environ.get("HELIX_UNLOAD_ON_EXIT", "1") != "0":
        _unloaded = {"done": False}

        def _unload_on_exit() -> None:
            if not _unloaded["done"]:
                _unloaded["done"] = True
                unload_model(model=args.model)

        atexit.register(_unload_on_exit)

    # Initialize workspace directories on boot
    init_user_workspace()

    plugin_manager = PluginManager()
    plugin_manager.discover_tools()

    engine = CoreEngine(
        client=client,
        plugin_manager=None if args.no_tools else plugin_manager,
    )

    # 1. Build plugin mode
    if args.build_plugin:
        prompt = (
            f"{PLUGIN_BUILDER_PROMPT}\n\n"
            f"Generate a dynamic plugin satisfying this request:\n{args.build_plugin}"
        )
        print(f"[*] Prompting {args.model} to author plugin for: {args.build_plugin}...")
        reply = engine.run_turn(prompt)
        blocks = parse_code_blocks(reply)
        if len(blocks) < 2:
            print("[ERROR] Expected at least 2 python code blocks (tool and test). Model returned:", file=sys.stderr)
            print(reply, file=sys.stderr)
            sys.exit(1)

        tool_code, test_code = blocks[0], blocks[1]
        # Infer name from class or regex
        name_match = re.search(r'name\s*:\s*str\s*=\s*["\']([^"\']+)["\']', tool_code)
        plugin_name = name_match.group(1) if name_match else "custom_tool"

        res = plugin_manager.create_or_update_plugin(plugin_name, tool_code, test_code)
        if res.status == "success":
            print(f"[PASS] Plugin '{plugin_name}' successfully built, tested, and mounted.")
            print(res.output)
            sys.exit(0)
        else:
            print(f"[FAIL] Plugin creation failed: {res.error_trace}", file=sys.stderr)
            sys.exit(1)

    # 2. Single turn mode
    if args.once:
        reply = engine.run_turn(args.once)
        print(reply)
        sys.exit(0)

    # 3. Interactive CLI loop
    print("=" * 60)
    print(f"  Helix Autonomous Coding Assistant (Model: {args.model})")
    print("  Type 'exit' or 'quit' to close. Press Ctrl+C to cancel a turn.")
    print("  GPU memory: /vram shared | /vram max | /vram cpu | /vram free | /vram status")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nHelix > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye.")
                break
            vram_reply = handle_vram_command(user_input, model=args.model)
            if vram_reply is not None:
                print(f"\n{vram_reply}")
                continue
            if is_unload_command(user_input):
                result = unload_model(model=args.model)
                print(f"\n{result['detail']} The next question reloads the model (about 40 seconds).")
                continue
            reply = engine.run_turn(user_input)
            print(f"\n{reply}")
        except KeyboardInterrupt:
            print("\nTurn canceled by user.")
        except EOFError:
            break


if __name__ == "__main__":
    main()
