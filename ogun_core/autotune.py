"""Startup hardware auto-tuning for Ogun: probe this machine, size the inference runtime to it, and report.

Kept out of main.py so the CLI stays small and the whole flow is testable. Nothing here runs at import time.
Every failure degrades to Ogun's static defaults (the original behaviour); auto-tuning never stops Ogun from starting.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from ogun_core import hardware
from ogun_core.config import apply_hardware_profile, get_vram_profile
from ogun_core.paths import OGUN_ROOT

PROFILE_PATH = OGUN_ROOT / "config" / "hardware_profile.json"


def add_arguments(parser) -> None:
    """Register the hardware flags on an argparse parser."""
    g = parser.add_argument_group("hardware auto-tuning")
    g.add_argument("--auto-tune", action="store_true",
                   help="Re-detect this machine and ignore any saved overrides (default: reuse saved overrides while the hardware is unchanged).")
    g.add_argument("--gpu-layers", type=str, default=None, metavar="N|all",
                   help="Force how many model layers run on the GPU (0 = CPU only, 'all' = everything).")
    g.add_argument("--force-cpu", action="store_true", help="Never use the GPU.")
    g.add_argument("--ctx", type=int, default=None, metavar="TOKENS", help="Force the context window (4096-16384 is the tuned range).")
    g.add_argument("--threads", type=int, default=None, metavar="N", help="Force the CPU thread count.")
    g.add_argument("--save-overrides", action="store_true", help="Remember the flags above for later runs on this machine.")
    g.add_argument("--no-hardware-probe", action="store_true", help="Skip auto-tuning and use Ogun's static defaults.")
    g.add_argument("--hardware-info", action="store_true", help="Print the detected hardware and the settings Ogun would use, then exit.")
    g.add_argument("--json", action="store_true", help="With --hardware-info: machine-readable output.")


def cli_overrides(args) -> Dict[str, Any]:
    """Translate parsed flags into auto_configure overrides. Unparseable values are ignored."""
    out: Dict[str, Any] = {}
    if getattr(args, "force_cpu", False):
        out["force_cpu"] = True
    raw = getattr(args, "gpu_layers", None)
    if raw is not None:
        if str(raw).strip().lower() in ("all", "auto", "max"):
            out["gpu_layers"] = 99
        else:
            try:
                out["gpu_layers"] = int(raw)
            except ValueError:
                pass
    if getattr(args, "ctx", None):
        out["ctx"] = int(args.ctx)
    if getattr(args, "threads", None):
        out["threads"] = int(args.threads)
    return out


def user_chose_gpu_split() -> bool:
    """True when the user already picked a GPU split through the environment: auto-tuning must not override that."""
    return bool(os.environ.get("OGUN_VRAM_PROFILE") or os.environ.get("OGUN_NUM_GPU"))


def _detect(host: str, model: str):
    hw = hardware.probe_hardware()
    oll = hardware.query_ollama(host, family=str(model).split(":")[0])
    # Memory Ollama already holds for the loaded model is replaced by our own load: it must not make the machine look smaller.
    hw = hardware.adjust_for_resident_models(hw, oll["resident_vram_gb"], oll["resident_ram_gb"])
    return hw, oll


def hardware_info(args, model: str, host: str) -> int:
    """--hardware-info: show what was detected and what would be used; saves the profile; never loads a model."""
    explicit = getattr(args, "model_explicit", None)
    overrides = cli_overrides(args)
    if explicit:
        overrides["model"] = explicit
    overrides["configured_model"] = model
    hw, oll = _detect(host, model)
    prof = hardware.auto_configure(hw, installed=oll["installed"] or None, overrides=overrides)
    hardware.save_profile(PROFILE_PATH, hw, prof, overrides=cli_overrides(args) if args.save_overrides else None)
    if getattr(args, "json", False):
        print(json.dumps({"fingerprint": hardware.fingerprint(hw), "hardware": hardware.hardware_to_dict(hw),
                          "profile": hardware.profile_to_dict(prof)}, indent=2))
    else:
        print(hardware.format_banner(hw, prof, profile_path=str(PROFILE_PATH)))
    return 0


def boot(args, model: str, host: str) -> Optional[Dict[str, Any]]:
    """Startup hook. Probes, tunes, applies the profile to the live config, saves it and prints the banner.

    Returns the applied profile dict, or None when auto-tuning was skipped or failed (static defaults stay in force).
    """
    if getattr(args, "no_hardware_probe", False):
        return None
    if getattr(args, "vram", None) or user_chose_gpu_split():
        print(f"[*] GPU split chosen explicitly ({get_vram_profile()}); hardware auto-tuning skipped. "
              "Run with --auto-tune and without --vram / OGUN_VRAM_PROFILE / OGUN_NUM_GPU to let Ogun choose.")
        return None
    try:
        hw, oll = _detect(host, model)
        saved = hardware.load_profile(PROFILE_PATH)
        saved_overrides: Dict[str, Any] = {}
        if isinstance(saved, dict) and saved.get("fingerprint") == hardware.fingerprint(hw) and not args.auto_tune:
            saved_overrides = saved.get("overrides", {}) or {}
        flags = cli_overrides(args)
        explicit = getattr(args, "model_explicit", None)
        if explicit:
            flags["model"] = explicit
        merged = {**saved_overrides, **flags}
        profile = hardware.auto_configure(hw, installed=oll["installed"] or None,
                                          overrides={**merged, "configured_model": model})
        applied = hardware.profile_to_dict(profile)
        apply_hardware_profile(applied)
        hardware.save_profile(PROFILE_PATH, hw, profile, overrides=merged if args.save_overrides else saved_overrides)
        print(hardware.format_banner(hw, profile, profile_path=str(PROFILE_PATH)))
        return applied
    except Exception as exc:  # auto-tuning must never stop Ogun from starting
        print(f"[Hardware Warning] Auto-tuning fell back to the default configuration: {exc}")
        return None
