"""Comprehensive offline tests for hardware profiling, dynamic tiering, auto-configuration,
persistence, diagnostic banners, and runtime integration.

Plain test runner style (no external network, no live model, no Ollama daemon required).
Run: python tests/test_hardware.py
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from ogun_core import hardware
from ogun_core.hardware import (
    GpuInfo,
    HardwareInfo,
    ModelSpec,
    ExecutionProfile,
    probe_hardware,
    catalogue,
    model_spec_from_show,
    choose_tier,
    auto_configure,
    format_banner,
    profile_to_dict,
    profile_from_dict,
    hardware_to_dict,
    fingerprint,
    save_profile,
    load_profile,
    recommend_model,
)


def _create_mock_hw(
    os_name="windows",
    cpu_name="Mock CPU 8-Core",
    physical=8,
    logical=16,
    ram_total=32.0,
    ram_avail=24.0,
    gpu_vendor="nvidia",
    gpu_name="NVIDIA GeForce RTX 4070",
    vram_total=12.0,
    vram_free=10.0,
    unified=False,
    driver="550.00",
    cc="8.9",
    isa=None,
    notes=None,
) -> HardwareInfo:
    """Helper to build HardwareInfo instances for offline tests."""
    isa = isa or {"avx": True, "avx2": True, "avx512": False, "neon": False}
    gpus = []
    if gpu_vendor != "none":
        gpus.append(
            GpuInfo(
                vendor=gpu_vendor,
                name=gpu_name,
                vram_total_gb=vram_total,
                vram_free_gb=vram_free,
                unified_memory=unified,
                driver=driver,
                compute_capability=cc,
                index=0,
            )
        )
    return HardwareInfo(
        os=os_name,
        arch="x86_64" if "arm" not in cpu_name.lower() else "arm64",
        cpu_name=cpu_name,
        cpu_physical=physical,
        cpu_logical=logical,
        isa=isa,
        ram_total_gb=ram_total,
        ram_available_gb=ram_avail,
        gpus=gpus,
        gpu=gpus[0] if gpus else None,
        probe_seconds=0.05,
        notes=notes or [],
    )


# -----------------------------------------------------------------------------
# Test 1: Hardware Probing & Diverse Scenarios
# -----------------------------------------------------------------------------

def test_hardware_probing_scenarios():
    """Verify hardware discovery with mocked system environments."""
    # 1. NVIDIA 24GB / 64GB RAM
    hw1 = _create_mock_hw(vram_total=24.0, vram_free=22.0, ram_total=64.0, ram_avail=50.0)
    assert hw1.gpu is not None
    assert hw1.gpu.vram_total_gb == 24.0
    assert hw1.ram_total_gb == 64.0

    # 2. This PC (12GB, 10.7GB free, 32GB RAM, Zen 3 24L/12P)
    hw2 = _create_mock_hw(
        cpu_name="AMD Ryzen 9 5900X 12-Core Processor",
        physical=12,
        logical=24,
        ram_total=32.0,
        ram_avail=26.0,
        gpu_name="NVIDIA GeForce RTX 5070",
        vram_total=12.0,
        vram_free=10.7,
    )
    assert hw2.cpu_physical == 12
    assert hw2.cpu_logical == 24
    assert hw2.gpu.vram_free_gb == 10.7

    # 3. 8GB GPU / 16GB RAM
    hw3 = _create_mock_hw(vram_total=8.0, vram_free=6.5, ram_total=16.0, ram_avail=10.0)
    assert hw3.gpu.vram_total_gb == 8.0

    # 4. 6GB GPU / 16GB RAM
    hw4 = _create_mock_hw(vram_total=6.0, vram_free=4.8, ram_total=16.0, ram_avail=11.0)
    assert hw4.gpu.vram_total_gb == 6.0

    # 5. 4GB GPU / 16GB RAM
    hw5 = _create_mock_hw(vram_total=4.0, vram_free=3.0, ram_total=16.0, ram_avail=12.0)
    assert hw5.gpu.vram_total_gb == 4.0

    # 6. 4GB GPU + 64GB RAM (Tier overlap)
    hw6 = _create_mock_hw(vram_total=4.0, vram_free=3.0, ram_total=64.0, ram_avail=54.0)
    assert hw6.gpu.vram_total_gb == 4.0
    assert hw6.ram_total_gb == 64.0

    # 7. AMD ROCm on Linux
    rocm_json = json.dumps({
        "0": {
            "VRAM Total Memory (B)": str(16 * 1024**3),
            "VRAM Total Used Memory (B)": str(2 * 1024**3),
        }
    })
    with patch("ogun_core.hardware._run_subproc", return_value=(0, rocm_json)):
        gpus_amd = hardware._probe_amd_gpus("linux", 2.0, [])
        assert len(gpus_amd) == 1
        assert gpus_amd[0].vendor == "amd"
        assert gpus_amd[0].vram_total_gb == 16.0
        assert gpus_amd[0].vram_free_gb == 14.0

    # 8. Intel iGPU only (< 6GB) -> CPU-only
    with patch("winreg.OpenKey", side_effect=Exception("No Intel Arc")):
        intel_gpus = hardware._probe_intel_gpus("windows", [])
        assert len(intel_gpus) == 0

    # 9. Apple Silicon M2 16GB unified
    hw_apple = hardware._probe_apple_silicon("darwin", "arm64", "Apple M2", 16.0, 11.0)
    assert hw_apple is not None
    assert hw_apple.vendor == "apple"
    assert hw_apple.unified_memory is True
    assert hw_apple.vram_total_gb == round(16.0 * 0.65, 2)

    # 10. CPU-only 8GB RAM
    hw_cpu8 = _create_mock_hw(gpu_vendor="none", ram_total=8.0, ram_avail=4.5)
    assert hw_cpu8.gpu is None

    # 11. CPU-only 64GB RAM
    hw_cpu64 = _create_mock_hw(gpu_vendor="none", ram_total=64.0, ram_avail=55.0)
    assert hw_cpu64.gpu is None

    # 12. nvidia-smi missing, timing out, or garbage
    with patch("ogun_core.hardware._find_nvidia_smi", return_value=None):
        notes = []
        g_none = hardware._probe_nvidia_gpus(2.0, notes)
        assert len(g_none) == 0

    with patch("ogun_core.hardware._find_nvidia_smi", return_value="nvidia-smi"):
        with patch("ogun_core.hardware._run_subproc", return_value=(-1, "timeout")):
            notes = []
            g_timeout = hardware._probe_nvidia_gpus(2.0, notes)
            assert len(g_timeout) == 0
            assert any("failed" in n for n in notes)

        with patch("ogun_core.hardware._run_subproc", return_value=(0, "not,a,valid,csv,output\nsecond,line,corrupt")):
            notes = []
            g_garbage = hardware._probe_nvidia_gpus(2.0, notes)
            assert len(g_garbage) == 0
            assert len(notes) > 0

    # 13. Probe where every internal call raises (must return CPU-only HardwareInfo without raising)
    with patch("ogun_core.hardware._probe_nvidia_gpus", side_effect=RuntimeError("GPU probe failed")), \
         patch("ogun_core.hardware._probe_amd_gpus", side_effect=RuntimeError("AMD probe failed")), \
         patch("ogun_core.hardware._probe_intel_gpus", side_effect=RuntimeError("Intel probe failed")), \
         patch("ogun_core.hardware._probe_ram", side_effect=RuntimeError("RAM probe failed")), \
         patch("ogun_core.hardware._probe_cpu_cores", side_effect=RuntimeError("Core probe failed")):
        hw_crash = hardware.probe_hardware()
        assert isinstance(hw_crash, HardwareInfo)
        assert hw_crash.gpu is None
        assert len(hw_crash.notes) > 0


# -----------------------------------------------------------------------------
# Test 2: Tier Determination Rules
# -----------------------------------------------------------------------------

def test_choose_tier_rules():
    """Verify tier assignment precedence: VRAM >= 16GB or RAM >= 64GB is high;
    (8-16GB VRAM or RAM >= 32GB) is mid; else low.
    """
    # High tier scenarios
    assert choose_tier(_create_mock_hw(vram_total=24.0, ram_total=64.0)) == "high"
    assert choose_tier(_create_mock_hw(vram_total=16.0, ram_total=16.0)) == "high"
    assert choose_tier(_create_mock_hw(vram_total=4.0, ram_total=64.0)) == "high"  # Tier overlap!
    assert choose_tier(_create_mock_hw(gpu_vendor="none", ram_total=64.0)) == "high"

    # Mid tier scenarios
    assert choose_tier(_create_mock_hw(vram_total=12.0, ram_total=32.0)) == "mid"
    assert choose_tier(_create_mock_hw(vram_total=8.0, ram_total=16.0)) == "mid"
    assert choose_tier(_create_mock_hw(vram_total=4.0, ram_total=32.0)) == "mid"
    assert choose_tier(_create_mock_hw(gpu_vendor="none", ram_total=32.0)) == "mid"

    # Low tier scenarios
    assert choose_tier(_create_mock_hw(vram_total=6.0, ram_total=16.0)) == "low"
    assert choose_tier(_create_mock_hw(vram_total=4.0, ram_total=16.0)) == "low"
    assert choose_tier(_create_mock_hw(gpu_vendor="none", ram_total=16.0)) == "low"
    assert choose_tier(_create_mock_hw(gpu_vendor="none", ram_total=8.0)) == "low"


# -----------------------------------------------------------------------------
# Test 3: Auto-Configuration Determinism & Invariants
# -----------------------------------------------------------------------------

def test_auto_configure_determinism_and_invariants():
    """Verify mathematical invariants, determinism, and monotonicity across 50 random setups."""
    hw = _create_mock_hw(vram_total=12.0, vram_free=10.0, ram_total=32.0, ram_avail=24.0)

    # 1. Determinism
    prof1 = auto_configure(hw)
    prof2 = auto_configure(hw)
    assert prof1 == prof2

    # 2. Invariants over a pseudo-random grid
    rng = random.Random(42)
    cat = catalogue()
    models = {m.tag: m for m in cat}
    model_14b = models["qwen2.5-coder:14b"]

    for _ in range(50):
        vram_tot = rng.choice([0.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0])
        vram_free = round(rng.uniform(0.0, vram_tot), 2) if vram_tot > 0 else 0.0
        ram_tot = rng.choice([8.0, 16.0, 32.0, 64.0, 128.0])
        ram_avail = round(rng.uniform(4.0, ram_tot), 2)
        cores = rng.randint(2, 32)

        hw_case = _create_mock_hw(
            gpu_vendor="nvidia" if vram_tot > 0 else "none",
            vram_total=vram_tot,
            vram_free=vram_free,
            ram_total=ram_tot,
            ram_avail=ram_avail,
            physical=cores,
        )

        prof = auto_configure(hw_case, model=model_14b)

        # Invariant A: VRAM budget constraint (when not forced CPU)
        if prof.mode in ("gpu", "hybrid"):
            assert prof.est_vram_gb <= max(0.0, hw_case.gpu.vram_free_gb - prof.reserve_gb) + 0.01

        # Invariant B: num_gpu is 99 (never blocks) when full offload, 0 when cpu
        if prof.mode == "gpu":
            assert prof.num_gpu == 99
            assert prof.gpu_units == prof.total_units
        elif prof.mode == "cpu":
            assert prof.num_gpu == 0
            assert prof.gpu_units == 0
        else:
            assert prof.num_gpu == prof.gpu_units

        # Invariant C: Context window constraints
        assert 4096 <= prof.num_ctx <= 16384
        assert prof.num_ctx % 1024 == 0
        assert prof.num_ctx <= model_14b.native_ctx

        # Invariant D: Threads allocation
        assert 1 <= prof.num_thread <= 32
        assert prof.num_thread <= hw_case.cpu_physical
        if prof.mode == "gpu":
            assert prof.num_thread <= 8

    # 3. Monotonicity in free VRAM: more free VRAM never yields fewer layers
    units_seen = []
    for free_gb in [1.0, 2.0, 3.5, 5.0, 7.0, 9.0, 11.0, 14.0]:
        h_mono = _create_mock_hw(vram_total=16.0, vram_free=free_gb, ram_total=32.0, ram_avail=25.0)
        p_mono = auto_configure(h_mono, model=model_14b)
        units_seen.append(p_mono.gpu_units)
    for a, b in zip(units_seen[:-1], units_seen[1:]):
        assert a <= b, f"Monotonicity violated: {a} > {b} with increasing free VRAM"

    # 4. Monotonicity in Context: larger context never yields more layers
    ctx_units = []
    for ctx_val in [4096, 8192, 12288, 16384]:
        h_ctx = _create_mock_hw(vram_total=12.0, vram_free=6.5, ram_total=32.0, ram_avail=25.0)
        p_ctx = auto_configure(h_ctx, model=model_14b, target_ctx=ctx_val)
        ctx_units.append(p_ctx.gpu_units)
    for a, b in zip(ctx_units[:-1], ctx_units[1:]):
        assert a >= b, f"Context monotonicity violated: {a} < {b} with larger ctx"


# -----------------------------------------------------------------------------
# Test 4: VRAM Math Calibration
# -----------------------------------------------------------------------------

def test_vram_math_calibration():
    """Calibration check: 14B Q4_K_M at ctx 8192:
    - 49 units on GPU estimates within 10% of 9.6 GB (range [8.64, 10.56])
    - 24 units on GPU estimates within 10% of 5.0 GB (range [4.50, 5.50])
    """
    cat = {m.tag: m for m in catalogue()}
    m14b = cat["qwen2.5-coder:14b"]

    # Calculate estimates using the mathematical formula:
    # (file_gb / U) * N + (KV_total * min(1.0, N / blocks)) + OVERHEAD
    u = m14b.blocks + 1  # 49
    kv_total_gb = (2 * m14b.blocks * m14b.head_count_kv * m14b.head_dim * 2 * 8192) / (1024**3)

    # 1. All units (49)
    est_all = (m14b.file_gb / u) * 49 + kv_total_gb * (49 / m14b.blocks) + hardware.CALIBRATED_VRAM_OVERHEAD_GB
    diff_all = abs(est_all - 9.6) / 9.6
    assert diff_all <= 0.10, f"49 units estimate {est_all:.2f}GB differs from 9.6GB by {diff_all*100:.1f}% (> 10%)"

    # 2. 24 units
    est_24 = (m14b.file_gb / u) * 24 + kv_total_gb * (24 / m14b.blocks) + hardware.CALIBRATED_VRAM_OVERHEAD_GB
    diff_24 = abs(est_24 - 5.0) / 5.0
    assert diff_24 <= 0.10, f"24 units estimate {est_24:.2f}GB differs from 5.0GB by {diff_24*100:.1f}% (> 10%)"


# -----------------------------------------------------------------------------
# Test 5: Overrides & Precedence
# -----------------------------------------------------------------------------

def test_overrides_and_precedence():
    """Verify CLI / user overrides take precedence and unsafe overrides warn."""
    hw = _create_mock_hw(vram_total=12.0, vram_free=8.0, ram_total=32.0, ram_avail=20.0)

    # 1. force_cpu override
    prof_cpu = auto_configure(hw, overrides={"force_cpu": True})
    assert prof_cpu.mode == "cpu"
    assert prof_cpu.num_gpu == 0
    assert prof_cpu.source == "override"

    # 2. explicit gpu_layers override
    prof_layers = auto_configure(hw, overrides={"gpu_layers": 16})
    assert prof_layers.gpu_units == 16
    assert prof_layers.num_gpu == 16
    assert prof_layers.source == "override"

    # 3. all / 99 layers override
    prof_all = auto_configure(hw, overrides={"gpu_layers": "all"})
    assert prof_all.gpu_units == 49
    assert prof_all.num_gpu == 99

    # 4. Context and thread overrides
    prof_custom = auto_configure(hw, overrides={"ctx": 4096, "threads": 4})
    assert prof_custom.num_ctx == 4096
    assert prof_custom.num_thread == 4

    # 5. Unsafe override produces warning
    prof_unsafe = auto_configure(hw, overrides={"gpu_layers": 49})
    assert any("Unsafe override" in w for w in prof_unsafe.warnings)


# -----------------------------------------------------------------------------
# Test 6: Persistence & Stable Fingerprint
# -----------------------------------------------------------------------------

def test_persistence_and_fingerprint():
    """Verify atomic write, UTF-8 BOM tolerance, corrupt file handling, and stable fingerprint."""
    hw1 = _create_mock_hw(vram_total=12.0, vram_free=10.0, ram_total=32.0, ram_avail=24.0)
    hw2_same = _create_mock_hw(vram_total=12.0, vram_free=4.0, ram_total=32.0, ram_avail=12.0)
    hw3_diff = _create_mock_hw(vram_total=16.0, vram_free=10.0, ram_total=32.0, ram_avail=24.0)

    # Fingerprint invariant: volatile free memory does NOT alter fingerprint
    fp1 = fingerprint(hw1)
    fp2 = fingerprint(hw2_same)
    fp3 = fingerprint(hw3_diff)
    assert fp1 == fp2, "Fingerprint should not change when only free memory changes"
    assert fp1 != fp3, "Fingerprint must change when total VRAM changes"

    # Save & load round trip
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "hardware_profile.json"
        prof = auto_configure(hw1)
        ok = save_profile(path, hw1, prof, overrides={"test": 123})
        assert ok is True
        assert path.exists()

        # Atomic write leaves no temporary files
        tmp_files = list(Path(tmpdir).glob("*.tmp.*"))
        assert len(tmp_files) == 0

        # Load round trip
        loaded = load_profile(path)
        assert loaded is not None
        assert loaded["fingerprint"] == fp1
        assert loaded["overrides"] == {"test": 123}

        # UTF-8 BOM tolerance
        bom_path = Path(tmpdir) / "bom_profile.json"
        with open(bom_path, "wb") as f:
            f.write(b"\xef\xbb\xbf" + json.dumps(loaded).encode("utf-8"))
        loaded_bom = load_profile(bom_path)
        assert loaded_bom is not None
        assert loaded_bom["fingerprint"] == fp1

        # Corrupt file backed up and ignored
        corrupt_path = Path(tmpdir) / "corrupt_profile.json"
        with open(corrupt_path, "w", encoding="utf-8") as f:
            f.write("{this is not valid json")
        res_corrupt = load_profile(corrupt_path)
        assert res_corrupt is None
        backups = list(Path(tmpdir).glob("corrupt_profile.json.corrupt-*"))
        assert len(backups) == 1


# -----------------------------------------------------------------------------
# Test 7: Banner ASCII & Safety
# -----------------------------------------------------------------------------

def test_format_banner_ascii_and_safety():
    """Verify banner is pure ASCII, handles None GPU safely, and formats expected sections."""
    hw_gpu = _create_mock_hw(vram_total=12.0, vram_free=10.0, ram_total=32.0, ram_avail=24.0)
    prof_gpu = auto_configure(hw_gpu)
    banner_gpu = format_banner(hw_gpu, prof_gpu)

    assert banner_gpu.isascii(), "Banner must be pure ASCII (no unicode box-drawing or emoji)"
    assert "OGUN HARDWARE PROFILE & RUNTIME" in banner_gpu
    assert "Tier:" in banner_gpu
    assert "Model:" in banner_gpu
    assert "Recommended:" in banner_gpu
    assert "Options:" in banner_gpu
    assert "Memory Est:" in banner_gpu

    # CPU-only safety (None GPU)
    hw_cpu = _create_mock_hw(gpu_vendor="none", ram_total=16.0, ram_avail=10.0)
    prof_cpu = auto_configure(hw_cpu)
    banner_cpu = format_banner(hw_cpu, prof_cpu)
    assert banner_cpu.isascii()
    assert "GPU:        None (CPU-only execution)" in banner_cpu


# -----------------------------------------------------------------------------
# Test 8: Integration & Static Guards
# -----------------------------------------------------------------------------

def test_integration_and_static_guards():
    """Live config follows the profile; --hardware-info works; no static copies of tunables."""
    from ogun_core import config as cfg
    from ogun_core.engine import ContextManager, handle_vram_command

    saved = dict(cfg._ACTIVE)
    try:
        assert cfg.apply_hardware_profile({"num_ctx": 4096, "num_gpu": 24, "num_thread": 4, "model_tag": cfg.MODEL_NAME})
        assert cfg.get_vram_profile() == "AUTO"
        opts = cfg.llm_options()
        assert opts["num_ctx"] == 4096 and opts["num_gpu"] == 24 and opts["num_thread"] == 4
        assert ContextManager().max_tokens == int(4096 * 0.85)      # context budget follows the live window
        assert not cfg.apply_hardware_profile("garbage")

        # a CPU-only profile sends num_gpu=0 (not omitted), and no thread option when unknown
        cfg.apply_hardware_profile({"num_ctx": 8192, "num_gpu": 0, "num_thread": None})
        o2 = cfg.llm_options()
        assert o2["num_gpu"] == 0 and "num_thread" not in o2
    finally:
        cfg._ACTIVE.clear()
        cfg._ACTIVE.update(saved)

    # /vram status reports the auto-tuned details
    try:
        cfg.apply_hardware_profile({"num_ctx": 6144, "num_gpu": 30, "gpu_units": 30, "total_units": 49, "mode": "hybrid",
                                    "model_tag": cfg.MODEL_NAME, "num_thread": 8})
        with patch("ogun_core.engine.ollama_ps", return_value=[]):
            reply = handle_vram_command("/vram status")
        assert "AUTO" in reply and "30/49" in reply and "6144" in reply, reply
    finally:
        cfg._ACTIVE.clear()
        cfg._ACTIVE.update(saved)

    # --hardware-info exits cleanly and never loads a model
    proc = subprocess.run([sys.executable, str(BASE / "main.py"), "--hardware-info"], cwd=str(BASE),
                          capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace")
    assert proc.returncode == 0, proc.stderr
    assert "OGUN HARDWARE PROFILE & RUNTIME" in proc.stdout
    assert "Goodbye" not in proc.stdout

    # static guard: nobody may copy NUM_CTX at import time (it would ignore the auto-tuned window)
    pat = re.compile(r"=\s*NUM_CTX|max_tokens\s*:\s*int\s*=\s*NUM_CTX|\"num_ctx\"\s*:\s*NUM_CTX")
    bad = []
    for path in (BASE / "ogun_core").glob("*.py"):
        if path.name == "config.py":
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if pat.search(line):
                bad.append(f"{path.name}:{i}: {line.strip()}")
    assert not bad, bad


def test_config_override_flags_and_boot_hook():
    """autotune.cli_overrides parses flags; boot() skips on --vram / env, applies a profile otherwise, never raises."""
    import argparse
    from ogun_core import autotune
    from ogun_core import config as cfg

    p = argparse.ArgumentParser()
    autotune.add_arguments(p)
    a = p.parse_args(["--gpu-layers", "all", "--ctx", "4096", "--threads", "6", "--force-cpu"])
    assert autotune.cli_overrides(a) == {"force_cpu": True, "gpu_layers": 99, "ctx": 4096, "threads": 6}
    assert autotune.cli_overrides(p.parse_args(["--gpu-layers", "junk"])) == {}
    assert autotune.cli_overrides(p.parse_args(["--gpu-layers", "12"])) == {"gpu_layers": 12}

    saved = dict(cfg._ACTIVE)
    try:
        a = p.parse_args([])
        a.vram = "max"
        assert autotune.boot(a, cfg.MODEL_NAME, cfg.OLLAMA_HOST) is None          # explicit split wins
        b = p.parse_args(["--no-hardware-probe"])
        b.vram = None
        assert autotune.boot(b, cfg.MODEL_NAME, cfg.OLLAMA_HOST) is None
        # probe failure must degrade to defaults, not raise
        c = p.parse_args([])
        c.vram = None
        with patch.dict(os.environ, {"OGUN_VRAM_PROFILE": "", "OGUN_NUM_GPU": ""}),              patch("ogun_core.autotune.hardware.probe_hardware", side_effect=RuntimeError("boom")):
            assert autotune.boot(c, cfg.MODEL_NAME, cfg.OLLAMA_HOST) is None
        # successful boot with a fake machine applies AUTO + saves the profile in a temp location
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"OGUN_VRAM_PROFILE": "", "OGUN_NUM_GPU": ""}),              patch("ogun_core.autotune.PROFILE_PATH", Path(td) / "hardware_profile.json"),              patch("ogun_core.autotune.hardware.probe_hardware", return_value=_nvidia_pc(total=24.0, free=22.0)),              patch("ogun_core.autotune.hardware.query_ollama", return_value={"reachable": False, "installed": [], "resident": [],
                                                                               "resident_vram_gb": 0.0, "resident_ram_gb": 0.0}):
            applied = autotune.boot(c, cfg.MODEL_NAME, cfg.OLLAMA_HOST)
            assert applied and cfg.get_vram_profile() == "AUTO" and cfg.get_num_ctx() == applied["num_ctx"]
            assert (Path(td) / "hardware_profile.json").exists()
    finally:
        cfg._ACTIVE.clear()
        cfg._ACTIVE.update(saved)


def _nvidia_pc(total=12.0, free=6.6, ram_total=64.0, ram_avail=40.0):
    g = hardware.GpuInfo(vendor="nvidia", name="Test RTX", vram_total_gb=total, vram_free_gb=free, unified_memory=False,
                         driver="1", compute_capability="12.0", index=0)
    return hardware.HardwareInfo(os="windows", arch="amd64", cpu_name="Test CPU", cpu_physical=12, cpu_logical=24,
                                 isa={"avx": True, "avx2": True, "avx512": False, "neon": False}, ram_total_gb=ram_total,
                                 ram_available_gb=ram_avail, gpus=[g], gpu=g, probe_seconds=0.1, notes=[])


def test_resident_ollama_model_does_not_shrink_the_machine():
    """Regression: a re-probe taken while our own model is loaded measured 1.9 GB free and chose CPU-only."""
    model = {m.tag: m for m in hardware.catalogue()}["qwen2.5-coder:14b"]
    seen_while_loaded = _nvidia_pc(free=1.9)                       # driver view: other apps + our own 4.5 GB model
    naive = hardware.auto_configure(seen_while_loaded, model=model)
    assert naive.gpu_units == 0 and naive.mode == "cpu"            # what the bug produced
    fixed_hw = hardware.adjust_for_resident_models(seen_while_loaded, resident_vram_gb=4.47, resident_ram_gb=5.6)
    assert fixed_hw.gpu.vram_free_gb == 6.37 and fixed_hw.ram_available_gb == 45.6
    assert any("held by models Ollama" in n for n in fixed_hw.notes)
    fixed = hardware.auto_configure(fixed_hw, model=model)
    assert fixed.gpu_units > 10 and fixed.mode == "hybrid", fixed
    # never beyond the card / the machine
    capped = hardware.adjust_for_resident_models(_nvidia_pc(total=12.0, free=11.0, ram_avail=63.0), 5.0, 9.0)
    assert capped.gpu.vram_free_gb == 12.0 and capped.ram_available_gb == 64.0
    # no-ops: nothing resident, tiny amounts, CPU-only box, unified memory, garbage input
    same = _nvidia_pc()
    assert hardware.adjust_for_resident_models(same, 0.0) is same and hardware.adjust_for_resident_models(same, 0.01) is same
    cpu_only = hardware.HardwareInfo(os="linux", arch="x86_64", cpu_name="c", cpu_physical=4, cpu_logical=8, isa={}, ram_total_gb=16.0,
                                     ram_available_gb=8.0, gpus=[], gpu=None, probe_seconds=0.0, notes=[])
    assert hardware.adjust_for_resident_models(cpu_only, 5.0).gpu is None
    assert hardware.adjust_for_resident_models(same, "x", None) is same or hardware.adjust_for_resident_models(same, "x", None) == same


def test_query_ollama_reads_installed_specs_and_resident_memory():
    import io
    import urllib.request

    GB = 1024 ** 3
    calls = []

    class Resp:
        def __init__(self, data):
            self.data = data

        def read(self):
            return json.dumps(self.data).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    show = {
        "qwen2.5-coder:14b": {"model_info": {"general.architecture": "qwen2", "qwen2.block_count": 48, "qwen2.attention.head_count": 40,
                                             "qwen2.attention.head_count_kv": 8, "qwen2.embedding_length": 5120, "qwen2.context_length": 32768},
                              "details": {"quantization_level": "Q4_K_M"}},
        "qwen2.5-coder:7b": {"model_info": {"qwen2.block_count": 28, "qwen2.attention.head_count": 28, "qwen2.attention.head_count_kv": 4,
                                            "qwen2.embedding_length": 3584, "qwen2.context_length": 32768}, "details": {}},
        "llava:latest": {"model_info": {"llama.block_count": 32, "llama.attention.head_count_kv": 8, "llama.embedding_length": 4096,
                                        "llama.context_length": 4096}, "details": {}},
    }

    def fake_urlopen(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        calls.append(url)
        if url.endswith("/api/tags"):
            return Resp({"models": [{"name": "qwen2.5-coder:14b", "size": int(9.0 * GB)}, {"name": "qwen2.5-coder:7b", "size": int(4.7 * GB)},
                                    {"name": "llava:latest", "size": int(4.7 * GB)}]})
        if url.endswith("/api/show"):
            return Resp(show[json.loads(req.data.decode("utf-8"))["model"]])
        if url.endswith("/api/ps"):
            return Resp({"models": [{"name": "qwen2.5-coder:14b", "size": int(10.0 * GB), "size_vram": int(4.5 * GB)}]})
        raise OSError("unexpected " + url)

    orig = urllib.request.urlopen
    urllib.request.urlopen = fake_urlopen
    try:
        out = hardware.query_ollama("http://127.0.0.1:11434", family="qwen2.5-coder")
        assert out["reachable"] and [m.tag for m in out["installed"]] == ["qwen2.5-coder:14b", "qwen2.5-coder:7b"]   # family filter: no llava
        spec14 = out["installed"][0]
        assert (spec14.blocks, spec14.head_count_kv, spec14.hidden, spec14.head_dim, spec14.file_gb) == (48, 8, 5120, 128, 9.0)
        spec7 = out["installed"][1]
        assert (spec7.blocks, spec7.head_count_kv, spec7.head_dim) == (28, 4, 128), spec7          # 3584 / 28 heads
        assert out["resident"] == ["qwen2.5-coder:14b"] and out["resident_vram_gb"] == 4.5 and out["resident_ram_gb"] == 5.5
        # no family filter: llava is read with ITS OWN architecture, not Qwen's
        llava = [m for m in hardware.query_ollama(family=None)["installed"] if m.tag == "llava:latest"][0]
        assert (llava.blocks, llava.native_ctx, llava.hidden) == (32, 4096, 4096), llava
    finally:
        urllib.request.urlopen = orig

    # unreachable / garbage: never raises, reports not reachable
    def dead(req, timeout=None):
        raise OSError("connection refused")

    urllib.request.urlopen = dead
    try:
        out = hardware.query_ollama()
        assert out == {"reachable": False, "installed": [], "resident": [], "resident_vram_gb": 0.0, "resident_ram_gb": 0.0}
        assert hardware.query_ollama(host="")["reachable"] is False
    finally:
        urllib.request.urlopen = orig


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                fails += 1
                import traceback
                print(f"FAIL {name}: {type(e).__name__}: {e}")
                traceback.print_exc()
    sys.exit(1 if fails else 0)
