"""Ogun unified path resolvers and directory configuration.

All paths are strictly relative to the Ogun installation root.
No paths are created at import time.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

# Detect if running as a frozen executable (e.g. PyInstaller)
IS_FROZEN = getattr(sys, "frozen", False)

if IS_FROZEN:
    OGUN_ROOT = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Ogun"
else:
    OGUN_ROOT = Path(__file__).resolve().parents[1]

# Core engine paths
OGUN_CORE_DIR = (OGUN_ROOT / "ogun_core").resolve()

# Dynamic plugin paths
PLUGINS_DIR = (OGUN_ROOT / "plugins").resolve()
CUSTOM_TOOLS_DIR = (PLUGINS_DIR / "custom_tools").resolve()
PLUGINS_TESTS_DIR = (PLUGINS_DIR / "tests").resolve()
QUARANTINE_DIR = (PLUGINS_DIR / "quarantine").resolve()
REGISTRY_PATH = (PLUGINS_DIR / "registry.json").resolve()

# User workspace
WORKSPACE_DIR = (OGUN_ROOT / "workspace").resolve()

# Training & dataset paths
TRAINING_DIR = (OGUN_ROOT / "training").resolve()
TRAINING_DATA_DIR = (TRAINING_DIR / "data").resolve()
TRAINING_RAW_DIR = (TRAINING_DATA_DIR / "raw").resolve()
TRAINING_TOKENIZED_DIR = (TRAINING_DATA_DIR / "tokenized").resolve()
TRAINING_EXPORTS_DIR = (TRAINING_DATA_DIR / "exports").resolve()
MERGED_DATASET_PATH = (TRAINING_DATA_DIR / "merged_dpo_dataset.jsonl").resolve()


def init_user_workspace() -> None:
    """Ensure dynamic directories exist when the system boots (never at import time)."""
    CUSTOM_TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    PLUGINS_TESTS_DIR.mkdir(parents=True, exist_ok=True)
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
