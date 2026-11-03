"""Ogun path security guard preventing AI operations from modifying engine or system files.

Strict path validation:
Any write tool invoked by the AI must resolve exclusively to:
  - plugins/custom_tools/
  - plugins/tests/
  - workspace/

All other destinations (including ogun_core/, training/, main.py, .bat scripts,
and plugins/registry.json) are strictly immutable.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

from ogun_core.paths import (
    CUSTOM_TOOLS_DIR,
    OGUN_CORE_DIR,
    OGUN_ROOT,
    PLUGINS_TESTS_DIR,
    REGISTRY_PATH,
    TRAINING_DIR,
    WORKSPACE_DIR,
)


class PathSecurityGuard:
    """Enforces write-path boundaries for AI operations."""

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        allowed_dirs: Optional[List[Path]] = None,
    ) -> None:
        self.base_dir = (base_dir or OGUN_ROOT).resolve()
        self.core_dir = (self.base_dir / "ogun_core").resolve()
        self.training_dir = (self.base_dir / "training").resolve()
        self.registry_path = (self.base_dir / "plugins" / "registry.json").resolve()
        self.main_file = (self.base_dir / "main.py").resolve()

        if allowed_dirs is None:
            self.allowed_dirs = [
                (self.base_dir / "plugins" / "custom_tools").resolve(),
                (self.base_dir / "plugins" / "tests").resolve(),
                (self.base_dir / "workspace").resolve(),
            ]
        else:
            self.allowed_dirs = [d.resolve() for d in allowed_dirs]

    def _resolve_target(self, filepath: Union[str, Path]) -> Path:
        raw = str(filepath).strip().strip('"').strip("'")
        if not raw:
            raise PermissionError("PermissionError: Empty file path is not allowed.")

        # Reject UNC paths
        if raw.startswith(("\\\\", "//")):
            raise PermissionError(f"PermissionError: UNC paths are forbidden: '{raw}'")

        # Reject drive-relative paths (e.g. C:file)
        if len(raw) >= 2 and raw[1] == ":" and (len(raw) == 2 or raw[2] not in ("\\", "/")):
            raise PermissionError(f"PermissionError: Drive-relative paths are forbidden: '{raw}'")

        p = Path(os.path.expandvars(os.path.expanduser(raw)))
        if not p.is_absolute():
            p = (self.base_dir / p).resolve()
        else:
            p = p.resolve()
        return p

    def is_inside(self, target: Path, parent: Path) -> bool:
        """Check if target path is strictly inside parent directory (case-normalized)."""
        try:
            norm_target = Path(os.path.normcase(str(target.resolve())))
            norm_parent = Path(os.path.normcase(str(parent.resolve())))
            norm_target.relative_to(norm_parent)
            return norm_target != norm_parent
        except ValueError:
            return False

    def is_in_core(self, target: Path) -> bool:
        norm_target = os.path.normcase(str(target.resolve()))
        norm_core = os.path.normcase(str(self.core_dir.resolve()))
        return norm_target == norm_core or self.is_inside(target, self.core_dir)

    def is_in_training(self, target: Path) -> bool:
        norm_target = os.path.normcase(str(target.resolve()))
        norm_train = os.path.normcase(str(self.training_dir.resolve()))
        return norm_target == norm_train or self.is_inside(target, self.training_dir)

    def is_allowed(self, filepath: Union[str, Path]) -> bool:
        try:
            self.validate_write(filepath)
            return True
        except PermissionError:
            return False

    def validate_write(self, filepath: Union[str, Path]) -> Path:
        """Validate write target.

        Raises PermissionError if:
          - The path targets ogun_core/ or any file within it
          - The path targets training/ or any file within it
          - The path targets main.py, registry.json, or any .bat file
          - The path is outside plugins/custom_tools/, plugins/tests/, or workspace/

        Returns the resolved Path if valid.
        """
        target = self._resolve_target(filepath)
        norm_target = os.path.normcase(str(target))

        # Check core directory
        if self.is_in_core(target):
            raise PermissionError(
                f"PermissionError: Modification of core engine files in '{target}' is strictly forbidden."
            )

        # Check training directory
        if self.is_in_training(target):
            raise PermissionError(
                f"PermissionError: Modification of training files in '{target}' is strictly forbidden for AI tools."
            )

        # Check registry.json
        if norm_target == os.path.normcase(str(self.registry_path)):
            raise PermissionError(
                f"PermissionError: Modification of '{self.registry_path}' via file tools is forbidden."
            )

        # Check main.py
        if norm_target == os.path.normcase(str(self.main_file)):
            raise PermissionError(
                f"PermissionError: Modification of entrypoint '{self.main_file}' is forbidden."
            )

        # Check .bat files
        if target.suffix.lower() == ".bat":
            raise PermissionError(
                f"PermissionError: Modification of Windows batch script '{target}' is forbidden."
            )

        # Check allowed directories
        for allowed in self.allowed_dirs:
            norm_allowed = os.path.normcase(str(allowed.resolve()))
            if norm_target == norm_allowed or self.is_inside(target, allowed):
                return target

        raise PermissionError(
            f"PermissionError: File write target '{target}' is restricted. "
            "AI writes are strictly restricted to: ['plugins/custom_tools/', 'plugins/tests/', 'workspace/']."
        )


_default_guard = PathSecurityGuard()


def validate_write_path(filepath: Union[str, Path]) -> Path:
    return _default_guard.validate_write(filepath)


def is_safe_write_path(filepath: Union[str, Path]) -> bool:
    return _default_guard.is_allowed(filepath)
