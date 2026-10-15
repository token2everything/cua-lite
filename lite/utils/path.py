"""
CUA-Lite Utility Functions

This module provides common utility functions for path resolution and environment handling.

Usage:
    from lite.utils.path import resolve_path

    path = resolve_path("data/file.json", "CUA_DATA_DIR")

Run:
    python -m lite.utils.path
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path


@cache
def project_root() -> Path:
    """The repo root: the directory holding this source tree's ``pyproject.toml``
    and the ``lite/`` package.

    Found by walking UP from this file to the first ancestor with both markers —
    NOT by counting ``__file__.parents[N]``, which silently breaks the moment a
    module is moved to a different depth (a nested ``pyproject.toml`` — e.g. an
    env's — is excluded by requiring a co-located ``lite/`` package marker, so only
    the real repo root matches). Use this
    anywhere you need a worktree-relative anchor (e.g. project-root-relative log
    paths, locating ``scripts/``). Cached.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "lite").is_dir():
            return parent
    raise RuntimeError(
        "project root not found: no ancestor of "
        f"{__file__} has both pyproject.toml and a lite/ package "
        "(this anchor assumes an editable/source checkout)."
    )


def resolve_path(
    path: str,
    env_var: str = "CUA_LITE_DATASETS_ROOT",
) -> str:
    """
    Resolve a path using an environment variable as the base directory.

    Use path as-is if it exists or is absolute; else prepend the directory from
    env_var (must be set). Raise if missing or the combined path does not exist.
    Returns a resolved absolute path string.

    Args:
        path: The path to resolve
        env_var: Environment variable name containing the base directory

    Returns:
        Resolved absolute path as str

    Raises:
        FileNotFoundError: If path does not exist as-is and env_var is not set, or if the resolved path does not exist

    Example:
        # With CUA_DATA_DIR=/data
        resolve_path("images/test.png", "CUA_DATA_DIR")
        # Returns "/data/images/test.png" if it exists
    """
    if os.path.exists(path):
        return str(Path(path).resolve())
    if os.path.isabs(path):
        return str(Path(path).resolve())

    root_val = os.environ.get(env_var, "").strip()
    if not root_val:
        raise FileNotFoundError(
            f"Path {path!r} does not exist as-is. For relative paths, {env_var} must be set."
        )
    candidate = Path(root_val) / path
    if not candidate.exists():
        raise FileNotFoundError(
            f"Path does not exist: {path!r} as-is, nor under {env_var}: {candidate}"
        )
    return str(candidate.resolve())

if __name__ == "__main__":
    print("CUA-Lite Utility Functions")
    print("=" * 40)
    print("\nFunctions:")
    print("  - resolve_path(path, env_var)")
