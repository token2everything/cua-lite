"""Git state of THIS checkout — anchored at ``project_root()``, not the CWD.

The shared home for checkout-anchored git queries. Rollout provenance uses
``git_commit()`` and can carry a ``-dirty`` suffix; HF upload's default tag uses
``run_git("rev-parse --short HEAD")`` because a Hub tag must be a clean ref
name. Raw preproc rows do not automatically carry ``others.commit``, so freeze
the producing tree when row provenance and published revision must line up.

Usage:
    from lite.utils.git import git_commit, run_git
"""

from __future__ import annotations

import subprocess

from lite.utils.path import project_root


def run_git(cmd: str) -> str:
    """Run a git command at the repo root and return stripped stdout.

    Anchored via :func:`project_root` — a process launched from outside the
    checkout must still report THIS repo's state (the output lands in
    trajectory rows and HF revision tags, not just logs). Returns ``""``
    when git or the repo is unavailable.
    """
    try:
        return subprocess.check_output(
            ["git"] + cmd.split(), cwd=project_root(),
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, RuntimeError):
        # RuntimeError: project_root() outside a source checkout (e.g. an
        # installed wheel) — same "unavailable" outcome as missing git.
        return ""


def git_commit() -> str:
    """Short HEAD hash, ``-dirty``-suffixed when the tree has uncommitted
    changes (e.g. ``"614d8c57-dirty"``); ``""`` outside a repo."""
    commit = run_git("rev-parse --short HEAD")
    if commit and run_git("status --porcelain --short"):
        commit += "-dirty"
    return commit
