"""image_freshness derives the repo root via project_root() (metadata contract section 11.P P9).

The prior ``Path(__file__).resolve().parents[3]`` depth count silently breaks
on module moves; this pins the marker-based derivation.

Run: uv run pytest tests/gym/utils/backend/test_image_freshness_root.py -v
"""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest

from lite.gym.utils.backend.freshness import (
    _REPO_ROOT,
    _excluded,
    _ignored,
    _normalize_dockerfile_bytes,
    tree_content_hash,
)
from lite.utils.path import project_root


def test_backend_freshness_owner_module() -> None:
    repo = Path(__file__).resolve().parents[4]

    spec = importlib.util.find_spec("lite.gym.utils.backend.freshness")
    assert spec is not None
    assert Path(spec.origin).resolve() == (
        repo / "lite" / "gym" / "utils" / "backend" / "freshness.py"
    )


def test_repo_root_is_project_root():
    assert _REPO_ROOT == project_root()
    assert (_REPO_ROOT / "pyproject.toml").is_file()
    assert (_REPO_ROOT / "lite").is_dir()


def test_dockerfile_hash_ignores_comments_and_blanks():
    """A comment/blank-only Dockerfile edit yields a byte-identical image, so it
    must NOT change the freshness hash (else host-side comment cleanups like C8a
    spuriously force a rebuild). Guards the image_freshness class fix."""
    base = b"# syntax=docker/dockerfile:1\nFROM x\n# a doc comment\nRUN echo hi\n"
    comment_and_blank_edit = b"# syntax=docker/dockerfile:1\nFROM x\nRUN echo hi\n\n\n"
    assert _normalize_dockerfile_bytes(base) == _normalize_dockerfile_bytes(comment_and_blank_edit)


def test_dockerfile_hash_still_trips_on_real_changes():
    """Negative-direction: anything that actually affects the built image must
    still change the hash — instruction edits, parser directives, and
    RUN-continuation/heredoc shell content (which is part of the built command)."""
    base = b"# syntax=docker/dockerfile:1\nFROM x\nRUN echo hi\n"
    # instruction body change
    assert _normalize_dockerfile_bytes(base) != _normalize_dockerfile_bytes(
        b"# syntax=docker/dockerfile:1\nFROM x\nRUN echo BYE\n"
    )
    # parser directive change (affects the build)
    assert _normalize_dockerfile_bytes(base) != _normalize_dockerfile_bytes(
        b"# escape=`\nFROM x\nRUN echo hi\n"
    )
    # a '#' inside a RUN line-continuation is a shell comment in the built command
    with_run_comment = b"RUN a \\\n  # part of the shell command\n  b\n"
    without = b"RUN a \\\n  b\n"
    assert _normalize_dockerfile_bytes(with_run_comment) != _normalize_dockerfile_bytes(without)
    # BuildKit heredoc contents are command/script/file contents, not Dockerfile
    # comments. Dropping them would under-hash real image changes.
    heredoc = b"RUN <<'EOF'\n#!/bin/bash\n# script comment\n\necho hi\nEOF\n"
    heredoc_changed = b"RUN <<'EOF'\n#!/usr/bin/env bash\n# script comment\n\necho hi\nEOF\n"
    assert _normalize_dockerfile_bytes(heredoc) != _normalize_dockerfile_bytes(
        heredoc_changed
    )


def test_dockerfile_hash_preserves_comments_and_blanks_inside_heredoc():
    base = b"RUN <<EOF\ncat <<'PY' > /tmp/app.py\n# generated file comment\nprint('hi')\nPY\nEOF\n"
    comment_edit = b"RUN <<EOF\ncat <<'PY' > /tmp/app.py\nprint('hi')\nPY\nEOF\n"
    blank_edit = (
        b"RUN <<EOF\ncat <<'PY' > /tmp/app.py\n"
        b"# generated file comment\n\nprint('hi')\nPY\nEOF\n"
    )
    assert _normalize_dockerfile_bytes(base) != _normalize_dockerfile_bytes(
        comment_edit
    )
    assert _normalize_dockerfile_bytes(base) != _normalize_dockerfile_bytes(blank_edit)
    numeric_dash_delim = b"RUN <<'EOF-1'\n# kept\nvalue\nEOF-1\n"
    numeric_dash_changed = b"RUN <<'EOF-1'\nvalue\nEOF-1\n"
    assert _normalize_dockerfile_bytes(numeric_dash_delim) != (
        _normalize_dockerfile_bytes(numeric_dash_changed)
    )


def test_gitignore_not_hashed():
    """.gitignore is a git artifact never COPY'd into any image."""
    assert _ignored("lite/gym/envs/lite/osworld/docker/.gitignore")
    assert not _ignored("lite/gym/envs/lite/osworld/docker/.dockerignore")


def test_exclude_supports_basename_and_path_suffixes():
    assert _excluded(
        "lite/gym/envs/osworld/docker/server.py",
        "server.py",
        ("server.py",),
    )
    assert _excluded(
        "lite/gym/envs/osworld/docker/server.py",
        "server.py",
        ("osworld/docker/server.py",),
    )
    assert not _excluded(
        "lite/gym/envs/androidworld/docker/server.py",
        "server.py",
        ("osworld/docker/server.py",),
    )


def test_tree_content_hash_tracks_external_tree_content(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='sample'\n")
    (root / "package.py").write_text("VALUE = 1\n")
    ignored = root / "__pycache__"
    ignored.mkdir()
    (ignored / "package.cpython-312.pyc").write_bytes(b"one")

    def _skip(base, path):
        return "__pycache__" in path.relative_to(base).parts

    first = tree_content_hash(root, skip=_skip)
    (ignored / "package.cpython-312.pyc").write_bytes(b"two")
    assert tree_content_hash(root, skip=_skip) == first
    (root / "package.py").write_text("VALUE = 2\n")
    assert tree_content_hash(root, skip=_skip) != first


def test_tree_content_hash_can_mark_absent_roots(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    assert tree_content_hash(root, roots=("missing",), missing="marker")


def test_tree_content_hash_errors_on_absent_roots_by_default(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        tree_content_hash(root, roots=("missing",))
