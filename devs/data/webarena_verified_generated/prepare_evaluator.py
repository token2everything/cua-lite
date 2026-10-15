#!/usr/bin/env python3
"""Build an isolated, source-pinned v0.2 evaluator without copying official tasks.

Run with ``--output-root <new-directory>``, then put that directory first in
PYTHONPATH when invoking generated-task replay. The installed dependency stays
unchanged; this build requires explicit generated-task configuration.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from importlib.metadata import distribution
from pathlib import Path
from typing import Any

from contract import sha256_file, sha256_json

BASE_VERSION = "1.2.3"
MODULE_PATH = Path("core/evaluation/evaluators/agent_response_evaluator.py")
SOURCE_MODULE_SHA256 = "8ae2caf59c6fafecf4ec259ea67bf79d27f19c7fcbdc33a312cea730c4e54c31"
PATCH_PATH = Path(__file__).parent / "patches" / "agent_response_null.patch"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def prepare_evaluator(output_root: Path) -> dict[str, Any]:
    """Copy package code, apply the owning-module patch, and pin the build receipt."""
    dependency = distribution("webarena-verified")
    if dependency.version != BASE_VERSION:
        raise ValueError(f"evaluator build requires {BASE_VERSION}, found {dependency.version}")
    source_root = Path(dependency.locate_file("webarena_verified")).resolve()
    source_module = source_root / MODULE_PATH
    source_hash = sha256_file(source_module)
    if source_hash != SOURCE_MODULE_SHA256:
        raise ValueError("pinned AgentResponseEvaluator source hash mismatch")

    output_root = output_root.resolve()
    if output_root.is_relative_to(source_root) or ".venv" in output_root.parts:
        raise ValueError("evaluator output must be outside the installed dependency and .venv")
    output_root.mkdir(parents=True, exist_ok=False)
    target_root = output_root / "webarena_verified"
    source_files = {}
    for source_path in sorted(source_root.rglob("*.py")):
        relative = source_path.relative_to(source_root)
        if "assets" in relative.parts or "__pycache__" in relative.parts:
            continue
        target_path = target_root / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        source_files[str(relative)] = sha256_file(source_path)

    patch_result = subprocess.run(
        [
            "patch",
            "--batch",
            "--forward",
            "--fuzz=0",
            "-p1",
            "--directory",
            str(output_root),
            "--input",
            str(PATCH_PATH.resolve()),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    patched_files = {name: sha256_file(target_root / name) for name in source_files}
    changed_files = [name for name in source_files if source_files[name] != patched_files[name]]
    if changed_files != [str(MODULE_PATH)]:
        raise ValueError(f"evaluator patch changed unexpected files: {changed_files}")

    manifest = {
        "build_id": "wavgen-v02-agent-response-null-v1",
        "base_version": BASE_VERSION,
        "source_package_root": str(source_root),
        "pythonpath": str(output_root),
        "source_module": str(MODULE_PATH),
        "source_module_sha256": source_hash,
        "patched_module_sha256": patched_files[str(MODULE_PATH)],
        "patch_sha256": sha256_file(PATCH_PATH),
        "source_code_sha256": sha256_json(source_files),
        "patched_code_sha256": sha256_json(patched_files),
        "source_files_sha256": source_files,
        "patched_files_sha256": patched_files,
        "copied_files": len(source_files),
        "excluded": ["assets", "__pycache__", "non-Python files"],
        "changed_files": changed_files,
        "patch_output": patch_result.stdout.strip(),
    }
    (output_root / "build_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    args = _parse_args()
    manifest = prepare_evaluator(args.output_root)
    print(
        json.dumps(
            {
                "build_id": manifest["build_id"],
                "pythonpath": manifest["pythonpath"],
                "manifest": str(args.output_root / "build_manifest.json"),
                "patched_module_sha256": manifest["patched_module_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
