"""Copy one source run's execution and review evidence into a shared delivery."""

from __future__ import annotations

import shutil
from pathlib import Path


def copy_run_evidence(source: Path, destination: Path) -> None:
    """Preserve run evidence bytes and paths without copying private credentials."""
    destination.mkdir()
    shutil.copytree(source / "release", destination / "release")
    for name in [
        "author", "author_provenance", "candidates", "baselines", "author_baselines",
        "cases", "commands", "model_results", "attempt_results", "attempt_quality",
        "execution_provenance", "reference_attempts", "glm_attempts", "model_probe", "plans",
        "runtime_gui_preflight",
    ]:
        path = source / name
        if path.exists():
            shutil.copytree(path, destination / name)
    for name in [
        "run.json", "release_manifest.json", "private.jsonl", "public.jsonl", "admitted.json",
        "replay_receipts.jsonl", "contamination_receipts.jsonl", "model_probe_pass.json",
    ]:
        shutil.copyfile(source / name, destination / name)

    # Current-run review artifacts are separate from archived execution provenance.
    for name in [
        "release.receipt.json", "runtime_gui_preflight_pass.json", "runtime_gui_preflight_script.py",
        "runtime_gui_preflight.log", "runtime_gui_preflight.coordinator.log",
        "runtime_gui_preflight.coordinator.receipt.json", "continuation.coordinator.log",
        "continuation.receipt.json", "continuation.status.json",
    ]:
        path = source / name
        if path.exists():
            shutil.copyfile(path, destination / name)
    for pattern in [
        "*_migration_*review_*.json", "visible_source_audit.*.json",
        "native_response_projection_review*.json", "audit_*_visible_proof.py",
        "review_*_native.py", "*.first_response_identity.json",
    ]:
        for path in sorted(source.glob(pattern)):
            shutil.copyfile(path, destination / path.name)

    resources = destination / "resources"
    resources.mkdir()
    workers = resources / "workers"
    workers.mkdir()
    for path in sorted((source / "infra/workers").glob("*.json")):
        shutil.copyfile(path, workers / path.name)
    shutil.copyfile(source / "prepared.json", resources / "prepared.json")
    for name in ["release.intent.json", "release.receipt.json"]:
        path = source / name
        if path.exists():
            shutil.copyfile(path, resources / name)
    receipts = destination / "archive_receipts"
    receipts.mkdir()
    for path in sorted((source / "archives").glob("*.receipt.json")):
        shutil.copyfile(path, receipts / path.name)
    for name in ["source-runtime.tar.gz", "source-runtime.tar.manifest.json"]:
        shutil.copyfile(source / "infra" / name, destination / name)
