"""Copy validated image-text V3 candidates and provenance into an audit bundle.

This publishes review candidates only. It does not mark human review complete
or promote records for training. An existing target is never overwritten.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import shutil
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

from clean_native_trajectories import sha256_file, verify_execution_evidence
from clean_v3_projection import audit_roundtrip
from internvalidata.data_models.messages_types.multimodal_image_text_messages import (
    MultimodalImageTextMessagesV3,
)
from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--clean-output", type=Path)
    source.add_argument("--existing-review-bundle", type=Path)
    source.add_argument("--absolute-jsonl", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--json-dir", type=Path)
    parser.add_argument("--jsonl-dir", type=Path)
    parser.add_argument("--require-trajectories", type=int, default=300)
    parser.add_argument("--technical-smoke", action="store_true")
    return parser.parse_args()


def absolute_image_paths(row: dict, media_root: Path) -> tuple[dict, list[dict]]:
    """Rebase all image references, recording exact reversible path changes."""
    output = copy.deepcopy(row)
    changes = []
    aliases = output["metadata"]["image_aliases"]
    archive = output["metadata"]["source_media_archive"]
    native_locations = {}
    for uri, reference in aliases.items():
        archived = archive[Path(uri).name]
        if archived["sha256"] != reference["sha256"]:
            raise ValueError("Native image and its visible alias contain different bytes")
        native_locations[uri] = str((media_root / archived["url"]).resolve())
    for index, message in enumerate(output["messages"]):
        if isinstance(message["content"], list):
            for part_index, part in enumerate(message["content"]):
                if part["type"] != "image":
                    continue
                old = part["image"]["url"]
                new = str((media_root / old).resolve())
                changes.append(
                    {
                        "path": ["messages", index, "content", part_index, "image", "url"],
                        "old": old,
                        "new": new,
                        "kind": "value",
                    }
                )
                part["image"]["url"] = new
        for boundary_index, boundary in enumerate(message["meta"]["native_projection"]):
            content = boundary["source_message"]["content"]
            if not isinstance(content, list):
                continue
            for part_index, part in enumerate(content):
                if part["type"] != "image":
                    continue
                old = part["image"]["url"]
                new = native_locations[old]
                changes.append(
                    {
                        "path": [
                            "messages",
                            index,
                            "meta",
                            "native_projection",
                            boundary_index,
                            "source_message",
                            "content",
                            part_index,
                            "image",
                            "url",
                        ],
                        "old": old,
                        "new": new,
                        "kind": "value",
                    }
                )
                part["image"]["url"] = new
    for field, references in (("image_aliases", aliases), ("source_media_archive", archive)):
        for old_key, reference in list(references.items()):
            old = reference["url"]
            new = str((media_root / old).resolve())
            changes.append(
                {
                    "path": ["metadata", field, old_key, "url"],
                    "old": old,
                    "new": new,
                    "kind": "value",
                }
            )
            reference["url"] = new
            new_key = native_locations[old_key] if field == "image_aliases" else new
            if new_key in references:
                raise ValueError(f"Absolute image mapping key collision: {new_key}")
            references[new_key] = references.pop(old_key)
            changes.append(
                {"path": ["metadata", field], "old": old_key, "new": new_key, "kind": "mapping_key"}
            )
    for change in changes:
        path = Path(change["new"])
        if not path.is_absolute():
            raise ValueError(f"Exported image reference is not absolute: {path}")
        path.relative_to(media_root)
        if not path.is_file():
            raise FileNotFoundError(path)
    return output, changes


def export_absolute_jsonl(source: Path, source_jsonl: Path, output_jsonl: Path) -> dict:
    """Export accepted V3 rows and audit readback against untouched native data."""
    source = source.resolve()
    output_jsonl = output_jsonl.resolve()
    output_jsonl.relative_to(source)
    if output_jsonl.exists():
        raise FileExistsError(f"Refusing to overwrite an existing JSONL: {output_jsonl}")
    source_sha = sha256_file(source_jsonl)
    verification = json.loads((source / "verification.json").read_text())
    if source_sha != verification["published_sha256"]:
        raise ValueError("Source JSONL differs from its fixed cleaning receipt")
    processed = source / "processed" / "003_gui_behavior_filters"
    accepted = {}
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    incomplete = output_jsonl.with_name(output_jsonl.name + ".incomplete")
    audit_path = output_jsonl.with_name(output_jsonl.stem + ".export_audit.json")
    if incomplete.exists() or audit_path.exists():
        raise FileExistsError("An incomplete export or audit already exists")
    rows = []
    with source_jsonl.open() as input_file, incomplete.open("x") as output_file:
        for line in input_file:
            row = json.loads(line)
            task_id = row["metadata"]["source_task_id"]
            processed_file = processed / f"{task_id}.json"
            record = json.loads(processed_file.read_text())
            if not record["metadata"]["internflow"]["accepted"]:
                raise ValueError("Source V3 trajectory was rejected by the GUI filter")
            if record["metadata"]["source_task_id"] != task_id:
                raise ValueError("Source V3 trajectory differs from the GUI-filter record")
            accepted[task_id] = sha256_file(processed_file)
            exported, changes = absolute_image_paths(row, source)
            output_file.write(
                json.dumps(exported, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            rows.append({"task_id": task_id, "image_path_changes": changes})
    task_ids = [entry["task_id"] for entry in rows]
    if len(set(task_ids)) != len(rows) or set(task_ids) != set(accepted):
        raise ValueError("JSONL trajectories differ from the accepted GUI-filter cohort")
    if len(rows) != verification["accepted"]:
        raise ValueError("JSONL row count differs from its cleaning receipt")
    # Reverse only recorded path edits, then run the unchanged native audit.
    with source_jsonl.open() as input_file, incomplete.open() as output_file:
        for index, pair in enumerate(zip_longest(input_file, output_file)):
            original_line, exported_line = pair
            if original_line is None or exported_line is None:
                raise ValueError("Exported JSONL omitted or added a trajectory")
            original = json.loads(original_line)
            exported = json.loads(exported_line)
            MultimodalImageTextMessagesV3.model_validate_json(exported_line)
            restored = copy.deepcopy(exported)
            for change in reversed(rows[index]["image_path_changes"]):
                if change["kind"] == "mapping_key":
                    container = restored
                    for component in change["path"]:
                        container = container[component]
                    container[change["old"]] = container.pop(change["new"])
                else:
                    container = restored
                    for component in change["path"][:-1]:
                        container = container[component]
                    if container[change["path"][-1]] != change["new"]:
                        raise ValueError("Serialized image path differs from export mapping")
                    container[change["path"][-1]] = change["old"]
            if restored != original:
                raise ValueError("Export changed trajectory fields other than image paths")
            task_id = rows[index]["task_id"]
            trajectory = source / "original" / task_id / "agent_trajectory.jsonl"
            if sha256_file(trajectory) != original["metadata"]["source_trajectory_sha256"]:
                raise ValueError("Native trajectory differs from its source receipt")
            rows[index]["roundtrip"] = audit_roundtrip(
                restored, json.loads(trajectory.read_text()), source
            )
    if sha256_file(source_jsonl) != source_sha:
        raise ValueError("Source JSONL changed during export")
    for task_id, digest in accepted.items():
        if sha256_file(processed / f"{task_id}.json") != digest:
            raise ValueError("Accepted intermediate JSON changed during export")
    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "target_format": "multimodal_image_text_dialogs_v3",
        "internvalidata": "0.4.17",
        "jsonl_path": str(output_jsonl),
        "jsonl_sha256": sha256_file(incomplete),
        "source_jsonl": str(source_jsonl),
        "source_jsonl_sha256": source_sha,
        "source_files_unchanged": True,
        "sample_rows": len(rows),
        "image_paths": "absolute_shared_storage",
        "image_path_references": sum(len(entry["image_path_changes"]) for entry in rows),
        "validated_trajectories": len(rows),
        "non_image_fields_equal": True,
        "image_files_hashes_dimensions_verified": True,
        "review_status": "pending_human_review",
        "training_promotion": "not_promoted",
        "rows": rows,
    }
    with audit_path.open("x") as audit_file:
        json.dump(manifest, audit_file, indent=2, ensure_ascii=False)
    incomplete.rename(output_jsonl)
    return {key: value for key, value in manifest.items() if key != "rows"} | {
        "audit_path": str(audit_path)
    }


def export_trajectory_files(jsonl: Path, output_dir: Path, file_format: str) -> dict:
    """Write each verified absolute-path V3 trajectory as JSON or one-line JSONL."""
    jsonl = jsonl.resolve()
    output_dir = output_dir.resolve()
    audit = json.loads(jsonl.with_name(jsonl.stem + ".export_audit.json").read_text())
    source_sha = sha256_file(jsonl)
    if audit["jsonl_sha256"] != source_sha or audit["image_paths"] != "absolute_shared_storage":
        raise ValueError("Trajectory export requires the verified absolute-path JSONL")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite trajectory files: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    files = {}
    with jsonl.open() as stream:
        for line in stream:
            row = json.loads(line)
            task_id = row["metadata"]["source_task_id"]
            path = output_dir / f"{task_id}.{file_format}"
            with path.open("x") as output:
                if file_format == "json":
                    json.dump(row, output, indent=2, ensure_ascii=False)
                else:
                    output.write(line.rstrip("\n"))
                output.write("\n")
            if json.loads(path.read_text()) != row:
                raise ValueError("Per-trajectory export differs from its source JSONL row")
            MultimodalImageTextMessagesV3.model_validate_json(path.read_text())
            files[path.name] = sha256_file(path)
    if len(files) != audit["sample_rows"] or sha256_file(jsonl) != source_sha:
        raise ValueError("Trajectory export changed the verified trajectory cohort")
    manifest = {
        "target_format": "multimodal_image_text_dialogs_v3",
        "internvalidata": "0.4.17",
        "file_format": file_format,
        "output_dir": str(output_dir),
        "source_jsonl_path": str(jsonl),
        "source_jsonl_sha256": source_sha,
        "trajectories": len(files),
        "trajectory_rows_equal_source_jsonl": True,
        "image_paths": "absolute_shared_storage",
        "review_status": "pending_human_review",
        "training_promotion": "not_promoted",
        "files_sha256": files,
    }
    manifest_path = output_dir.parent / f"{output_dir.name}.export_manifest.json"
    with manifest_path.open("x") as output:
        json.dump(manifest, output, indent=2, ensure_ascii=False)
    return manifest


def main() -> int:
    """Verify serialized rows and media before copying the candidate bundle."""
    args = _parse_args()
    if importlib.metadata.version("internvalidata") != "0.4.17":
        raise ValueError("The audit bundle requires InternValidata 0.4.17")
    if args.absolute_jsonl is not None:
        if (
            (args.json_dir is None and args.jsonl_dir is None)
            or args.target is not None
            or args.output_jsonl is not None
        ):
            raise ValueError("Absolute JSONL conversion requires --json-dir or --jsonl-dir")
        manifest = {}
        for file_format, output_dir in (("json", args.json_dir), ("jsonl", args.jsonl_dir)):
            if output_dir is not None:
                exported = export_trajectory_files(args.absolute_jsonl, output_dir, file_format)
                manifest[file_format] = {
                    key: value for key, value in exported.items() if key != "files_sha256"
                }
        print(json.dumps(manifest, ensure_ascii=False))
        return 0
    if args.existing_review_bundle is not None:
        if args.output_jsonl is None or args.target is not None:
            raise ValueError("Existing review conversion requires --output-jsonl only")
        source = args.existing_review_bundle.resolve()
        manifest = export_absolute_jsonl(
            source, source / "multimodal_image_text_dialogs_v3.jsonl", args.output_jsonl
        )
        if args.json_dir is not None:
            manifest["json_export"] = export_trajectory_files(
                args.output_jsonl, args.json_dir, "json"
            )
        if args.jsonl_dir is not None:
            manifest["jsonl_export"] = export_trajectory_files(
                args.output_jsonl, args.jsonl_dir, "jsonl"
            )
        print(json.dumps(manifest, ensure_ascii=False))
        return 0
    if (
        args.target is None
        or args.output_jsonl is not None
        or args.json_dir is not None
        or args.jsonl_dir is not None
    ):
        raise ValueError("A new review bundle requires --target only")
    source = args.clean_output.resolve()
    target = args.target.resolve()
    verification = json.loads((source / "verification.json").read_text())
    expected_scope = "technical_smoke_not_v03_data" if args.technical_smoke else "v03_formal"
    if verification["scope"] != expected_scope:
        raise ValueError("Technical smoke and formal datasets must remain separate")
    if verification.get("format_projection") != "reversible_image_v3_v2_1":
        raise ValueError("Publication requires the reversible, audited native projection")
    execution_sha = None
    if not args.technical_smoke:
        execution_sha = verification["model_execution_sha256"]
        if sha256_file(source / "model_execution.json") != execution_sha:
            raise ValueError("Archived batch teacher snapshot differs from the cleaning receipt")
        batch_execution = json.loads((source / "model_execution.json").read_text())
        if batch_execution["teacher"] != verification["expected_teacher"]:
            raise ValueError("Archived teacher parameters differ from the cleaning receipt")
    if verification["accepted"] != args.require_trajectories:
        raise ValueError(
            f"Expected {args.require_trajectories} accepted trajectories; "
            f"found {verification['accepted']}"
        )
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite an existing audit bundle: {target}")
    data_file = source / "multimodal_image_text_dialogs_v3.jsonl"
    data_sha = sha256_file(data_file)
    if data_sha != verification["published_sha256"]:
        raise ValueError("Published data checksum no longer matches cleaning receipt")
    images = json.loads((source / "image_manifest.json").read_text())
    seen_images = set()
    task_ids = set()
    row_count = 0
    audits = []
    for line in data_file.open():
        checked = MultimodalImageTextMessagesV3.model_validate_json(line)
        row = json.loads(line)
        task_id = checked.metadata["source_task_id"]
        task_ids.add(task_id)
        original = source / "original" / task_id
        trajectory = original / "agent_trajectory.jsonl"
        native = json.loads(trajectory.read_text())
        if sha256_file(trajectory) != row["metadata"]["source_trajectory_sha256"]:
            raise ValueError("Preserved native trajectory differs from its source receipt")
        tools_path = original / "canonical_tools.json"
        if sha256_file(tools_path) != row["metadata"]["canonical_tools_sha256"]:
            raise ValueError("Preserved owning tool schema differs from source receipt")
        if row["tools"] != json.loads(tools_path.read_text()):
            raise ValueError("Published tool contract differs from the owning canonical schemas")
        if not args.technical_smoke and native["metadata"]["tools"] != row["tools"]:
            raise ValueError("Formal native tool contract differs from its frozen owner schema")
        audit = audit_roundtrip(row, native, source)
        public = json.loads((original / "public_task.json").read_text())
        row_execution_sha = None if args.technical_smoke else row["metadata"]["execution_evidence"]["model_execution_sha256"]
        row_execution_path = None if args.technical_smoke else source / batch_execution["source_executions"][row_execution_sha]
        execution = verify_execution_evidence(
            {
                "task_id": task_id,
                "request_dir": str(original / "requests"),
                "score_receipt": str(original / "reference_receipt.json"),
                "api_type": row["metadata"]["source_api_type"],
                "model_execution": str(row_execution_path),
                "model_execution_sha256": row_execution_sha,
                "evidence_kind": row["metadata"].get("evidence_kind"),
            },
            native,
            public,
            not args.technical_smoke,
        )
        if (
            not args.technical_smoke
            and execution["model_execution_sha256"]
            != row["metadata"]["execution_evidence"]["model_execution_sha256"]
        ):
            raise ValueError("Row teacher snapshot differs from its archived frozen execution")
        if (
            execution["request_evidence_sha256"]
            != row["metadata"]["execution_evidence"]["request_evidence_sha256"]
        ):
            raise ValueError("Archived complete request evidence differs from source hashes")
        if len(public["sites"]) != 1 and execution.get("evidence_kind") != "official_task_rewrite":
            raise ValueError("Formal audit bundle is single-site only")
        audits.append(
            {
                "task_id": task_id,
                "roundtrip": audit,
                "model_id": execution["model_id"],
                "score_checks": execution["score_checks"],
                **({key: execution[key] for key in (
                    "evidence_kind", "official_task_id", "benchmark_derived", "evaluation_scope",
                    "independent_final_state", "pass_rule",
                )} if execution.get("evidence_kind") == "official_task_rewrite" else {}),
            }
        )
        row_count += 1
        for message in checked.messages:
            if not isinstance(message.content, list):
                continue
            for part in message.content:
                if part.type != "image":
                    continue
                relative = part.image.url
                image_path = (source / relative).resolve()
                image_path.relative_to(source)
                if relative in seen_images:
                    continue
                with Image.open(image_path) as image:
                    image.load()
                    if tuple(image.size) != tuple(part.image.image_wh):
                        raise ValueError(f"Image dimensions differ: {relative}")
                if sha256_file(image_path) != images[relative]["sha256"]:
                    raise ValueError(f"Image checksum differs: {relative}")
                seen_images.add(relative)
    if len(task_ids) != args.require_trajectories:
        raise ValueError("Accepted trajectories do not match unique source task count")
    target.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source / "verification.json", target / "verification.json")
    shutil.copyfile(source / "image_manifest.json", target / "image_manifest.json")
    shutil.copyfile(source / "profile.yaml", target / "profile.yaml")
    shutil.copyfile(source / "prepared_plan.json", target / "prepared_plan.json")
    if not args.technical_smoke:
        shutil.copyfile(source / "model_execution.json", target / "model_execution.json")
        shutil.copytree(source / "model_executions", target / "model_executions")
    shutil.copytree(source / "original", target / "original")
    shutil.copytree(source / "processed", target / "processed")
    shutil.copytree(source / "rejected", target / "rejected")
    shutil.copytree(source / "audits", target / "audits")
    copied_projection = target / "audits" / "cleaning_projection.relative.jsonl"
    shutil.copyfile(data_file, copied_projection)
    shutil.copyfile(source / "input_manifest.jsonl", target / "input_manifest.jsonl")
    # The copied manifest also covers images retained from rejected inputs.
    for relative in sorted(images):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, destination)
    # Validate the copied bundle so relative image references work at its final location.
    copied_sha = sha256_file(copied_projection)
    if copied_sha != data_sha:
        raise ValueError("Copied JSONL differs from validated source")
    for relative in images:
        if sha256_file(target / relative) != images[relative]["sha256"]:
            raise ValueError(f"Copied image differs: {relative}")
    copied_audits = []
    for line in copied_projection.open():
        row = json.loads(line)
        original = target / "original" / row["metadata"]["source_task_id"]
        trajectory = original / "agent_trajectory.jsonl"
        if sha256_file(trajectory) != row["metadata"]["source_trajectory_sha256"]:
            raise ValueError("Copied native source differs from original evidence")
        native = json.loads(trajectory.read_text())
        roundtrip = audit_roundtrip(row, native, target)
        row_execution_sha = None if args.technical_smoke else row["metadata"]["execution_evidence"]["model_execution_sha256"]
        row_execution_path = None if args.technical_smoke else target / batch_execution["source_executions"][row_execution_sha]
        execution = verify_execution_evidence(
            {
                "task_id": row["metadata"]["source_task_id"],
                "request_dir": str(original / "requests"),
                "score_receipt": str(original / "reference_receipt.json"),
                "api_type": row["metadata"]["source_api_type"],
                "model_execution": str(row_execution_path),
                "model_execution_sha256": row_execution_sha,
                "evidence_kind": row["metadata"].get("evidence_kind"),
            },
            native,
            json.loads((original / "public_task.json").read_text()),
            not args.technical_smoke,
        )
        if (
            not args.technical_smoke
            and execution["model_execution_sha256"]
            != row["metadata"]["execution_evidence"]["model_execution_sha256"]
        ):
            raise ValueError("Copied row teacher snapshot differs from its original evidence")
        if (
            execution["request_evidence_sha256"]
            != row["metadata"]["execution_evidence"]["request_evidence_sha256"]
        ):
            raise ValueError("Copied complete request evidence differs from original hashes")
        copied_audits.append(
            {
                "task_id": row["metadata"]["source_task_id"],
                "roundtrip": roundtrip,
                "request_evidence_equal": True,
                "formal_score_artifact_hashes_verified": not args.technical_smoke,
            }
        )
    absolute_export = export_absolute_jsonl(target, copied_projection, target / data_file.name)
    json_export = export_trajectory_files(target / data_file.name, target / "json_absolute", "json")
    jsonl_export = export_trajectory_files(
        target / data_file.name, target / "jsonl_absolute", "jsonl"
    )
    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "scope": expected_scope,
        "target_format": "multimodal_image_text_dialogs_v3",
        "internvalidata": "0.4.17",
        "candidate_trajectories": len(task_ids),
        "sample_rows": row_count,
        "referenced_images": len(seen_images),
        "jsonl_sha256": absolute_export["jsonl_sha256"],
        "source_cleaning_jsonl_sha256": data_sha,
        "jsonl_path": absolute_export["jsonl_path"],
        "image_paths": "absolute_shared_storage",
        "absolute_image_path_audit": absolute_export["audit_path"],
        "json_dir": json_export["output_dir"],
        "json_trajectories": json_export["trajectories"],
        "jsonl_dir": jsonl_export["output_dir"],
        "jsonl_trajectories": jsonl_export["trajectories"],
        "review_status": "pending_human_review",
        "training_promotion": "not_promoted",
        **({"benchmark_derived": True, "evaluation_scope": "evaluation_only"}
           if any(audit.get("evidence_kind") == "official_task_rewrite" for audit in audits)
           else {}),
        "roundtrip_audited_trajectories": len(audits),
        "verified_owner_score_receipts": len(audits),
        "copied_bundle_verified_trajectories": len(copied_audits),
        "model_execution_sha256": execution_sha,
        "expected_teacher": None if args.technical_smoke else verification["expected_teacher"],
        "source_cleaning_receipt": str(source / "verification.json"),
        "review_notes": "原始轨迹保留在 original/；格式投影与规则结果见 verification.json、"
        "prepared_plan.json 和 processed/。每条清洗轨迹同时导出到 json_absolute/ 和 "
        "jsonl_absolute/；两种格式的全部图片引用为本共享目录的绝对路径。",
    }
    (target / "review_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (target / "export_audits.json").write_text(json.dumps(audits, indent=2))
    (target / "copied_bundle_audits.json").write_text(json.dumps(copied_audits, indent=2))
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
