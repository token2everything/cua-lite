"""Read-only raw rollout/debug log contract checks.

This module validates persisted ``trajectory.parquet`` rows produced under
rollout sample directories. It intentionally does not import migration code or
mutate logs; callers get a list of human-readable contract errors.

This is a debug checker, not the canonical publish contract owner. Row-level
semantics are delegated to the data-owned raw rollout validators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from lite.core.errors import LiteContractError
from lite.core.tools.results import (
    TOOL_RESULT_ERROR_SECTION_HEADER,
    project_tool_result_text,
)
from lite.data.staging import coerce_messages, coerce_meta
from lite.data.utils.rows import clean_nones
from lite.data.utils.rows import validate_raw_rollout_rows
from lite.infer.debug.log_layout import (
    ACTIONS_JSON,
    ENV_RESULT_IMAGES_DIR,
    PROMPT_IMAGES_ANNOTATED_DIR,
    PROMPT_IMAGES_DIR,
    RESULTS_JSON,
    TURN_REQUIRED_FILES,
    turn_dirs,
    turn_results_path,
)
from lite.utils.path import project_root

_TRUNCATED_LOG_TEXT_RE = re.compile(r"^(?P<prefix>.*)\.\.\. \[(?P<n>\d+) chars total\]$", re.S)
def check_log_contract(log_root: Path | str) -> list[str]:
    """Return rollout log contract violations under *log_root*.

    ``log_root`` may be either a rollout root containing ``sample_*`` dirs or a
    single sample dir. A directory named ``sample_*`` is contract-checked and
    must contain ``trajectory.parquet``.
    """

    root = Path(log_root)
    sample_dirs = _sample_dirs(root)
    if not sample_dirs:
        return [f"{root}: no sample_* trajectory directories found"]

    errors: list[str] = []
    for sample_dir in sample_dirs:
        errors.extend(_check_sample_dir(sample_dir))
    return errors


def _sample_dirs(root: Path) -> list[Path]:
    if root.is_dir() and (
        root.name.startswith("sample_") or (root / "trajectory.parquet").exists()
    ):
        return [root]
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("sample_*") if p.is_dir())


def _check_sample_dir(sample_dir: Path) -> list[str]:
    path = sample_dir / "trajectory.parquet"
    if not path.is_file():
        if _is_explicit_envblocked_void_sample(sample_dir):
            return []
        return [f"{sample_dir}: missing trajectory.parquet"]

    try:
        import pandas as pd

        rows = pd.read_parquet(path).to_dict("records")
    except Exception as exc:  # pragma: no cover - exact parquet exceptions vary.
        return [f"{path}: failed to read trajectory.parquet: {exc}"]

    errors: list[str] = []
    sample_images: list[str] | None = None
    if len(rows) != 1:
        errors.append(
            f"{path}: rollout sample trajectory.parquet must contain exactly one row; "
            f"found {len(rows)}"
        )
    projected_tool_text_by_call_id: dict[str, str] = {}
    for row_idx, row in enumerate(rows):
        row = _plain(row)
        where = f"{path}: row {row_idx}"
        metadata = _coerce_metadata_for_check(row.get("metadata"), where, errors)
        messages = _coerce_messages_for_check(row.get("messages"), where, errors)
        images = row.get("images")
        errors.extend(_check_images(images, where, sample_dir))
        if sample_images is None and isinstance(images, list):
            sample_images = [
                str(image)
                for image in images
                if isinstance(image, (str, Path)) and str(image)
            ]
        if isinstance(messages, list) and isinstance(metadata, dict) and isinstance(images, list):
            validation_messages = clean_nones(messages)
            validation_metadata = clean_nones(metadata)
            try:
                validate_raw_rollout_rows(
                    [{
                        "images": images,
                        "messages": validation_messages,
                        "metadata": validation_metadata,
                    }],
                    where,
                )
            except (LiteContractError, ValueError) as exc:
                errors.append(f"{where}: {exc}")
            _merge_tool_texts(projected_tool_text_by_call_id, _tool_text_by_call_id(messages))
        else:
            if not isinstance(messages, list):
                errors.append(f"{where}: messages must be a list")
            if not isinstance(metadata, dict):
                errors.append(f"{where}: metadata must be an object")
    errors.extend(
        _check_turn_debug_artifacts(
            sample_dir,
            projected_tool_text_by_call_id,
            sample_images or [],
        )
    )
    return errors


def _is_explicit_envblocked_void_sample(sample_dir: Path) -> bool:
    summary_path = sample_dir / "summary.json"
    error_path = sample_dir / "error.txt"
    if not summary_path.is_file() or not error_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text())
        error_text = error_path.read_text()
    except Exception:  # noqa: BLE001 - malformed sidecars still fail as missing parquet.
        return False
    if not isinstance(summary, dict):
        return False
    summary_error = summary.get("error")
    return (
        isinstance(summary_error, str)
        and bool(summary_error)
        and summary_error == error_text
        and "lite.gym.errors.EnvBlocked" in error_text
        and summary.get("n_turns") == 0
        and summary.get("terminated") is False
        and summary.get("truncated") is False
    )


def _check_images(images: Any, where: str, sample_dir: Path) -> list[str]:
    if images is None:
        return [f"{where}: images must be a list"]
    if not isinstance(images, list):
        return [f"{where}: images must be a list"]
    errors: list[str] = []
    for idx, image in enumerate(images):
        image_where = f"{where}: images[{idx}]"
        if not isinstance(image, str) or not image:
            errors.append(f"{image_where} must be a non-empty string path")
            continue
        resolved = _resolve_logged_image(sample_dir, image)
        if not resolved.is_file():
            errors.append(f"{image_where}: image path does not exist: {image}")
    return errors


def _resolve_logged_image(sample_dir: Path, image: str) -> Path:
    path = Path(image)
    if path.is_absolute():
        return path
    for candidate in (Path.cwd() / path, project_root() / path, sample_dir / path):
        if candidate.is_file():
            return candidate
    return Path.cwd() / path


def _coerce_metadata_for_check(value: Any, where: str, errors: list[str]) -> Any:
    if value is None or isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return coerce_meta(value)
        except Exception as exc:  # noqa: BLE001 - diagnostics only.
            errors.append(f"{where}: metadata JSON is invalid: {exc}")
            return value
    return value


def _coerce_messages_for_check(value: Any, where: str, errors: list[str]) -> Any:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return coerce_messages(value)
        except Exception as exc:  # noqa: BLE001 - diagnostics only.
            errors.append(f"{where}: messages JSON is invalid: {exc}")
            return value
    return value


def _check_turn_debug_artifacts(
    sample_dir: Path,
    projected_tool_text_by_call_id: dict[str, str],
    sample_images: list[str],
) -> list[str]:
    dirs = turn_dirs(sample_dir)
    if not dirs:
        return []

    errors: list[str] = []
    stale_files = (
        "00_screenshot.png",
        "04_screenshot_annotated.png",
        "05_results.json",
        "06_timing.json",
    )
    stale_dirs = ("images", "annotated", "result_images")
    for turn_dir in dirs:
        where = str(turn_dir)
        for name in TURN_REQUIRED_FILES:
            if not (turn_dir / name).is_file():
                errors.append(f"{where}: missing {name}")
        results_path = turn_results_path(turn_dir)
        if results_path is None:
            errors.append(f"{where}: missing {RESULTS_JSON}")
        for name in stale_files:
            if (turn_dir / name).exists():
                errors.append(f"{where}: stale legacy artifact {name}")
        for name in stale_dirs:
            if (turn_dir / name).exists():
                errors.append(f"{where}: stale legacy artifact {name}/")

        actions_path = turn_dir / ACTIONS_JSON
        if actions_path.is_file():
            actions_payload = _load_json(actions_path, errors)
            if actions_payload is not None and not isinstance(actions_payload, dict):
                errors.append(f"{actions_path}: {ACTIONS_JSON} must be an object")

        if results_path is not None:
            results_payload = _load_json(results_path, errors)
            if isinstance(results_payload, dict):
                _check_turn_results_payload(
                    results_payload,
                    str(results_path),
                    results_path.name,
                    turn_dir,
                    sample_dir,
                    sample_images,
                    errors,
                    projected_tool_text_by_call_id,
                )
            elif results_payload is not None:
                errors.append(f"{results_path}: {results_path.name} must be an object")

        _check_prompt_image_cache_pair(
            turn_dir,
            PROMPT_IMAGES_DIR,
            PROMPT_IMAGES_ANNOTATED_DIR,
            errors,
        )
        env_result_image_dir = turn_dir / ENV_RESULT_IMAGES_DIR
        if env_result_image_dir.exists() and not env_result_image_dir.is_dir():
            errors.append(f"{where}: {ENV_RESULT_IMAGES_DIR} must be a directory")
    return errors


def _check_prompt_image_cache_pair(
    turn_dir: Path,
    image_dir_name: str,
    annotated_dir_name: str,
    errors: list[str],
) -> None:
    where = str(turn_dir)
    image_dir = turn_dir / image_dir_name
    annotated_dir = turn_dir / annotated_dir_name
    if annotated_dir.exists() and not image_dir.exists():
        errors.append(f"{where}: {annotated_dir_name}/ exists without {image_dir_name}/")
    if image_dir.exists() and not image_dir.is_dir():
        errors.append(f"{where}: {image_dir_name} must be a directory")
    if annotated_dir.exists() and not annotated_dir.is_dir():
        errors.append(f"{where}: {annotated_dir_name} must be a directory")
    if image_dir.is_dir() and annotated_dir.is_dir():
        image_names = {p.name for p in image_dir.glob("*.png") if p.is_file()}
        for annotated in annotated_dir.glob("*.png"):
            if annotated.name not in image_names:
                errors.append(
                    f"{annotated}: annotated image has no matching "
                    f"{image_dir_name}/{annotated.name}"
                )


def _load_json(path: Path, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 - diagnostics only.
        errors.append(f"{path}: failed to read JSON: {exc}")
        return None


def _check_turn_results_payload(
    payload: dict[str, Any],
    where: str,
    result_file_name: str,
    turn_dir: Path,
    sample_dir: Path,
    sample_images: list[str],
    errors: list[str],
    projected_tool_text_by_call_id: dict[str, str],
) -> None:
    for key in ("reward", "terminated", "truncated", "results", "info"):
        if key not in payload:
            errors.append(f"{where}: missing top-level key {key!r}")
    if "terminated" in payload and not isinstance(payload["terminated"], bool):
        errors.append(f"{where}: terminated must be a bool")
    if "truncated" in payload and not isinstance(payload["truncated"], bool):
        errors.append(f"{where}: truncated must be a bool")
    if "info" in payload and not isinstance(payload["info"], dict):
        errors.append(f"{where}: info must be an object")
    results = payload.get("results")
    if results is None:
        errors.append(f"{where}: missing results")
        return
    if not isinstance(results, list):
        errors.append(f"{where}: results must be a list")
        return
    required_result_keys = {"tool_call_id", "images", "text", "metadata", "error"}
    seen_result_call_ids: set[str] = set()
    for idx, result in enumerate(results):
        result_where = f"{where}: results[{idx}]"
        if not isinstance(result, dict):
            errors.append(f"{result_where} must be an object")
            continue
        missing = required_result_keys - set(result)
        if missing:
            errors.append(f"{result_where}: missing keys {sorted(missing)}")
            continue
        call_id = result.get("tool_call_id")
        if isinstance(call_id, str) and call_id:
            if call_id in seen_result_call_ids:
                errors.append(
                    f"{result_where}: duplicate {result_file_name} result for tool_call_id {call_id!r}"
                )
            seen_result_call_ids.add(call_id)
        _check_result_native_text_error(result, result_where, errors)
        _check_projected_result_text(
            result,
            result_where,
            errors,
            projected_tool_text_by_call_id,
        )
        _check_result_images_payload(
            result["images"],
            idx,
            result,
            result_where,
            result_file_name,
            turn_dir,
            errors,
        )
    _check_result_image_orphans(turn_dir, results, where, result_file_name, errors)


def _check_result_images_payload(
    image_infos: Any,
    idx: int,
    result: dict[str, Any],
    where: str,
    result_file_name: str,
    turn_dir: Path,
    errors: list[str],
) -> None:
    if not isinstance(image_infos, list):
        errors.append(f"{where}: images must be a list")
        return
    for image_idx, image_info in enumerate(image_infos):
        image_where = f"{where}: images[{image_idx}]"
        if not isinstance(image_info, dict):
            errors.append(f"{image_where} must be an object")
            continue
        _check_env_result_image_payload(
            image_info,
            idx,
            image_idx,
            result,
            image_where,
            turn_dir,
            errors,
        )


def _check_env_result_image_payload(
    image_info: dict[str, Any],
    result_idx: int,
    image_idx: int,
    result: dict[str, Any],
    where: str,
    turn_dir: Path,
    errors: list[str],
) -> None:
    expected = {"path", "source", "bytes", "sha1"}
    missing = expected - set(image_info)
    if missing:
        errors.append(f"{where}: image missing keys {sorted(missing)}")
        return
    if image_info["source"] != ENV_RESULT_IMAGES_DIR:
        errors.append(f"{where}: image.source must be {ENV_RESULT_IMAGES_DIR!r}")
    rel_path = image_info["path"]
    if not isinstance(rel_path, str) or not rel_path:
        errors.append(f"{where}: image.path must be a non-empty string")
        return
    path = Path(rel_path)
    if (
        path.is_absolute()
        or not path.parts
        or path.parts[0] != ENV_RESULT_IMAGES_DIR
        or ".." in path.parts
    ):
        errors.append(f"{where}: image.path must be relative under {ENV_RESULT_IMAGES_DIR}/")
        return
    file_path = turn_dir / path
    if not file_path.is_file():
        errors.append(f"{where}: image.path does not exist: {rel_path}")
        return
    data = file_path.read_bytes()
    if image_info["bytes"] != len(data):
        errors.append(f"{where}: image.bytes does not match file size")
    digest = hashlib.sha1(data).hexdigest()
    if image_info["sha1"] != digest:
        errors.append(f"{where}: image.sha1 does not match file content")
    call_id = result.get("tool_call_id") or f"result_{result_idx:04d}"
    safe = "".join(
        ch if ("a" <= ch <= "z" or "A" <= ch <= "Z" or "0" <= ch <= "9" or ch in "._-") else "_"
        for ch in str(call_id)
    )
    expected_name = f"{result_idx:04d}_{image_idx:04d}_from_{safe}.png"
    if path.name != expected_name:
        errors.append(
            f"{where}: image filename {path.name!r} does not match result ordinal/tool_call_id "
            f"{expected_name!r}"
        )


def _check_result_image_orphans(
    turn_dir: Path,
    results: list[Any],
    where: str,
    result_file_name: str,
    errors: list[str],
) -> None:
    result_image_dir = turn_dir / ENV_RESULT_IMAGES_DIR
    if not result_image_dir.is_dir():
        return
    referenced: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        for image_info in result.get("images", []):
            if isinstance(image_info, dict) and isinstance(image_info.get("path"), str):
                referenced.append(image_info["path"])
    seen_refs: set[str] = set()
    duplicate_refs: set[str] = set()
    for rel in referenced:
        if rel in seen_refs:
            duplicate_refs.add(rel)
        seen_refs.add(rel)
    for rel in sorted(duplicate_refs):
        errors.append(f"{where}: duplicate result image path {rel}")
    referenced_set = set(referenced)
    for path in sorted(result_image_dir.glob("*.png")):
        rel = str(Path(ENV_RESULT_IMAGES_DIR) / path.name)
        if rel not in referenced_set:
            errors.append(f"{path}: orphan result image not referenced by {result_file_name}")


def _tool_text_by_call_id(messages: Any) -> dict[str, str]:
    if not isinstance(messages, list):
        return {}

    by_call_id: dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        call_id = msg.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id:
            continue
        text_parts = _text_content_parts(msg.get("content"))
        if not text_parts:
            continue
        text = "\n".join(text_parts)
        by_call_id[call_id] = (
            f"{by_call_id[call_id]}\n{text}" if call_id in by_call_id else text
        )
    return by_call_id


def _text_content_parts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    return [
        item["text"]
        for item in content
        if (
            isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        )
    ]


def _merge_tool_texts(dst: dict[str, str], src: dict[str, str]) -> None:
    for call_id, text in src.items():
        dst[call_id] = f"{dst[call_id]}\n{text}" if call_id in dst else text


def _check_result_native_text_error(
    result: dict[str, Any],
    where: str,
    errors: list[str],
) -> None:
    native_text = result.get("text")
    native_error = result.get("error")
    if native_text is not None and not isinstance(native_text, str):
        errors.append(f"{where}: text must be a string or null")
    if native_error is not None and not isinstance(native_error, str):
        errors.append(f"{where}: error must be a string or null")


def _check_projected_result_text(
    result: dict[str, Any],
    where: str,
    errors: list[str],
    projected_tool_text_by_call_id: dict[str, str],
) -> None:
    call_id = result.get("tool_call_id")
    native_text = result.get("text")
    native_error = result.get("error")
    if not isinstance(call_id, str) or not call_id:
        return
    if native_text is not None and not isinstance(native_text, str):
        return
    if native_error is not None and not isinstance(native_error, str):
        return

    expected = project_tool_result_text(native_text, native_error)
    if expected is None:
        return
    projected = projected_tool_text_by_call_id.get(call_id)
    if not projected:
        errors.append(
            f"{where}: native result for tool_call_id {call_id!r} is not projected "
            "into trajectory.parquet messages"
        )
        return
    if not _contains_projected_text(projected, expected):
        errors.append(
            f"{where}: trajectory.parquet role:tool message for tool_call_id {call_id!r} "
            "does not match projected native text/error"
        )


def _contains_projected_text(projected_text: str, expected: str) -> bool:
    if expected in projected_text:
        return True
    error_marker = f"\n\n{TOOL_RESULT_ERROR_SECTION_HEADER}\n"
    if error_marker in expected:
        expected_text, expected_error = expected.split(error_marker, 1)
        match = _TRUNCATED_LOG_TEXT_RE.fullmatch(expected_text)
        if match:
            prefix = match.group("prefix").rstrip()
            error_section = f"{TOOL_RESULT_ERROR_SECTION_HEADER}\n{expected_error}"
            return bool(prefix) and prefix in projected_text and error_section in projected_text
    match = _TRUNCATED_LOG_TEXT_RE.fullmatch(expected)
    if not match:
        return False
    prefix = match.group("prefix").rstrip()
    return bool(prefix) and prefix in projected_text


def _plain(value: Any) -> Any:
    if hasattr(value, "as_py"):
        value = value.as_py()
    elif hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        try:
            value = value.tolist()
        except TypeError:
            pass
    elif hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass

    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate rollout trajectory.parquet and per-turn debug artifacts.",
    )
    parser.add_argument("log_root", type=Path)
    args = parser.parse_args(argv)

    errors = check_log_contract(args.log_root)
    if not errors:
        return 0
    for error in errors:
        print(error, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
