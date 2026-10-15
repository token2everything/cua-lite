"""Debug-only helpers for reading trajectory artifact layout."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

TURN_DIR_RE = re.compile(r"^turn_(\d+)$")
ARTIFACT_ORDINAL_RE = re.compile(r"^(\d+)")
PROMPT_TXT = "01_prompt.txt"
RESPONSE_TXT = "02_response.txt"
ACTIONS_JSON = "03_actions.json"
#: The three artifacts every turn dir must contain. Ordered as the writer
#: numbers them, which is also the order a reader walks a turn.
TURN_REQUIRED_FILES = (PROMPT_TXT, RESPONSE_TXT, ACTIONS_JSON)
RESULTS_JSON = "04_results.json"
TIMING_JSON = "05_timing.json"
PROMPT_IMAGES_DIR = "prompt_images"
PROMPT_IMAGES_ANNOTATED_DIR = "prompt_images_annotated"
ENV_RESULT_IMAGES_DIR = "env_result_images"


def turn_index(path: Path) -> int | None:
    """Return the numeric turn index for ``turn_N`` / ``turn_NNNN`` paths."""

    match = TURN_DIR_RE.match(path.name)
    return int(match.group(1)) if match else None


def turn_sort_key(path: Path) -> tuple[int, int, str]:
    """Sort ``turn_9``, ``turn_010``, ``turn_100`` by numeric turn index."""

    index = turn_index(path)
    if index is None:
        return (1, 0, path.name)
    return (0, index, path.name)


def turn_dirs(sample_dir: Path) -> list[Path]:
    """Return turn directories sorted by numeric turn index."""

    return sorted(
        (path for path in sample_dir.glob("turn_*") if path.is_dir()),
        key=turn_sort_key,
    )


def artifact_sort_key(path: Path) -> tuple[int, int, str]:
    """Sort artifacts by leading ordinal, with non-ordinal names last."""

    match = ARTIFACT_ORDINAL_RE.match(path.name)
    if match:
        return (0, int(match.group(1)), path.name)
    return (1, 0, path.name)


def sorted_pngs(paths: Iterable[Path]) -> list[Path]:
    """Return existing PNG paths sorted by artifact ordinal."""

    return sorted((path for path in paths if path.is_file()), key=artifact_sort_key)


def turn_image_paths(turn_dir: Path) -> list[Path]:
    """Return prompt-image debug artifacts for a turn."""

    image_dir = turn_dir / PROMPT_IMAGES_DIR
    return sorted_pngs(image_dir.glob("*.png")) if image_dir.is_dir() else []


def turn_images(sample_dir: Path) -> list[tuple[Path, Path]]:
    """Return ``(turn_dir, image_path)`` pairs for all turn images."""

    images: list[tuple[Path, Path]] = []
    for turn_dir in turn_dirs(sample_dir):
        images.extend((turn_dir, image) for image in turn_image_paths(turn_dir))
    return images


def last_turn_image(turn_dir: Path) -> Path | None:
    """Return the last image artifact for one turn directory."""

    images = turn_image_paths(turn_dir)
    return images[-1] if images else None


def matching_annotation_path(turn_dir: Path, image_path: Path) -> Path | None:
    """Return the annotation artifact for ``image_path`` when one exists."""

    if image_path.parent.name == PROMPT_IMAGES_DIR:
        annotated = turn_dir / PROMPT_IMAGES_ANNOTATED_DIR / image_path.name
        return annotated if annotated.is_file() else None
    return None


def turn_results_path(turn_dir: Path) -> Path | None:
    """Return the current turn result payload path."""

    current = turn_dir / RESULTS_JSON
    return current if current.is_file() else None


def turn_timing_path(turn_dir: Path) -> Path | None:
    """Return the current turn timing payload path."""

    current = turn_dir / TIMING_JSON
    return current if current.is_file() else None
