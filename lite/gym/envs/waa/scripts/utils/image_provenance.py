#!/usr/bin/env python3
"""Create and validate provenance for the prepared WAA qcow2 image.

Run: uv run python lite/gym/envs/waa/scripts/utils/image_provenance.py \\
       verify --image <qcow2> --iso <iso>   # or: write ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ENV_DIR = Path(__file__).resolve().parents[2]
PREP_DIR = ENV_DIR / "docker" / "prep"
DEFAULT_PREP_IMAGE = (
    "windowsarena/winarena@sha256:869afffe384f0e0356dbfcdf78486611e31d62e692544cbfd9a6c08606f07440"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prep_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in PREP_DIR.rglob("*") if item.is_file()):
        # Keep the logical prep/ prefix used before these sources moved under
        # docker/, so the directory-only refactor does not stale existing disks.
        logical_path = Path("prep") / path.relative_to(PREP_DIR)
        digest.update(logical_path.as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _expected(iso: Path, prep_image: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "iso_sha256": _sha256(iso),
        "prep_source_sha256": _prep_sha256(),
        "upstream_prep_image": prep_image,
    }


def _sidecar(image: Path) -> Path:
    return Path(f"{image}.provenance.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verify", "write"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--iso", type=Path, required=True)
    parser.add_argument("--prep-image", default=DEFAULT_PREP_IMAGE)
    args = parser.parse_args()
    image = args.image.expanduser().resolve()
    iso = args.iso.expanduser().resolve()
    expected = _expected(iso, args.prep_image)
    sidecar = _sidecar(image)

    if args.command == "write":
        if not image.is_file() or image.stat().st_size == 0:
            raise FileNotFoundError(f"cannot record missing WAA image: {image}")
        payload = {
            **expected,
            "image_size": image.stat().st_size,
        }
        tmp = Path(f"{sidecar}.tmp")
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(sidecar)
        print(sidecar)
        return

    if not image.is_file() or image.stat().st_size == 0 or not sidecar.is_file():
        raise SystemExit(1)
    actual = json.loads(sidecar.read_text(encoding="utf-8"))
    valid = all(actual.get(key) == value for key, value in expected.items())
    valid = valid and actual.get("image_size") == image.stat().st_size
    raise SystemExit(0 if valid else 1)


if __name__ == "__main__":
    main()
