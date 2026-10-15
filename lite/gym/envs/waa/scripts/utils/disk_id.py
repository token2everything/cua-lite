#!/usr/bin/env python3
"""Identity of the distributable WAA disk, derivable from the checkout ALONE.

    python disk_id.py                          # checkout disk id
    python disk_id.py --from-sidecar PATH      # disk id implied by a provenance sidecar
    python disk_id.py --write-sidecar IMAGE --prep-image IMG   # stamp a pulled disk

The disk id is ``sha256(pinned_iso_sha256 || prep_source_sha256)`` — the two
inputs that determine the prepared qcow2's contents, both derivable WITHOUT the
16 GB image or the Windows ISO (the ISO sha is pinned in
``download_windows_iso.sh``; the prep sources are git-tracked). It rides on the
published ``cua-lite/waa-disk`` image as the ``lite.waa_disk_id`` label so
``install.sh pull`` gates a pulled disk exactly like the runner's
``lite.src_hash`` — refusing a disk built from a different ISO or prep sources
than the current checkout expects. ``upstream_prep_image`` is deliberately NOT
part of the id: it names the base tool, not the disk bytes (the patches applied
to it live in the prep sources), so it must not gate the pull.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from image_provenance import DEFAULT_PREP_IMAGE, _prep_sha256

_ISO_SCRIPT = Path(__file__).resolve().parent / "download_windows_iso.sh"


def pinned_iso_sha256() -> str:
    m = re.search(r'EXPECTED_SHA="([0-9a-f]{64})"', _ISO_SCRIPT.read_text(encoding="utf-8"))
    if not m:
        raise SystemExit(f"could not find EXPECTED_SHA in {_ISO_SCRIPT}")
    return m.group(1)


def _mix(iso_sha: str, prep_sha: str) -> str:
    h = hashlib.sha256()
    h.update(iso_sha.encode())
    h.update(b"\0")
    h.update(prep_sha.encode())
    return h.hexdigest()


def checkout_disk_id() -> str:
    return _mix(pinned_iso_sha256(), _prep_sha256())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-sidecar", type=Path, help="print the disk id a sidecar implies")
    ap.add_argument("--write-sidecar", type=Path, help="stamp a pulled disk's provenance sidecar")
    ap.add_argument("--prep-image", default=DEFAULT_PREP_IMAGE)
    args = ap.parse_args()

    if args.from_sidecar:
        s = json.loads(args.from_sidecar.read_text(encoding="utf-8"))
        print(_mix(s["iso_sha256"], s["prep_source_sha256"]))
        return

    if args.write_sidecar:
        image = args.write_sidecar.expanduser().resolve()
        payload = {
            "image_size": image.stat().st_size,
            "iso_sha256": pinned_iso_sha256(),
            "prep_source_sha256": _prep_sha256(),
            "schema_version": 1,
            "upstream_prep_image": args.prep_image,
        }
        sidecar = Path(f"{image}.provenance.json")
        tmp = Path(f"{sidecar}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(sidecar)
        print(sidecar)
        return

    print(checkout_disk_id())


if __name__ == "__main__":
    main()
