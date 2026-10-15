#!/usr/bin/env python3
"""Build and verify the content-addressed WindowsAgentArena task asset cache.

Run: uv run python lite/gym/envs/waa/scripts/utils/assets.py \\
       verify --cache-dir ~/.cache/cua-lite/waa/assets   # or: install / refresh
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

ENV_DIR = Path(__file__).resolve().parents[2]
TASKS_PATH = ENV_DIR / "data" / "tasks.json"
MANIFEST_PATH = ENV_DIR / "data" / "assets.json"
WINARENAFILES_COMMIT = "2b8a27c3b561381e6f25cada0f97a8fd5ed95a75"

# The content-addressed blobs are re-hosted on cua-lite's own HF dataset
# (revision-pinned) for availability + reproducibility, mirroring lite.osworld —
# the upstream GitHub/Drive URLs in the manifest stay a per-asset fallback. Bump
# ASSETS_REV after re-uploading (see scripts/utils help / the dataset README).
ASSETS_REPO = "cua-lite/waa-assets"
ASSETS_REV = "30ad48d0f9338158d6357bde108e2c2f1e6ecc4a"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_asset_urls() -> list[str]:
    rows = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    urls: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str) and (
            "raw.githubusercontent.com/rogeriobonatti/winarenafiles/" in value
            or value.startswith("https://drive.google.com/uc?")
        ):
            urls.add(value)

    for row in rows:
        visit(row["config"])
    return sorted(urls)


def _immutable_url(source_url: str) -> str:
    prefix = "https://raw.githubusercontent.com/rogeriobonatti/winarenafiles/main/"
    if source_url.startswith(prefix):
        return source_url.replace(
            prefix,
            "https://raw.githubusercontent.com/rogeriobonatti/"
            f"winarenafiles/{WINARENAFILES_COMMIT}/",
            1,
        )
    return source_url


def _download(url: str, destination: Path) -> None:
    with httpx.stream(
        "GET",
        url,
        follow_redirects=True,
        timeout=httpx.Timeout(30.0, read=300.0),
        headers={"User-Agent": "cua-lite-waa/1"},
    ) as response:
        response.raise_for_status()
        with destination.open("wb") as stream:
            for chunk in response.iter_bytes():
                stream.write(chunk)


def _manifest_digest(manifest: dict[str, Any]) -> str:
    payload = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = _task_asset_urls()
    actual = sorted(entry["source_url"] for entry in manifest["assets"])
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(
            "WAA asset manifest is out of sync with tasks.json; "
            f"missing={missing}, extra={extra}. Run assets.py refresh."
        )
    return manifest


def _refresh(cache_dir: Path, workers: int) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    urls = _task_asset_urls()

    def fetch(source_url: str) -> dict[str, Any]:
        download_url = _immutable_url(source_url)
        fd, tmp_name = tempfile.mkstemp(prefix=".waa-asset-", dir=cache_dir)
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            _download(download_url, tmp)
            sha256 = _sha256(tmp)
            target = cache_dir / sha256
            if target.exists():
                tmp.unlink()
            else:
                tmp.replace(target)
            return {
                "source_url": source_url,
                "download_url": download_url,
                "sha256": sha256,
                "size": target.stat().st_size,
            }
        finally:
            tmp.unlink(missing_ok=True)

    entries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch, url): url for url in urls}
        for index, future in enumerate(as_completed(futures), start=1):
            entry = future.result()
            entries.append(entry)
            print(f"[{index}/{len(urls)}] {entry['source_url']}", flush=True)

    manifest = {
        "schema_version": 1,
        "winarenafiles_commit": WINARENAFILES_COMMIT,
        "assets": sorted(entries, key=lambda entry: entry["source_url"]),
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_complete(cache_dir, manifest)
    print(f"wrote {len(entries)} assets to {MANIFEST_PATH}")


def _install_one(entry: dict[str, Any], cache_dir: Path) -> int:
    target = cache_dir / entry["sha256"]
    if target.is_file() and target.stat().st_size == entry["size"]:
        if _sha256(target) == entry["sha256"]:
            return target.stat().st_size
        target.unlink()

    fd, tmp_name = tempfile.mkstemp(prefix=".waa-asset-", dir=cache_dir)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        _download(entry["download_url"], tmp)
        actual = _sha256(tmp)
        if actual != entry["sha256"]:
            raise RuntimeError(
                f"asset checksum mismatch for {entry['source_url']}: "
                f"expected {entry['sha256']}, got {actual}"
            )
        if tmp.stat().st_size != entry["size"]:
            raise RuntimeError(
                f"asset size mismatch for {entry['source_url']}: "
                f"expected {entry['size']}, got {tmp.stat().st_size}"
            )
        tmp.replace(target)
        return target.stat().st_size
    finally:
        tmp.unlink(missing_ok=True)


def _write_complete(cache_dir: Path, manifest: dict[str, Any]) -> None:
    complete = {
        "schema_version": 1,
        "manifest_sha256": _manifest_digest(manifest),
        "asset_count": len(manifest["assets"]),
        "total_bytes": sum(entry["size"] for entry in manifest["assets"]),
    }
    marker = cache_dir / ".complete.json"
    tmp = cache_dir / ".complete.json.tmp"
    tmp.write_text(
        json.dumps(complete, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(marker)


def _fetch_from_hf(cache_dir: Path) -> None:
    """Best-effort: pre-populate the cache with the content-addressed blobs from
    cua-lite's HF dataset (the fast, stable source). Never hard-fails — any blob
    that is missing/corrupt afterwards is re-fetched from its upstream URL by
    ``_install_one``, and the sha256 verification there is authoritative."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return
    try:
        snapshot_download(
            repo_id=ASSETS_REPO,
            repo_type="dataset",
            revision=ASSETS_REV,
            local_dir=str(cache_dir),
            ignore_patterns=["README.md", ".gitattributes"],
        )
    except Exception as exc:  # noqa: BLE001 — availability fallback, not integrity
        print(
            f"[assets] HF mirror unavailable ({type(exc).__name__}); using upstream URLs",
            flush=True,
        )


def _install(cache_dir: Path, workers: int) -> None:
    manifest = _load_manifest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    _fetch_from_hf(cache_dir)  # HF-first; _install_one falls back to upstream per asset
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_install_one, entry, cache_dir) for entry in manifest["assets"]]
        for index, future in enumerate(as_completed(futures), start=1):
            future.result()
            print(f"[assets] verified {index}/{len(futures)}", flush=True)
    _write_complete(cache_dir, manifest)


def _verify(cache_dir: Path) -> None:
    manifest = _load_manifest()
    marker_path = cache_dir / ".complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = _manifest_digest(manifest)
    if marker.get("manifest_sha256") != expected:
        raise RuntimeError(
            f"WAA asset cache marker is stale: {marker_path}. Run assets.py install."
        )
    expected_count = len(manifest["assets"])
    expected_bytes = sum(entry["size"] for entry in manifest["assets"])
    if marker.get("asset_count") != expected_count or marker.get("total_bytes") != expected_bytes:
        raise RuntimeError(
            f"WAA asset cache marker metadata is stale: {marker_path}. Run assets.py install."
        )
    seen: set[str] = set()
    for entry in manifest["assets"]:
        if entry["sha256"] in seen:
            continue
        seen.add(entry["sha256"])
        path = cache_dir / entry["sha256"]
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise RuntimeError(f"WAA asset missing or truncated: {path}")
        actual = _sha256(path)
        if actual != entry["sha256"]:
            raise RuntimeError(
                f"WAA asset checksum mismatch: {path}; expected {entry['sha256']}, got {actual}"
            )
    print(f"WAA asset cache verified: {expected_count} logical assets, {expected_bytes} bytes")


def _upload(cache_dir: Path) -> None:
    """Maintainers only: mirror the local content-addressed blobs to the HF dataset
    (needs HF_TOKEN with write to the cua-lite org). Prints the new revision to pin
    as ASSETS_REV. Workflow: refresh -> upload -> bump ASSETS_REV."""
    from huggingface_hub import CommitOperationAdd, HfApi

    blobs = sorted(
        p for p in cache_dir.iterdir()
        if p.is_file() and len(p.name) == 64 and all(c in "0123456789abcdef" for c in p.name)
    )
    if not blobs:
        raise SystemExit(f"no content-addressed blobs in {cache_dir}; run install/refresh first")
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(ASSETS_REPO, repo_type="dataset", private=False, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo=p.name, path_or_fileobj=str(p)) for p in blobs]
    api.create_commit(
        repo_id=ASSETS_REPO, repo_type="dataset", operations=ops,
        commit_message=f"Sync {len(blobs)} WAA asset blobs",
    )
    print(f"uploaded {len(blobs)} blobs; pin ASSETS_REV = {api.dataset_info(ASSETS_REPO).sha}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("refresh", "install", "verify", "upload"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    if args.command == "refresh":
        _refresh(cache_dir, args.workers)
    elif args.command == "install":
        _install(cache_dir, args.workers)
    elif args.command == "upload":
        _upload(cache_dir)
    else:
        _verify(cache_dir)


if __name__ == "__main__":
    main()
