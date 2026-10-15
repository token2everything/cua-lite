"""Canonical local format for cua-lite datasets.

A *staged* (a.k.a. canonical) dataset directory looks like:

    $CUA_LITE_DATASETS_ROOT/cua-lite/<DatasetName>/
        images/<hash[:2]>/<hash>.<ext>                       # CA image store
        <platform>/<task_type>/<split>/<variant>.parquet     # one file per variant
        README.md  (optional)
        stats.json (optional)

Every parquet row uses the canonical :class:`LiteSample` storage shape:

    {
        "images":   [<rel-path>, ...],          # rel to $CUA_LITE_DATASETS_ROOT
        "messages": [...],
        "metadata": {                            # mirrors lite.core metadata.to_dict()
            "metadata_kind": "cua",
            "dims":          ["desktop"|"browser"|"mobile",
                              "understanding"|"grounding.action"|"grounding.point"
                              |"grounding.bbox"|"use"],
            "extra_tool_schemas":   [],          # default; preserve when persisted tools require it
            "valid_actions": None,               # default; keep env-owned surface
            "others":        {...},             # dataset-specific keys
        },
    }

Both raw-data preproc adapters and the HF download tool produce this layout;
both the HF upload tool and ``lite.train.export.export_sft`` consume it.

For CUA metadata, ``dims[1]`` is the task-type literal
(``grounding.action``, etc.). It is filename-safe — ``.`` is allowed in path
components on every supported filesystem and isn't on HF's blacklist
(``<>:/\\|?*``) — so it doubles as the directory component verbatim. The CUA
``<platform>@<task_type>`` suffix is also the registry-key suffix produced by
``compose_key(agent_id, *metadata.dims)``.

``<split>`` ∈ {``train``, ``validation``}. Local parquets are never sharded
— pyarrow handles multi-GB files and there are no embedded image bytes
locally.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lite.utils.path import project_root


def prepare_output_dir(
    out_dir: Path | str,
    *,
    overwrite: bool = False,
    label: str = "output directory",
    protected_roots: tuple[Path | None, ...] = (),
) -> Path:
    """Create a fresh output directory or explicitly replace an old one.

    Dataset staging/download/filter tools write partitioned trees. Reusing a
    non-empty root can leave stale parquets or repo metadata from an earlier
    run, so callers must either point at a fresh directory or opt into replacing
    the whole tree.
    """
    out = Path(out_dir).resolve()
    for root in protected_roots:
        if root is None:
            continue
        protected = Path(root).resolve()
        if out == protected or out.is_relative_to(protected) or protected.is_relative_to(out):
            raise ValueError(
                f"{label} {out} must not overlap protected input root {protected}"
            )
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{label} {out} already exists and is not empty; "
                "pass --overwrite to replace it"
            )
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_artifact_path(path: str | Path, *, anchor_path: Path) -> Path:
    """Resolve a row artifact path using the producer path before cwd fallback."""
    artifact = Path(path)
    if artifact.is_absolute():
        return artifact

    candidates = []
    for base in [*anchor_path.resolve().parents, project_root(), Path.cwd()]:
        candidate = base / artifact
        if candidate.exists():
            candidates.append(candidate)
    unique = list(dict.fromkeys(candidate.resolve() for candidate in candidates))
    if len(unique) > 1:
        raise ValueError(
            f"ambiguous relative artifact path {str(path)!r}: found multiple matches "
            f"from anchor {anchor_path}: {unique}"
        )
    if unique:
        return unique[0]
    return Path.cwd() / artifact


def coerce_image_paths(images) -> list[str]:
    """Normalize a parquet row's image-path field to a list of strings."""
    value = to_plain(images)
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped.startswith("["):
            raise ValueError("images must be a list of image paths, not a bare string")
        value = json.loads(stripped)
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError("images must be a list of image paths")
    out: list[str] = []
    for path in value:
        if not isinstance(path, (str, Path)):
            raise ValueError("images must contain only path strings")
        out.append(str(path))
    return out


# ---------------------------------------------------------------------------
# Read-side coercion for current canonical rows.
#
# Current writers persist ``messages`` / ``metadata`` as opaque JSON strings
# (see ``serialize_opaque_json_fields``). Some in-process callers hand the already
# decoded canonical objects through the same helpers. These helpers parse those
# current shapes and normalize only lossless storage scalar drift such as
# integral float image indices.
#
# Legacy Arrow/HF struct materialization cleanup is owned by
# ``devs/migration``. Staging must not decide that ``None`` or ``[]`` is padding:
# in canonical JSON those values are producer evidence and downstream validators
# should see them unchanged.
# ---------------------------------------------------------------------------

def to_plain(obj):
    """Recursively convert numpy containers → plain python (json-serializable).

    No-op on values that are already plain (the pyarrow / JSON-string paths),
    so it is safe to apply unconditionally before JSON parsing.
    """
    if isinstance(obj, np.ndarray):
        return [to_plain(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def coerce_meta(md) -> dict:
    """Normalize a row's ``metadata`` (any of the three read shapes) → plain dict.

    Returns ``{}`` for falsy input (``None`` / missing / empty), matching every
    call site's prior ``… or {}`` guard.
    """
    md = to_plain(md)
    if not md:  # None / "" / {} → {} (short-circuit before json.loads, matching
        return {}  # export_sft's prior ``raw.get("metadata") or {}`` guard)
    if isinstance(md, str):
        md = json.loads(md)
    return md or {}


def coerce_messages(msgs) -> list:
    """Parse a canonical row's ``messages`` into a plain list.

    This is not a legacy parquet repair path. It preserves producer-visible
    keys and values other than lossless image-index scalar normalization.
    """
    msgs = to_plain(msgs)
    if isinstance(msgs, str):
        msgs = json.loads(msgs)
    if not isinstance(msgs, list):
        raise ValueError("messages JSON must decode to a list")
    return _canonicalize_image_refs(msgs)


def _canonicalize_image_refs(messages: list) -> list:
    """Normalize lossless image-index scalars, leaving message envelopes alone.

    Shared by both read shapes: image refs are storage-typed scalars, not
    provider evidence, so a ``0.0`` index must land as ``0`` whether the cell
    arrived as an Arrow struct or as an opaque JSON string.
    """
    if not isinstance(messages, list):
        return messages
    out = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            out.append(message)
            continue
        msg = dict(message)
        msg["content"] = [_canonicalize_content_part(part) for part in content]
        out.append(msg)
    return out


def _canonicalize_messages_for_write(messages):
    """Return messages in canonical write order without accepting old keys."""
    if not isinstance(messages, list):
        return messages
    out = []
    for message in _canonicalize_image_refs(messages):
        if not isinstance(message, dict):
            out.append(message)
            continue
        msg = dict(message)
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            msg["tool_calls"] = [
                _canonicalize_tool_call_for_write(call) for call in tool_calls
            ]
        if msg.get("role") == "tool" and "tool_call_id" in msg:
            ordered = {}
            for key in ("role", "tool_call_id", "content"):
                if key in msg:
                    ordered[key] = msg.pop(key)
            ordered.update(msg)
            msg = ordered
        out.append(msg)
    return out


def _canonicalize_tool_call_for_write(call):
    if not isinstance(call, dict) or "function" not in call:
        return call
    out = dict(call)
    function = out.get("function")
    if isinstance(function, dict):
        function_out = dict(function)
        ordered_function = {}
        for key in ("name", "arguments"):
            if key in function_out:
                ordered_function[key] = function_out.pop(key)
        ordered_function.update(function_out)
        out["function"] = ordered_function
    ordered = {}
    for key in ("id", "type", "function"):
        if key in out:
            ordered[key] = out.pop(key)
    ordered.update(out)
    return ordered


def _canonicalize_content_part(part):
    if not isinstance(part, dict):
        return part
    if part.get("type") != "image" or "index" not in part:
        return part
    out = dict(part)
    out["index"] = _coerce_storage_image_index(out["index"])
    return out


def _coerce_storage_image_index(value):
    """Coerce only mathematically integral float image refs introduced by storage.

    Private, but ONE approved importer lives outside this module:
    ``devs/data/utils.py`` (dev tooling, not shipped) reuses it so the storage
    index-coercion rule has a single spelling. A rename here is fine — it fails
    that import loudly — but a deletion is not: check that caller first.
    """
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return value


OPAQUE_JSON_FIELDS = ("messages", "metadata")


def serialize_opaque_json_fields(row: dict) -> dict:
    """Return *row* with transport-opaque columns serialized to JSON strings.

    ``messages`` and ``metadata`` carry heterogeneous tool calls, tool-result
    messages, and nested tool schemas. Keeping them as opaque JSON avoids Arrow /
    datasets schema inference adding nullable fields or reshaping nested calls:
    PyArrow infers one unified struct schema across all rows, so rows carrying
    tool calls with different argument shapes (click vs. type vs. key …) union
    into a struct with every possible field, missing ones filled with null, and
    numpy arrays inside arguments cast to their ``repr()`` string.
    """
    out = dict(row)
    for key in OPAQUE_JSON_FIELDS:
        if key in out and not isinstance(out[key], str):
            value = out[key]
            if key == "messages":
                value = _canonicalize_messages_for_write(value)
            out[key] = json.dumps(value, default=_json_default)
    return out


# ---------------------------------------------------------------------------
# Layout root resolvers
# ---------------------------------------------------------------------------

ORG = "cua-lite"


def dataset_root(name: str, *, root: Path | str | None = None) -> Path:
    """Absolute path to a staged dataset directory.

    Defaults to ``$CUA_LITE_DATASETS_ROOT/cua-lite/<name>``. Pass *root* to
    override (used by tests and by the HF round-trip tool).
    """
    if root is None:
        env = os.environ.get("CUA_LITE_DATASETS_ROOT")
        if not env:
            raise RuntimeError(
                "CUA_LITE_DATASETS_ROOT must be set, or pass root= explicitly"
            )
        root = env
    return Path(root) / ORG / name


def image_rel_prefix(name: str) -> str:
    """Path prefix written into row ``images`` lists for *name*.

    Rows store image paths relative to ``$CUA_LITE_DATASETS_ROOT`` so a
    multi-dataset SFT mix can use a single ``--image-root``.
    """
    return f"{ORG}/{name}/images"


# ---------------------------------------------------------------------------
# Image dedup
# ---------------------------------------------------------------------------

def sha256_of_file(path: Path, _buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_buf):
            h.update(chunk)
    return h.hexdigest()


class CorruptImageError(RuntimeError):
    """A source image failed Pillow decoding or verification."""


class ImageStore:
    """Content-addressed image store at ``<root>/<hash[:2]>/<hash>.<ext>``.

    Two ingest modes:

    * ``put(path)``   — file on disk; sha256 streamed; ext copied from
      the source filename.
    * ``put(blob, ext=...)`` — raw bytes; caller supplies the extension
      (HF download path; the upstream HF Image feature carries no filename).

    Returned path is relative to ``$CUA_LITE_DATASETS_ROOT`` if *rel_prefix*
    was set at construction (the default uses ``image_rel_prefix(name)``);
    otherwise returns the in-store rel path (``<hash[:2]>/<hash>.<ext>``).

    Thread-safe: ``put`` may be called concurrently. ``_seen`` is guarded by
    a lock; file writes are idempotent (same content → same destination).
    """

    def __init__(self, root: Path, *, rel_prefix: str | None = None, verify: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.rel_prefix = (rel_prefix or "").rstrip("/")
        self.verify = verify
        self._seen_files: dict[str, str] = {}  # abs source path -> rel path
        self._lock = threading.Lock()

    # ----- ingest -----

    def put(self, src: Path | str | bytes, *, ext: str | None = None) -> str:
        """Add *src* to the store and return the rel path used in row ``images``.

        For ``Path`` / ``str`` inputs, the file extension is taken from the
        source filename. For ``bytes``, ``ext`` is required (no leading dot).
        """
        if isinstance(src, (bytes, bytearray)):
            if not ext:
                raise ValueError("ext is required when src is bytes")
            return self._put_bytes(bytes(src), ext.lower().lstrip("."))
        return self._put_file(Path(src))

    def _put_file(self, src: Path) -> str:
        key = str(src)
        with self._lock:
            cached = self._seen_files.get(key)
        if cached:
            return cached
        if self.verify:
            from PIL import Image
            try:
                with Image.open(src) as img:
                    img.verify()
            except (OSError, SyntaxError, ValueError) as e:
                raise CorruptImageError(f"unreadable image: {src}") from e
        digest = sha256_of_file(src)
        ext = src.suffix.lower().lstrip(".") or "bin"
        rel = self._materialize(digest, ext, lambda dst: shutil.copyfile(src, dst))
        with self._lock:
            self._seen_files[key] = rel
        return rel

    def _put_bytes(self, blob: bytes, ext: str) -> str:
        digest = hashlib.sha256(blob).hexdigest()
        return self._materialize(digest, ext or "bin", lambda dst: dst.write_bytes(blob))

    def _materialize(self, digest: str, ext: str, write_fn: Callable[[Path], object]) -> str:
        in_store = f"{digest[:2]}/{digest}.{ext}"
        dst = self.root / in_store
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                dir=dst.parent, prefix=f".{digest}.", suffix=".tmp"
            )
            os.close(fd)
            tmp = Path(tmp_name)
            try:
                write_fn(tmp)
                os.replace(tmp, dst)
            finally:
                tmp.unlink(missing_ok=True)
        return f"{self.rel_prefix}/{in_store}" if self.rel_prefix else in_store

    def put_many(self, srcs: list[Path], *, max_workers: int = 32) -> list[str]:
        """Parallel ingest for I/O-bound workloads (e.g. NFS sources)."""
        if not srcs:
            return []
        if len(srcs) == 1:
            return [self.put(srcs[0])]
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(self.put, srcs))

    # ----- introspection -----

    def path_of(self, rel: str) -> Path:
        """Resolve a row-image rel path back to an absolute filesystem path.

        Accepts both forms: ``cua-lite/<name>/images/ab/abc.png`` (with prefix)
        and ``ab/abc.png`` (in-store only).
        """
        rel = rel.lstrip("/")
        if self.rel_prefix and rel.startswith(self.rel_prefix + "/"):
            rel = rel[len(self.rel_prefix) + 1:]
        return self.root / rel

    def count(self) -> int:
        return sum(1 for p in self.root.rglob("*") if p.is_file())

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def hash_split(key: str, *, val_frac: float = 0.02, seed: int = 42) -> str:
    """Deterministic split: ~val_frac of *key* values go to ``validation``."""
    h = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return "validation" if int(h[:8], 16) % 10_000 < int(val_frac * 10_000) else "train"


def content_fingerprint(row: dict) -> bytes:
    """Content identity of a canonical row: its ``images`` and its ``messages``.

    Two rows share a fingerprint exactly when they are the same *sample* —
    which is how upstream corpora re-publish one sample under two keys, so the
    metadata (``others.id`` / ``source_id``, the split key itself) is
    deliberately excluded. ``images`` has already been rewritten to the
    content-addressed store paths by the time a row reaches
    :meth:`SplitAssigner.assign`, so identical pixels compare equal without
    reading any bytes here.

    Deliberately NOT the image hash alone: one screen legitimately backs
    thousands of *distinct* rows, so an image-keyed fingerprint would treat a
    whole cohort as one duplicate group.
    """
    h = hashlib.sha1()
    h.update(json.dumps(row["images"], separators=(",", ":")).encode())
    h.update(b"\0")
    h.update(json.dumps(row["messages"], sort_keys=True, separators=(",", ":")).encode())
    return h.digest()


class SplitAssigner:
    """Assign ``train`` / ``validation`` per row, with a per-bucket cap.

    * ``canonical_fn(row)`` — if set and returns a non-None split label,
      that label is used (after passing through ``_UPSTREAM_SPLIT_MAP``).
      Use this when an upstream dataset ships its own canonical split *and* the
      adapter honors it; the adapter signals that by writing the transient
      ``metadata.others.split`` hint which ``SourceStaging.stage_entry`` reads
      through here and then strips. Exactly one adapter does so today —
      ``lite/data/preproc/guiact``, whose source files come as
      ``(path, split)`` pairs. Multimodal-Mind2Web, which this example used to
      name, is the counter-case rather than an instance: it *has* upstream
      splits and deliberately ignores them, reading only ``train`` (its 3
      ``test_*`` splits are benchmark holdouts, and ``test`` is reserved — see
      below), so it writes no hint and the hash split decides.
    * Otherwise, ``hash_split(key_fn(row))`` decides; raw row decisions assigned
      to validation after ``val_cap`` per ``bucket_fn(row) + bucket_extra`` fall
      back to train. Content co-location is applied afterwards, so the final
      physical validation row count may be above or below that decision cap.
    * Either way, rows that are the **same sample** land in the same split:
      the first member of a :func:`content_fingerprint` group fixes the answer
      for the group. Without this the split key is per-row while the *sample*
      is not, so an upstream corpus that publishes one sample under two ids
      hashes it into both splits and the held-out slice is not held out.
      Upstream's own split label does not save you either, which is why the
      co-location wraps ``canonical_fn`` too rather than only the hash.

    HF naming: we always emit ``train`` / ``validation``. ``test`` is reserved
    for separate out-of-distribution benchmark repositories.
    """

    # Map upstream split labels onto our two canonical names.
    _UPSTREAM_SPLIT_MAP: dict[str, str] = {
        "train": "train",
        "val": "validation",
        "validation": "validation",
        "dev": "validation",
        "test": "validation",
        "test_task": "validation",
        "test_website": "validation",
        "test_domain": "validation",
    }

    def __init__(
        self,
        *,
        key_fn: Callable[[dict], str],
        key_desc: str,
        bucket_fn: Callable[[dict], tuple],
        bucket_desc: str,
        canonical_fn: Callable[[dict], str | None] | None = None,
        val_frac: float = 0.02,
        val_cap: int = 2000,
        seed: int = 42,
    ):
        self.canonical_fn = canonical_fn
        self.key_fn = key_fn
        self.key_desc = key_desc
        self.bucket_fn = bucket_fn
        self.bucket_desc = bucket_desc
        self.val_frac = val_frac
        self.val_cap = val_cap
        self.seed = seed
        self._val_counts: dict[tuple, int] = {}
        self._group_split: dict[bytes, str] = {}

    def describe(self) -> str:
        """How this carve was produced, for a reader who has only the artifact.

        Recorded rather than recomputed, because nothing in a published row says
        which split it landed in — that is the partition path's job — so a
        consumer who wants to audit or reproduce the carve has no other source.
        ``key_desc`` / ``bucket_desc`` are required for the same reason:
        ``val_frac`` and ``seed`` are useless without knowing *what* was hashed,
        and a lambda cannot say.

        The last clause is the load-bearing one. Where ``val_cap`` binds, the
        validation decision budget is consumed by the first ``val_cap``
        hash-selected rows in source iteration order, so it is not a function of
        these parameters at all and any upstream reorder or filter change silently
        re-draws it. Content co-location can move physical rows after that decision,
        so the final validation row count alone cannot prove whether the cap bound.
        """
        return (
            f"split: content-identical rows co-located, then hash_split on {self.key_desc} "
            f"with val_frac={self.val_frac}, seed={self.seed}, "
            f"val_cap={self.val_cap} per {self.bucket_desc}"
            + (" (an upstream split label, where the source ships one, wins over the hash)"
               if self.canonical_fn is not None else "")
            + ". A cap-bound carve depends on source iteration order and these "
            "parameters do not reproduce it; content co-location may "
            "make the final physical validation row count differ from the cap."
        )

    def assign(self, row: dict, *, bucket_extra: tuple = ()) -> str:
        """The split for *row*, co-located with the rest of its content group.

        ``setdefault`` evaluates :meth:`_assign_unique` for **every** row, not
        just the first of a group, and that is load-bearing rather than
        incidental: the ``val_cap`` ledger therefore advances exactly as it
        would without co-location, so a row with no content twin cannot move.
        Only a later member of a group can get an answer other than its own.
        """
        return self._group_split.setdefault(
            content_fingerprint(row),
            self._assign_unique(row, bucket_extra=bucket_extra),
        )

    def _assign_unique(self, row: dict, *, bucket_extra: tuple) -> str:
        if self.canonical_fn is not None:
            raw = self.canonical_fn(row)
            if raw is not None:
                mapped = self._UPSTREAM_SPLIT_MAP.get(str(raw).lower())
                if mapped is not None:
                    return mapped
        split = hash_split(self.key_fn(row), val_frac=self.val_frac, seed=self.seed)
        if split == "validation":
            bucket = self.bucket_fn(row) + tuple(bucket_extra)
            count = self._val_counts.get(bucket, 0)
            if count >= self.val_cap:
                return "train"
            self._val_counts[bucket] = count + 1
        return split


# ---------------------------------------------------------------------------
# Path layout — single source of truth
#
# task_type values are themselves filename-safe (``grounding.action`` /
# ``grounding.point`` / ``grounding.bbox`` / ``understanding`` /
# ``use``), so they're used verbatim as path components. The string
# ``mobile.grounding.action`` is the same dotted shape as the HF config_name.
# ---------------------------------------------------------------------------

# The only two split names a canonical partition path may carry. ``test`` is
# reserved for separate out-of-distribution benchmark repositories (see
# ``SplitAssigner``), so it is deliberately absent.
CANONICAL_SPLITS: tuple[str, ...] = ("train", "validation")


def partition_path(
    root: Path,
    *,
    platform: str,
    task_type: str,
    split: str,
    variant: str,
    shard_idx: int | None = None,
    shard_total: int | None = None,
) -> Path:
    """Canonical output path for a single partition.

    Always ``<platform>/<task_type>/<split>/<variant>[…]``. The variant is part
    of the path because it is the only record of which producer wrote a file --
    without it a later run cannot tell whether it covers that file's rows, and
    must either publish two layouts or delete rows nobody rewrote. Sharded files
    use ``shard-NNNNN-of-NNNNN.parquet``; local writers always pass
    shard_idx/shard_total as None -- sharding is HF-only.
    """
    # ``variant`` reaches here from ``stage --config-names``, i.e. user input.
    # An empty one writes a dotfile ``iter_partitions`` skips; a separator writes
    # a depth ``parse_partition_path`` cannot read back, so the rows are invisible
    # to stats and to upload; ``..`` climbs out and lands on the collapsed shape
    # this layout exists to avoid. All three lose data silently, which is why the
    # check is here rather than at each caller.
    if not variant or "/" in variant or "\\" in variant or variant in (".", ".."):
        raise ValueError(
            f"variant must be a single non-empty path component, got {variant!r}"
        )
    base = root / platform / task_type / split
    if shard_idx is not None and shard_total is not None:
        return base / variant / f"shard-{shard_idx:05d}-of-{shard_total:05d}.parquet"
    # NOT ``with_suffix``: a variant may contain a dot (``cfg.rl`` from
    # ``stage --config-names``) and ``with_suffix`` would eat it as an extension.
    return base / f"{variant}.parquet"


def parse_partition_path(rel: Path) -> tuple[str, str, str, str | None, bool] | None:
    """Inverse of :func:`partition_path` for non-sharded files.

    Returns ``(platform, task_type, split, variant_or_None, collapsed)`` or
    ``None`` for unexpected layouts. Sharded files are not handled here —
    callers that read from HF snapshots merge shard groups first. The
    second path component is the task_type literal verbatim
    (``grounding.action``, ``understanding``, ``use``, …).
    """
    parts = rel.parts
    if len(parts) == 3:
        platform, task_type, fname = parts
        if not fname.endswith(".parquet"):
            return None
        split = Path(fname).stem
        if split not in CANONICAL_SPLITS:
            return None
        return platform, task_type, split, None, True
    if len(parts) == 4:
        platform, task_type, split, fname = parts
        if not fname.endswith(".parquet") or split not in CANONICAL_SPLITS:
            return None
        return platform, task_type, split, Path(fname).stem, False
    return None


def split_of_partition_file(rel: Path) -> str | None:
    """The canonical split *rel* sits in, or ``None`` if it carries no split.

    Reads the split marker wherever :func:`partition_path` may have written it: the
    parent directory for a variant or sharded partition, the file stem for a
    collapsed one (``<split>.parquet``). Unlike :func:`parse_partition_path` it fixes
    no number of leading components, so it also answers for a path taken relative to
    a *sub*directory of a dataset, and answers ``None`` for a file with no partition
    layout at all (a rollout ``trajectory.parquet``).

    The parent wins over the stem because a *variant* may be named ``train`` while
    the split above it is ``validation`` (``desktop/use/validation/train.parquet``);
    a ``task_type`` never can be, so the collapsed form stays unambiguous.
    """
    if rel.parent.name in CANONICAL_SPLITS:
        return rel.parent.name
    if rel.stem in CANONICAL_SPLITS:
        return rel.stem
    return None






# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------

def iter_parquet_rows(path: Path, batch_size: int = 64) -> Iterator[dict]:
    """Stream rows from a parquet file one-by-one.

    Uses ``iter_batches`` to dodge pyarrow's "Nested data conversions not
    implemented" error on parquets with deeply nested message schemas.
    """
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def _json_default(obj):
    """Handle numpy types inside tool_call arguments."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_partition(rows: list[dict], out_path: Path) -> None:
    """Write rows to a parquet file. Caller guarantees homogeneous schema."""
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = [serialize_opaque_json_fields(r) for r in rows]
    table = pa.Table.from_pylist(serialized)
    fd, tmp_name = tempfile.mkstemp(
        dir=out_path.parent, prefix=f".{out_path.name}.", suffix=".tmp"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        pq.write_table(table, tmp, compression="snappy")
        os.replace(tmp, out_path)
    finally:
        tmp.unlink(missing_ok=True)


def iter_partitions(staging_dir: Path) -> Iterator[tuple[str, str, str, str | None, Path]]:
    """Walk a staging dir, yielding ``(platform, task_type, split, variant, parquet_path)``.

    Skips ``images/``, dotfiles, README, stats. Only handles non-sharded
    paths (the canonical local layout). Sharded HF layouts are merged
    by ``hub_download`` before they reach this function.
    """
    if not staging_dir.is_dir():
        return
    # The local canonical layout has exactly three (collapsed) or four
    # (variant) components.  Do not ``rglob`` here: it descends through the
    # content-addressed image store before the later ``images`` guard can reject
    # it, turning a handful of partition lookups into millions of directory
    # entries on large datasets.
    candidates: list[Path] = []
    for platform_dir in staging_dir.iterdir():
        if (
            not platform_dir.is_dir()
            or platform_dir.name == "images"
            or platform_dir.name.startswith(".")
        ):
            continue
        candidates.extend(platform_dir.glob("*/*.parquet"))
        candidates.extend(platform_dir.glob("*/*/*.parquet"))
    for p in sorted(candidates):
        rel = p.relative_to(staging_dir)
        if any(seg.startswith(".") for seg in rel.parts):
            continue
        if rel.parts[0] == "images":
            continue
        parsed = parse_partition_path(rel)
        if parsed is None:
            continue
        platform, task_type, split, variant, _collapsed = parsed
        yield platform, task_type, split, variant, p


def _reject_legacy_web_platform_roots(
    staging_dir: Path,
    partitions: list[tuple[str, str, str, str | None, Path]],
) -> None:
    platforms = {platform for platform, *_ in partitions}
    if "web" in platforms:
        raise ValueError(
            f"{staging_dir} contains legacy web/ platform partitions. "
            "Regenerate those rows under browser/. For legacy flat sources, run "
            "devs/migration before staging/upload."
        )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class DatasetStats:
    rows_in: int = 0
    rows_out: int = 0
    rows_dropped: int = 0
    unique_images: int = 0
    image_store_bytes: int = 0
    by_partition: dict[tuple, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_dropped": self.rows_dropped,
            "unique_images": self.unique_images,
            "image_store_bytes": self.image_store_bytes,
            "by_partition": {
                "::".join(map(str, k)): v
                for k, v in sorted(self.by_partition.items())
            },
        }


def write_stats(out_dir: Path, stats: DatasetStats) -> None:
    """Persist a ``DatasetStats`` to ``<out_dir>/stats.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.json").write_text(json.dumps(stats.to_dict(), indent=2))


def collect_stats(buffers: dict[tuple, list[dict]], store: ImageStore) -> DatasetStats:
    """Build a ``DatasetStats`` from per-partition buffers + an image store."""
    s = DatasetStats()
    for key, rows in buffers.items():
        s.by_partition[key] = len(rows)
        s.rows_out += len(rows)
    s.unique_images = store.count()
    s.image_store_bytes = store.size_bytes()
    return s


def collect_stats_from_disk(staging_dir: Path) -> DatasetStats:
    """Build a ``DatasetStats`` by walking an existing staging directory.

    Uses parquet metadata for row counts (no row materialization) and the
    image store directory for image stats. Variant for collapsed cohorts
    falls back to the task_type subtype (e.g. ``grounding:point`` →
    ``point``) so the stats table always has a non-None variant column.
    """
    s = DatasetStats()
    partitions = list(iter_partitions(staging_dir))
    _reject_legacy_web_platform_roots(staging_dir, partitions)
    for platform, task_type, split, variant, parquet_path in partitions:
        n = pq.ParquetFile(parquet_path).metadata.num_rows
        v = variant or task_type.split(".")[-1]
        s.by_partition[(platform, task_type, split, v)] = (
            s.by_partition.get((platform, task_type, split, v), 0) + n
        )
        s.rows_out += n
    img_dir = staging_dir / "images"
    if img_dir.is_dir():
        files = [p for p in img_dir.rglob("*") if p.is_file()]
        s.unique_images = len(files)
        s.image_store_bytes = sum(p.stat().st_size for p in files)
    return s


# ---------------------------------------------------------------------------
# Adapter helper: flush per-partition buffers to disk
# ---------------------------------------------------------------------------

def flush_buffers(out_dir: Path, buffers: dict[tuple, list[dict]]) -> None:
    """Write each partition buffer to its canonical path under *out_dir*.


    Every partition is written EXPANDED, as ``<split>/<variant>.parquet``. The
    collapsed spelling was a single-variant convenience, but it does not record
    which variant produced ``<split>.parquet``, so a later run with a different
    variant set could neither keep it (two layouts published at once) nor remove
    it (rows nobody rewrote, deleted silently). With the variant always in the
    filename a run replaces exactly what it writes and nothing else.

    Adapters call this after they've buffered rows by
    ``(platform, task_type, split, variant)``.
    """
    # One scan before any write. A collapsed ``<split>.parquet`` from an older
    # layout cannot coexist with this one -- ``parse_partition_path`` accepts
    # both spellings, so ``iter_partitions`` would read the split twice and
    # publish duplicate rows. Its own rows are unattributable (the collapsed name
    # records no variant), so they can be neither merged nor safely dropped.
    # Scanning the WHOLE tree, not just the cohorts this run writes, is what
    # makes the check mean what it says; doing it first is what keeps a refusal
    # from leaving a half-written tree behind.
    platform_dirs = out_dir.iterdir() if out_dir.is_dir() else ()
    stale = sorted(
        p
        for platform_dir in platform_dirs
        if (
            platform_dir.is_dir()
            and platform_dir.name != "images"
            and not platform_dir.name.startswith(".")
        )
        for p in platform_dir.glob("*/*.parquet")
    )
    if stale:
        raise FileExistsError(
            f"{', '.join(str(p) for p in stale)} use the older collapsed layout, which "
            "records no variant and so cannot be merged with or safely replaced by "
            "this run. Delete them and rerun; the image store beside them is "
            "content-addressed and is reused, so nothing is re-hashed."
        )
    for (platform, task_type, split, variant), rows in buffers.items():
        if not rows:
            continue
        path = partition_path(
            out_dir,
            platform=platform,
            task_type=task_type,
            split=split,
            variant=variant,
        )
        write_partition(rows, path)


__all__ = [
    "ORG",
    "to_plain",
    "coerce_meta",
    "coerce_messages",
    "serialize_opaque_json_fields",
    "dataset_root",
    "image_rel_prefix",
    "CorruptImageError",
    "ImageStore",
    "hash_split",
    "SplitAssigner",
    "CANONICAL_SPLITS",
    "partition_path",
    "parse_partition_path",
    "split_of_partition_file",
    "prepare_output_dir",
    "resolve_artifact_path",
    "coerce_image_paths",
    "iter_parquet_rows",
    "write_partition",
    "iter_partitions",
    "DatasetStats",
    "write_stats",
    "collect_stats",
    "collect_stats_from_disk",
    "flush_buffers",
]
