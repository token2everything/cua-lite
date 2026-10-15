#!/usr/bin/env python3
"""Offline layers of the lite.cuaworld validation sweep — the generator for
``data/validation_excludes.json``.

    # dry run, report only (writes JSON reports under --report)
    uv run python lite/gym/envs/lite/cuaworld/scripts/utils/validation_sweep.py

    # regenerate the offline layers of data/validation_excludes.json in place
    uv run python lite/gym/envs/lite/cuaworld/scripts/utils/validation_sweep.py --write

    # one layer / one software, for triage
    uv run python .../validation_sweep.py --layers forged --software gvsig_desktop -v

WHY THIS EXISTS
---------------
The original sweep is a **no-op sweep**: reset a task (runs ``pre_task``), take no
agent action, run the verifier, and flag anything that does not score a clean 0.
It has two structural blind spots that the offline sweep cannot close by itself:

1. **Hollow artifacts.** A task that correctly scores 0 for "nothing happened" but
   scores a PASS the moment a plausible-looking but empty/garbage file appears at
   the expected output path. The no-op never creates that file, so it never looks.
2. **Setup aborts.** A task whose ``pre_task`` hook exits non-zero part-way
   through. Its expected agent-free baseline is 0 — and its actual score is also 0,
   because the episode starts from a half-built desktop. Invisible *by
   construction* to a sweep whose only signal is "baseline != 0".

Trajectory-backed augmentations in ``data/validation_excludes.json`` cover those
blind spots. After ``--write`` regenerates the offline layers, preserve or
reapply those trajectory-backed entries before committing the lock file.

This module adds one layer for each, both **fully offline and deterministic** (no
container, no image, no VLM, no network), so a re-run reproduces the same entries:

``forged``    Execute every registered verifier host-side against a synthesized
              "lazy agent" filesystem: every file the verifier asks the container
              for EXISTS, is non-trivial in size, carries a plausible extension and
              a fresh mtime, and is empty/garbage inside. A PASS means the
              deliverable's CONTENT is never checked.
``setup_rc``  Replay the ``pre_task`` hook in a throwaway root (user namespace +
              chroot, no network) whose ``/workspace`` is populated exactly the way
              ``install.sh`` stages it, and ASSERT rc == 0. A non-zero rc is a task
              that can never be solved.

The layer that needs a live container — the original no-op sweep — is NOT run
here. It is recorded in ``_meta._layers`` as an external layer with its own
per-software coverage, so a software that has never been swept is no longer
byte-identical to a clean one (``geogebra`` and ``qgis`` were exactly that: zero
entries, and ``src/software.py::_exclude_reasons`` does ``doc.items()``, so a
missing key silently yields ``{}``).

REGENERATE SEMANTICS
--------------------
``--write`` rewrites ONLY the reason codes this module owns (``_OFFLINE_REASONS``):
every existing entry carrying one of them is dropped, and the layers' fresh
findings are written back. Entries produced by the live no-op sweep are carried
through untouched. ``_meta._total`` is recomputed (a test asserts it matches).
"""
from __future__ import annotations

import argparse
import base64
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

# Importable as a script from anywhere: bootstrap sys.path by MARKER lookup, then
# hand repo-root duty to the shared helper (no `parents[N]` depth counting, which
# breaks silently the moment this file moves).
for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "pyproject.toml").is_file():
        sys.path.insert(0, str(_candidate))
        break

from lite.utils.path import project_root  # noqa: E402

REPO_ROOT = project_root()
ENV_DIR = REPO_ROOT / "lite/gym/envs/lite/cuaworld"
MATERIALS_DIR = ENV_DIR / ".cache"

from lite.gym.envs.lite.cuaworld.src.software import (  # noqa: E402
    VALIDATION_EXCLUDES_PATH as EXCLUDES_PATH,
    is_excludes_metadata_key,
)

#: Reason codes produced by the layers in this file. ``--write`` regenerates
#: exactly these and leaves every other code (the live no-op sweep's) alone.
FORGED_REASON = "hollow_artifact"
SETUP_ABORT_REASON = "setup_aborts"
_OFFLINE_REASONS = frozenset({FORGED_REASON, SETUP_ABORT_REASON})

LAYERS = ("forged", "setup_rc")

#: What ``scripts/install.sh::build`` copies into the image's ``/workspace``. A
#: hook that sources anything else under ``/workspace`` sources a file that does
#: not exist in ANY image — see openlca/chemical_reaction_stoichiometry, which
#: reads ``/workspace/utils/task_utils.sh`` while the on-disk ``utils/`` holds a
#: lone ``.gitkeep``.
STAGED_PAYLOAD = ("scripts", "data", "config", "assets")


# ─────────────────────────── task inventory ────────────────────────────────


def _registered_ids(registered_json: Path) -> dict[str, str]:
    """``task_id -> split`` for one env's ``registered.json``."""
    raw = json.loads(registered_json.read_text())
    splits: dict[str, list[str]] = {
        name: ids
        for name, ids in raw.items()
        if name != "additional_splits" and isinstance(ids, list)
    }
    for name, ids in (raw.get("additional_splits") or {}).items():
        splits.setdefault(name, ids)
    out: dict[str, str] = {}
    for split, ids in splits.items():
        for task_id in ids:
            out.setdefault(task_id, split)
    return out


def iter_tasks(software: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """Every REGISTERED task with an on-disk task dir, sorted, deterministic.

    The registered basis (3083 of the 3216 on-disk task dirs) is the one a rollout
    can actually sample, so it is the only basis an exclude entry can act on.
    """
    rows: list[dict[str, Any]] = []
    for registered_json in sorted(MATERIALS_DIR.glob("*/*/registered.json")):
        env_root = registered_json.parent
        sw = registered_json.relative_to(MATERIALS_DIR).parts[0]
        if software and sw not in software:
            continue
        for task_id, split in sorted(_registered_ids(registered_json).items()):
            task_dir = env_root / "tasks" / task_id
            if not task_dir.is_dir():
                continue
            try:
                spec = json.loads((task_dir / "task.json").read_text())
            except (OSError, ValueError):
                continue
            rows.append(
                {
                    "key": f"{sw}/{task_id}",
                    "software": sw,
                    "task_id": task_id,
                    "split": split,
                    "env_root": str(env_root),
                    "task_dir": str(task_dir),
                    "spec": spec,
                }
            )
    return rows


def _verifier_target(spec: dict) -> str | None:
    try:
        target = spec["success"]["spec"]["program"]
    except (KeyError, TypeError):
        return None
    return target if isinstance(target, str) and target.strip() else None


# ─────────────────────── layer 1: forged artifacts ─────────────────────────

#: A real 640x400 8-bit greyscale PNG, one flat mid-grey plane. Plausible
#: container, plausible dimensions, ZERO information — exactly the artifact a
#: lazy agent produces. Built here (not pasted as a blob) so it stays readable
#: and byte-identical on every run.
def _forged_png(width: int = 640, height: int = 400) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            len(payload).to_bytes(4, "big")
            + kind
            + payload
            + zlib.crc32(kind + payload).to_bytes(4, "big")
        )

    header = width.to_bytes(4, "big") + height.to_bytes(4, "big") + bytes((8, 0, 0, 0, 0))
    raw = (b"\x00" + b"\x80" * width) * height
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


_PNG = _forged_png()
#: Deterministic "garbage bytes" filler — the binary equivalent of
#: `head -c 2000 /dev/urandom > ~/my_turbine.wpa`, but reproducible.
_GARBAGE = bytes((i * 167 + 13) % 256 for i in range(4096))
_TEXT_FILLER = "".join(f"line {i}\n" for i in range(400))

_IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
)
_TEXT_SUFFIXES = frozenset(
    {".txt", ".csv", ".tsv", ".log", ".md", ".xml", ".svg", ".html", ".yaml",
     ".yml", ".ini", ".cfg", ".py", ".sh", ".sql", ".r", ".m", ".dat", ".out"}
)

#: Keys a TRUTHFUL exporter would still answer YES to for a hollow artifact: the
#: file is there, it is big enough, it was written during the episode. Every
#: SUBSTANTIVE key is deliberately absent so the verifier's OWN default applies.
#: Anchored on purpose — the loose first cut of this regex (which also matched
#: `found`, `present`, `valid`, `screenshot`, `count`, ...) answered YES to the
#: content questions too and ran at ~1% precision.
_TRIVIAL_KEY = re.compile(
    r"(?i)(^exists$|_exists$|^file_exists|is_file$|isfile$|created_during"
    r"|during_task|^mtime$|_mtime$|^modified$|^timestamp$|^task_start"
    r"|^start_time$|^size$|_size$|^bytes$|^file_created)"
)

_FUTURE_MTIME = 4_102_444_800.0  # 2100-01-01, deterministic "after task start"


def _trivial_value(key: str) -> Any:
    k = key.lower()
    if "mtime" in k or k in {"timestamp", "modified", "task_start", "start_time"}:
        return _FUTURE_MTIME
    if "size" in k or k == "bytes":
        return len(_GARBAGE)
    return True


class _HollowResult(dict):
    """The result JSON of a hollow artifact: answers ONLY the trivial keys."""

    def _hit(self, key: Any) -> bool:
        return isinstance(key, str) and bool(_TRIVIAL_KEY.search(key))

    def get(self, key: Any, default: Any = None) -> Any:
        return _trivial_value(key) if self._hit(key) else default

    def __getitem__(self, key: Any) -> Any:
        if self._hit(key):
            return _trivial_value(key)
        raise KeyError(key)

    def __contains__(self, key: Any) -> bool:
        return self._hit(key)

    def keys(self):  # noqa: ANN201 - dict protocol
        return []

    def items(self):  # noqa: ANN201 - dict protocol
        return []

    def values(self):  # noqa: ANN201 - dict protocol
        return []

    def __iter__(self):  # noqa: ANN204 - dict protocol
        return iter(())

    def __len__(self) -> int:
        return 1

    def __bool__(self) -> bool:
        return True


def _materialise(container_src: str, host_dst: Path) -> None:
    """Write the forged stand-in for ``container_src`` at ``host_dst``."""
    host_dst.parent.mkdir(parents=True, exist_ok=True)
    suffix = Path(container_src).suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        host_dst.write_bytes(_PNG)
    elif suffix == ".json":
        host_dst.write_text("{}")
    elif suffix in _TEXT_SUFFIXES or not suffix:
        host_dst.write_text(_TEXT_FILLER)
    else:
        host_dst.write_bytes(_GARBAGE)
    os.utime(host_dst, (time.time(), time.time()))


class _Pipe:
    """In-process stand-in for the adapter's verifier RPC pipes."""

    def __init__(self, handler=None) -> None:
        self._handler = handler
        self._queue: list[Any] = []

    def send(self, payload: Any) -> None:
        if self._handler is None:
            self._queue.append(payload)
        else:
            self._handler(payload)

    def recv(self) -> Any:
        return self._queue.pop(0)


#: The three corners of the probe.
#:
#: ``absent_no``  reproduces the no-op sweep's own input — nothing was written and
#:                the judge says no. Must score a clean 0; if it does not, the
#:                task is the LIVE layer's finding, not ours, and any forged pass
#:                read off it is not evidence.
#: ``forged_no``  the lazy agent's filesystem, judge still says no. A PASS here is
#:                the finding: no judge is involved, so the deliverable's CONTENT
#:                is provably never checked.
#: ``forged_yes`` same, but the judge is fooled into answering yes to everything.
#:                Reported as a DIAGNOSTIC only, never as an exclude. A pass that
#:                needs a fooled judge is not evidence of a hollow-artifact
#:                defect — it is the ordinary shape of a VLM-judged task (a few
#:                existence points programmatically, the rest from the judge), and
#:                treating it as a defect flags ~26% of the VLM-heavy softwares.
#:                The claim it rests on ("a real judge shown a blank 640x400 grey
#:                PNG answers yes") is exactly the thing we have no evidence for.
PROBE_MODES = ("absent_no", "forged_no", "forged_yes")


def _probe_child(task_dir: str, target: str, mode: str, workdir: str, out) -> None:
    try:
        out.put(_probe(Path(task_dir), target, mode, Path(workdir)))
    except Exception:  # noqa: BLE001 - report, never hang the pool
        out.put({"outcome": "harness_error", "detail": traceback.format_exc()[-400:]})


def _probe(task_dir: Path, target: str, mode: str, workdir: Path) -> dict:
    """Run ONE verifier through the adapter's own worker against forged inputs.

    Fidelity comes from reusing ``adapter._verifier_worker`` / ``_load_verifier``
    / ``_load_trajectory`` verbatim: the verifier sees exactly the ``env_info``,
    ``traj`` and ``task_info`` shapes a real episode gives it. Only the three
    container-facing edges are synthesized — file reads, shell reads, and the
    judge.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from lite.gym.envs.lite.cuaworld.src import adapter, vlm

    forge, judge_says_yes = mode.split("_")[0] == "forged", mode.endswith("_yes")
    workdir.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(workdir)
    os.chdir(workdir)

    frames = []
    for index in range(3):
        frame = workdir / f"frame_{index:05d}.png"
        frame.write_bytes(_PNG)
        frames.append(frame)
    (workdir / "final.png").write_bytes(_PNG)
    (workdir / "post_verification.png").write_bytes(_PNG)

    # ── the judge ───────────────────────────────────────────────────────────
    class _Verdict(dict):
        def get(self, key: Any, default: Any = None) -> Any:
            if not judge_says_yes:
                return False if isinstance(default, bool) or default is None else default
            k = str(key).lower()
            if any(w in k for w in ("score", "count", "num", "rating", "pct")):
                return 100
            if "confidence" in k:
                return "high"
            if any(w in k for w in ("reason", "explanation", "text", "detail")):
                return "yes"
            return True

        def __getitem__(self, key: Any) -> Any:
            return self.get(key)

        def __contains__(self, key: Any) -> bool:
            return True

        def keys(self):  # noqa: ANN201 - dict protocol
            return []

        def items(self):  # noqa: ANN201 - dict protocol
            return []

        def __bool__(self) -> bool:
            return True

    def query_vlm(prompt=None, images=None, image=None, **kwargs):  # noqa: ANN001
        if any(kwargs.get(k) for k in ("return_json", "json_response", "output_schema")):
            return _Verdict()
        return {
            "success": True,
            "response": "yes" if judge_says_yes else "no",
            "parsed": _Verdict(),
            "error": "",
        }

    vlm.query_vlm = query_vlm
    vlm.parse_vlm_json = lambda _text: _Verdict()
    vlm.sample_trajectory_frames = lambda *a, **k: [str(p) for p in frames]
    vlm.get_final_screenshot = lambda *a, **k: str(workdir / "final.png")
    vlm.get_first_screenshot = lambda *a, **k: str(frames[0])

    # ── the container edges ─────────────────────────────────────────────────
    forged_paths: set[str] = set()
    asked_for: list[str] = []
    commands: list[str] = []

    def handle(message: Any) -> None:
        request_id, operation, args = message
        if operation == "copy_from_env":
            container_src, host_dst = args
            asked_for.append(str(container_src))
            if not forge:
                # What the real bridge does for a path that is not there: the
                # RPC fails and the worker's `rpc` raises.
                responses.send(
                    (request_id, False, f"FileNotFoundError: {container_src}")
                )
                return
            _materialise(str(container_src), Path(host_dst))
            forged_paths.add(str(Path(host_dst).resolve()))
            value = None
        elif operation == "copy_to_env":
            value = None
        elif operation == "run_command":
            # A hollow artifact answers nothing on the shell channel; forging
            # command OUTPUT would be a guess about the app, not about the
            # deliverable. Under-reports exec-only verifiers on purpose.
            commands.append(str(args[0])[:160])
            value = ""
        else:  # pragma: no cover - the adapter has no other operation
            raise RuntimeError(operation)
        responses.send((request_id, True, value))

    requests = _Pipe(handle)
    responses = _Pipe()
    results = _Pipe()

    # ── the result-JSON edge ────────────────────────────────────────────────
    # Verifiers copy the exporter's result JSON out and read it. Answer ONLY the
    # trivial keys, and ONLY for the files we actually forged — shimming every
    # json.load in the module (the first cut) corrupted the verifier's own config
    # reads and produced a wave of bogus `'bool' object is not iterable` errors.
    real_load_verifier = adapter._load_verifier

    def load_verifier(path: Path, verifier_target: str):  # noqa: ANN202
        func = real_load_verifier(path, verifier_target)
        namespace = getattr(func, "__globals__", {})
        real_json = namespace.get("json")
        if real_json is None or not hasattr(real_json, "load"):
            return func
        import types

        shim = types.ModuleType("json")
        for attribute in dir(real_json):
            try:
                setattr(shim, attribute, getattr(real_json, attribute))
            except (AttributeError, TypeError):
                pass

        def load(fp, *a, **k):  # noqa: ANN001, ANN202
            name = getattr(fp, "name", None)
            if isinstance(name, str) and str(Path(name).resolve()) in forged_paths:
                return _HollowResult()
            return real_json.load(fp, *a, **k)

        def loads(text, *a, **k):  # noqa: ANN001, ANN202
            if isinstance(text, (str, bytes)) and text.strip() in ("{}", b"{}"):
                return _HollowResult()
            return real_json.loads(text, *a, **k)

        shim.load = load
        shim.loads = loads
        namespace["json"] = shim
        return func

    adapter._load_verifier = load_verifier

    traj = adapter._load_trajectory(workdir)
    spec = json.loads((task_dir / "task.json").read_text())
    task_info = {
        "task_id": spec.get("id", task_dir.name),
        "metadata": spec.get("metadata") or {},
        "task_spec": spec,
        "_env_id": "validation-sweep",
        "_lite_env_id": "validation-sweep",
        "_container": None,
    }

    started = time.monotonic()
    adapter._verifier_worker(
        str(task_dir / "verifier.py"),
        target,
        traj,
        task_info,
        str(workdir),
        requests,
        responses,
        results,
    )
    payload = results.recv()
    elapsed = round(time.monotonic() - started, 2)
    if "error" in payload:
        kind = "load_error" if payload["error"].startswith("load:") else "verifier_error"
        return {"outcome": kind, "detail": payload["error"][:200], "secs": elapsed}
    result = payload["result"]
    if not isinstance(result, dict):
        return {"outcome": "bad_return", "detail": repr(result)[:150], "secs": elapsed}
    raw = result.get("score", result.get("raw_score"))
    passed = bool(result.get("passed"))
    reward, reward_type = adapter._final_reward(
        spec, raw_score=float(raw) if isinstance(raw, (int, float)) else (100.0 if passed else 0.0),
        passed=passed,
    )
    return {
        "outcome": "ok",
        "passed": passed,
        "score": raw,
        "reward": reward,
        "reward_type": reward_type,
        "feedback": str(result.get("feedback"))[:400],
        "asked_for": asked_for[:8],
        "commands": commands[:6],
        "secs": elapsed,
    }


def _run_probe(job: tuple[str, str, str, str]) -> dict:
    task_dir, target, mode, key = job
    workdir = Path(tempfile.mkdtemp(prefix="cuaworld-forged-"))
    queue: Any = mp.Queue()
    process = mp.Process(
        target=_probe_child, args=(task_dir, target, mode, str(workdir), queue)
    )
    process.start()
    try:
        result = queue.get(timeout=60)
    except Exception:  # noqa: BLE001 - empty queue == the child died or hung
        result = {"outcome": "timeout"}
    finally:
        process.join(2)
        if process.is_alive():
            process.terminate()
            process.join(2)
            if process.is_alive():
                process.kill()
        shutil.rmtree(workdir, ignore_errors=True)
    result["key"] = key
    result["mode"] = mode
    return result


def run_forged_layer(tasks: list[dict], jobs: int) -> dict[str, dict]:
    """``key -> {mode: result}`` over every registered verifier, all four modes."""
    from concurrent.futures import ThreadPoolExecutor

    queue: list[tuple[str, str, str, str]] = []
    for row in tasks:
        target = _verifier_target(row["spec"])
        if target is None or not (Path(row["task_dir"]) / "verifier.py").is_file():
            continue
        for mode in PROBE_MODES:
            queue.append((row["task_dir"], target, mode, row["key"]))

    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for result in pool.map(_run_probe, queue):
            out.setdefault(result["key"], {})[result["mode"]] = result
    return out


# ─────────────────────── layer 2: pre_task rc assertion ────────────────────

#: Commands stubbed to rc 0 inside the replay. Two families, both on purpose:
#: the app/desktop binaries that only exist in the image (an abort there would
#: say "openLCA is not installed on the host", not "this hook is broken"), and
#: the privileged/interpreted ones whose host behaviour cannot match the
#: container's. Everything ELSE runs for real — bash builtins, coreutils, grep,
#: awk, du — which is what makes the shell control flow authentic. Bash's
#: `command_not_found_handle` stubs every remaining unknown binary the same way,
#: so the replay can only ever UNDER-report an abort.
_STUBBED = (
    "chown chgrp su sudo systemctl service apt apt-get dpkg snap flatpak "
    "python python3 pip pip3 java xdotool wmctrl xrandr xdpyinfo xset scrot "
    "import gnome-screenshot ffmpeg convert wget curl git ssh scp sleep "
    "pkill killall setsid nohup notify-send dbus-send gsettings"
).split()

_REPLAY_DRIVER = r"""
set -u
ROOT="$1"; PLAN="$2"; OUT="$3"
mkdir -p "$ROOT"/{proc,dev,tmp,home,workspace,opt,root,run,var,etc,usr,bin,sbin,lib,lib64,stub}
for d in usr bin sbin lib lib64 etc var run; do
  [ -e "/$d" ] || continue
  mount --bind "/$d" "$ROOT/$d" 2>/dev/null || continue
  mount -o remount,ro,bind "$ROOT/$d" 2>/dev/null || true
done
mount -t proc proc "$ROOT/proc" 2>/dev/null || true
mount --rbind /dev "$ROOT/dev" 2>/dev/null || true

for stub in __STUBS__; do
  printf '#!/bin/sh\nexit 0\n' > "$ROOT/stub/$stub"
  chmod 755 "$ROOT/stub/$stub"
done

: > "$OUT"
while IFS=$'\t' read -r task payload hook; do
  # A fresh, empty desktop for every task: /home /tmp /opt are throwaway tmpfs.
  for m in home tmp opt; do
    mountpoint -q "$ROOT/$m" && umount -l "$ROOT/$m"
    mount -t tmpfs tmpfs "$ROOT/$m"
  done
  mkdir -p "$ROOT/home/ga/Desktop" "$ROOT/home/ga/Documents"
  # /workspace exactly as scripts/install.sh stages it.
  mountpoint -q "$ROOT/workspace" && umount -l "$ROOT/workspace"
  mount -t tmpfs tmpfs "$ROOT/workspace"
  for d in __PAYLOAD__; do
    [ -d "$payload/$d" ] || continue
    mkdir -p "$ROOT/workspace/$d"
    mount --bind "$payload/$d" "$ROOT/workspace/$d" 2>/dev/null || true
  done
  [ -f "$payload/env.json" ] && cp "$payload/env.json" "$ROOT/workspace/env.json"
  mkdir -p "$ROOT/tmp/cuaworld_task_sources"
  cp -r "$payload/../__GUARDED__/$task" "$ROOT/tmp/cuaworld_task_sources/$task"
  chmod -R 755 "$ROOT/tmp/cuaworld_task_sources" 2>/dev/null || true
  printf '%s\n' \
    'export DISPLAY=:1' \
    'export PATH=/stub:/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin' \
    'command_not_found_handle() { return 0; }' \
    'export -f command_not_found_handle' \
    "$hook" > "$ROOT/tmp/cuaworld_setup.sh"
  timeout -k 5 45 chroot "$ROOT" /bin/bash /tmp/cuaworld_setup.sh \
    > "$ROOT/tmp/.sweep.out" 2>&1
  rc=$?
  printf '%s\t%s\t%s\n' "$task" "$rc" \
    "$(tr '\n\t' '  ' < "$ROOT/tmp/.sweep.out" | tail -c 400)" >> "$OUT"
done < "$PLAN"
"""


#: `source`/`.` of an absolute `/workspace/<top>/…` path.
_WORKSPACE_SOURCE = re.compile(
    r"(?m)^[ \t]*(?:source|\.)[ \t]+(/workspace/([A-Za-z0-9_.-]+)[^\s;&|]*)"
)


def _unstageable_source(task_dir: Path) -> str | None:
    """The `/workspace/…` path this task's setup sources that NO image can have.

    `install.sh::build` assembles the whole of `/workspace` from `STAGED_PAYLOAD`
    plus `env.json`; anything else under `/workspace` is absent from every image
    ever built, in every software, forever. That makes this the one abort cause
    that is decidable from the checkout alone and INDEPENDENT of what the image
    bakes — which is exactly the property the replay itself lacks (see
    `run_setup_rc_layer`), so it is what promotes a replay rc into a finding.
    """
    hook = task_dir / "setup_task.sh"
    if not hook.is_file():
        return None
    text = hook.read_text(encoding="utf-8", errors="replace")
    for path, top in _WORKSPACE_SOURCE.findall(text):
        if top not in STAGED_PAYLOAD and top != "env.json":
            return path
    return None


def _sandbox_available() -> bool:
    if not shutil.which("unshare"):
        return False
    probe = subprocess.run(
        ["unshare", "--map-root-user", "--mount", "--net", "--pid", "--fork",
         "sh", "-c", "mount -t tmpfs tmpfs /home && echo ok"],
        capture_output=True, text=True,
    )
    return probe.stdout.strip().endswith("ok")


def _pre_task_hook(row: dict) -> str | None:
    """The command the adapter runs for ``pre_task``, with the task-source path
    rewritten exactly as ``_run_task_hook`` rewrites it."""
    from lite.gym.envs.lite.cuaworld.src.adapter import _task_hook

    task_id = row["task_id"]
    hook = _task_hook(row["spec"], "pre_task")
    if hook is None:
        legacy = Path(row["task_dir"]) / "setup_task.sh"
        return f"bash /tmp/cuaworld_task_sources/{task_id}/setup_task.sh" if legacy.is_file() else None
    return hook.replace(
        f"/workspace/tasks/{task_id}", f"/tmp/cuaworld_task_sources/{task_id}"
    )


def _stage_guarded_sources(task_dir: Path, destination: Path) -> None:
    """Copy a task dir through the SAME normalizations ``_run_task_hook`` applies
    on upload, so the replay runs the bytes the container would run.

    This is what keeps the layer honest as the adapter's guards grow: an abort the
    adapter already neutralizes (the `task_utils.sh` non-fatal `return 1` class,
    say) disappears from the findings by construction, instead of being excluded
    forever on the strength of a stale reproduction.
    """
    from lite.gym.envs.lite.cuaworld.src.adapter import _guard_hook_body, _hook_helpers

    helpers = _hook_helpers(task_dir)
    source_root = f"/workspace/tasks/{task_dir.name}"
    remote_root = f"/tmp/cuaworld_task_sources/{task_dir.name}"
    for source in sorted(task_dir.rglob("*")):
        if not source.is_file():
            continue
        target = destination / source.relative_to(task_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        if source.suffix == ".sh":
            try:
                text = data.decode()
            except UnicodeDecodeError:
                pass
            else:
                data = (
                    _guard_hook_body(text, helpers)
                    .replace(source_root, remote_root)
                    .encode()
                )
        target.write_bytes(data)
        target.chmod(0o755)


def run_setup_rc_layer(tasks: list[dict], workroot: Path) -> dict[str, dict]:
    """``key -> {"rc", "tail", "unstageable_source"}`` for every registered task.

    THE RC IS A CANDIDATE, NOT A VERDICT — and the reason is worth stating in full,
    because the obvious reading of this layer is wrong. The replay root is an EMPTY
    desktop: bash, coreutils and the staged `/workspace` are real, but nothing the
    software's own image bakes is there. Measured over the pinned materials, 1451 of
    3083 hooks exit non-zero here, and almost all of them are that absence talking —
    `ERROR: Source data missing: /opt/fits_samples/…` (astroimagej), `WARNING:
    Eclipse not detected` under `set -e`, `ERROR: GeoGebra not found!`. Every one of
    those runs fine in the real image. Excluding on a bare rc would delete a third of
    the corpus on the strength of a sandbox artifact.
    So `_classify` promotes an rc to a finding only when it is corroborated by a
    cause that is decidable WITHOUT the image (`_unstageable_source`).
    To make the rc itself trustworthy, run the same replay inside
    `cua-lite/lite.cuaworld.<software>` instead of this chroot — the driver is
    already just "populate /workspace, run the guarded hook, record rc", so only the
    root changes. That is the natural home for a real pre_task rc ASSERTION, and it
    belongs beside the live no-op sweep (which already has the container open).
    """
    by_env: dict[str, list[dict]] = {}
    for row in tasks:
        by_env.setdefault(row["env_root"], []).append(row)

    out: dict[str, dict] = {}
    driver = (
        _REPLAY_DRIVER.replace("__STUBS__", " ".join(_STUBBED))
        .replace("__PAYLOAD__", " ".join(STAGED_PAYLOAD))
        .replace("__GUARDED__", "_sweep_guarded")
    )
    driver_path = workroot / "replay.sh"
    driver_path.write_text(driver)

    for env_root, rows in sorted(by_env.items()):
        payload = Path(env_root)
        # Staged INSIDE the env's `.cache/` (gitignored) so the chroot can bind it,
        # and cleared up front as well as at the end: a run killed mid-env would
        # otherwise leave a `_sweep_guarded` tree behind in the materials.
        guarded_root = payload.parent / "_sweep_guarded"
        shutil.rmtree(guarded_root, ignore_errors=True)
        plan_lines: list[str] = []
        for row in rows:
            hook = _pre_task_hook(row)
            if hook is None:
                continue
            _stage_guarded_sources(Path(row["task_dir"]), guarded_root / row["task_id"])
            plan_lines.append(f"{row['task_id']}\t{payload}\t{hook}")
        if not plan_lines:
            continue
        plan = workroot / f"plan-{payload.name}.tsv"
        plan.write_text("\n".join(plan_lines) + "\n")
        result_path = workroot / f"rc-{payload.name}.tsv"
        root = Path(tempfile.mkdtemp(prefix="cuaworld-replay-", dir=workroot))
        subprocess.run(
            ["unshare", "--map-root-user", "--mount", "--net", "--pid", "--fork",
             "--kill-child", "bash", str(driver_path), str(root), str(plan),
             str(result_path)],
            capture_output=True, text=True,
        )
        software = rows[0]["software"]
        # A hook's own stdout is arbitrary bytes (openrocket dumps a jar banner).
        raw = (
            result_path.read_text(encoding="utf-8", errors="replace")
            if result_path.is_file()
            else ""
        )
        for line in raw.splitlines():
            task_id, _, rest = line.partition("\t")
            rc_text, _, tail = rest.partition("\t")
            out[f"{software}/{task_id}"] = {
                "rc": int(rc_text) if rc_text.isdigit() else 0,
                "tail": "".join(c for c in tail if c.isprintable())[-300:],
                "unstageable_source": _unstageable_source(
                    payload / "tasks" / task_id
                ),
            }
        shutil.rmtree(root, ignore_errors=True)
        if guarded_root.exists():
            shutil.rmtree(guarded_root, ignore_errors=True)
    return out


# ───────────────────────────── the exclude file ─────────────────────────────


def live_layer_excludes() -> dict[str, str]:
    """``key -> reason`` for entries the live no-op sweep owns (not ours)."""
    document = json.loads(EXCLUDES_PATH.read_text())
    return {
        f"{software}/{task}": reason
        for software, entries in document.items()
        if not is_excludes_metadata_key(software)
        for task, reason in entries.items()
        if reason not in _OFFLINE_REASONS
    }


def _classify(
    forged: dict[str, dict],
    setup_rc: dict[str, dict],
    layers: tuple[str, ...],
) -> dict[str, str]:
    """``key -> reason`` for everything the offline layers found.

    A task the live no-op sweep already flagged is skipped, not relabelled: its
    reason carries more information than ours would (an unconditional-pass stub
    is `gameable_full` whether or not a hollow file exists), and clobbering it
    would silently rewrite the live layer from an offline run.
    """
    already = live_layer_excludes()
    findings: dict[str, str] = {}
    if "setup_rc" in layers:
        for key, result in sorted(setup_rc.items()):
            if result["rc"] and result["unstageable_source"] and key not in already:
                findings[key] = SETUP_ABORT_REASON
    if "forged" in layers:
        for key, modes in sorted(forged.items()):
            if key in findings or key in already:
                continue  # a task that never starts is not "gameable"
            if set(modes) != set(PROBE_MODES):
                continue
            baseline = modes["absent_no"]
            # The no-op baseline must be a clean, scoring 0 — anything else is
            # the LIVE sweep's finding, and reading a forged pass off a verifier
            # that already errors or already passes is not evidence.
            if baseline["outcome"] != "ok" or baseline["passed"]:
                continue
            if modes["forged_no"]["outcome"] == "ok" and modes["forged_no"]["passed"]:
                findings[key] = FORGED_REASON
    return findings


def judge_gated_passes(forged: dict[str, dict]) -> list[str]:
    """Diagnostic: tasks that pass on a hollow artifact ONLY once the judge is
    fooled. Deliberately NOT an exclude — see ``PROBE_MODES``."""
    already = live_layer_excludes()
    return sorted(
        key
        for key, modes in forged.items()
        if set(modes) == set(PROBE_MODES)
        and key not in already
        and modes["absent_no"]["outcome"] == "ok"
        and not modes["absent_no"]["passed"]
        and not modes["forged_no"].get("passed")
        and modes["forged_yes"]["outcome"] == "ok"
        and modes["forged_yes"]["passed"]
    )


#: The software the LIVE no-op sweep (reset + verify in a real container, the layer
#: this module cannot run) has actually been run over, and the reason codes it owns.
#: A CONSTANT, not something derived from the file: a software with zero findings and
#: a software that was never swept both look like `{}` in the data, and that
#: indistinguishability is the bug — `geogebra` and `qgis` sat outside the file
#: entirely, and `_exclude_reasons` does `doc.items()`, so a missing key silently
#: yields `{}`. Extend this list in the same change that runs the live sweep on a new
#: software; `--write` republishes it into `_meta._layers` on every regenerate.
LIVE_NOOP_SOFTWARE = (
    "ardour astroimagej blender3d coppeliasim dbeaver diagrams_net eclipse freecad "
    "gcompris gmat gpredict gretl gvsig_desktop hec_ras imagej jstock knime "
    "kstars_sim librecad libreoffice_calc moodle odoo openemr openlca openrocket "
    "openvsp pycharm pymol qblade slicer3d solvespace sumo sweet_home_3d ugene "
    "vlc_media_player vscode webots wordpress"
).split()
LIVE_NOOP_REASONS = (
    "nonzero_baseline missing_verifier gameable_full "
    "broken_export_query_live_instance verifier_runtime_pip verifier_crash "
    "verifier_nameerror verifier_syntaxerror slow_timeout verifier_pdb other"
).split()


def write_excludes(findings: dict[str, str], swept: list[str], layers: tuple[str, ...]) -> dict:
    """Regenerate the offline layers in place; carry the live layer through."""
    document = json.loads(EXCLUDES_PATH.read_text())
    metadata = {k: v for k, v in document.items() if is_excludes_metadata_key(k)}
    meta = metadata.setdefault("_meta", {})
    swept_set = set(swept)
    stale = _regenerated(layers)

    table: dict[str, dict[str, str]] = {
        software: {
            task: reason
            for task, reason in entries.items()
            if not (software in swept_set and reason in stale)
        }
        for software, entries in document.items()
        if not is_excludes_metadata_key(software)
    }
    for software in swept_set:
        table.setdefault(software, {})
    for key, reason in findings.items():
        software, _, task = key.partition("/")
        table.setdefault(software, {})[task] = reason

    ordered = {
        software: dict(sorted(entries.items()))
        for software, entries in sorted(table.items())
    }
    meta["_total"] = sum(len(entries) for entries in ordered.values())
    layer_meta = meta.setdefault("_layers", {})
    layer_meta["live_noop"] = {
        "runs": "reset + verify in a real container, no agent action, no VLM",
        "reasons": sorted(LIVE_NOOP_REASONS),
        "software": sorted(LIVE_NOOP_SOFTWARE),
        "never_swept": sorted(set(ordered) - set(LIVE_NOOP_SOFTWARE)),
    }
    for layer in layers:
        layer_meta[layer] = {
            "runs": "scripts/utils/validation_sweep.py --layers " + layer,
            "reasons": sorted(_regenerated((layer,))),
            "software": sorted(swept_set),
            "never_swept": sorted(set(ordered) - swept_set),
        }
    EXCLUDES_PATH.write_text(json.dumps({**metadata, **ordered}, indent=2) + "\n")
    return {**metadata, **ordered}


def _regenerated(layers: tuple[str, ...]) -> frozenset[str]:
    reasons: set[str] = set()
    if "forged" in layers:
        reasons.add(FORGED_REASON)
    if "setup_rc" in layers:
        reasons.add(SETUP_ABORT_REASON)
    return frozenset(reasons)


# ──────────────────────────────── driver ────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layers", default=",".join(LAYERS),
                        help=f"comma-separated subset of {LAYERS}")
    parser.add_argument("--software", default="", help="comma-separated filter")
    parser.add_argument("--jobs", type=int, default=24)
    parser.add_argument("--report", default=".data/validation_sweep")
    parser.add_argument(
        "--reuse", default="",
        help="report dir of an earlier run: load `setup_rc.json` from it instead of "
             "replaying. The replay is ~9 min of serial chroot work and its rc is "
             "input-independent, so re-running it to re-derive the same rcs is pure "
             "cost. `unstageable_source` — the only part that decides a finding — is "
             "always recomputed from the checkout, so a stale cache cannot change a "
             "verdict, only the diagnostic tail it is reported with.",
    )
    parser.add_argument("--write", action="store_true",
                        help="regenerate the offline layers of validation_excludes.json")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    layers = tuple(name for name in args.layers.split(",") if name)
    unknown = set(layers) - set(LAYERS)
    if unknown:
        parser.error(f"unknown layer(s): {sorted(unknown)}")
    software = tuple(name for name in args.software.split(",") if name)

    report = REPO_ROOT / args.report
    report.mkdir(parents=True, exist_ok=True)

    tasks = iter_tasks(software)
    swept = sorted({row["software"] for row in tasks})
    print(f"registered tasks: {len(tasks)} across {len(swept)} software", flush=True)

    forged: dict[str, dict] = {}
    if "forged" in layers:
        started = time.monotonic()
        forged = run_forged_layer(tasks, args.jobs)
        (report / "forged.json").write_text(json.dumps(forged, indent=1, sort_keys=True))
        outcomes = Counter(
            modes["forged_no"]["outcome"]
            for modes in forged.values()
            if "forged_no" in modes
        )
        print(f"forged: {len(forged)} verifiers in {time.monotonic() - started:.0f}s "
              f"{dict(outcomes)}", flush=True)
        judge_gated = judge_gated_passes(forged)
        (report / "judge_gated.json").write_text(json.dumps(judge_gated, indent=1))
        print(f"forged: {len(judge_gated)} more pass ONLY with a fooled judge "
              f"(diagnostic, not excluded — see PROBE_MODES)", flush=True)

    setup_rc: dict[str, dict] = {}
    if "setup_rc" in layers:
        started = time.monotonic()
        cached = Path(args.reuse).expanduser() / "setup_rc.json" if args.reuse else None
        if cached is not None and cached.is_file():
            setup_rc = json.loads(cached.read_text())
            for key, result in setup_rc.items():
                software, _, task_id = key.partition("/")
                task_dir = next(
                    MATERIALS_DIR.glob(f"{software}/*/tasks/{task_id}"), None
                )
                result["unstageable_source"] = (
                    _unstageable_source(task_dir) if task_dir else None
                )
            verb = f"reused from {cached}"
        else:
            if not _sandbox_available():
                parser.error(
                    "setup_rc needs an unprivileged user+mount namespace "
                    "(`unshare --map-root-user --mount`); rerun with --layers forged"
                )
            workroot = Path(tempfile.mkdtemp(prefix="cuaworld-setup-rc-"))
            try:
                setup_rc = run_setup_rc_layer(tasks, workroot)
            finally:
                shutil.rmtree(workroot, ignore_errors=True)
            verb = f"replayed in {time.monotonic() - started:.0f}s"
        (report / "setup_rc.json").write_text(json.dumps(setup_rc, indent=1, sort_keys=True))
        aborts = sum(1 for r in setup_rc.values() if r["rc"])
        attributed = sum(1 for r in setup_rc.values() if r["rc"] and r["unstageable_source"])
        print(f"setup_rc: {len(setup_rc)} hooks {verb}, {aborts} non-zero rc, "
              f"{attributed} of them attributable without the image", flush=True)

    findings = _classify(forged, setup_rc, layers)
    (report / "findings.json").write_text(json.dumps(findings, indent=1, sort_keys=True))
    print(f"findings: {len(findings)} {dict(Counter(findings.values()))}")
    if args.verbose:
        for key, reason in sorted(findings.items()):
            detail = forged.get(key, {}).get("forged_no", {})
            print(f"  {key:58s} {reason:21s} score={detail.get('score')} "
                  f"reward={detail.get('reward')} {setup_rc.get(key, {}).get('tail', '')[:80]}")

    if args.write:
        document = write_excludes(findings, swept, layers)
        print(f"wrote {EXCLUDES_PATH.relative_to(REPO_ROOT)} "
              f"(_total={document['_meta']['_total']})")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
