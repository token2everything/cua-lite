#!/usr/bin/env python3
"""Build a reusable "ready snapshot" so episodes restore in seconds instead of
cold-booting Windows (~60-90s).

Run once after the base qcow2 exists:

    uv run python lite/gym/envs/waa/scripts/utils/prepare_snapshot.py

It cold-boots one VM to the WAA-ready state, migrate-saves its RAM+device state
to a file, and freezes the ready disk + device files. The result under
``~/.cache/cua-lite/waa/snapshot/`` is used automatically by qemu.py whenever it is present:

    snapshot/
      ready.qcow2            # ready-state disk overlay (backed by base.qcow2), shared read-only
      ready.state            # migrated RAM + device state, shared read-only, fed to -incoming
      device/windows.*       # small per-VM device files (MAC, secure-boot vars, TPM, ...)
      manifest.json          # base-disk identity so a stale snapshot is rejected

Requires the snapshot-capable runner image (Dockerfile drops +invtsc and teaches
dockur's boot watchdog the restore marker).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

RUNNER = "cua-lite/waa:latest"
GUEST_IP = "20.20.20.21"
CONTAINER = "waa-snapshot-prep"
# /storage files that are NOT per-VM device state (excluded from device/).
_NON_DEVICE = {"data.qcow2", "ready.state", "windows.boot"}


def _sh(*a, check=True, **k):
    r = subprocess.run(a, text=True, capture_output=True, **k)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(a)}\n{r.stderr[-2000:]}")
    return r


def _sha256_head(path: Path, limit: int = 256 << 20) -> str:
    """Hash up to `limit` bytes — enough to identify the base disk cheaply."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = limit
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            remaining -= len(chunk)
            if remaining <= 0:
                break
    return h.hexdigest()


def _running(name: str) -> bool:
    out = _sh("docker", "inspect", "-f", "{{.State.Running}}", name, check=False)
    return out.stdout.strip() == "true"


def _health(name: str) -> str:
    out = _sh("docker", "inspect", "-f", "{{.State.Health.Status}}", name, check=False)
    return out.stdout.strip()


def _hmp(name: str, cmd: str, timeout: int = 600) -> str:
    """Run one HMP monitor command inside the container (127.0.0.1:7100)."""
    py = f"""
import socket, time, re
s = socket.create_connection(('127.0.0.1', 7100), timeout=5); s.settimeout(0.5); time.sleep(0.2)
try:
    while True: s.recv(65536)
except Exception: pass
cmd = {cmd!r}
s.sendall((cmd + chr(10)).encode()); buf = b''; s.settimeout({timeout})
while True:
    try: d = s.recv(65536)
    except socket.timeout: break
    if not d: break
    buf += d
    if buf.rstrip().endswith(b'(qemu)'): break
clean = re.sub(r'\\x1b\\[[0-9;]*[A-Za-z]|\\x1b\\[K|\\r', '', buf.decode('utf-8', 'replace'))
print(clean.strip()[-400:])
"""
    return _sh("docker", "exec", "-i", name, "python3", "-c", py, check=False).stdout.strip()


def _boot_ready(
    work: Path, base: Path, assets: Path, vcpus: int, memory_gb: int, timeout: int
) -> None:
    storage = work / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    _sh(
        "docker", "run", "--rm", "--entrypoint", "qemu-img",
        "--mount", f"type=bind,src={base},dst=/images/base.qcow2,readonly",
        "--mount", f"type=bind,src={storage},dst=/storage",
        RUNNER, "create", "-f", "qcow2", "-F", "qcow2",
        "-b", "/images/base.qcow2", "/storage/data.qcow2",
    )
    _sh("docker", "rm", "-f", "-v", CONTAINER, check=False)
    _sh(
        "docker", "run", "-d", "--name", CONTAINER,
        "--device=/dev/kvm", "--device=/dev/net/tun", "--cap-add", "NET_ADMIN",
        "--sysctl", "net.ipv4.ip_forward=1", "--shm-size=2g",
        "--mount", f"type=bind,src={base},dst=/images/base.qcow2,readonly",
        "--mount", f"type=bind,src={storage},dst=/storage",
        "--mount", f"type=bind,src={assets},dst=/opt/waa-assets,readonly",
        "--publish", "127.0.0.1:0:5050", "--publish", "127.0.0.1:0:8006",
        "--env", f"RAM_SIZE={memory_gb}G", "--env", f"CPU_CORES={vcpus}",
        "--env", "CPU_MODEL=host", "--env", "HV=N",
        "--env", f"VM_NET_IP={GUEST_IP}", "--env", f"WAA_GUEST_IP={GUEST_IP}",
        "--env", f"WAA_GUEST_READY_TIMEOUT_S={timeout}", "--env", "HOST_PORTS=5050",
        "--env", "ARGUMENTS=-qmp tcp:0.0.0.0:7200,server,nowait",
        RUNNER,
    )
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not _running(CONTAINER):
            logs = _sh("docker", "logs", "--tail", "20", CONTAINER, check=False).stderr
            raise RuntimeError(f"prep VM exited before ready:\n{logs[-1500:]}")
        if _health(CONTAINER) == "healthy":
            print(f"[snapshot] cold boot to ready in {time.time() - t0:.1f}s", flush=True)
            return
        time.sleep(3)
    raise RuntimeError("prep VM did not become ready in time")


def main() -> None:
    cache = Path(os.path.expanduser("~/.cache/cua-lite/waa"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-disk", type=Path, default=cache / "images" / "windows11-waa.qcow2")
    ap.add_argument("--assets-dir", type=Path, default=cache / "assets")
    ap.add_argument("--out", type=Path, default=cache / "snapshot")
    ap.add_argument("--vcpus", type=int, default=8)
    ap.add_argument("--memory-gb", type=int, default=8)
    ap.add_argument("--ready-timeout-s", type=int, default=900)
    args = ap.parse_args()

    base = args.base_disk.expanduser().resolve()
    assets = args.assets_dir.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if not base.is_file():
        raise SystemExit(f"base disk not found: {base} (run install.sh first)")
    _sh("docker", "image", "inspect", RUNNER)

    work = cache / "snapshot-build"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    storage = work / "storage"

    try:
        print("[snapshot] booting a fresh VM to the ready state ...", flush=True)
        _boot_ready(work, base, assets, args.vcpus, args.memory_gb, args.ready_timeout_s)

        print("[snapshot] migrate-saving RAM + device state ...", flush=True)
        _hmp(CONTAINER, "migrate_set_parameter max-bandwidth 10G")
        t0 = time.time()
        _hmp(CONTAINER, 'migrate -d "exec:cat > /storage/ready.state"')
        for _ in range(args.ready_timeout_s):
            st = _hmp(CONTAINER, "info migrate")
            if "completed" in st:
                break
            if "failed" in st:
                raise RuntimeError(f"migrate failed: {st}")
            time.sleep(1)
        else:
            raise RuntimeError("migrate did not complete in time")
        size = (storage / "ready.state").stat().st_size
        print(f"[snapshot] state saved: {size / 1e9:.2f} GB in {time.time() - t0:.1f}s", flush=True)
        _sh("docker", "rm", "-f", "-v", CONTAINER, check=False)

        # Freeze the bundle atomically.
        staging = out.with_suffix(".building")
        if staging.exists():
            shutil.rmtree(staging)
        (staging / "device").mkdir(parents=True)
        # ready disk overlay (still backed by /images/base.qcow2)
        shutil.move(str(storage / "data.qcow2"), str(staging / "ready.qcow2"))
        shutil.move(str(storage / "ready.state"), str(staging / "ready.state"))
        for item in storage.iterdir():
            if item.name in _NON_DEVICE or item.suffix == ".iso":
                continue
            shutil.copy2(item, staging / "device" / item.name)
        (staging / "manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "base_disk": str(base),
            "base_disk_size": base.stat().st_size,
            "base_disk_head_sha256": _sha256_head(base),
            "ram_state_size": size,
            "vcpus": args.vcpus,
            "memory_gb": args.memory_gb,
            "device_files": sorted(p.name for p in (staging / "device").iterdir()),
        }, indent=2) + "\n", encoding="utf-8")

        if out.exists():
            shutil.rmtree(out)
        staging.rename(out)
        print(f"[snapshot] ready snapshot written to {out}", flush=True)
        devices = ", ".join(sorted(p.name for p in (out / "device").iterdir()))
        print(f"[snapshot] device files: {devices}")
    finally:
        _sh("docker", "rm", "-f", "-v", CONTAINER, check=False)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
