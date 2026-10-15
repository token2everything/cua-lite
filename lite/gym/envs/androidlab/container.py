"""Docker-container-per-worker emulator container + factory for androidlab.

Why one container per worker: AndroidLab's Quick Boot snapshot was
saved inside a container at paths ``/root/.android/system-images/...``
and ``/root/.android/avd/...``. The paths are baked into
``snapshot.pb``; reading them from user-mode on our host is
non-trivial. Running each worker emulator inside its own container
makes the paths match natively
— the only straightforward way to load the bundled snapshot and recover
the ABC/AAA contacts, the Pink-Floyd MP3s, and the SQLite DBs every
judge keys off of.

Version pinning: the snapshot is tied to emulator build 11906825
(34.2.15). Today's ``sdkmanager 'emulator'`` installs 36.5.10, which
rejects the 34.x snapshot. Our ``cua-lite/androidlab:latest`` image
swaps 34.2.15 in at build time so the snapshot load is reliable.

Port allocation uses the shared ``lite.gym.utils.backend.ports`` allocator
(random-offset scan + port-bind-check prune). Port ranges don't overlap
across envs (lite.osworld 20000-20999, androidlab 21000-21999), so
sharing one reservation file is safe. See the port-range map at the top
of ``lite/gym/utils/backend/ports.py`` for the full layout.

We grab 1 host port per container: the in-container env-server's HTTP
API. The emulator's own console / adb / gRPC ports bind to the
container's loopback and don't need a host-side publish — the
env-server (``docker/server.py``) owns the local adb client and the
gRPC channel, both entirely container-side. Host-side ``docker exec``
is used only for one-shot setup (boot wait, adb root, geo/date pin),
NOT on the per-step hot path.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from lite.gym.container import LiteContainerBase, boot_with_retry
from lite.gym.errors import CapacityExhausted
from lite.gym.utils import config as env_config
from lite.gym.utils.config.naming import format_container_name as _format_container_name
from lite.gym.utils.backend.docker import _rm_argv, docker_run_detached
from lite.gym.utils.backend.ports import allocate_ports

logger = logging.getLogger(__name__)

# Infra timeouts / memory cap — see ../configs/default.yaml server_kwargs
# (override the whole config via ANDROID_LAB_CONFIG=<path>). ENV_DIR is the
# env package root (package root).
ENV_DIR = str(Path(__file__).resolve().parents[0])
CFG = env_config.load(ENV_DIR)

# ============================================================================
# Config defaults — every value below is read once from configs/default.yaml
# via env_config.load(ENV_DIR). Swap the whole file at startup with
# ANDROID_LAB_CONFIG=<abs-path | bundled-name>. A rollout's env_kwargs still
# override per run; these are only registration defaults.
# ============================================================================
# --- env_kwargs (per-instance) ---
#: Image tag — single source is configs/default.yaml's env_kwargs.image
#: (must match what ``scripts/install.sh`` tags).
DEFAULT_IMAGE = CFG.env_kwargs["image"]
# --- server_kwargs (per-deployment) ---
#: Cold-boot deadline. Empirically the cua-lite/androidlab:latest snapshot
#: load takes 300-500 s on an essentially idle host (load avg < 10% of
#: cores). The slow phase is inside qemu (swiftshader GPU init + a11y
#: reverse-grpc wrapper attach + service re-spawn) so it doesn't get
#: faster with more cores. Older default 180 s was too tight — first
#: cold boot failed even after the 2-attempt retry. 600 s gives
#: comfortable margin while still failing fast on a truly stuck boot.
#: See server_kwargs.boot_timeout in ../configs/default.yaml.
BOOT_TIMEOUT_S = CFG.server_kwargs["boot_timeout"]
#: In-container HTTP API readiness deadline. See server_kwargs.api_timeout.
API_READY_TIMEOUT_S = CFG.server_kwargs["api_timeout"]
#: Android-system-services readiness deadline. Mirrors
#: :data:`lite.gym.envs.androidworld.container.ANDROID_READY_TIMEOUT_S`
#: — ``Successfully loaded snapshot`` (the boot signal androidlab
#: uses) lets us know qemu finished restoring the VM, but the
#: in-Android PackageManager is still rebuilding its package list for
#: several seconds after that. ``/init`` on the in-container server
#: runs ``adb shell wm size`` plus follow-on judge / app-launch
#: machinery that all depend on PM being available; calling them too
#: early surfaces as opaque adb errors / 500s. Probing ``pm path``
#: closes that gap. See devs/envs/androidworld.md "Known Issue"
#: section for the discovery trail. See server_kwargs.android_ready_timeout.
ANDROID_READY_TIMEOUT_S = CFG.server_kwargs["android_ready_timeout"]
#: Per-container memory cap. androidlab needs slightly more than
#: androidworld because the AOSP image baked into the AVD is bigger
#: (~3.5 GB RAM in steady state). memory == memory-swap disables swap;
#: swappiness=0 makes the kernel prefer OOM-kill over host swap thrash.
#: See server_kwargs.memory_limit in ../configs/default.yaml.
_MEMORY_LIMIT = CFG.server_kwargs["memory_limit"]
# ============================================================================

#: AVD name is baked into the image (snapshot.pb references this exact
#: directory name). Don't change without rebuilding the image.
DEFAULT_AVD_NAME = "Pixel_7_Pro_API_33"

# Inside-container ports are fixed at the Android convention (5554 console,
# 5555 adb, 8554 gRPC). Only the published host-side ports vary per worker.
_CPORT_CONSOLE = 5554
_CPORT_ADB = 5555
_CPORT_GRPC = 8554
_CPORT_API = 9554        # docker/server.py listens here (in-container)

# Paths inside the cua-lite/androidlab image.
_ADB_BIN = "/root/.android/platform-tools/adb"
_EMU_BIN = "/root/.android/emulator/emulator"

# Reserve 21000-21999, sitting just above cb (lite.osworld) at 20000-20999
# and well clear of androidworld emulator pool whose grpc
# projection (console+3000) tops out at 10554. 1000 ports / 1 per
# container = 1000 concurrent androidlab containers.
_PORT_RANGE_START = 21000
_PORT_RANGE_END = 21999

# SESSION_ID-scoped name prefix matches the repo-wide Docker cleanup
# convention (lite/gym/envs/<env>/scripts/cleanup.sh). Orphan cleanup:
#   bash lite/gym/envs/androidlab/scripts/cleanup.sh
# which filters on "lite-env-${SESSION_ID}-androidlab-". Name assembly
# lives in :mod:`lite.gym.utils.config.naming` (imported above).

@dataclass
class AndroidLabContainer(LiteContainerBase):
    """One running emulator inside one docker container.

    Each ``factory.acquire()`` returns a fresh container with ``default_boot``
    just loaded. Between episodes on the same env, call
    :meth:`reset_snapshot` to restore pristine seeded state in ~3-5s
    without re-spawning the container.
    """

    name: str
    api_port: int
    image: str = DEFAULT_IMAGE
    avd_name: str = DEFAULT_AVD_NAME

    #: Device serial as seen by adb INSIDE the container — fixed because
    #: the in-container emulator console port is fixed at 5554.
    adb_serial: str = field(
        default=f"emulator-{_CPORT_CONSOLE}", init=False
    )

    #: Ports this container currently owns — released on ``destroy()``.
    _ports_owned: tuple[int, ...] = field(default=(), repr=False)

    rm_label: ClassVar[str] = "androidlab"

    @property
    def base_url(self) -> str:
        """Host URL of the in-container env-server (docker/server.py).

        Used by main.py's ``_RemoteJudge`` proxy to run
        ``judge.judge()`` + XML compression + ``find_package`` inside
        the container without importing androidlab on host.
        """
        return f"http://localhost:{self.api_port}"

    # -------- lifecycle --------------------------------------------------

    def start(self) -> None:
        """``docker run`` + boot the emulator + wait for snapshot load."""
        # Register BEFORE the container-creating call — covers the whole boot
        # path below (docker run + the multi-minute emulator boot waits): a
        # KeyboardInterrupt there is a BaseException that acquire()'s
        # ``except Exception`` won't catch; destroy() is idempotent, so
        # failures/retries stay clean. The docker-create semaphore is
        # acquired in acquire() (wraps the WHOLE boot path), not here.
        self._register()
        kvm_gid = _kvm_gid()
        # NOTE: androidworld disables IPv6 here to work around a qemu
        # modem-chardev ``getaddrinfo("::1:<port>")`` bug that strands
        # the SIM. androidlab inherits the same qemu modem warning at
        # boot, but disabling IPv6 here breaks the AndroidLab Quick
        # Boot snapshot restore (snapshot.pb baked in a state that
        # expects the lo ::1 address; without it the emulator hangs
        # with "Failed to setup emulator in a timely fashion" and
        # never finishes loading default_boot — 133/138 replay
        # timeouts under MAI-UI-8B). androidlab has no SMS tasks
        # today, so the missing SIM is benign; leave IPv6 enabled.
        # Forward LLM-judge env vars from host into the container. The
        # ~39 query_detect androidlab tasks (bluecoins / contacts /
        # cantook / clock / map_me / pimusic Q&A) ground-truth their
        # answers via an OpenAI-style chat call — see
        # android_lab.evaluation.task._llm_judge. Endpoint resolution
        # in the in-container judge: AZURE_API_BASE+AZURE_API_KEY →
        # AzureOpenAI; else OPENAI_API_KEY (+ OPENAI_BASE_URL) →
        # OpenAI. Without forwarding, the judge fast-fails on missing
        # credentials and the task silently scores reward=0 even when
        # the model's answer is correct.
        judge_env = {
            env_var: val
            for env_var in ("OPENAI_API_KEY", "OPENAI_BASE_URL",
                            "AZURE_API_BASE", "AZURE_API_KEY")
            if (val := os.environ.get(env_var))
        }
        # Only the in-container env-server's HTTP API needs to be reachable
        # from host. adb goes via ``docker exec``; gRPC + emulator console
        # stay inside the container's network namespace.
        docker_run_detached(
            name=self.name, image=self.image,
            auto_remove=True,
            devices=("/dev/kvm",),
            memory=_MEMORY_LIMIT,   # see _MEMORY_LIMIT comment above
            group_add=(kvm_gid,) if kvm_gid is not None else (),
            env=judge_env,
            ports=((self.api_port, _CPORT_API),),
            command=("sleep", "86400"),
            redact=("OPENAI_API_KEY", "AZURE_API_KEY"),
            timeout=180.0,
            label="androidlab",
        )

        # Launch emulator (detached). ``-no-snapshot-save`` loads
        # default_boot on boot but doesn't write dirty state back, so
        # every container instance starts from the same seeded state.
        emu_cmd = (
            f"cd /root && {_EMU_BIN} -avd {self.avd_name} "
            "-no-snapshot-save -no-audio -no-window -gpu swiftshader_indirect "
            f"-ports {_CPORT_CONSOLE},{_CPORT_ADB} -grpc {_CPORT_GRPC} "
            "> /tmp/emu.log 2>&1"
        )
        subprocess.run(
            # bash -c (not -lc) — same rationale as _start_server below.
            ["docker", "exec", "-d", self.name, "bash", "-c", emu_cmd],
            check=True, capture_output=True, timeout=30,
        )

        self._wait_until_booted()
        self._wait_until_android_ready()
        # ``_wait_until_booted`` already ran ``adb devices`` which auto-starts
        # the adb server, so no explicit ``start-server`` is needed here.
        self._start_server()
        self._wait_until_api_ready()
        # NOTE: _register() happens right after ``docker run`` above (so
        # the atexit backstop covers the boot window), NOT here.

    def _start_server(self) -> None:
        """Spawn ``docker/server.py`` (FastAPI) inside the container.

        Backgrounded so it lives as long as the container. Hosts the
        judge / xml-compression / find_package / adb endpoints the host
        process drives via HTTP RPC.

        We use ``bash -c`` (NOT ``bash -lc``): a login shell re-sources
        ``/etc/profile`` and clobbers the Dockerfile's ``ENV PATH``
        with Debian's default. The in-container ``server.py`` shells
        out to ``adb`` via ``subprocess.run(['adb', ...])`` which needs
        ``/root/.android/platform-tools`` on PATH — and the Dockerfile
        IS the canonical place that puts it there.
        """
        cmd = (
            f"python /usr/local/bin/server.py --port {_CPORT_API} "
            "> /tmp/server.log 2>&1"
        )
        # 10 → 30 s: at c=32+ concurrent emulator boots the rootless-docker
        # daemon serialises ``docker exec`` setup, pushing individual call
        # latencies from <100 ms baseline to 5-15 s.
        r = subprocess.run(
            ["docker", "exec", "-d", self.name, "bash", "-c", cmd],
            capture_output=True, timeout=30,
        )
        if r.returncode != 0:
            stderr = (r.stderr or b"").decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"docker exec (launch server) failed for {self.name}: {stderr}"
            )

    def _wait_until_api_ready(self) -> None:
        """Poll /healthz until uvicorn is up (typically <5 s)."""
        import urllib.request
        deadline = time.monotonic() + API_READY_TIMEOUT_S
        last_err = ""
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    f"{self.base_url}/healthz", timeout=2.0
                ) as resp:
                    if resp.status == 200:
                        logger.info(
                            "server ready on %s", self.base_url,
                        )
                        return
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
            # 0.5s poll: /healthz is cheap, detects readiness sooner.
            time.sleep(0.5)
        try:
            log_tail = subprocess.run(
                ["docker", "exec", self.name, "tail", "-100", "/tmp/server.log"],
                capture_output=True, timeout=5,
            ).stdout.decode("utf-8", errors="replace")
        except Exception:
            log_tail = "(could not read server.log)"
        # Still warming: the in-container uvicorn just hasn't bound /healthz
        # yet. Recoverable — the client should retry the SAME instance while it
        # finishes coming up (CapacityExhausted → 503 + Retry-After), not get a
        # terminal 500 that ends the episode.
        raise CapacityExhausted.warming(
            f"androidlab server in {self.name} not HTTP-ready in "
            f"{API_READY_TIMEOUT_S}s (last error: {last_err}).\n"
            f"--- server.log tail ---\n{log_tail}"
        )

    def _wait_until_booted(self) -> None:
        deadline = time.monotonic() + BOOT_TIMEOUT_S
        last_log = ""
        while time.monotonic() < deadline:
            last_log = self._read_log()
            if "Successfully loaded snapshot" in last_log:
                break
            if "incompatible version" in last_log or "Failed to setup emulator" in last_log:
                tail = last_log[-1500:] if last_log else "(empty log)"
                raise RuntimeError(
                    f"Emulator in container {self.name} failed to load snapshot.\n"
                    f"  Likely cause: the image's emulator binary doesn't match the\n"
                    f"  Quick Boot snapshot (expected build 11906825 / 34.2.15).\n"
                    f"  Rebuild {self.image} via lite/gym/envs/androidlab/scripts/install.sh.\n"
                    f"  Log tail:\n{tail}"
                )
            # Fast-fail: under high-concurrency cold-boot bursts in
            # rootless docker, occasionally one container's kvm access
            # gets denied at runtime (qemu fails on KVM_CREATE_VM or
            # similar) despite ``--group-add <kvm_gid>`` being passed.
            # The two known emu.log signatures:
            #   1. Android emulator wrapper's preflight: "doesn't have
            #      permissions to use KVM" + "requires hardware
            #      acceleration" (falls back to TCG → never boots).
            #   2. qemu directly: "Could not access KVM kernel module:
            #      Permission denied" + "failed to initialize KVM"
            #      (process exits with code 1).
            # Both surface within ~5-10 s of the docker run. Fast-fail
            # so the caller can destroy + retry instead of waiting out
            # BOOT_TIMEOUT_S (600 s) of either software emulation or
            # the exited-qemu silence.
            if ("requires hardware acceleration" in last_log
                    or "doesn't have permissions to use KVM" in last_log
                    or "Could not access KVM kernel module" in last_log
                    or "failed to initialize KVM" in last_log):
                tail = last_log[-1500:] if last_log else "(empty log)"
                raise RuntimeError(
                    f"Emulator in container {self.name} can't access /dev/kvm "
                    f"despite --group-add. Intermittent rootless docker race; "
                    f"caller should destroy + retry. Fast-fail saves "
                    f"~{BOOT_TIMEOUT_S}s of post-failure silence.\n"
                    f"  Log tail:\n{tail}"
                )
            # 1s poll: each iter reads the emulator log via docker exec
            # (~50-200 ms). Halved from 2s for snappier snapshot-load
            # detection. The throttle bounds concurrent pollers.
            time.sleep(1.0)
        else:
            # Snapshot load timed out without an incompatible-version /
            # KVM-permission signature — the emulator is just still warming up
            # (cold snapshot restore under host contention can exceed the
            # deadline). Recoverable: retrying the same instance plausibly
            # succeeds, so signal warming (503 + Retry-After), not a 500.
            raise CapacityExhausted.warming(
                f"emulator in container {self.name} not finished loading "
                f"snapshot in {BOOT_TIMEOUT_S}s. Log tail:\n{last_log[-1500:]}"
            )

        # Confirm adb sees the device before we return.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            r = subprocess.run(
                ["docker", "exec", self.name, _ADB_BIN, "devices"],
                capture_output=True, text=True, timeout=30,
            )
            if f"{self.adb_serial}\tdevice" in r.stdout:
                return
            # 1s poll (was 2s): adb devices is a cheap docker exec.
            time.sleep(1.0)
        # Snapshot loaded but adb hasn't registered the device yet — a brief
        # post-restore race, still warming. Recoverable: retry the same
        # instance rather than terminally 500.
        raise CapacityExhausted.warming(
            f"emulator in {self.name} booted but adb hasn't seen the device yet."
        )

    def _wait_until_android_ready(self) -> None:
        """Poll until Android PackageManager is usable after snapshot
        restore.

        ``_wait_until_booted`` returns as soon as qemu logs
        ``Successfully loaded snapshot`` — that's the VM-restore
        signal, but in-Android service rehydration (PackageManager
        rebuilding its package list, AccessibilityManagerService
        coming back online, etc.) continues for several seconds
        afterward on a contended host. Calling ``/init`` (which
        runs ``adb shell wm size``) or any of the downstream
        adb-driven per-step calls before PM is reachable surfaces
        as opaque adb errors / 500s.

        The androidlab in-container server doesn't use a11y_grpc
        (unlike androidworld), so we only probe PackageManager.
        Mirrors :meth:`AndroidWorldContainer._wait_until_android_ready`
        — see devs/envs/androidworld.md "Known Issue" for the full
        discovery.
        """
        deadline = time.monotonic() + ANDROID_READY_TIMEOUT_S
        last_pm = ""
        while time.monotonic() < deadline:
            try:
                r = subprocess.run(
                    ["docker", "exec", self.name, _ADB_BIN,
                     "-s", self.adb_serial,
                     "shell", "pm", "path", "com.android.settings"],
                    capture_output=True, timeout=15,
                )
                last_pm = (r.stdout or b"").decode(errors="replace").strip()
                if r.returncode == 0 and last_pm.startswith("package:"):
                    logger.info("android services ready in %s", self.name)
                    return
            except subprocess.SubprocessError:
                pass
            time.sleep(1.0)
        # Still warming: snapshot restored but PackageManager hasn't finished
        # rehydrating its package list yet. Recoverable — retry the same
        # instance.
        raise CapacityExhausted.warming(
            f"android services in {self.name} not ready in "
            f"{ANDROID_READY_TIMEOUT_S}s (pm.path={last_pm!r})."
        )

    def _read_log(self) -> str:
        # Swallow timeouts/errors: this runs in a 2-second polling loop
        # inside ``_wait_until_booted``; one stuck ``docker exec`` under
        # heavy daemon load shouldn't abort the whole boot. Next poll will
        # likely succeed. The outer BOOT_TIMEOUT_S governs real timeouts.
        try:
            r = subprocess.run(
                ["docker", "exec", self.name, "cat", "/tmp/emu.log"],
                capture_output=True, text=True, timeout=30,
            )
            return r.stdout if r.returncode == 0 else ""
        except subprocess.SubprocessError:
            return ""

    # -------- per-episode reset -----------------------------------------

    def reset_snapshot(self) -> bool:
        """Hot-reload ``default_boot`` in ~3-5s.

        Restores seeded contacts / events / playlists / transactions to
        their pristine state without re-spawning the container. Returns
        False on failure; caller should then ``destroy()`` + fresh
        ``factory.acquire()``.
        """
        try:
            r = subprocess.run(
                ["docker", "exec", self.name, _ADB_BIN, "-s", self.adb_serial,
                 "emu", "avd", "snapshot", "load", "default_boot"],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.SubprocessError as e:
            logger.warning("reset_snapshot raised on %s: %s", self.name, e)
            return False
        combined = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0 or "KO" in combined:
            logger.warning("reset_snapshot failed on %s: %r", self.name, combined.strip())
            return False
        # Settle window. Snapshot load is fast (~2 s qemu side), but the
        # Android stack inside the restored VM needs more time to come
        # back online:
        # * SurfaceFlinger re-attaches GL/Vulkan contexts
        # * a11y service rebinds the reverse-gRPC channel
        # * settings provider re-reads /data/system/users/0/settings_*.xml
        # * launcher repaints
        # Bumped 1 s → 3 s to match androidworld's empirically-tuned
        # value (world's M1 5-rep test saw ~50% reward=0 on
        # AudioRecorderRecordAudio with 1 s, fixed at 3 s). Lab's
        # in-container RPC layer absorbs more of the race than world's
        # direct adb/a11y path, so 1 s may have been sufficient — but
        # 2 s extra per reset is negligible vs the risk of snapshot-reuse
        # flakiness under load.
        time.sleep(3.0)
        return True

    # -------- adb passthrough -------------------------------------------

    def adb(self, *args: str, timeout: float = 15.0) -> subprocess.CompletedProcess:
        """Run ``adb -s emulator-5554 <args>`` inside the container.

        Host adb can't talk to the container's emulator directly — rootless
        docker's port publish doesn't forward the adb handshake, and the
        emulator binds 127.0.0.1 inside the container. ``docker exec`` adds
        ~10ms latency per call, negligible compared to the post-action
        3s UI-settle wait.
        """
        return subprocess.run(
            ["docker", "exec", self.name, _ADB_BIN, "-s", self.adb_serial, *args],
            capture_output=True, timeout=timeout, check=False,
        )

    def adb_shell(self, *args: str, timeout: float = 15.0) -> str:
        r = self.adb("shell", *args, timeout=timeout)
        return (r.stdout or b"").decode("utf-8", errors="replace").strip()

    def _exec_out(self, *args: str, timeout: float = 15.0) -> bytes:
        """``adb exec-out`` — binary-safe (screencap, raw XML file cat)."""
        r = subprocess.run(
            ["docker", "exec", self.name, _ADB_BIN, "-s", self.adb_serial, "exec-out", *args],
            capture_output=True, timeout=timeout, check=False,
        )
        return r.stdout or b""

    # -------- cleanup ----------------------------------------------------
    # destroy() inherited from LiteContainerBase (template: rm -f -v @ 60 s —
    # qemu's post-SIGKILL unwind on a stuck emulator can take 30-90 s —
    # → release _ports_owned → de-register; idempotent, never raises).


class AndroidLabContainerFactory:
    """Spawn one fresh ``AndroidLabContainer`` per :meth:`acquire`.

    Holds shared kwargs (``image``, ``avd_name``, ``port_range``,
    ``session_id``, ``token_hash``, ``server_port``) across multiple
    ``acquire()`` calls. Cross-episode reuse is **not** done here — the
    value-add is :meth:`AndroidLabContainer.reset_snapshot` (AVD
    snapshot reload) on a kept-alive container.
    """

    def __init__(
        self,
        image: str | None = None,
        avd_name: str = DEFAULT_AVD_NAME,
        port_range: tuple[int, int] = (_PORT_RANGE_START, _PORT_RANGE_END),
        task_id: str | None = None,
        session_id: str | None = None,
        token_hash: str | None = None,
        server_port: int | None = None,
    ):
        self.image = image or DEFAULT_IMAGE
        self.avd_name = avd_name
        self.port_range = port_range
        self.task_id = task_id
        # Per-factory session_id + token_hash + server_port (set by the env
        # when running under a multi-tenant env server). session_id
        # falls back to $SESSION_ID for single-tenant use; token_hash
        # and server_port fall back to absent (no segment in container
        # name). ``server_port`` + ``token_hash`` together isolate two
        # same-token env-server instances on the same host.
        self.session_id = session_id
        self.token_hash = token_hash
        self.server_port = server_port

    def _make_name(self, api_port: int) -> str:
        """Canonical name: ``lite-env-[{server_port}-][{token_hash}-]{session_id}-androidlab-[{task_id}-]{api_port}``.

        Full segment contract + sanitization lives in
        :func:`lite.gym.utils.config.naming.format_container_name`. ``api_port``
        is the unique per-instance suffix (each container binds a
        different host port).
        """
        return _format_container_name(
            env_id="androidlab",
            task_id=self.task_id,
            suffix=str(api_port),
            session_id=self.session_id or os.environ.get("SESSION_ID"),
            token_hash=self.token_hash,
            server_port=self.server_port,
        )

    def acquire(self, *, max_attempts: int = 3) -> AndroidLabContainer:
        """Pick a free api port, spawn container, wait for snapshot boot.

        Retries up to ``max_attempts`` times on transient boot failures
        (emulator snapshot-load timeout, adb-not-ready timeout, in-container
        API readiness timeout, KVM permission race). Failed attempts are
        torn down completely before the next attempt — fresh port, fresh
        container name. ``RuntimeError`` + warming ``CapacityExhausted`` are retried; image-missing /
        docker-daemon errors propagate immediately.

        Why 3 attempts: rootless docker under concurrent cold-boot bursts
        intermittently denies KVM access to one container in the batch
        (~1-2 % rate). The fast-fail KVM check in ``_wait_until_booted``
        catches this in <10 s, so 3 attempts cap the worst-case acquire
        wall time at ~30 s of failures + one successful boot. Two
        attempts left a long-running stress run with ~0.01 % unrecoverable
        rate; three drives it under 0.0001 % at negligible cost.

        Concurrent acquires across the whole process are gated by
        :func:`lite.gym.remote.admission.docker_create_slot` — a
        process-wide threading semaphore (default 8) that bounds
        ``docker run`` parallelism so the daemon doesn't thrash. The
        hold spans the WHOLE boot pipeline (docker run + snapshot
        load + adb ready + API ready), because the disk-IO bottleneck
        during snapshot restore is heavier than the daemon-side
        bottleneck.
        """
        from lite.gym.remote.admission import docker_create_slot

        def _build() -> AndroidLabContainer:
            (api,) = allocate_ports(
                n=1, range_start=self.port_range[0], range_end=self.port_range[1]
            )
            name = self._make_name(api)
            # Nuke any same-named zombie from a previous crash before we
            # ``docker run`` a fresh one. 60 s matches ``destroy()`` —
            # qemu's post-SIGKILL unwind on a stuck emulator can take 30-90 s.
            subprocess.run(_rm_argv(name), capture_output=True, timeout=60)
            return AndroidLabContainer(
                name=name,
                api_port=api,
                image=self.image,
                avd_name=self.avd_name,
                _ports_owned=(api,),
            )

        def _start(c: AndroidLabContainer) -> None:
            # structural docker semaphore — bounds concurrent ``docker run``
            # so the daemon doesn't thrash. Held across c.start() (docker run
            # + snapshot load + adb ready + API ready).
            with docker_create_slot():
                c.start()

        # Retry classification (transient RuntimeError/CapacityExhausted →
        # fresh build + retry; else destroy + raise) is single-sourced in
        # boot_with_retry.
        return boot_with_retry(
            _build, start=_start, max_attempts=max_attempts, label="androidlab",
        )


def _kvm_gid() -> int | None:
    """Host kvm group gid, for ``--group-add``."""
    try:
        import grp
        return grp.getgrnam("kvm").gr_gid
    except Exception:
        return None
