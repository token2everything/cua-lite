"""OSWorld v1 VM-in-Docker container lifecycle — spawn / name / boot-wait / reap.

One `cua-lite/osworld` container = one QEMU/KVM VM booting `Ubuntu.qcow2` + an in-container
eval server (`docker/server.py`) that hosts OSWorld v1's `desktop_env` and drives the VM
locally. The cua-lite HOST needs ZERO `desktop_env` — it talks to the container over one
HTTP-RPC port (the androidworld pattern), so the host uv carries no OSWorld dep at all.

Only the HOST-side API port is dynamic (mapped to the fixed in-container :6000). The VM's own
ports (5000/9222/8080) stay INTERNAL — server.py reaches them at the guest IP 20.20.20.21. cua-lite owns the
container (naming + reaping); names go through :func:`lite.gym.utils.config.naming.format_container_name`
(`env_id="osworld"`, disjoint from `-osworld_2-` and lite `.osworld-`).

Run: not directly. Imported by ``lite/gym/envs/osworld/main.py``.
"""

from __future__ import annotations

import logging
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import requests

from lite.gym.container import LiteContainerBase
from lite.gym.errors import CapacityExhausted
from lite.gym.utils.config.identity import EnvIdentity
from lite.gym.utils.config.naming import format_container_name
from lite.gym.utils.backend.docker import _rm_argv, docker_run_detached
from lite.gym.utils.backend.ports import allocate_ports

logger = logging.getLogger(__name__)

_IMAGE_DEFAULT = "cua-lite/osworld:latest"
_CPORT_API = 6000       # docker/server.py listens here (fixed in-container)
# Dedicated host-port band (only the RPC API port is published; VM ports stay in the container).
# Non-overlapping per lite/gym/utils/backend/ports.py's range map — do NOT use the default 20000-20999
# (that's the sandbox noVNC band). osworld_2 uses 23000-23999.
_API_PORT_RANGE = (22000, 22999)
_DOCKER_RUN_TIMEOUT_S = 180.0


def _kvm_gid() -> int | None:
    try:
        import grp
        return grp.getgrnam("kvm").gr_gid
    except Exception:
        return None


def _server_py_path() -> str:
    return str(Path(__file__).resolve().parent / "docker" / "server.py")


# ── Container ─────────────────────────────────────────────────────────────────
@dataclass
class OSWorldContainer(LiteContainerBase):
    """One VM-in-Docker container + its in-container eval server.

    `OSWorldContainerFactory.build()` returns an un-started container (api port + name
    reserved, NOT booted); `container.start()` `docker run`s it, launches `docker/server.py`,
    and waits for `/healthz` vm_ready. The host drives eval over `http://localhost:{api_port}`.
    Registry / atexit / destroy() (rm -f -v + port release) come from
    :class:`LiteContainerBase`."""

    name: str
    api_port: int
    image: str = _IMAGE_DEFAULT
    qcow2_path: str = ""
    ram_size: str = "4G"
    cpu_cores: str = "4"
    disk_size: str = "32G"
    boot_timeout: float = 300.0
    screen_width: int = 1920
    screen_height: int = 1080
    client_password: str = "password"
    _ports_owned: tuple[int, ...] = field(default=(), repr=False)

    rm_label: ClassVar[str] = "osworld"

    @property
    def host(self) -> str:
        return "localhost"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.api_port}"

    def start(self) -> None:
        from lite.gym.remote.admission import docker_create_slot
        with docker_create_slot():
            self._start_locked()

    def _start_locked(self) -> None:
        self._register()   # atexit backstop covers the whole boot window below
        kvm_gid = _kvm_gid()
        docker_run_detached(
            name=self.name, image=self.image,
            auto_remove=True,
            cap_add=("NET_ADMIN",),
            devices=("/dev/kvm", "/dev/net/tun"),
            sysctls=("net.ipv4.ip_forward=1",),
            group_add=(kvm_gid,) if kvm_gid is not None else (),
            env={
                "DISK_SIZE": self.disk_size,
                "RAM_SIZE": self.ram_size,
                "CPU_CORES": self.cpu_cores,
                "SCREEN_W": str(self.screen_width),
                "SCREEN_H": str(self.screen_height),
                "CLIENT_PASSWORD": self.client_password,
                "SERVER_PORT": str(_CPORT_API),
                # Avoid qemu-docker falling back to user networking when host
                # inotify limits prevent dnsmasq from watching resolv.conf.
                "DNSMASQ_OPTS": "--no-resolv --no-poll",
            },
            volumes=(
                f"{self.qcow2_path}:/System.qcow2:ro",
                f"{_server_py_path()}:/usr/local/bin/server.py:ro",
            ),
            ports=((self.api_port, _CPORT_API),),
            # CLIENT_PASSWORD is a fixed literal here, but keep it redacted so
            # a future threaded secret can't leak (parity with osworld_2).
            redact=("CLIENT_PASSWORD", "OPENAI_API_KEY", "GITLAB_TOKEN"),
            timeout=_DOCKER_RUN_TIMEOUT_S,
            label="osworld",
        )
        self._launch_server()
        self._wait_ready()

    def _exclude_api_dnat(self) -> None:
        """The qemu-docker base image DNATs ALL container ports (except its own VNC) to the
        guest VM at 20.20.20.21, and (re)builds that DNAT DURING boot — flushing an earlier
        rule. So RE-ASSERT (idempotent check-then-insert) a PREROUTING RETURN for the API port,
        called each poll until it sticks — else host→:6000 is hijacked to the guest
        (RemoteDisconnected) and _wait_ready never sees the server."""
        try:
            subprocess.run(
                ["docker", "exec", self.name, "sh", "-c",
                 f"iptables -t nat -C PREROUTING -p tcp --dport {_CPORT_API} -j RETURN 2>/dev/null "
                 f"|| iptables -t nat -I PREROUTING 1 -p tcp --dport {_CPORT_API} -j RETURN"],
                capture_output=True, timeout=30,
            )
        except Exception:
            pass  # best-effort; re-asserted each poll, so a transient docker-exec stall just retries next iteration

    def _launch_server(self) -> None:
        r = subprocess.run(
            ["docker", "exec", "-d", self.name,
             "/opt/venv/bin/python", "/usr/local/bin/server.py"],
            capture_output=True, timeout=30,
        )
        if r.returncode != 0:
            stderr = (r.stderr or b"").decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"launch eval server in {self.name} failed: {stderr}")

    def _wait_ready(self) -> None:
        url = f"{self.base_url}/healthz"
        deadline = time.monotonic() + self.boot_timeout
        last = "no response"
        while time.monotonic() < deadline:
            self._exclude_api_dnat()   # re-assert until the boot-time DNAT stops flushing it
            try:
                resp = requests.get(url, timeout=(5, 5))
                if resp.status_code == 200 and resp.json().get("vm_ready"):
                    logger.info("osworld VM+server ready in %s", self.name)
                    return
                last = f"HTTP {resp.status_code} {resp.text[:80]}" if resp.status_code == 200 else f"HTTP {resp.status_code}"
            except Exception as e:
                last = f"{type(e).__name__}: {str(e)[:80]}"
            time.sleep(2.0)
        raise CapacityExhausted.warming(
            what=f"osworld VM/server in {self.name} not ready in {self.boot_timeout}s (last: {last})",
        )

# destroy() inherited from LiteContainerBase (template: _pre_destroy hook →
# docker rm -f -v → release _ports_owned → de-register; idempotent, never raises).


# ── Factory ───────────────────────────────────────────────────────────────────
@dataclass
class OSWorldContainerFactory:
    """Reserves ONE host port (api) + assigns the container name + pre-reaps any same-named
    zombie (all in `build()`, fast/sync); `container.start()` boots the VM + eval server."""

    qcow2_path: str
    image: str = _IMAGE_DEFAULT
    ram_size: str = "4G"
    cpu_cores: str = "4"
    disk_size: str = "32G"
    boot_timeout: float = 300.0
    screen_width: int = 1920
    screen_height: int = 1080
    client_password: str = "password"
    session_id: str | None = None
    token_hash: str | None = None
    server_port: int | None = None
    task_id: str | None = None

    def _make_name(self, suffix: str) -> str:
        identity = EnvIdentity(
            session_id=self.session_id, token_hash=self.token_hash, server_port=self.server_port,
        )
        return format_container_name(
            env_id="osworld", task_id=self.task_id, suffix=suffix,
            session_id=identity.resolved_session_id(), token_hash=identity.token_hash,
            server_port=identity.server_port,
        ).lower()

    def build(self) -> OSWorldContainer:
        (api_port,) = allocate_ports(n=1, range_start=_API_PORT_RANGE[0], range_end=_API_PORT_RANGE[1])
        name = self._make_name(uuid.uuid4().hex[:6])
        try:
            subprocess.run(_rm_argv(name), capture_output=True, timeout=60)
        except Exception:
            pass  # best-effort pre-reap (name is uuid-unique); never leak the reserved port on a daemon stall
        return OSWorldContainer(
            name=name, api_port=api_port, image=self.image, qcow2_path=self.qcow2_path,
            ram_size=self.ram_size, cpu_cores=self.cpu_cores, disk_size=self.disk_size,
            boot_timeout=self.boot_timeout, screen_width=self.screen_width,
            screen_height=self.screen_height, client_password=self.client_password,
            _ports_owned=(api_port,),
        )
