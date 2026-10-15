"""Setup function for lite_osworld tasks.

Executes OSWorld config steps via the shared dispatch (computer.interface).

Usage:
    from lite.gym.envs.lite.osworld.src.utils.setup import setup_fn
"""

from __future__ import annotations

import asyncio
import logging

from lite.gym.envs.lite.osworld.src.utils.dispatch import dispatch_actions
from lite.gym.sandbox.types import SandboxTaskConfig

logger = logging.getLogger(__name__)


async def _wait_flask_ready(computer, timeout: float = 60.0) -> None:
    """Block until the in-container osworld Flask server (port 5000) is bound.

    Config/postconfig/oracle ``execute`` + ``launch`` actions POST to
    http://localhost:5000 (dispatch.py:_flask_post). Reset gates on a single
    ``/tmp/gnome-ready`` marker probe (base.py READY_MARKER), but
    ``gnome-ready`` (supervisor priority 31) fires ~4-11s BEFORE
    ``osworld-server`` (priority 50) binds port 5000. Without this wait, the
    first config ``execute``/``launch`` steps fired right after reset would hit
    a not-yet-listening socket → ``curl`` rc=7 → the POST silently returns "" →
    setup.sh never runs (no testDir → oracle ``find`` rc=1), and chrome never
    launches + the History sqlite is never created (→ check_history_deleted
    trivially returns 1.0 = false trivial_pass). Regression surfaced on
    osworld_os_4d117223 + osworld_chrome_44ee5668.

    Measured: gnome-ready at +7s, port 5000 at +11.5s. This precondition closes
    that gap (osworld-only; the Flask server is the osworld image). download/
    run_command steps go over exec-stdio and don't need this, but config lists
    interleave execute steps, so we gate the whole config dispatch on it.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        r = await computer.interface.run_command(
            "curl -s -m 3 -o /dev/null -w '%{http_code}' -X POST "
            "http://localhost:5000/setup/execute "
            "-H 'Content-Type: application/json' "
            "-d '{\"command\":\"true\",\"shell\":true}'"
        )
        if r.stdout.strip() == "200":
            return
        await asyncio.sleep(1.0)
    logger.warning("osworld Flask server (port 5000) not ready after %.0fs; "
                   "config execute/launch steps may fail", timeout)


async def setup_fn(task: SandboxTaskConfig, computer) -> None:
    """Set up a lite_osworld task via OSWorld config steps."""
    # Wait for the Flask server to bind port 5000 before dispatching config
    # steps (see _wait_flask_ready docstring).
    await _wait_flask_ready(computer)
    # git/su/LibreOffice all run as the desktop user now (server is the user), so no
    # dubious-ownership workaround, no ownership-handover step, and the LO profile dir
    # is user-owned by construction — nothing to pre-arrange.
    # Suppress LibreOffice "You are running version X.Y for the first time" info-bar.
    # LO compares ooSetupLastVersion in setup.xcu against the installed version
    # (7.3.7.2); if missing it shows the bar every run.
    await computer.interface.run_command(
        "mkdir -p /home/user/.config/libreoffice/4/user && "
        "echo '<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<oor:items xmlns:oor=\"http://openoffice.org/2001/registry\">"
        "<item oor:path=\"/org.openoffice.Setup/Product\">"
        "<prop oor:name=\"ooSetupLastVersion\" oor:op=\"fuse\">"
        "<value>7.3.7.2</value></prop></item></oor:items>' "
        "> /home/user/.config/libreoffice/4/user/setup.xcu",
    )
    config = task.metadata.get("config", [])
    if config:
        await dispatch_actions(computer, config)
        # (No chown needed: every config step runs as the desktop user, so its
        # files are user-owned by construction — nothing is left root-owned to
        # hand back.)
        # Always disable VS Code workspace trust after all config steps run.
        # Tasks that open VS Code via terminal (execute type, not launch type)
        # also need this. For non-VS Code tasks this write is harmless.
        # Run last so task setup that writes settings.json doesn't overwrite it.
        await computer.interface.run_command(
            "mkdir -p /home/user/.config/Code/User && "
            "python3 -c \"import json, os; "
            "p = '/home/user/.config/Code/User/settings.json'; "
            "s = json.load(open(p)) if os.path.exists(p) else {}; "
            "s['security.workspace.trust.enabled'] = False; "
            "json.dump(s, open(p, 'w'), indent=2)\""
        )
        # Wait for UI to settle before taking the initial screenshot.
        # Apps (VS Code, LibreOffice, GIMP) may still be rendering.
        await asyncio.sleep(5)
    else:
        await _setup_via_commands(computer, task.metadata)


async def _setup_via_commands(computer, metadata: dict) -> None:
    """Fallback for train tasks using setup_commands."""
    for dir_path in metadata.get("create_dirs", []):
        await computer.interface.run_command(f"mkdir -p '{dir_path}'")

    for path, content in metadata.get("initial_files", {}).items():
        parent = path.rsplit("/", 1)[0]
        if parent:
            await computer.interface.run_command(f"mkdir -p '{parent}'")
        escaped = content.replace("'", "'\\''")
        await computer.interface.run_command(f"printf '%s' '{escaped}' > '{path}'")

    for cmd in metadata.get("setup_commands", []):
        trimmed = cmd.strip()
        if trimmed.startswith("sleep "):
            try:
                await asyncio.sleep(float(trimmed.split()[1]))
            except (IndexError, ValueError):
                await computer.interface.run_command(cmd)
        else:
            await computer.interface.run_command(cmd)
