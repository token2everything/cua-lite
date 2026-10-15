"""lite.demo — Simple demo tasks for CUA-Lite.

A handful of representative desktop tasks that demonstrate different CUA
capabilities. Each task is simple (<10 steps), has a deterministic evaluator,
and an oracle for validation.

Usage:
    uv run python -c "import lite.gym as gym; print(gym.registry.task_ids('lite.demo'))"
    uv run python scripts/rollout.py --model-id gpt-5.5 --env-id lite.demo --task-id create_file
"""

from __future__ import annotations

from pathlib import Path

from lite.gym.utils import config as env_config
from lite.gym.sandbox import SandboxTaskConfig, register_tasks

ENV_DIR = str(Path(__file__).parent)
CFG = env_config.load(ENV_DIR)

# ============================================================================
# Config defaults — every value below is read once from configs/default.yaml
# via env_config.load(ENV_DIR). Swap the whole file at startup with
# LITE_DEMO_CONFIG=<abs-path | bundled-name>. A rollout's env_kwargs still
# override per run; these are only registration defaults.
# ============================================================================
# --- env_kwargs (per-instance) ---
_POST_ACTION_DELAY = CFG.env_kwargs["post_action_delay"]
_IMAGE = CFG.env_kwargs["computer"]["image"]
# Composed computer_config (reused per task). ``display`` is the "WxH" STRING
# the container shape expects, built from the yaml display_resolution list.
# ``timeout`` is the boot-readiness ceiling — how long the DockerProvisioner
# polls for desktop readiness over ``docker exec``.
_c = CFG.env_kwargs["computer"]
_COMPUTER_CONFIG = {
    "image": _IMAGE,
    "display": f"{_c['display_resolution'][0]}x{_c['display_resolution'][1]}",
    "memory": _c["memory"],
    "cpu": _c["cpu"],
    "timeout": _c["timeout"],
}
# server_kwargs: none — reaper/pool/admission are framework-owned.
# ============================================================================


def _ensure_services(env_id: str) -> None:
    """env-server startup hook: verify the demo docker image is pulled.

    Symmetric with lite.osworld / android_{lab,world}'s same-named hook.
    The image is the sandbox family desktop base, built locally from
    ``lite/gym/sandbox/docker/Dockerfile.linux``.
    """
    from lite.gym.utils.backend.docker import require_image_present
    from lite.gym.utils.backend.freshness import image_for
    require_image_present(image_for("lite.demo", tag=_IMAGE))


# =============================================================================
# Task 1: create_file — Terminal file creation
# =============================================================================

async def _create_file_setup(task: SandboxTaskConfig, computer) -> None:
    await computer.interface.run_command("rm -f /home/user/Desktop/hello.txt")


async def _create_file_eval(task: SandboxTaskConfig, computer) -> float:
    result = await computer.interface.run_command(
        "cat /home/user/Desktop/hello.txt 2>/dev/null"
    )
    content = getattr(result, "stdout", "").strip()
    if "Hello, World!" in content:
        return 1.0
    if content:
        return 0.5  # file exists but wrong content
    return 0.0


async def _create_file_solve(task: SandboxTaskConfig, computer) -> None:
    await computer.interface.run_command(
        "echo 'Hello, World!' > /home/user/Desktop/hello.txt"
    )


# =============================================================================
# Task 2: rename_file — File rename
# =============================================================================

async def _rename_setup(task: SandboxTaskConfig, computer) -> None:
    await computer.interface.run_command(
        "mkdir -p /home/user/Documents && "
        "echo 'quarterly report draft' > /home/user/Documents/report.txt && "
        "rm -f /home/user/Documents/summary.txt"
    )


async def _rename_eval(task: SandboxTaskConfig, computer) -> float:
    result = await computer.interface.run_command(
        "test -f /home/user/Documents/summary.txt && "
        "! test -f /home/user/Documents/report.txt && "
        "echo PASS || echo FAIL"
    )
    output = getattr(result, "stdout", "").strip()
    return 1.0 if "PASS" in output else 0.0


async def _rename_solve(task: SandboxTaskConfig, computer) -> None:
    await computer.interface.run_command(
        "mv /home/user/Documents/report.txt /home/user/Documents/summary.txt"
    )


# =============================================================================
# Task 3: change_wallpaper — GNOME system settings
# =============================================================================

async def _wallpaper_setup(task: SandboxTaskConfig, computer) -> None:
    # Reset to the default image-based wallpaper (GNOME: a non-empty
    # picture-uri means an image backdrop is in effect).
    await computer.interface.run_command(
        "export DBUS_SESSION_BUS_ADDRESS=$(cat /tmp/dbus-session-bus-address); "
        "gsettings set org.gnome.desktop.background picture-uri "
        "'file:///usr/share/backgrounds/warty-final-ubuntu.png'"
    )


async def _wallpaper_eval(task: SandboxTaskConfig, computer) -> float:
    # Solid color = empty picture-uri (no image backdrop).
    result = await computer.interface.run_command(
        "export DBUS_SESSION_BUS_ADDRESS=$(cat /tmp/dbus-session-bus-address); "
        "gsettings get org.gnome.desktop.background picture-uri"
    )
    output = getattr(result, "stdout", "").strip()
    return 1.0 if output in ("''", '""') else 0.0


async def _wallpaper_solve(task: SandboxTaskConfig, computer) -> None:
    await computer.interface.run_command(
        "export DBUS_SESSION_BUS_ADDRESS=$(cat /tmp/dbus-session-bus-address); "
        "gsettings set org.gnome.desktop.background picture-uri \"\""
    )


# =============================================================================
# Task 4: find_and_write — Multi-step: count files + write result
# =============================================================================

async def _find_write_setup(task: SandboxTaskConfig, computer) -> None:
    await computer.interface.run_command(
        "rm -rf /home/user/Documents/project && "
        "mkdir -p /home/user/Documents/project && "
        "touch /home/user/Documents/project/{notes,todo,readme,changelog,license}.txt && "
        "rm -f /home/user/Desktop/count.txt"
    )


async def _find_write_eval(task: SandboxTaskConfig, computer) -> float:
    result = await computer.interface.run_command(
        "cat /home/user/Desktop/count.txt 2>/dev/null"
    )
    output = getattr(result, "stdout", "").strip()
    return 1.0 if output == "5" else 0.0


async def _find_write_solve(task: SandboxTaskConfig, computer) -> None:
    await computer.interface.run_command(
        "ls /home/user/Documents/project/*.txt | wc -l > /home/user/Desktop/count.txt"
    )


# =============================================================================
# Task 5: create_directory_structure — Nested directory creation
# =============================================================================

async def _mkdir_setup(task: SandboxTaskConfig, computer) -> None:
    await computer.interface.run_command(
        "rm -rf /home/user/Desktop/myproject"
    )


async def _mkdir_eval(task: SandboxTaskConfig, computer) -> float:
    result = await computer.interface.run_command(
        "test -d /home/user/Desktop/myproject/src && "
        "test -d /home/user/Desktop/myproject/docs && "
        "test -d /home/user/Desktop/myproject/tests && "
        "echo PASS || echo FAIL"
    )
    output = getattr(result, "stdout", "").strip()
    return 1.0 if "PASS" in output else 0.0


async def _mkdir_solve(task: SandboxTaskConfig, computer) -> None:
    await computer.interface.run_command(
        "mkdir -p /home/user/Desktop/myproject/{src,docs,tests}"
    )


# =============================================================================
# Registration
# =============================================================================

register_tasks("lite.demo", {
    "eval": [
        SandboxTaskConfig(
            task_id="create_file",
            instruction=(
                "Create a text file called hello.txt on the Desktop "
                "with the content 'Hello, World!' (exactly). "
                "You can use the terminal."
            ),
            computer=_COMPUTER_CONFIG,
            max_steps=10,
            setup_fn=_create_file_setup,
            evaluate_final_fn=_create_file_eval,
            solve_fn=_create_file_solve,
        ),
        SandboxTaskConfig(
            task_id="rename_file",
            instruction=(
                "Rename the file report.txt in ~/Documents to summary.txt. "
                "The original file should no longer exist after renaming."
            ),
            computer=_COMPUTER_CONFIG,
            max_steps=10,
            setup_fn=_rename_setup,
            evaluate_final_fn=_rename_eval,
            solve_fn=_rename_solve,
        ),
        SandboxTaskConfig(
            task_id="change_wallpaper",
            instruction=(
                "Change the desktop wallpaper to a solid color "
                "(remove the background image)."
            ),
            computer=_COMPUTER_CONFIG,
            max_steps=10,
            setup_fn=_wallpaper_setup,
            evaluate_final_fn=_wallpaper_eval,
            solve_fn=_wallpaper_solve,
        ),
        SandboxTaskConfig(
            task_id="find_and_write",
            instruction=(
                "Count the number of .txt files in ~/Documents/project/ "
                "and write just the number to ~/Desktop/count.txt. "
                "For example, if there are 3 files, count.txt should contain '3'."
            ),
            computer=_COMPUTER_CONFIG,
            max_steps=10,
            setup_fn=_find_write_setup,
            evaluate_final_fn=_find_write_eval,
            solve_fn=_find_write_solve,
        ),
        SandboxTaskConfig(
            task_id="create_directories",
            instruction=(
                "Create a project directory structure on the Desktop: "
                "~/Desktop/myproject/ with three subdirectories: src, docs, and tests."
            ),
            computer=_COMPUTER_CONFIG,
            max_steps=10,
            setup_fn=_mkdir_setup,
            evaluate_final_fn=_mkdir_eval,
            solve_fn=_mkdir_solve,
        ),
    ],
}, factory_kwargs={"post_action_delay": _POST_ACTION_DELAY})


from lite.gym.remote.reaper import ContainerServices  # noqa: E402
from lite.gym.services import register_services  # noqa: E402


class LiteDemoServices(ContainerServices):
    """Env-server capabilities for lite.demo: docker-image presence check
    (``ensure``) + per-container reconciliation (``live_ids``/``reap`` inherited
    from :class:`ContainerServices`)."""

    def ensure(self, env_id: str) -> None:
        _ensure_services(env_id)


register_services("lite.demo", LiteDemoServices())
from lite.gym.services import BackendFamily, register_family  # noqa: E402
register_family("lite.demo", BackendFamily.DEDICATED)
