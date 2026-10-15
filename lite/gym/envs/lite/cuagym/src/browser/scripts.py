"""Browser setup/evaluation functions for ``lite.cuagym``.

Browser tasks run in the same Docker desktop substrate as the desktop task family:
one Sandbox exec-stdio container per rollout, driven by coordinate actions, so
the agent sees a real Chrome window rather than a headless Playwright page. The
CUA-Gym-Hub mocks are baked into the shared ``cua-lite/lite.cuagym`` image
(``/opt/mocks/<app>_mock``) and served per task with ``vite preview`` (the mock's
state API is a Vite middleware).

Each task reuses its official bundle without semantic patches:
  - ``initial_setup.py`` POSTs the task's state to the local mock and launches
    ``google-chrome <url>`` (our wrapper opens it maximised on DISPLAY :0) — that
    IS the setup; the agent then drives the page by coordinate clicks.
  - ``reward.py`` GETs ``/go`` and prints ``REWARD: x``.

The only rewrite is materializing ``__CUA_GYM_<APP>_(URL|HOST)__`` placeholders to
the local ``vite preview`` port for each referenced app. One container per rollout
means the bundle's global ``/tmp/task_web_sid`` is already isolated — no salting.

Usage:
    from lite.gym.envs.lite.cuagym.src.browser.scripts import setup_fn, evaluate_final_fn
"""

from __future__ import annotations

import ast
import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from lite.gym.envs.lite.cuagym.src.utils.container import write_text
from lite.gym.envs.lite.cuagym.src.utils.display import bridge_display
from lite.gym.envs.lite.cuagym.src.utils.reward import REWARD_RE, parse_reward
from lite.gym.envs.lite.cuagym.src.utils.runtime import (
    ENV_PY,
    bundle_needs_judge,
    install_reward_judge,
    wait_for_new_window,
    window_ids,
)
from lite.gym.errors import CuaGymTaskError
from lite.gym.sandbox.exec_stdio.client import ExecStdioError

logger = logging.getLogger(__name__)

_COMMAND_TIMEOUT = 300.0
_VITE = "/opt/mocks/node_modules/vite/bin/vite.js"
_MOCKS = "/opt/mocks"
_PLACEHOLDER_RE = re.compile(r"__CUA_GYM_([A-Z0-9_]+)_(URL|HOST)__")
_URL_TEMPLATE = "__CUA_GYM_MOCK_APP_URL_TEMPLATE__"
#: A non-loopback http endpoint left behind by a bundle that hardcoded its author's
#: dev box instead of using the placeholder above. Loopback is excluded so a bundle
#: that legitimately names 127.0.0.1/localhost keeps its own port. The port is
#: captured because it, not the bundle's app list, says WHICH mock the URL meant.
_DEV_HOST_RE = re.compile(
    r"http://(?!127\.0\.0\.1|localhost)\d{1,3}(?:\.\d{1,3}){3}:(?P<port>\d+)"
)
#: Port each mock listened on upstream: CUA-Gym-Hub's ``deploy-all.sh`` assigns
#: ``BASE_PORT=8000`` + the mock's alphabetical deploy index, and each mock's
#: ``SCHEMA.md`` records the resulting base URL. Only used to attribute a hardcoded
#: ``_DEV_HOST_RE`` URL to ONE of a bundle's apps. Deliberately partial: the mocks
#: whose pinned SCHEMA.md omits its port (discord, google_docs, google_drive,
#: google_sheets, notion, shopify_admin, stripe_dashboard) or contradicts a sibling's
#: (gitlab and gmail both claim google_calendar's 8017) are left out — an app that is
#: absent here simply never wins the port match below.
_UPSTREAM_PORTS: dict[str, int] = {
    "asana": 8002, "docusign": 8012, "facebook": 8014, "github": 8015,
    "google_calendar": 8017, "hubspot": 8022, "instacart": 8023, "instagram": 8024,
    "linkedin": 8027, "microsoft_teams": 8028, "jira": 8029, "monday": 8030,
    "outlook_web": 8033, "pinterest": 8036, "postman": 8037, "reddit": 8043,
    "salesforce": 8044, "slack": 8047, "trello": 8049, "twitter": 8050,
    "uber_eats": 8052, "wechat": 8057,
}
_KNOWN_APPS = (
    "asana", "discord", "docusign", "facebook", "github", "gitlab", "gmail",
    "google_calendar", "google_docs", "google_drive", "google_sheets", "hubspot",
    "instacart", "instagram", "jira", "linkedin", "microsoft_teams", "monday",
    "notion", "outlook_web", "pinterest", "postman", "reddit", "salesforce",
    "shopify_admin", "slack", "stripe_dashboard", "trello", "twitter",
    "uber_eats", "wechat",
)


def _canonical_app(app: str) -> str:
    return app.lower().removesuffix("_mock")


def _port_for(app: str) -> int:
    """Deterministic port per app — setup and evaluate compute the same one, so
    evaluate reuses the server setup started (both run in one container)."""
    app = _canonical_app(app)
    try:
        return 22000 + _KNOWN_APPS.index(app)
    except ValueError as exc:
        raise ValueError(f"unsupported CUA-Gym mock app: {app!r}") from exc


def _referenced_apps(src: str, extra_apps: list[str] | tuple[str, ...] = ()) -> list[str]:
    """Every mock app a bundle references (``__CUA_GYM_<APP>_URL__`` → ``<app>``).
    ~13% of tasks cross-reference other apps' mocks, so we scan, not assume one."""
    apps = {_canonical_app(app) for app in extra_apps}
    apps.update(_canonical_app(m.group(1)) for m in _PLACEHOLDER_RE.finditer(src))
    return sorted(apps)


def _materialize(src: str, apps: list[str] | tuple[str, ...] = ()) -> str:
    """Replace every placeholder with the app's local ``vite preview`` base."""

    def repl(m: re.Match) -> str:
        app = _canonical_app(m.group(1))
        base = f"http://127.0.0.1:{_port_for(app)}"
        return base.removeprefix("http://") if m.group(2) == "HOST" else base

    out = _PLACEHOLDER_RE.sub(repl, src)
    # 13 bundles hardcode the upstream author's dev-box endpoint (e.g.
    # http://172.17.46.46:8028) instead of the __CUA_GYM_<APP>_URL__ placeholder, so
    # the substitution above has nothing to rewrite and the script dials an address
    # that does not exist here — `ConnectionError: No route to host` in setup (fatal),
    # in a chrome step, or a deterministic `REWARD: 0.0`. Loopback and
    # placeholder-derived URLs are left alone.
    #
    # 12 of the 13 reference exactly one app, but the 13th
    # (5efc831f-7c65-5aa2-950f-a5f2991e478b) is a cross-app slack+github bundle whose
    # hardcoded chrome landing URL is on slack's upstream port 8047; picking sorted
    # `apps[0]` sent the agent to the github mock instead. So attribute the URL by its
    # PORT, which is the only part of it that identifies a mock.
    if apps:
        candidates = [_canonical_app(app) for app in apps]

        def dev_host_repl(m: re.Match) -> str:
            port = int(m.group("port"))
            matched = [app for app in candidates if _UPSTREAM_PORTS.get(app) == port]
            if not matched:
                # No upstream port match: fine when the bundle names one app (the URL
                # can only have meant that mock — this is the other 12), a coin flip
                # otherwise, so refuse rather than rewrite to the wrong mock.
                if len(candidates) != 1:
                    raise ValueError(
                        f"cannot attribute hardcoded {m.group(0)} to one of {candidates}"
                    )
                matched = candidates
            return f"http://127.0.0.1:{_port_for(matched[0])}"

        out = _DEV_HOST_RE.sub(dev_host_repl, out)
    if _URL_TEMPLATE in out:
        canonical = _referenced_apps(out, apps)
        if not canonical:
            raise ValueError(f"{_URL_TEMPLATE} requires task metadata.others.apps")
        ports = {app: _port_for(app) for app in canonical}
        helper = (
            f"__cua_gym_ports = {ports!r}\n"
            "def __cua_gym_mock_url(app):\n"
            "    return f\"http://127.0.0.1:{__cua_gym_ports[app.lower().removesuffix('_mock')]}\"\n"
        )
        tree = ast.parse(out)
        app_variables = {
            node.args.args[0].arg
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.args.args
            and _URL_TEMPLATE in (ast.get_source_segment(out, node) or "")
        }
        if len(app_variables) != 1:
            raise ValueError(
                f"{_URL_TEMPLATE} must appear in one function with an app argument"
            )
        app_variable = app_variables.pop()
        out = helper + out.replace(
            _URL_TEMPLATE,
            f"{{__cua_gym_mock_url({app_variable})}}",
        )
    unresolved = sorted(set(re.findall(r"__CUA_GYM_[A-Z0-9_]+__", out)))
    if unresolved:
        raise ValueError(f"unresolved CUA-Gym placeholders: {unresolved}")
    return out


def _load_bundle_script(
    path: Path,
    apps: list[str] | tuple[str, ...],
    *,
    phase: str,
) -> tuple[str, str]:
    try:
        source = path.read_text()
        return source, _materialize(source, apps)
    except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
        raise CuaGymTaskError(
            f"lite.cuagym browser {phase} bundle is invalid: "
            f"{type(exc).__name__}: {exc}",
            phase=phase,
            kind="invalid_bundle",
        ) from exc


async def _start_mock(computer, app: str) -> None:
    """Start ``vite preview`` for one mock (idempotent — skip if already up)."""
    port = _port_for(app)
    app_dir = f"{_MOCKS}/{app}_mock"
    result = await computer.interface.run_command(
        f"test -f {app_dir}/dist/index.html || exit 66; "
        f"curl -sf http://127.0.0.1:{port}/state?sid=__h__ -o /dev/null 2>/dev/null "
        f"|| (cd {app_dir} && setsid node {_VITE} preview --host 127.0.0.1 --port {port} "
        f"--strictPort >/tmp/mock_{app}.log 2>&1 < /dev/null &)"
    )
    if result.returncode != 0:
        raise CuaGymTaskError(
            f"lite.cuagym browser mock {app!r} is unavailable in the pinned image",
            phase="setup",
            kind="missing_mock",
            returncode=result.returncode,
        )


async def _wait_mock(computer, app: str, timeout: int = 40) -> bool:
    port = _port_for(app)
    for _ in range(timeout):
        r = await computer.interface.run_command(
            f"curl -sf http://127.0.0.1:{port}/state?sid=__h__ -o /dev/null && echo UP || echo DOWN"
        )
        if "UP" in (r.stdout or ""):
            return True
        await asyncio.sleep(1)
    logger.warning(
        "lite.cuagym browser task: mock %s did not come up on port %d",
        app,
        port,
    )
    return False


async def _ensure_mocks(
    computer,
    src: str,
    extra_apps: list[str] | tuple[str, ...] = (),
) -> list[str]:
    apps = _referenced_apps(src, extra_apps)
    try:
        for app in apps:
            _port_for(app)
    except ValueError as exc:
        raise CuaGymTaskError(
            f"lite.cuagym browser bundle references an unsupported mock: {exc}",
            phase="setup",
            kind="invalid_bundle",
        ) from exc
    for app in apps:
        await _start_mock(computer, app)
    for app in apps:
        if not await _wait_mock(computer, app):
            raise CuaGymTaskError(
                f"lite.cuagym browser mock {app!r} failed to start on "
                f"port {_port_for(app)}",
                phase="setup",
                kind="mock_start_failed",
            )
    return apps


async def _open_missing_app_tabs(computer, apps: list[str]) -> None:
    """Make every cross-app target discoverable without changing task semantics."""
    if len(apps) < 2:
        return
    ports = " ".join(str(_port_for(app)) for app in apps)
    command = f"""
set -e
# Tolerate a missing/odd sid file. This block only OPENS EXTRA TABS so a cross-app
# task's other mocks are discoverable; under `set -e` a bundle that writes per-app
# sid files (or none) aborted the whole setup and discarded the episode. Losing the
# extra tabs is a far smaller harm than losing the task.
sid="$(cat /tmp/task_web_sid 2>/dev/null | head -1 || true)"
pages="$(curl -sf http://127.0.0.1:9222/json)"
urls=()
for port in {ports}; do
  if ! printf '%s' "$pages" | /opt/env/bin/jq -e --arg port ":$port/" \
      'any(.[]; (.url // "") | contains($port))' >/dev/null; then
    case "$port" in
"""
    for app in apps:
        port = _port_for(app)
        command += f'      {port}) urls+=("http://127.0.0.1:{port}/?sid=$sid") ;;\n'
    command += """    esac
  fi
done
if [ "${#urls[@]}" -gt 0 ]; then
  DISPLAY=:0 google-chrome --new-tab "${urls[@]}"
  sleep 1
  for _ in "${urls[@]}"; do
    DISPLAY=:0 /opt/env/bin/xdotool key ctrl+shift+Tab
  done
fi
"""
    # jq/xdotool are env-vendored (→ /opt/env/bin, resolved via the server's PATH);
    # the bare `google-chrome --new-tab` runs as the desktop user and targets the
    # user's running browser.
    result = await _run_task_command(
        computer,
        command,
        "cross-app tabs",
        error_type=CuaGymTaskError,
    )
    _raise_for_command(
        result,
        "cross-app tabs",
        error_type=CuaGymTaskError,
    )


def _raise_for_command(
    result,
    phase: str,
    *,
    error_type: type[CuaGymTaskError] = CuaGymTaskError,
) -> None:
    if getattr(result, "returncode", 0) == 0:
        return
    detail = (getattr(result, "stderr", "") or getattr(result, "stdout", "") or "").strip()
    raise error_type(
        f"lite.cuagym browser {phase} failed with rc={result.returncode}: {detail[-500:]}",
        phase=phase,
        kind="timeout" if result.returncode in {124, 137} else "command_failed",
        returncode=result.returncode,
    )


async def _run_task_command(
    computer,
    command: str,
    phase: str,
    *,
    error_type: type[CuaGymTaskError] = CuaGymTaskError,
):
    try:
        return await computer.interface.run_command(
            command,
            timeout=_COMMAND_TIMEOUT,
        )
    except (TimeoutError, ExecStdioError) as exc:
        if isinstance(exc, TimeoutError) or "timeoutexpired" in str(exc).lower():
            raise error_type(
                f"lite.cuagym browser {phase} exceeded {_COMMAND_TIMEOUT:.0f}s",
                phase=phase,
                kind="timeout",
            ) from exc
        raise


async def setup_fn(task, computer) -> None:
    """Start the task's mock(s), run the official initial_setup.py (inject state +
    open Chrome maximised on DISPLAY :0)."""
    # CUA-Gym scripts hardcode DISPLAY=:0; the image's Xvnc is :1 — bridge it.
    await bridge_display(computer)
    existing_windows = await window_ids(computer)
    apps = task.metadata.get("others", {}).get("apps", [])
    src, rendered = _load_bundle_script(
        Path(task.metadata["setup"]),
        apps,
        phase="setup",
    )
    referenced_apps = await _ensure_mocks(computer, src, apps)
    await write_text(computer, rendered, "/tmp/initial_setup.py")
    # The browser setup script runs as the desktop user under ENV_PY (py3.12), so its
    # `google-chrome <url>` launch targets the user's browser directly. Web bundles
    # write only to /tmp + POST to the local mocks and never import UNO/gi, so ENV_PY
    # always.
    interp = ENV_PY
    result = await _run_task_command(
        computer,
        f"cd /tmp && DISPLAY=:0 {interp} /tmp/initial_setup.py",
        "setup",
        error_type=CuaGymTaskError,
    )
    _raise_for_command(result, "setup", error_type=CuaGymTaskError)
    if not await wait_for_new_window(computer, existing_windows):
        raise CuaGymTaskError(
            "lite.cuagym browser Chrome window did not appear",
            phase="setup",
            kind="no_task_window",
        )
    await _open_missing_app_tabs(computer, referenced_apps)
    page_health = await _run_task_command(
        computer,
        "/opt/env/bin/lite-cuagym-page-health --timeout 20",
        "page render",
        error_type=CuaGymTaskError,
    )
    _raise_for_command(
        page_health,
        "page render",
        error_type=CuaGymTaskError,
    )


async def evaluate_final_fn(task, computer) -> tuple[float, dict[str, Any]]:
    """Run the task's official reward.py against the local mock(s), parse REWARD.

    Returns ``(reward, eval_info)`` unconditionally; see the desktop counterpart for
    why this does not take the sibling envs' ``debug`` flag.
    """
    apps = task.metadata.get("others", {}).get("apps", [])
    src, rendered = _load_bundle_script(
        Path(task.metadata["reward"]),
        apps,
        phase="reward",
    )
    await _ensure_mocks(computer, src, apps)  # usually already up from setup; cheap no-op
    # 36 of the 1261 live browser tasks import `/tmp/reward_judge.py` for their LLM-judged
    # components. Install it only for those, so the other 1225 rewards are unchanged.
    if bundle_needs_judge(src):
        await install_reward_judge(computer)
    await write_text(computer, rendered, "/tmp/reward.py")
    # The browser reward runs as the desktop user under ENV_PY (browser bundles never import
    # gi/uno); it reads the mock's HTTP state over localhost.
    interp = ENV_PY
    # PYTHONIOENCODING mirrors the desktop path: these bundles are LLM-authored and
    # at least one desktop reward emits a surrogate PAIR as two lone `\uXXXX`
    # escapes in a print banner, which UnicodeEncodeErrors against the redirected
    # stdout and kills the whole evaluation. No browser bundle carries one in the
    # current pinned snapshot
    # (0 of 1505), so this is pre-emptive — but the asymmetry between the two
    # reward invocations was unintended, and a materials bump could introduce one.
    r = await _run_task_command(
        computer,
        f"cd /tmp && PYTHONIOENCODING=utf-8:backslashreplace {interp} /tmp/reward.py",
        "reward",
    )
    _raise_for_command(r, "reward")
    out = r.stdout or ""
    if not REWARD_RE.search(out):
        raise CuaGymTaskError(
            f"lite.cuagym browser reward produced no REWARD sentinel: {out[-500:]}",
            phase="reward",
            kind="no_reward",
        )
    try:
        # Carry the reward script's own per-component diagnostics into
        # `info["eval"]` -> `04_results.json`; see the desktop counterpart for why
        # it is stored as bounded lines rather than one oversized blob.
        return parse_reward(out), {"reward_stdout": out.splitlines()[-40:]}
    except ValueError as exc:
        raise CuaGymTaskError(
            f"lite.cuagym browser reward is invalid: {exc}",
            phase="reward",
            kind="invalid_reward",
        ) from exc
