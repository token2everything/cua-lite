"""CAPTCHA environments — agent must solve visual CAPTCHA challenges.

One env_id (``captcha``), eight challenge categories — one per Flask server
in ``servers/`` (Playwright-driven, no Docker): text_captcha_4 / slider /
rotation / icon_click / math / image_select / icon_match / paged. Every
task carries ``metadata.others["category"]``; select categories at
parquet-export time via ``--filter`` (no per-category env_ids).

Split semantics ARE distribution semantics: train + eval share the default
`train_eval.json` distribution; split="test" is always out-of-distribution.
Per-category task families:
  - `<id>_local`               (split=eval)  — fixed seed `_EVAL_SEED`.
  - `<id>_local_eval{0..31}`   (split=eval)  — 32 prime-spaced seeded variants.
  - `random_<id>_local`        (split=train) — fresh random challenge per reset.
  - `random_<id>_<mode>_local`  (split=train) — per extra `train_<mode>.json`.
  - `<id>_held_out_local_test{0..31}` (split=test, when `test/held_out.json`
    exists)                                     — OOD final test.

``metadata.others["mode"]`` is the fine-grained filter key below category:
it tags content modes — image_select's data organizations (crop / full /
halligan; halligan = its real-data OOD test set) and alternative training
distributions (`easy` from `train_easy.json`). Tasks on the default
`train_eval` distribution carry no tag; image_select is the exception with
no untagged default (crop and full are both standard).

Prerequisites (deps + chromium + assets, idempotent):
    uv run --no-sync bash lite/gym/envs/captcha/scripts/install.sh

Usage:
    uv run python -c "import lite.gym as gym; print(gym.registry.task_ids('captcha'))"
"""

from __future__ import annotations

import asyncio
import atexit
import dataclasses
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, ClassVar

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.calls import RuntimeEnvAction
from lite.core.tools.extra_tools import LiteFinishToolSet
from lite.core.tools.schemas import BaseTools
from lite.gym.base import LiteBaseEnv
from lite.gym.errors import EnvDepsMissingError
from lite.gym.registry import register, registry
from lite.gym.services import EnvServerResource, EnvServices, register_services
from lite.gym.types import (
    EXECUTED_ACTIONS_INFO_KEY,
    LiteEnvObservation,
    LiteEnvStepResult,
    LiteExecutedAction,
)
from lite.gym.utils import config as env_config
from lite.gym.utils.backend.coordinate import norm_to_pixel
from lite.gym.utils.backend.model_inputs import (
    coerce_model_duration,
    project_model_keys,
)
from lite.gym.utils.backend.reaper import reap
from lite.gym.utils.feedback.errors import (
    MODEL_ACTION_ERROR_TYPES,
    ToolErrorFeedback,
    append_feedback,
    error_only_feedback,
    record_model_action_error,
    record_tool_execution_error,
    unsupported_action_message,
)
from lite.gym.utils.feedback.ingress import (
    invalid_action_message,
    prepare_env_tool_calls,
    standalone_tool_call_feedback_with_reason,
)
from lite.gym.utils.feedback.results import (
    build_tool_results_from_decisions,
    ordered_tool_call_ids,
)
from lite.gym.utils.feedback.surface import (
    copy_valid_actions,
    resolve_extra_tools,
    resolve_valid_actions,
)
from lite.gym.utils.server.routing import routing_server_url
from lite.gym.wrappers import overlay_cursor_px
from lite.utils.image import encode_png

logger = logging.getLogger(__name__)

# Cursor origin at session start, in the canonical [0, 1000] normalized space:
# the viewport CENTRE. A real desktop session warps the pointer to screen centre
# at start and a browser inherits it, so turn 0 must show the cursor there — not
# at top-left (0, 0), which is the "never moved" sentinel and not a real state.
# Same seed as waa / cua.bench / cua.sandbox (all ``w // 2, h // 2``) and as
# ``norm_to_pixel``'s own malformed-coordinate fallback.
_CURSOR_ORIGIN_NORM: tuple[int, int] = (500, 500)

ENV_DIR = str(Path(__file__).parent)
CFG = env_config.load(ENV_DIR)

# ============================================================================
# Config defaults — every value below is read once from configs/default.yaml
# via env_config.load(ENV_DIR). Swap the whole file at startup with
# CAPTCHA_CONFIG=<abs-path | bundled-name>. A rollout's env_kwargs still
# override per run; these are only registration defaults.
# ============================================================================
# --- env_kwargs (per-instance) ---
# null: captcha registers MANY categories whose budget differs
# (10 or 15), so the concrete value is hardcoded at the registration site
# (_register_local per _CATEGORIES). A concrete int here overrides every category.
_MAX_STEPS          = CFG.env_kwargs["max_steps"]
_POST_ACTION_DELAY  = CFG.env_kwargs["post_action_delay"]
_DISPLAY_RESOLUTION = tuple(CFG.env_kwargs["display_resolution"])
# null: unseeded by default; the eval split pins a fixed
# per-category seed at registration (_EVAL_SEED / _TEST_SEED below).
_SEED               = CFG.env_kwargs["seed"]
_HEADLESS           = CFG.env_kwargs["headless"]
_EXTRA_TOOLS        = CFG.env_kwargs["extra_tools"]
# --- server_kwargs (per-deployment) ---
_SERVER_START_TIMEOUT = CFG.server_kwargs["server_start_timeout"]
# ============================================================================

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CAPTCHA_DIR = Path(__file__).parent

# Fixed seed used for eval variants (chosen arbitrarily; must be deterministic).
_EVAL_SEED = 42

# Disjoint seed base for the final held-out test set (split="test"). Must be
# far from _EVAL_SEED so generative captchas draw fresh samples the training
# loop has never seen. 10_000_003 is prime and sits well outside the
# _EVAL_SEED + i*7919 range (max ≈ 245k for i=31).
_TEST_SEED = 10_000_003

# ---------------------------------------------------------------------------
# Asset auto-download
# ---------------------------------------------------------------------------

# Runtime assets live under the env's git-ignored ``.cache/`` (same convention as
# mobilegym / browsergym); ``scripts/install.sh`` populates it. They are NEVER
# auto-downloaded at import — that blocking snapshot_download froze the event loop
# via the reaper's _import_all(); _register_tasks() fails loud if they're absent.
_ASSETS_DIR = _CAPTCHA_DIR / ".cache" / "assets"


# ---------------------------------------------------------------------------
# LocalCaptchaEnv — Playwright-driven episode runtime
# ---------------------------------------------------------------------------

# Flask challenge servers live in servers/; each is spawned in a fresh
# subprocess, loaded by file path so they stay self-contained.
_SERVER_DIR = Path(__file__).parent / "servers"

# The GUI interaction verbs CAPTCHA solving uses (incl. ``drag`` —
# slider/rotation/icon_match REQUIRE it; ``_dispatch_action`` handles them).
# Finish tools are exposed only via explicit env_kwargs.extra_tools.
_VALID_ACTIONS = resolve_valid_actions(
    CFG.env_kwargs.get("valid_actions"),
    env_name="captcha", platform="browser",
)
class CaptchaTools(BaseTools):
    """What captcha declares beyond the GUI surface: nothing — CAPTCHA solving is
    GUI-only, and this env still declares the (empty) set so a yaml naming an
    unavailable extra tool raises instead of resolving to a silent no-op."""

    _SCHEMAS: ClassVar[dict[str, dict[str, Any]]] = {}


#: Finish tools cannot live in an env's own set, so the union is not optional.
_KNOWN_STANDALONE_TOOL_NAMES = CaptchaTools.get_tool_names() | LiteFinishToolSet.get_tool_names()


# Playwright receives projected wire key names for specials (e.g. "Enter", "ArrowUp").
def _server_scope() -> str:
    """Isolation id = the owning env-server's listen port (``str``), or
    ``"direct"`` when there's no env-server in the loop (direct rollout).

    ``CUA_LITE_ENV_SERVER_PORT`` is exported into every worker by
    ``lite.gym.remote.server.create_app`` (same ambient var webgym/mobilegym
    scope by). Baked into the Flask subprocess marker + ``/tmp`` file names so
    a concurrent env-server's boot reap can't kill our procs or delete our
    in-flight result files. Must match ``scope.server_port`` (str-cast) that
    ``_reap_zombies`` greps for."""
    p = os.environ.get("CUA_LITE_ENV_SERVER_PORT")
    return p if p else "direct"


def _scope_str(server_port: int | None) -> str:
    """Normalize a scope's ``server_port`` (the value the env-server threads
    into ``reap``/``live_ids``) to the SAME string ``_server_scope`` derives
    from the ambient ``CUA_LITE_ENV_SERVER_PORT``. ``None`` (no env-server in
    the loop) maps to ``"direct"`` so direct-mode and server-mode share one
    id-space."""
    return str(server_port) if server_port is not None else "direct"


def _server_marker(scope: str) -> str:
    """The cmdline tag baked into each Flask challenge-server subprocess.

    The ``_es<scope>_`` infix makes every pgrep/reap SERVER-SCOPED: a
    concurrent env-server (different listen port) matches only its OWN procs,
    so its boot reap can't kill our live Flask servers. Single source of truth
    shared by the writer (``reset``'s ``Popen``) and all readers
    (``_reap_zombies``, ``_flask_server_procs``)."""
    return f"CUA_LITE_CAPTCHA_SERVER_es{scope}_ = 1"


def _find_free_port() -> int:
    """Find a free TCP port from the OS ephemeral range.

    captcha spawns a short-lived Flask server PER EPISODE, so it needs a large,
    auto-recycling port pool. The shared flock allocator (lite.gym.utils.backend.ports)
    targets long-lived docker envs and only covers ~1000 ports, which exhausts
    under captcha's per-episode churn — so we use the OS ephemeral range here.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── Process-wide registry + atexit hook ──────────────────────────────────────
# A LocalCaptchaEnv spawns a Flask challenge server (a plain Popen that does NOT
# self-terminate on parent death — it would listen forever) plus a Playwright
# chromium. close() reclaims both, but a crash/exit between reset() and close()
# would orphan the Flask proc. Mirror the atexit registries in
# androidworld / androidlab / sandbox.base (same snapshot-under-lock structure):
# track live envs and terminate their Flask proc at interpreter exit. The
# ``_LIVE_LOCK`` guards the set because the env-server mutates it from
# ``asyncio.to_thread`` workers. (The chromium is reclaimed by Playwright's
# driver-pipe teardown on exit and, if orphaned mid-crash, by the PPid==1
# ``_reap_zombies`` backstop; SIGKILL is inherently uncoverable — atexit
# doesn't fire.)
_LIVE_ENVS: set[LocalCaptchaEnv] = set()
_LIVE_LOCK = threading.Lock()


def _terminate_env_server(env: "LocalCaptchaEnv") -> None:
    proc = env._server_proc
    if proc is not None and proc.poll() is None:
        proc.terminate()


@atexit.register
def _cleanup_live_captcha_servers() -> None:
    reap(_LIVE_ENVS, _LIVE_LOCK, _terminate_env_server)


class LocalCaptchaEnv(LiteBaseEnv, EnvServerResource):
    """Playwright-based CAPTCHA environment. No Docker required.

    Starts a Flask CAPTCHA server locally and controls a headless Chromium
    browser via Playwright. The agent receives screenshots and sends
    click/drag/type actions through the browser coordinate action surface.
    """

    EXTRA_TOOLS: ClassVar[type[BaseTools]] = CaptchaTools

    def __init__(
        self,
        *,
        # ── identity / required ──
        server_file: str,
        instruction: str,
        # ── user-facing tunables ──
        max_steps: int | None = _MAX_STEPS,
        post_action_delay: float = _POST_ACTION_DELAY,
        display_resolution: tuple[int, int] = _DISPLAY_RESOLUTION,
        seed: int | None = _SEED,
        headless: bool = _HEADLESS,
        valid_actions: list[str] | None = _VALID_ACTIONS,
        extra_tools: list[str] | None = _EXTRA_TOOLS,
        # ── env-owned make config (registry._ENV_MAKE_KWARGS) ──
        cursor: bool = True,
        # ── internal / registration ──
        env_vars: dict[str, str] | None = None,
        server_start_timeout: float = _SERVER_START_TIMEOUT,
        # Task record (bound by _register_local's entry_point; feeds the
        # metadata builder — see _task_metadata).
        category: str = "",
        mode: str | None = None,
        **kwargs,
    ):
        self._server_file = server_file
        self._instruction = instruction
        self._category = category
        self._mode = mode
        self._max_steps = max_steps
        self._post_action_delay = post_action_delay
        # Browser viewport size; named display_resolution to match the repo-wide
        # env kwarg of the render-controlling envs (mobilegym / lite.osworld).
        # captcha owns its Playwright browser, so it genuinely controls the
        # render — see docs/envs.md.
        self._display_resolution = tuple(display_resolution)
        self._seed = seed
        self._headless = headless
        # Unconditional assignment through the shared resolver — captcha used
        # to swallow ``valid_actions`` in ``**kwargs`` and silently ignore it.
        self._valid_actions = resolve_valid_actions(
            valid_actions, env_name="captcha", platform="browser",
        )
        self._extra_tool_schemas = type(self).extra_tool_schemas(extra_tools)
        self._env_vars = dict(env_vars) if env_vars else {}
        self._server_start_timeout = server_start_timeout
        self._step_count = 0
        self._server_proc: subprocess.Popen | None = None
        self._server_log_fd = None
        self._playwright = None
        self._browser = None
        self._page = None
        self._port = 0
        # Track cursor position so drag-without-start_coordinate works
        # (mirrors computer.interface.get_cursor_position() in Sandbox) AND so
        # the capture overlay knows where to paint the pointer.
        self._cursor = cursor
        # Pre-reset placeholder only: no browser exists yet, so no pointer exists
        # to be anywhere. reset() calls _reset_cursor(), which makes the centre
        # real before the first capture.
        self._cursor_x, self._cursor_y = self._center_px()

    def _center_px(self) -> tuple[int, int]:
        w, h = self._display_resolution
        return self._to_pixel(list(_CURSOR_ORIGIN_NORM), w, h)

    async def _reset_cursor(self) -> None:
        """Park the REAL pointer at the viewport centre and track it there.

        INVARIANT: what ``_take_screenshot`` composites is the real pointer
        position, never a guess. Playwright exposes no mouse-position getter
        (``page.mouse`` is a write-only channel), so rather than *assume* the
        session starts at the centre we *establish* it: the ``mouse.move`` below
        actually puts Chromium's pointer there, and every subsequent captcha
        action is itself a coordinate move that keeps the tracked value true.

        CONTRACT: ``cursor=False`` keeps the harness OFF the pointer — zero
        Playwright mouse events from reset. Only the tracked value is reset;
        the real move is guarded, exactly as ``waa`` guards its ``_park_cursor``
        call (``lite/gym/envs/waa/main.py``, "only when we actually paint a
        cursor: otherwise this would perturb the guest"). Here it is more than
        perturbation: this is a CAPTCHA benchmark, pointer telemetry is
        precisely what anti-bot heuristics read (see ``_CURSOR_ORIGIN_NORM``),
        so an operator who opted out must not get a genuine mousemove
        delivered to the page. Nothing reads the tracked value in that mode
        either — ``_take_screenshot`` returns the raw capture — so leaving the
        pointer where Chromium put it costs nothing.
        """
        self._cursor_x, self._cursor_y = self._center_px()
        if not self._cursor:
            return
        await self._page.mouse.move(self._cursor_x, self._cursor_y)

    @staticmethod
    def _task_metadata(
        category: str,
        mode: str | None = None,
        *,
        extra_tools: list[str] | None = _EXTRA_TOOLS,
    ) -> LiteCUAMetadata:
        """Same-source metadata builder.
        ``valid_actions`` is GUI-only; response/terminate are finish tools
        selected via explicit ``env_kwargs.extra_tools``.
        ``others`` carries ``category`` (the primary filter key) and, for
        image_select variants / alternative distributions, ``mode``."""
        others: dict[str, str] = {"category": category}
        if mode is not None:
            others["mode"] = mode
        return LiteCUAMetadata(
            dims=("browser", "use"),
            extra_tool_schemas=resolve_extra_tools(
                extra_tools, tools=CaptchaTools, env_name="captcha",
            ),
            valid_actions=copy_valid_actions(_VALID_ACTIONS),
            others=others,
        )

    def _runtime_metadata(self) -> LiteCUAMetadata:
        return dataclasses.replace(
            self._task_metadata(self._category, self._mode),
            valid_actions=self._valid_actions,
            extra_tool_schemas=list(self._extra_tool_schemas),
        )

    @property
    def external_resource_id(self) -> str | None:
        """Flask server port — the framework reconcile loop matches this
        instance to its subprocess via ``live_ids`` (exact ext-id match)."""
        return f"port:{self._port}" if self._server_proc else None

    async def reset(self) -> LiteEnvObservation:
        await self.close()

        self._port = _find_free_port()
        try:
            # Server-scoped /tmp names: ``captcha_<server_port>_<flask_port>.*``
            # so a concurrent env-server's boot reap (scoped to its OWN
            # server_port) never touches our in-flight result/log files.
            self._server_scope = _server_scope()
            self._result_file = f"/tmp/captcha_{self._server_scope}_{self._port}.json"
            server_path = _SERVER_DIR / self._server_file
            env = {**os.environ, "CAPTCHA_RESULT_FILE": self._result_file}
            # Make the owning env-server port visible IN the subprocess env so
            # the marker pgrep is server-scoped (see the cmdline marker below).
            env["CUA_LITE_ENV_SERVER_PORT"] = self._server_scope
            if self._seed is not None:
                env["CAPTCHA_SEED"] = str(self._seed)
                # Disable Python hash randomization so dict/set iteration is
                # deterministic — needed for byte-identical CAPTCHAs across resets.
                env["PYTHONHASHSEED"] = str(self._seed)
            # Task-specific env (e.g. CAPTCHA_MODE for image_select variants).
            env.update(self._env_vars)
            # Redirect server stdout/stderr to a per-instance log file. Using
            # subprocess.PIPE deadlocks once the OS pipe buffer (~64 KB) fills,
            # because nothing in this process drains the pipes.
            self._server_log = f"/tmp/captcha_{self._server_scope}_{self._port}.log"
            self._server_log_fd = open(self._server_log, "w")
            # Load the challenge module by file path (spec_from_file_location)
            # rather than by module name — servers/math.py would otherwise
            # collide with the stdlib ``math``. The leading no-op assignment
            # tags the cmdline so the boot orphan sweep can pgrep leaked
            # servers unambiguously; the ``_es<server_port>_`` infix makes that
            # pgrep SERVER-SCOPED (a concurrent env-server's reap matches only
            # its own marker), and ``port=N`` lets ``live_ids`` match procs to
            # instances.
            self._server_proc = subprocess.Popen(
                [
                    sys.executable, "-c",
                    f"{_server_marker(self._server_scope)}; "
                    f"import importlib.util as u; "
                    f"s = u.spec_from_file_location('captcha_challenge_server', {str(server_path)!r}); "
                    f"m = u.module_from_spec(s); s.loader.exec_module(m); "
                    f"m.app.run(host='127.0.0.1', port={self._port}, debug=False)",
                ],
                stdout=self._server_log_fd,
                stderr=subprocess.STDOUT,
                env=env,
            )
            # The Flask Popen does NOT self-terminate on parent death (it would
            # listen forever). Register for the atexit backstop the INSTANT it
            # exists — a hard signal between Popen returning and this .add would
            # otherwise orphan the subprocess (atexit can't reap what isn't in
            # _LIVE_ENVS). Mirrors sandbox/base.py adding to _LIVE_CONTAINERS
            # before ``docker run``.
            with _LIVE_LOCK:
                _LIVE_ENVS.add(self)

            # Clean stale result file from previous runs
            try:
                os.remove(self._result_file)
            except FileNotFoundError:
                pass

            # Wait for the server to accept connections, bounded by a real
            # wall-clock budget. The old fixed 50×(0.5+0.2) loop could burn
            # ~35s while the error message claimed "10s"; server_start_timeout
            # is configurable so contended hosts can widen it.
            deadline = time.monotonic() + self._server_start_timeout
            connected = False
            while time.monotonic() < deadline:
                if self._server_proc.poll() is not None:
                    log = Path(self._server_log).read_text(errors="replace")[-4000:]
                    raise RuntimeError(
                        f"CAPTCHA server exited (code {self._server_proc.returncode}) "
                        f"before accepting connections on port {self._port}:\n{log}"
                    )
                try:
                    with socket.create_connection(("127.0.0.1", self._port), timeout=0.5):
                        connected = True
                        break
                except (ConnectionRefusedError, OSError):
                    await asyncio.sleep(0.2)
            if not connected:
                log = Path(self._server_log).read_text(errors="replace")[-4000:]
                raise RuntimeError(
                    f"CAPTCHA server failed to start on port {self._port} within "
                    f"{self._server_start_timeout:.0f}s:\n{log}"
                )

            # Launch Playwright browser
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=self._headless)
            self._page = await self._browser.new_page(
                viewport={"width": self._display_resolution[0],
                          "height": self._display_resolution[1]},
            )
            await self._page.goto(f"http://127.0.0.1:{self._port}")
            await asyncio.sleep(1)  # wait for page to fully render
        except BaseException:
            # A failed reset() must not leak the server proc / browser / log fd
            # / reserved port for an env the caller then discards.
            await self.close()
            raise

        self._step_count = 0
        # Before the first capture, so the frame is taken with the pointer
        # already at the centre — capture and cursor agree, as in the lite.*
        # sandbox path.
        await self._reset_cursor()
        screenshot = await self._take_screenshot()
        return LiteEnvObservation(
            image=screenshot,
            text=self._instruction,
        )

    async def step(self, actions: list[RuntimeEnvAction]) -> LiteEnvStepResult:
        input_actions = actions
        result_call_ids = ordered_tool_call_ids(input_actions)
        metadata = self.metadata
        actions, ingress_errors = prepare_env_tool_calls(actions, metadata)
        terminated = False
        # Model-emitted calls that ENDED the episode. They get no continuation
        # observation: devs/migration/verify.py forbids a tool result for a
        # terminal call. Keyed on ``action["call_id"]`` and NOT on
        # ``result_call_id``, because an INTERNAL finish call has no ``call_id``
        # and the loop-detect wrapper's injected ``terminate`` carries the
        # intercepted NON-finish model call's id as ``_result_call_id`` -- that
        # call must still be answered.
        terminal_call_ids: set[str] = set()
        executed_actions: list[LiteExecutedAction] = []
        step_screenshots: list[bytes] = []
        action_errors: dict[str, ToolErrorFeedback] = dict(ingress_errors)
        w, h = self._display_resolution

        for action, result_call_id in actions:
            name = action["name"]
            args = action["arguments"]

            tool_feedback, tool_reason = standalone_tool_call_feedback_with_reason(
                action,
                _KNOWN_STANDALONE_TOOL_NAMES,
                metadata.extra_tool_schemas,
            )
            if tool_feedback is not None:
                if result_call_id:
                    append_feedback(action_errors, result_call_id, tool_feedback)
                executed_actions.append({
                    "call": "noop",
                    "args": {
                        "name": name,
                        "reason": tool_reason,
                    },
                })
                # The shared layer already decided the SURFACE: ``current`` is a
                # GUI slot the model got wrong, which owes a frame even though
                # nothing ran (R2a + R3); ``error_only`` is a text tool, which
                # owes none. Read that decision rather than re-deriving it.
                if tool_feedback.carrier == "current":
                    step_screenshots.append(await self._take_screenshot())
                continue

            if name in LiteFinishToolSet.get_tool_names():
                terminated = True
                if action.get("call_id"):
                    terminal_call_ids.add(action["call_id"])
                break

            invalid_action = invalid_action_message(action, metadata.valid_actions)
            if invalid_action:
                if result_call_id:
                    action_errors[result_call_id] = error_only_feedback(invalid_action)
                executed_actions.append({
                    "call": "noop",
                    "args": {"name": name, "reason": invalid_action},
                })
                continue

            try:
                calls = await self._dispatch_action(name, args, w, h)
            except MODEL_ACTION_ERROR_TYPES as e:
                record_model_action_error(
                    action_errors, result_call_id, e, action_name=name
                )
                executed_actions.append({"call": "noop", "args": {"name": name, "reason": str(e)}})
                break
            except Exception as e:
                record_tool_execution_error(
                    action_errors, result_call_id, e, action_name=name
                )
                executed_actions.append({"call": "noop", "args": {"name": name, "reason": str(e)}})
                break
            executed_actions.extend(calls)
            if not calls and name not in ("wait", "screenshot"):
                # No branch in the dispatch ladder ran, so this action executed
                # nothing and is owed no frame. ``wait`` and ``screenshot`` also
                # record no call but DID run, so they fall through and capture.
                if result_call_id:
                    action_errors[result_call_id] = error_only_feedback(
                        unsupported_action_message(name)
                    )
                continue

            # One frame PER EXECUTED ACTION, captured after that action's settle
            # delay so the frame records the settled page. The count is a pure
            # function of how many actions ran -- ``screenshot`` and
            # ``cursor_position`` are not exceptions. An aborted batch (the
            # ``break`` arms above) captured a frame for each action that DID
            # run, which is the honest record.
            if self._post_action_delay > 0:
                await asyncio.sleep(self._post_action_delay)
            step_screenshots.append(await self._take_screenshot())

        if not step_screenshots:
            # Nothing executed (empty batch, a terminal-only call, or every call
            # rejected). The turn still owes the model a current observation.
            if self._post_action_delay > 0:
                await asyncio.sleep(self._post_action_delay)
            step_screenshots.append(await self._take_screenshot())

        self._step_count += 1
        # L2 dead-backend fail-fast: the challenge is
        # CLIENT-SIDE rendered, so if the per-episode Flask server DIES
        # mid-episode the chromium page keeps showing a stale challenge — the
        # agent would grind to max_steps and _evaluate_final silently scores 0
        # (a dead backend looks identical to a wrong answer, polluting data).
        # The Flask server is a single owned subprocess, so a dead poll() is an
        # unambiguous outage: truncate LOUDLY now instead of burning the
        # remaining steps' model turns on a corpse. ``pool_unreachable`` in info
        # lets the rollout distinguish this from a genuine reward-0 task failure.
        server_dead = (self._server_proc is not None
                       and self._server_proc.poll() is not None)
        if server_dead and not terminated:
            logger.error(
                "captcha: challenge Flask server (port %s) exited mid-episode "
                "(code %s) — truncating; the episode cannot be evaluated",
                self._port, self._server_proc.returncode,
            )
        truncated = not terminated and (
            (self._max_steps is not None and self._step_count >= self._max_steps)
            or server_dead
        )

        done = terminated or truncated
        # Match androidworld: reward is None on intermediate steps, evaluated
        # only once at terminal. Avoids double-counting a single correct
        # submission (once via step-level, again via final-level).
        # _evaluate_final() does a blocking urllib GET (up to 2s); offload to a
        # thread so the terminal /step doesn't stall the event loop and serialize
        # co-running envs (mobilegym uses the same pattern).
        reward = (
            await asyncio.to_thread(self._evaluate_final)
            if done and not (server_dead and not terminated)
            else None
        )
        return build_tool_results_from_decisions(
            LiteEnvStepResult(
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info={EXECUTED_ACTIONS_INFO_KEY: executed_actions,
                      "pool_unreachable": server_dead},
            ),
            ordered_call_ids=result_call_ids,
            continue_call_ids=[
                call_id for call_id in result_call_ids
                if call_id not in terminal_call_ids
            ],
            images=step_screenshots,
            feedback=action_errors,
        )

    async def close(self) -> None:
        # Swallow Playwright shutdown errors so the server proc still gets killed.
        try:
            if self._page:
                try: await self._page.close()
                except Exception: pass
                self._page = None
            if self._browser:
                try: await self._browser.close()
                except Exception: pass
                self._browser = None
            if self._playwright:
                try: await self._playwright.stop()
                except Exception: pass
                self._playwright = None
        finally:
            with _LIVE_LOCK:
                _LIVE_ENVS.discard(self)
            if self._server_proc:
                self._server_proc.terminate()
                try:
                    self._server_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._server_proc.kill()
                    self._server_proc.wait()   # reap the SIGKILL'd child (no zombie)
                self._server_proc = None
            if getattr(self, "_server_log_fd", None) and not self._server_log_fd.closed:
                self._server_log_fd.close()
            # Clean up per-instance result file and server log
            for path_attr in ("_result_file", "_server_log"):
                path = getattr(self, path_attr, None)
                if path:
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass

    # ------------------------------------------------------------------
    # Screenshot
    # ------------------------------------------------------------------

    async def _take_screenshot(self) -> bytes:
        png_bytes = encode_png(await self._page.screenshot())
        if not self._cursor:
            return png_bytes
        # INVARIANT: the composited coordinate is the real pointer position.
        # Headless Chromium's capture buffer carries no pointer, so composite the
        # tracked one host-side (same sprite as waa / cua.* via overlay_cursor_px)
        # — and that tracked value is only ever set by a mouse.move we issued
        # (reset's parking move, or a coordinate action), never assumed.
        return await asyncio.to_thread(
            overlay_cursor_px, png_bytes, self._cursor_x, self._cursor_y
        )

    # ------------------------------------------------------------------
    # Evaluation (via server HTTP API)
    # ------------------------------------------------------------------

    def _fetch_result(self) -> dict:
        """GET /result from the in-container Flask server.

        Raises on infra failure (server unreachable / timeout) rather than
        swallowing it — a transient failure must NOT be graded the same as a
        wrong answer (which would silently drop a real solve as 0.0). The legit
        no-answer case is a REACHABLE response with ``correct=False`` (see
        ``servers/math.py`` ``/result``), so an exception here means genuine
        infra trouble → let it propagate so the rollout errors + retries.
        """
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self._port}/result", timeout=2
        ) as resp:
            return json.loads(resp.read())

    def _evaluate_final(self) -> float:
        result = self._fetch_result()
        return 1.0 if result.get("correct") else 0.0

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _to_pixel(self, coord: list[int] | None, w: int, h: int) -> tuple[int, int]:
        """Convert [0, 1000] normalized coordinates to pixel coordinates
        (canonical round+clamp)."""
        return norm_to_pixel(coord, w, h, on_malformed="raise")

    async def _dispatch_action(
        self, name: str, args: dict, w: int, h: int
    ) -> list[LiteExecutedAction]:
        """Dispatch a CUA-Lite action to Playwright.

        Mirrors _dispatch_desktop_action in sandbox/base.py so that
        every action the agent adapter can emit is handled.
        """
        calls: list[LiteExecutedAction] = []
        page = self._page

        # -- Mouse actions --
        if name == "click":
            x, y = self._to_pixel(args.get("coordinate"), w, h)
            button = args.get("button", "left")
            clicks = args.get("clicks", 1)
            await page.mouse.click(x, y, button=button, click_count=clicks)
            self._cursor_x, self._cursor_y = x, y
            calls.append({"call": "mouse.click", "args": {"x": x, "y": y, "button": button, "clicks": clicks}})

        elif name == "mouse_move":
            x, y = self._to_pixel(args.get("coordinate"), w, h)
            await page.mouse.move(x, y)
            self._cursor_x, self._cursor_y = x, y
            calls.append({"call": "mouse.move", "args": {"x": x, "y": y}})

        elif name == "mouse_down":
            button = args.get("button", "left")
            coord = args.get("coordinate")
            if coord:
                x, y = self._to_pixel(coord, w, h)
                await page.mouse.move(x, y)
                self._cursor_x, self._cursor_y = x, y
            await page.mouse.down(button=button)
            calls.append({"call": "mouse.down", "args": {"button": button}})

        elif name == "mouse_up":
            button = args.get("button", "left")
            coord = args.get("coordinate")
            if coord:
                x, y = self._to_pixel(coord, w, h)
                await page.mouse.move(x, y)
                self._cursor_x, self._cursor_y = x, y
            await page.mouse.up(button=button)
            calls.append({"call": "mouse.up", "args": {"button": button}})

        elif name == "drag":
            start = args.get("start_coordinate")
            end = args.get("coordinate")
            if end:
                if start:
                    sx, sy = self._to_pixel(start, w, h)
                else:
                    # Use tracked cursor position (from preceding mouse_move),
                    # matching Sandbox's interface.get_cursor_position() behavior.
                    sx, sy = self._cursor_x, self._cursor_y
                ex, ey = self._to_pixel(end, w, h)
                await page.mouse.move(sx, sy)
                await page.mouse.down()
                steps = 20
                for i in range(1, steps + 1):
                    ix = sx + (ex - sx) * i // steps
                    iy = sy + (ey - sy) * i // steps
                    await page.mouse.move(ix, iy)
                    await asyncio.sleep(0.02)
                await page.mouse.up()
                self._cursor_x, self._cursor_y = ex, ey
                calls.append({"call": "mouse.drag", "args": {"start": [sx, sy], "end": [ex, ey]}})

        elif name == "scroll":
            direction = args.get("direction", "down")
            amount = args.get("amount", 3)
            coord = args.get("coordinate")
            if coord:
                x, y = self._to_pixel(coord, w, h)
                await page.mouse.move(x, y)
                self._cursor_x, self._cursor_y = x, y
            if direction in ("up", "down"):
                delta_y = amount * 100 * (-1 if direction == "up" else 1)
                await page.mouse.wheel(0, delta_y)
            else:
                delta_x = amount * 100 * (-1 if direction == "left" else 1)
                await page.mouse.wheel(delta_x, 0)
            calls.append({"call": "mouse.wheel", "args": {"direction": direction, "amount": amount}})

        elif name == "cursor_position":
            calls.append({"call": "cursor_position", "args": {"x": self._cursor_x, "y": self._cursor_y}})

        # -- Keyboard actions --
        elif name == "type":
            text = args.get("text", "")
            if not isinstance(text, str):
                raise TypeError("type.arguments.text must be a string")
            if text:
                await page.keyboard.type(text)
                calls.append({"call": "keyboard.type", "args": {"text": text}})

        elif name == "key":
            keys = project_model_keys(
                args.get("keys", []),
                action_name=name,
                backend="playwright",
            )
            # Press as a CHORD (down-in-order, release-in-reverse): Playwright
            # press() takes the "Mod+Key" combo form directly.
            await page.keyboard.press("+".join(keys))
            calls.append({"call": "keyboard.press", "args": {"keys": keys}})

        elif name == "key_down":
            keys = project_model_keys(
                args.get("keys", []),
                action_name=name,
                backend="playwright",
            )
            for k in keys:
                await page.keyboard.down(k)
            calls.append({"call": "keyboard.down", "args": {"keys": keys}})

        elif name == "key_up":
            keys = project_model_keys(
                args.get("keys", []),
                action_name=name,
                backend="playwright",
            )
            for k in reversed(keys):
                await page.keyboard.up(k)
            calls.append({"call": "keyboard.up", "args": {"keys": keys}})

        elif name == "hold_key":
            keys = project_model_keys(
                args.get("keys", []),
                action_name=name,
                backend="playwright",
            )
            duration = coerce_model_duration(
                args.get("duration", 1.0),
                action_name="hold_key",
            )
            for k in keys:
                await page.keyboard.down(k)
            await asyncio.sleep(duration)
            for k in reversed(keys):
                await page.keyboard.up(k)
            calls.append({
                "call": "keyboard.hold",
                "args": {"keys": keys, "duration": duration},
            })

        # -- Utility actions --
        elif name == "wait":
            duration = coerce_model_duration(
                args.get("duration", 1.0),
                action_name="wait",
            )
            await asyncio.sleep(duration)

        elif name == "screenshot":
            pass  # screenshot is taken after every step

        else:
            logger.warning("Unknown action: %s(%s), skipping", name, args)

        return calls


# ---------------------------------------------------------------------------
# Category definitions — a single source of truth for all variants
# ---------------------------------------------------------------------------

_INSTRUCTION_TEXT = (
    "A CAPTCHA challenge is displayed in the browser. "
    "Read the distorted text in the image, type it into the input box, "
    "and click the Submit button."
)
_INSTRUCTION_SLIDER = (
    "A slider CAPTCHA puzzle is displayed in the browser. "
    "There is a background image with a puzzle-piece-shaped gap "
    "and a slider bar at the bottom. Drag the slider handle to "
    "move the puzzle piece horizontally until it fills the gap."
)
_INSTRUCTION_ROTATION = (
    "A rotation CAPTCHA is displayed in the browser. "
    "A circular image has been randomly rotated. Drag the slider "
    "to rotate the image back to its correct, upright orientation, "
    "then click the Submit button."
)
_INSTRUCTION_ICON_CLICK = (
    "An icon-click CAPTCHA is displayed in the browser. "
    "The image contains various icons (animals, shapes, etc.). "
    "Read the prompt to find out which icon category to click, "
    "then click on ALL icons of that category. "
    "Finally, click the Submit button."
)
_INSTRUCTION_MATH = (
    "A math CAPTCHA is displayed in the browser. "
    "The image shows a simple arithmetic expression (e.g., 38 + 57 = ?). "
    "Calculate the answer, type it into the input box, "
    "and click the Submit button."
)
_INSTRUCTION_IMAGE_SELECT = (
    "An image-select CAPTCHA is displayed in the browser. "
    "A 3x3 grid of 9 tiles is shown with a prompt naming a target "
    "category (e.g., 'crosswalks'). Click every tile whose content "
    "includes the target — clicking a tile toggles its selection — "
    "then click the Verify button."
)
_INSTRUCTION_ICON_MATCH = (
    "An icon-match CAPTCHA is displayed in the browser. "
    "A canvas shows several scattered icons (different shapes and colors). "
    "Among them, exactly TWO icons are visually identical (same shape AND "
    "same color). Find the matching pair and DRAG one of them onto the other. "
    "The drop must land within ~30 pixels of the target icon's center."
)
_INSTRUCTION_PAGED = (
    "A multi-page CAPTCHA is displayed in the browser. "
    "A carousel shows one card at a time, each card containing a colored "
    "shape. The prompt names a target shape+color (e.g., 'red triangle'). "
    "Use the left/right arrow buttons or the navigation dots below the "
    "card to flip between pages until the target card is shown, then "
    "click the Submit button."
)

# One row per task-family. `id` is the task-name stem; `category` (default:
# id) is the metadata category — 1:1 with the server file. image_select has
# two rows (one server, two data organizations) distinguished by
# `others.mode` instead of separate categories, so filters can treat
# image_select as one challenge type and pick a data mode orthogonally.
# `env_vars` — extra env vars injected into the challenge-server subprocess
# (e.g. CAPTCHA_MODE selecting the image_select data source).
_CATEGORIES: list[dict] = [
    {"id": "text_captcha_4",     "server": "text_captcha_4.py", "instruction": _INSTRUCTION_TEXT,"max_steps": 10},
    {"id": "slider",             "server": "slider.py",  "instruction": _INSTRUCTION_SLIDER,     "max_steps": 15},
    {"id": "rotation",           "server": "rotation.py","instruction": _INSTRUCTION_ROTATION,   "max_steps": 15},
    {"id": "icon_click",         "server": "icon_click.py","instruction": _INSTRUCTION_ICON_CLICK,"max_steps": 15},
    {"id": "math",               "server": "math.py",    "instruction": _INSTRUCTION_MATH,       "max_steps": 10},
    {"id": "image_select_crop",  "category": "image_select", "server": "image_select.py","instruction": _INSTRUCTION_IMAGE_SELECT,"max_steps": 15, "env_vars": {"CAPTCHA_MODE": "crop"}, "others": {"mode": "crop"}},
    {"id": "image_select_full",  "category": "image_select", "server": "image_select.py","instruction": _INSTRUCTION_IMAGE_SELECT,"max_steps": 15, "env_vars": {"CAPTCHA_MODE": "full"}, "others": {"mode": "full"}},
    {"id": "icon_match",         "server": "icon_match.py","instruction": _INSTRUCTION_ICON_MATCH, "max_steps": 10},
    {"id": "paged",              "server": "paged.py",   "instruction": _INSTRUCTION_PAGED,      "max_steps": 15},
]

# ---------------------------------------------------------------------------
# Task registration (Playwright-based)
# ---------------------------------------------------------------------------


def _ensure_deps() -> None:
    """Fail registration fast (with an install hint) when captcha runtime
    deps are missing, instead of erroring at the first ``reset()`` —
    playwright is imported lazily and flask/PIL only inside the Flask
    subprocess, so without this probe a missing dep surfaces mid-rollout."""
    import importlib.util
    missing = [m for m in ("playwright", "flask", "PIL")
               if importlib.util.find_spec(m) is None]
    if missing:
        raise EnvDepsMissingError(
            what=f"captcha deps missing: {', '.join(missing)}",
            install="uv run --no-sync bash lite/gym/envs/captcha/scripts/install.sh",
            see="lite/gym/envs/captcha/README.md",
        )


def _register_tasks() -> None:
    """Register every captcha task variant (direct mode only)."""
    # Env-wide make() defaults from default.yaml make_kwargs.
    registry.set_env_make_kwargs("captcha", CFG.make_kwargs)
    # Registration globs installed asset files under _ASSETS_DIR (train_*.json,
    # test/held_out.json, halligan) — the OOD `test` split + rotation-`easy` exist
    # ONLY as files. Assets are NOT auto-downloaded at import anymore (that
    # blocking snapshot_download froze the event loop via _import_all); they come
    # from scripts/install.sh. Fail LOUD when absent instead of silently
    # registering a partial task set (mirrors osworld's EnvDepsMissingError on a
    # missing catalog) — a partial set can't be self-healed (an unregistered
    # task_id can't be make()'d → reset() never runs → deadlock).
    if not (_ASSETS_DIR / "icon_click" / "train_eval.json").exists():
        raise EnvDepsMissingError(
            what="captcha assets are not installed (icon_click/train_eval.json absent)",
            install="uv run --no-sync bash lite/gym/envs/captcha/scripts/install.sh",
            see="lite/gym/envs/captcha/README.md",
        )

    def _register_local(category: str, task_id: str, server_file: str, instruction: str, max_steps: int,
                        seed: int | None, split: str,
                        env_vars: dict[str, str] | None = None,
                        mode: str | None = None) -> None:
        # Bind loop variables into the closure via default args. All bound
        # values are merged via setdefault so callers (e.g. gym.make with
        # extra kwargs, or slime's --group-shared-seed) can override any of
        # them without colliding with the explicit positional pass-through.
        # category/mode thread the task record into the instance so its
        # metadata builder serves the same others as the registered copy.
        def entry_point(_sf=server_file, _ins=instruction, _ms=max_steps,
                        _sd=seed, _se=env_vars, _cat=category, _md=mode, **kw):
            kw.setdefault("server_file", _sf)
            kw.setdefault("instruction", _ins)
            kw.setdefault("max_steps", _ms)
            kw.setdefault("seed", _sd)
            kw.setdefault("category", _cat)
            kw.setdefault("mode", _md)
            if _se:
                kw.setdefault("env_vars", _se)
            return LocalCaptchaEnv(**kw)
        register(
            f"captcha@{task_id}",
            entry_point,
            split=split,
            # Same-source contract: registered copy == the env's
            # builder output.
            metadata=LocalCaptchaEnv._task_metadata(category, mode),
        )

    # Eval-set size per category (controls eval statistical granularity).
    # Each variant is registered with a distinct prime-spaced seed so eval
    # samples see 32 different captcha instances (mirrors androidworld's
    # eval parquet approach — one row per seed).
    _EVAL_VARIANTS = 32
    _EVAL_SEED_STEP = 7919  # prime, avoids accidental seed collisions

    for _cat in _CATEGORIES:
        _env = _cat.get("env_vars")
        # Metadata category defaults to the task-name stem; image_select's
        # crop/full rows share category="image_select" + others["mode"].
        _category = _cat.get("category", _cat["id"])
        _default_mode = (_cat.get("others") or {}).get("mode")
        # Primary eval task (seed=_EVAL_SEED=42) — kept for backward compat.
        _register_local(
            category=_category,
            task_id=f"{_cat['id']}_local",
            server_file=_cat["server"],
            instruction=_cat["instruction"],
            max_steps=_cat["max_steps"],
            seed=_EVAL_SEED,
            split="eval",
            env_vars=_env,
            mode=_default_mode,
        )
        # Expanded eval set: 32 distinct seeded variants per category for
        # finer eval granularity (0.03 instead of 0.2 with a single task).
        for _i in range(_EVAL_VARIANTS):
            _register_local(
                category=_category,
                task_id=f"{_cat['id']}_local_eval{_i}",
                server_file=_cat["server"],
                instruction=_cat["instruction"],
                max_steps=_cat["max_steps"],
                seed=_EVAL_SEED + _i * _EVAL_SEED_STEP,
                split="eval",
                env_vars=_env,
                mode=_default_mode,
            )
        _register_local(
            category=_category,
            task_id=f"random_{_cat['id']}_local",
            server_file=_cat["server"],
            instruction=_cat["instruction"],
            max_steps=_cat["max_steps"],
            seed=None,
            split="train",
            env_vars=_env,
            mode=_default_mode,
        )

        # ----- Alternative training distributions (`train_<mode>.json`) -----
        # Any extra `train_*.json` next to the default `train_eval.json`
        # registers a `random_<id>_<mode>_local` train task with
        # CAPTCHA_MODE baked in at registration (like held_out/halligan) —
        # NOT a config-time env_kwargs override. E.g. rotation ships
        # `train_easy.json` (20-degree tolerance vs the standard 10) for RL
        # bootstrapping. ``others["mode"]`` is the single fine-grained
        # filter key (same key image_select uses for crop/full/halligan):
        #   --filter "lambda m: m.others.get('mode') == 'easy'"
        # Eval tasks are unaffected (separate task ids, default mode).
        #
        # Probing by ``_cat["id"]`` (not ``_category``) is deliberate: the
        # two image_select rows resolve to non-existent dirs, so this
        # mechanism (and the held-out probe below) is inert for them —
        # image_select's CAPTCHA_MODE slot is already taken by its data
        # organization (crop/full), which a `train_<mode>.json` override
        # would clobber; its OOD test set is the halligan block instead.
        for _mode_file in sorted((_ASSETS_DIR / _cat["id"]).glob("train_*.json")):
            if _mode_file.stem == "train_eval":
                continue  # the default mode, registered above
            _mode = _mode_file.stem.removeprefix("train_")
            _register_local(
                category=_category,
                task_id=f"random_{_cat['id']}_{_mode}_local",
                server_file=_cat["server"],
                instruction=_cat["instruction"],
                max_steps=_cat["max_steps"],
                seed=None,
                split="train",
                env_vars={**(_env or {}), "CAPTCHA_MODE": _mode_file.stem},
                mode=_mode,
            )

        # ----- Held-out OOD test (`test/held_out.json` per captcha) -----
        # Split semantics are distribution semantics: train/eval share the
        # default `train_eval.json` distribution; split="test" is ALWAYS
        # out-of-distribution and is never touched by the training loop —
        # only a post-training eval script reads it. Content disjoint from
        # train_eval: new vocabulary for icon_click / icon_match / paged;
        # disjoint background image pool for slider / rotation. Seeds use the
        # `_TEST_SEED` range (far from the eval segment) so generative
        # captchas draw fresh samples. Opt-in by `test/held_out.json`.
        _held_out_path = _ASSETS_DIR / _cat["id"] / "test" / "held_out.json"
        if _held_out_path.is_file():
            _test_env = {**(_env or {}), "CAPTCHA_MODE": "test/held_out"}
            for _i in range(_EVAL_VARIANTS):
                _register_local(
                    category=_category,
                    task_id=f"{_cat['id']}_held_out_local_test{_i}",
                    server_file=_cat["server"],
                    instruction=_cat["instruction"],
                    max_steps=_cat["max_steps"],
                    seed=_TEST_SEED + _i * _EVAL_SEED_STEP,
                    split="test",
                    env_vars=_test_env,
                    mode=_default_mode,
                )

    # ----- Halligan static benchmark (reCAPTCHA v2 binary subset) -----
    # image_select's OOD test set: real reCAPTCHA v2 challenges (Teoh et
    # al.'s Halligan benchmark, USENIX Security '25) instead of a synthetic
    # `test/held_out.json` — one task per challenge id, all split="test"
    # (split=test ⟺ OOD, same as the held-out sets above). Registration is
    # conditional on the dir existing so contributors who deliberately
    # delete the data (e.g. to test the re-import flow) don't crash.
    _HALLIGAN_DIR = _ASSETS_DIR / "image_select" / "test" / "halligan"
    _halligan_ids = sorted(
        int(p.name) for p in _HALLIGAN_DIR.glob("[0-9]" * 3)
        if p.is_dir() and (p / "meta.json").is_file()
    ) if _HALLIGAN_DIR.is_dir() else []
    for _hid in _halligan_ids:
        _register_local(
            category="image_select",
            task_id=f"image_select_halligan_local_test{_hid}",
            server_file="image_select.py",
            instruction=_INSTRUCTION_IMAGE_SELECT,
            max_steps=15,
            seed=None,  # halligan is fully deterministic by id, no seed needed
            split="test",
            env_vars={
                "CAPTCHA_MODE": "halligan",
                "CAPTCHA_HALLIGAN_ID": str(_hid),
            },
            mode="halligan",
        )
    if _halligan_ids:
        logger.debug(
            "Registered %d Halligan test tasks (ids %d..%d)",
            len(_halligan_ids), _halligan_ids[0], _halligan_ids[-1],
        )
    else:
        logger.debug(
            "No Halligan challenges found at %s — this should not happen "
            "after a clean clone; re-run _import_halligan.py to restore the "
            "static eval benchmark.", _HALLIGAN_DIR,
        )


if routing_server_url():
    # env-server client mode: the registry fetches the task list over HTTP
    # (register_from_server) and never needs local deps / assets. Keep the
    # module importable (helpers, LocalCaptchaEnv, reap hooks) without
    # side effects. ``routing_server_url`` (not the raw env var) so the
    # server's own lazy import of this module — which happens AFTER
    # ``serve_locally()`` — correctly registers local even though the server
    # inherited ``CUA_LITE_ENV_SERVER_URL`` from the shell.
    logger.debug("captcha: client mode — skipping local registration")
else:
    # NO network at module import — the reaper's registry._import_all() runs this
    # on the server's asyncio loop, and a blocking call there freezes the whole
    # server. _ensure_deps() is a local importlib probe; _register_tasks() only
    # globs assets that scripts/install.sh already put on disk and fails loud
    # (EnvDepsMissingError) if they're absent — it never downloads.
    _ensure_deps()
    _register_tasks()


# ---------------------------------------------------------------------------
# orphan-proc + /tmp sweep — backs CaptchaServices.reap (every tick; PPid==1-safe,
# so boot-independent), and runs once at boot via recover_all.
# ---------------------------------------------------------------------------

def _orphan_pids(pattern: str) -> list[int]:
    """PIDs owned by the current user whose cmdline matches ``pattern``
    and whose parent died (``PPid == 1``) — i.e. leaked by a prior
    server lifetime. Mirrors the uid + orphan filter of
    :func:`lite.gym.envs.mobilegym.main._orphan_chromium_pids`."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["pgrep", "-u", str(os.geteuid()), "-f", pattern], text=True, timeout=10,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return []
    pids: list[int] = []
    for line in out.splitlines():
        pid = int(line)
        try:
            status = Path(f"/proc/{pid}/status").read_text()
        except OSError:
            continue
        for sline in status.splitlines():
            if sline.startswith("PPid:") and sline.split()[1] == "1":
                pids.append(pid)
                break
    return pids


def _reap_zombies(port: int | None, token: str | None) -> int:
    """env-server startup hook: kill captcha subprocesses leaked by a
    prior server lifetime.

    Each episode spawns a Flask challenge server (plain ``Popen``) and a
    Playwright chromium. On graceful shutdown the server's lifespan hook
    closes every env, but a SIGKILL / OOM reparents both to init and the
    Flask proc keeps listening forever. Mirrors mobilegym's boot orphan
    sweep — reached via ``CaptchaServices.reap`` and run by ``recover_all``
    during the env-server's lifespan startup.

    ``port`` is THIS env-server's listen port — the isolation scope. We reap
    only Flask procs / ``/tmp`` files tagged with our own ``_es<scope>_`` marker
    (a prior incarnation of this server on the same port), so a concurrent
    env-server's leftovers on a different port are left untouched. ``token``
    accepted for hook parity but unused. Multi-tenant safe — other users'
    processes are excluded by the ``pgrep -u`` match; live envs of a co-tenant
    env-server (different port, same user) are excluded by BOTH the scoped
    marker and the ``PPid == 1`` orphan filter.
    """
    scope = _scope_str(port)
    killed = 0
    # Flask challenge servers — cmdline tagged by LocalCaptchaEnv's Popen with
    # our server-scoped marker (only a prior incarnation on THIS port matches).
    # Headless chromiums — Playwright's profile-dir marker (unscoped, but the
    # PPid==1 filter in _orphan_pids means we only ever touch genuinely-leaked
    # orphans whose owning server is already dead — never a live co-tenant's).
    for pattern in (_server_marker(scope), "playwright_chromiumdev"):
        for pid in _orphan_pids(pattern):
            try:
                os.kill(pid, 9)
                killed += 1
            except OSError:
                pass
    # Per-instance result/log files of dead servers accumulate in /tmp. Names
    # are ``captcha_<scope>_<flaskport>.{json,log}`` (set in reset); glob OUR
    # scope so we only GC this server's own leftovers. 24 h cutoff is a belt-
    # and-suspenders age gate: live episodes are minutes long (idle TTL 1 h),
    # so anything older than a day is certainly dead.
    import glob
    cutoff = time.time() - 86400
    for path in (glob.glob(f"/tmp/captcha_{scope}_*.json")
                 + glob.glob(f"/tmp/captcha_{scope}_*.log")):
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            pass
    if killed:
        logger.info("captcha reap_zombies: killed %d orphan proc(s)", killed)
    return killed


# ---------------------------------------------------------------------------
# env-server reconcile: live_ids (liveness oracle) helper
# ---------------------------------------------------------------------------


def _flask_server_procs(scope: str) -> dict[int, tuple[int, int, float]] | None:
    """Map Flask challenge-server port → ``(pid, ppid, age_s)`` for every
    proc tagged with THIS server's ``_es<scope>_`` marker, owned by the current
    user. Scoping to ``scope`` means a co-tenant env-server's live procs
    (different port) don't pollute our liveness set — over-inclusion would only
    suppress ghosting, but exact scoping keeps the oracle precise.

    Returns ``None`` on a GENUINE ``pgrep`` failure (binary missing, syntax/
    fatal exit ≥2, or timeout) so :func:`_live_flask_port_ids` can fail closed
    — a partial/empty set on a failed scan would mass-false-ghost live captcha
    instances. An EMPTY ``dict`` means ``pgrep`` ran fine and matched nothing
    (exit 1)."""
    import re
    try:
        out = subprocess.check_output(
            ["pgrep", "-u", str(os.geteuid()), "-f", _server_marker(scope)],
            text=True, timeout=10,
        ).strip()
    except subprocess.CalledProcessError as e:
        if e.returncode == 1:
            return {}        # pgrep ran, no matching procs — genuinely empty
        return None          # exit ≥2 (syntax/fatal) — fail closed
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None          # pgrep missing / hung — fail closed
    uptime = float(Path("/proc/uptime").read_text().split()[0])
    hz = os.sysconf("SC_CLK_TCK")
    procs: dict[int, tuple[int, int, float]] = {}
    for line in out.splitlines():
        pid = int(line)
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
            stat = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            continue
        m = re.search(r"port=(\d+)", cmdline)
        if not m:
            continue  # marker mentioned in some other cmdline, not a real server
        # /proc/<pid>/stat: fields after the ")" of comm — ppid is field 4
        # (index 1), starttime field 22 (index 19), in clock ticks.
        fields = stat.rsplit(")", 1)[1].split()
        ppid = int(fields[1])
        age = uptime - float(fields[19]) / hz
        procs[int(m.group(1))] = (pid, ppid, age)
    return procs


def _live_flask_port_ids(scope: str) -> set[str] | None:
    """The set of currently-live ``external_resource_id``s for captcha, in the
    same id-space as :attr:`LocalCaptchaEnv.external_resource_id` (``port:<n>``).

    Reconstructed EXACTLY from the live marker-tagged Flask procs via a local
    ``/proc`` + ``pgrep`` enumeration (:func:`_flask_server_procs`), enabling
    exact-match ghost detection (a tracked ``port:<n>`` whose Flask proc is gone
    is a genuine ghost). Over-inclusion of a co-tenant's live procs is harmless
    (it only PREVENTS ghosting). **Fails closed**: returns ``None`` if the
    ``pgrep`` scan genuinely failed (vs. an exact empty set when it ran and
    matched nothing), so a transient scan failure skips ghost detection rather
    than false-ghosting every live instance."""
    procs = _flask_server_procs(scope)
    if procs is None:
        return None
    return {f"port:{p}" for p in procs}


def _captcha_health_check(env_id: str) -> None:
    """Per-category asset probe — surfaces in ``GET /envs/captcha``. Verifies
    every registered category has its asset dir, so partial-install hosts
    report ``available: false`` instead of failing at reset-time.

    The probed dir is the entry's ``category`` (falling back to ``id``) — the
    SAME key reset-time registration resolves the asset dir from. Sibling rows
    that share one asset dir (``image_select_crop`` / ``image_select_full`` →
    ``image_select/``) collapse to a single probe; using ``id`` here would
    false-report a complete install as missing those rows' non-existent dirs."""
    missing = sorted({
        entry.get("category", entry["id"]) for entry in _CATEGORIES
        if not (_ASSETS_DIR / entry.get("category", entry["id"])).exists()
    })
    if missing:
        raise EnvDepsMissingError(
            what=f"captcha assets missing for category: {', '.join(missing)}",
            install="uv run --no-sync bash lite/gym/envs/captcha/scripts/install.sh",
            see="lite/gym/envs/captcha/README.md",
        )


class CaptchaServices(EnvServices):
    """Env-server capabilities for captcha: asset health probe (``health``) +
    per-instance flask-port reconciliation (``live_ids`` + ``reap``). No shared
    backend to start (each episode spawns its own flask server in ``reset``), so
    no ``ensure``/``shutdown``.

    Liveness oracle (``live_ids``): the env's ``external_resource_id`` is
    ``port:<flaskport>``. The live set is reconstructed EXACTLY from the live
    marker-tagged Flask procs (a local ``/proc`` + ``pgrep`` scan, not a flaky
    remote oracle), so we return the exact set — enabling exact-match ghost
    detection (the framework ghosts a tracked ``port:<n>`` whose proc is gone).
    Ghost reporting that the old ``reap_drift`` did via the slot-count heuristic
    is now the framework's job, driven by this ``live_ids``."""

    def health(self, env_id: str) -> None:
        _captcha_health_check(env_id)

    def live_ids(self, env_id: str, scope) -> set[str] | None:
        # Raise (not return None) on a pgrep probe failure so the reconcile loop skips
        # the cycle fail-closed; None is reserved for "no per-instance world".
        # captcha always has a per-instance world, so it never returns None.
        ids = _live_flask_port_ids(_scope_str(scope.server_port))
        if ids is None:
            from lite.gym.errors import ReconcileProbeError
            raise ReconcileProbeError(f"live_ids({env_id}): flask-port pgrep probe failed")
        return ids

    def reap(self, env_id: str, scope, in_use: set[str], *, boot: bool = False) -> int:
        # captcha's orphan policy = the old reap_zombies body: kill leaked
        # marker-tagged Flask/chromium procs whose parent died (PPid==1) +
        # GC stale ``/tmp/captcha_*`` files. Keys off orphan-PID + file age,
        # NOT untracked id, so ``in_use`` is ignored. The PPid==1 filter makes
        # it safe every tick (a live proc is parented by the alive env-server /
        # node driver, PPid≠1), so the boot/steady split is irrelevant here —
        # ``boot`` is accepted for protocol parity and unused.
        return _reap_zombies(scope.server_port, None)


register_services("captcha", CaptchaServices())
from lite.gym.services import BackendFamily, register_family  # noqa: E402

register_family("captcha", BackendFamily.DEDICATED)
