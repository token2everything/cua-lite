"""Cursor config boundary.

Cursor repaint is env-owned capture behavior configured through
``make_kwargs.cursor``. The old ``cursor_overlay`` name is retired, and browser
DOM capture paths must use the shared cursor asset rather than a red dot.

Run:  uv run pytest tests/gym/utils/config/test_cursor_default.py -q
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
import yaml

from lite.gym.registry import _WRAPPER_KWARG_DEFAULTS, CARRIED_SPEC_KWARGS, registry

ROOT = Path(__file__).resolve().parents[4]

_CURSOR_CONFIGS = (
    "lite/gym/envs/browsergym/configs/default.yaml",
    "lite/gym/envs/browsergym/configs/isolation.yaml",
    "lite/gym/envs/captcha/configs/default.yaml",
    "lite/gym/envs/cua/bench/configs/default.yaml",
    "lite/gym/envs/cua/sandbox/configs/default.yaml",
    "lite/gym/envs/online_mind2web/configs/default.yaml",
    "lite/gym/envs/waa/configs/default.yaml",
    "lite/gym/envs/webgym/configs/default.yaml",
    "lite/gym/envs/webharbor/webvoyager/configs/default.yaml",
)

_BROWSER_CONTAINER_SERVERS = (
    "lite/gym/envs/webgym/docker/patches/playwright_instance.py",
    "lite/gym/envs/online_mind2web/docker/server.py",
    "lite/gym/envs/webharbor/webvoyager/docker/server.py",
)


def _read_yaml(relpath: str) -> dict[str, Any]:
    data = yaml.safe_load((ROOT / relpath).read_text(encoding="utf-8")) or {}
    assert isinstance(data, dict), relpath
    return data


def _make_kwargs(relpath: str) -> dict[str, Any]:
    make_kwargs = _read_yaml(relpath).get("make_kwargs") or {}
    assert isinstance(make_kwargs, dict), relpath
    return make_kwargs


def _source_function(relpath: str, name: str) -> str:
    text = (ROOT / relpath).read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            source = ast.get_source_segment(text, node)
            assert source is not None
            return source
    raise AssertionError(f"{relpath} missing {name}")


def _exec_source_function(source: str, name: str):
    ns: dict[str, Any] = {}
    exec("from __future__ import annotations\n" + source, ns)
    return ns[name]


def test_env_configs_do_not_mention_cursor_overlay():
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in sorted((ROOT / "lite/gym/envs").glob("**/configs/*.yaml"))
        if "cursor_overlay" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_cursor_configs_default_make_kwarg_true():
    offenders = [
        relpath
        for relpath in _CURSOR_CONFIGS
        if _make_kwargs(relpath).get("cursor") is not True
    ]

    assert offenders == []


def test_registry_cursor_is_carried_but_not_wrapper():
    assert "cursor_overlay" not in _WRAPPER_KWARG_DEFAULTS
    assert "cursor_overlay" not in CARRIED_SPEC_KWARGS
    assert "cursor" not in _WRAPPER_KWARG_DEFAULTS
    assert "cursor" in CARRIED_SPEC_KWARGS


def test_generic_cursor_overlay_wrapper_is_removed():
    source = (ROOT / "lite/gym/wrappers.py").read_text(encoding="utf-8")

    assert "class CursorOverlayWrapper" not in source
    assert "def overlay_cursor_px" in source
    assert "def load_cursor_sprite" in source


def test_browser_capture_paths_use_shared_cursor_not_red_dot():
    owners = {
        "lite/gym/envs/browsergym/main.py": "overlay_cursor_px",
        "lite/gym/envs/webgym/docker/patches/_playwright_controller.py": "_CAPTURE_CURSOR_B64",
        "lite/gym/envs/online_mind2web/docker/server.py": "_CAPTURE_CURSOR_B64",
        "lite/gym/envs/webharbor/webvoyager/docker/server.py": "_CAPTURE_CURSOR_B64",
    }

    missing = [
        relpath
        for relpath, marker in owners.items()
        if marker not in (ROOT / relpath).read_text(encoding="utf-8")
    ]
    red_dot_markers = [
        relpath
        for relpath in owners
        for bad in ("red-cursor", "backgroundColor = 'red'", "borderRadius = '50%'")
        if bad in (ROOT / relpath).read_text(encoding="utf-8")
    ]

    assert missing == []
    assert red_dot_markers == []


def test_captcha_composites_the_shared_cursor_like_its_browser_siblings():
    """captcha owns a headless Chromium whose capture buffer has NO native
    pointer, so it MUST repaint the cursor env-side — same contract as every
    other browser-backed env, and the more so because captcha is a
    click-PRECISION benchmark.

    This previously asserted the opposite (``"cursor" not in config``), which
    pinned a regression: the retired ``CursorOverlayWrapper`` had composited
    captcha's cursor via ``make_kwargs.cursor_overlay: true``, and the rename to
    ``cursor`` reached the seven siblings but skipped captcha.
    """
    config = _make_kwargs("lite/gym/envs/captcha/configs/default.yaml")
    source = (ROOT / "lite/gym/envs/captcha/main.py").read_text(encoding="utf-8")

    assert config.get("cursor") is True
    assert "cursor_overlay" not in source
    assert "overlay_cursor_px" in source


def test_cursor_session_origin_is_viewport_centre_everywhere():
    """ONE session-start pointer origin across every cursor env: the CENTRE.

    Three things justify centre over the alternatives:
      * the retired ``CursorOverlayWrapper`` seeded ``(500, 500)`` normalized on
        ``reset()`` — i.e. centre — uniformly for every desktop/browser env, so it is
        the baseline every trajectory was collected under;
      * a real desktop session warps the pointer to screen centre at start and a
        browser inherits it. ``(0, 0)`` is the "pointer never moved" sentinel,
        not a state a real session is ever in;
      * ``norm_to_pixel``'s own malformed-coordinate fallback is ``(w // 2, h // 2)``.

    The regression this pins had the origin diverge THREE ways at once: waa /
    cua.* kept centre, webgym seeded top-left, and online_mind2web +
    webharbor.webvoyager seeded ``None`` — which made ``_show_capture_cursor``
    return ``False`` and shipped a turn-0 frame with no cursor at all.

    Per-env behavioural coverage lives next to each env (``tests/gym/envs/``);
    this is the cross-env table that makes a lone drifting env fail loudly.
    """
    # Envs that seed a PIXEL cursor directly: the literal must be a centre halving.
    pixel_origin_owners = {
        "lite/gym/envs/waa/main.py": "(self._screen_w // 2, self._screen_h // 2)",
        "lite/gym/envs/cua/sandbox/env.py": "(self._wh[0] // 2, self._wh[1] // 2)",
        "lite/gym/envs/cua/bench/main.py": "(self._wh[0] // 2, self._wh[1] // 2)",
        "lite/gym/envs/webgym/docker/patches/_playwright_controller.py": (
            "viewport_width / 2.0"
        ),
    }
    # Envs that seed through the shared [0, 1000] projection: centre is (500, 500).
    norm_origin_owners = (
        "lite/gym/envs/captcha/main.py",
        "lite/gym/envs/online_mind2web/docker/server.py",
        "lite/gym/envs/webharbor/webvoyager/docker/server.py",
    )

    missing_pixel = [
        relpath
        for relpath, marker in pixel_origin_owners.items()
        if marker not in (ROOT / relpath).read_text(encoding="utf-8")
    ]
    missing_norm = [
        relpath
        for relpath in norm_origin_owners
        if "_CURSOR_ORIGIN_NORM: tuple[int, int] = (500, 500)"
        not in (ROOT / relpath).read_text(encoding="utf-8")
    ]
    # A container env that builds its instance without seeding the pointer ships a
    # cursor-less turn-0 frame — the exact online_mind2web / webvoyager regression.
    unseeded_containers = [
        relpath
        for relpath in (
            "lite/gym/envs/online_mind2web/docker/server.py",
            "lite/gym/envs/webharbor/webvoyager/docker/server.py",
        )
        if "last_cursor=" not in (ROOT / relpath).read_text(encoding="utf-8")
    ]

    assert missing_pixel == []
    assert missing_norm == []
    assert unseeded_containers == []


def test_browser_container_cursor_bool_parser_is_shared_and_strict():
    sources = {
        relpath: _source_function(relpath, "_coerce_cursor_bool")
        for relpath in _BROWSER_CONTAINER_SERVERS
    }
    reference_path = _BROWSER_CONTAINER_SERVERS[0]
    reference = sources[reference_path]
    drifted = [relpath for relpath, source in sources.items() if source != reference]

    assert drifted == [], f"_coerce_cursor_bool drifted from {reference_path}: {drifted}"

    # Byte-identity across all three copies is asserted above, so exercising the
    # reference copy covers every image.
    parser = _exec_source_function(reference, "_coerce_cursor_bool")
    for value in (False, "false", "False", 0, "0", " no ", "off", ""):
        assert parser(value) is False
    for value in (True, "true", "True", 1, "1", " yes ", "on", None):
        assert parser(value) is True

    for bad in ("2", 2, [], object()):
        try:
            parser(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"cursor parser accepted {bad!r}")


def test_browser_container_reset_step_use_cursor_bool_parser():
    expected = {
        "lite/gym/envs/webgym/docker/patches/playwright_instance.py": (
            'cursor = _coerce_cursor_bool(args.get("cursor"), default=True)',
            1,
        ),
        "lite/gym/envs/online_mind2web/docker/server.py": (
            'cursor = _coerce_cursor_bool(body.get("cursor"), default=True)',
            2,
        ),
        "lite/gym/envs/webharbor/webvoyager/docker/server.py": (
            'cursor = _coerce_cursor_bool(body.get("cursor"), default=True)',
            2,
        ),
    }
    offenders = [
        relpath
        for relpath, (snippet, count) in expected.items()
        if (ROOT / relpath).read_text(encoding="utf-8").count(snippet) != count
    ]

    assert offenders == []


@pytest.mark.parametrize("bad_key", ["cursor", "cursor_overlay", "definitely_unknown"])
def test_grounding_envs_reject_unsupported_cursor_kwargs(bad_key):
    from lite.gym.envs.osworld_g.main import OSWorldGEnv
    from lite.gym.envs.screenspot_pro.main import ScreenSpotProEnv

    screenspot_annotation = {
        "img_filename": "unused.png",
        "instruction": "click the target",
        "bbox": [10, 10, 20, 20],
        "img_size": [100, 100],
    }
    osworld_g_annotation = {
        "img_filename": "unused.png",
        "instruction": "click the target",
        "box_type": "bbox",
        "box_coordinates": [10, 10, 20, 20],
        "img_size": [100, 100],
    }

    with pytest.raises(TypeError, match=bad_key):
        ScreenSpotProEnv(
            annotation=screenspot_annotation,
            images_dir=Path("/tmp"),
            **{bad_key: False},
        )
    with pytest.raises(TypeError, match=bad_key):
        OSWorldGEnv(
            annotation_original=osworld_g_annotation,
            annotation_refined=osworld_g_annotation,
            images_dir=Path("/tmp"),
            **{bad_key: False},
        )


@pytest.mark.parametrize("bad_key", ["cursor", "cursor_overlay", "definitely_unknown"])
def test_osworld_envs_reject_unsupported_cursor_kwargs(bad_key):
    from lite.gym.envs.osworld.main import OSWorldConfig, OSWorldEnv
    from lite.gym.envs.osworld_2.worker_env import OSWorldWorkerEnv

    with pytest.raises(TypeError, match=bad_key):
        OSWorldEnv(
            config=OSWorldConfig(domain="chrome", task_id="task"),
            **{bad_key: False},
        )
    with pytest.raises(TypeError, match=bad_key):
        OSWorldWorkerEnv(
            task_id="054",
            **{bad_key: False},
        )


def test_webgym_accepts_cursor_make_kwarg_without_services():
    from lite.gym.envs.webgym.main import WebGymEnv

    env = WebGymEnv(task={"difficulty": 2}, cursor=False)

    assert env._cursor is False


@pytest.mark.parametrize("bad_key", ["cursor_overlay", "definitely_unknown"])
def test_webgym_rejects_unknown_cursor_kwargs_without_services(bad_key):
    from lite.gym.envs.webgym.main import WebGymEnv

    with pytest.raises(TypeError, match=bad_key):
        WebGymEnv(task={"difficulty": 2}, **{bad_key: False})


def test_env_make_kwargs_preserves_cursor_and_filters_old_name():
    env_id = "faketest_cursor_make_kwarg"
    try:
        registry.set_env_make_kwargs(
            env_id,
            {"cursor": False, "cursor_overlay": True, "step_timeout": 7.0, "bogus": 1},
        )

        assert registry.env_make_kwargs(env_id) == {"cursor": False, "step_timeout": 7.0}
    finally:
        import sys

        sys.modules["lite.gym.registry"]._env_make_kwargs.pop(env_id, None)
