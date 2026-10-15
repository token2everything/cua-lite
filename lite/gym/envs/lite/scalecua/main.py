"""lite.scalecua — ScaleCUA OSWorld tasks on the lite.osworld desktop substrate."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from lite.gym.envs.lite.osworld.main import LiteOsworldEnv
from lite.gym.envs.lite.scalecua.src.osworld.setup import setup_fn
from lite.gym.envs.lite.scalecua.src.osworld.verify import evaluate_final_fn
from lite.gym.envs.lite.scalecua.src.utils import dataset
from lite.gym.errors import EnvDepsMissingError
from lite.gym.sandbox import register_jsonl_tasks
from lite.gym.utils import config as env_config

logger = logging.getLogger(__name__)

_ENV_ID = "lite.scalecua"
_DIR = Path(__file__).parent
CFG = env_config.load(str(_DIR))
_MAX_STEPS = CFG.env_kwargs["max_steps"]
_POST_ACTION_DELAY = CFG.env_kwargs["post_action_delay"]
_c = CFG.env_kwargs["computer"]
_IMAGE = _c["image"]
_COMPUTER_CONFIG = {
    "image": _IMAGE,
    "display": f"{_c['display_resolution'][0]}x{_c['display_resolution'][1]}",
    "memory": _c["memory"],
    "cpu": _c["cpu"],
    "timeout": _c["timeout"],
    "hostname": "user-virtual-machine",
}
_MAX_RESETS_PER_CONTAINER = CFG.server_kwargs["max_resets_per_container"]
_README = "lite/gym/envs/lite/scalecua/README.md"


def _computer() -> dict[str, Any]:
    return dict(_COMPUTER_CONFIG)


def _dep_error(what: str) -> EnvDepsMissingError:
    return EnvDepsMissingError(
        what=what,
        install=(
            "uv run --no-sync bash "
            "lite/gym/envs/lite/scalecua/scripts/install.sh"
        ),
        see=_README,
    )


def _check_desktop_env() -> None:
    try:
        import desktop_env  # noqa: F401
    except ImportError as exc:
        raise EnvDepsMissingError(
            what="OSWorld desktop_env package not installed — required for lite.scalecua evaluators",
            install="uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/install.sh",
            see=_README,
        ) from exc


def _check_base_image() -> None:
    from lite.gym.utils.backend.docker import require_image_present
    from lite.gym.utils.backend.freshness import image_for

    require_image_present(image_for("lite.osworld", tag=_IMAGE))


class LiteScaleCuaEnv(LiteOsworldEnv):
    """ScaleCUA task env using the existing lite.osworld container image."""

    def __init__(
        self,
        task: Any = None,
        *,
        display_resolution: tuple[int, int] = (1920, 1080),
        image: str | None = None,
        vnc_port: int | None = None,
        env_id: str | None = _ENV_ID,
        setup_fn: Any = None,
        evaluate_step_fn: Any = None,
        evaluate_final_fn: Any = None,
        container: str | None = None,
        computer_config: dict[str, Any] | None = None,
        post_action_delay: float = _POST_ACTION_DELAY,
        debug: bool = False,
        max_steps: int | None = None,
        noise: bool = False,
        seed: int | None = None,
        valid_actions: list[str] | None = None,
        extra_tools: list[str] | None = None,
        expose_oracle: bool = False,
        domain_overrides: dict[str, dict] | None = None,
    ) -> None:
        resolution = tuple(display_resolution)
        if resolution != (1920, 1080):
            raise ValueError(
                "lite.scalecua tasks target the 1920x1080 OSWorld desktop contract"
            )
        if image is not None and image != _IMAGE:
            raise ValueError(
                "lite.scalecua always uses the configured lite.osworld image "
                f"{_IMAGE!r}; pass image=None or that exact image"
            )
        _check_base_image()
        if computer_config is None:
            computer_config = dict(_COMPUTER_CONFIG)
        super().__init__(
            task,
            display_resolution=resolution,
            image=None,
            vnc_port=vnc_port,
            env_id=env_id,
            setup_fn=setup_fn,
            evaluate_step_fn=evaluate_step_fn,
            evaluate_final_fn=evaluate_final_fn,
            container=container,
            computer_config=computer_config,
            post_action_delay=post_action_delay,
            debug=debug,
            max_steps=max_steps,
            noise=noise,
            seed=seed,
            valid_actions=valid_actions,
            extra_tools=extra_tools,
            expose_oracle=expose_oracle,
            domain_overrides=domain_overrides,
        )
        self._max_resets_per_container = _MAX_RESETS_PER_CONTAINER


_TASK_COMPUTER = _computer()
_catalog_registered = False


def _register_tasks() -> None:
    global _catalog_registered
    if _catalog_registered:
        return
    missing = [
        dataset.catalog_path(split)
        for split in dataset.RUNTIME_SPLITS
        if not dataset.catalog_path(split).is_file()
    ]
    if missing:
        raise _dep_error(
            "lite.scalecua generated catalogs are missing: "
            + ", ".join(str(path) for path in missing)
        )
    try:
        dataset.validate_catalog_lock()
        dataset.validate_runtime_cache()
        for split in dataset.RUNTIME_SPLITS:
            dataset.validate_catalog(dataset.catalog_path(split), expected_split=split)
    except Exception as exc:
        raise _dep_error(f"lite.scalecua generated catalogs are invalid: {exc}") from exc

    common = {
        "env_id": _ENV_ID,
        "computer": _TASK_COMPUTER,
        "setup_fn": setup_fn,
        "evaluate_final_fn": evaluate_final_fn,
        "env_class": LiteScaleCuaEnv,
        "factory_kwargs": {
            "computer_config": _TASK_COMPUTER,
            "post_action_delay": _POST_ACTION_DELAY,
        },
        "default_max_steps": _MAX_STEPS,
        "platform": "desktop",
    }
    for split in dataset.RUNTIME_SPLITS:
        register_jsonl_tasks(
            dataset.registration_catalog_path(split),
            split=split,
            **common,
        )
    _catalog_registered = True


def ensure_services(env_id: str) -> None:
    _check_desktop_env()


from lite.gym.registry import registry  # noqa: E402
from lite.gym.services import BackendFamily, register_family, register_services  # noqa: E402
from lite.gym.remote.reaper import ContainerServices  # noqa: E402


class LiteScaleCuaServices(ContainerServices):
    def register_tasks(self, env_id: str) -> None:
        _register_tasks()

    def ensure(self, env_id: str) -> None:
        ensure_services(env_id)


register_services(_ENV_ID, LiteScaleCuaServices())
register_family(_ENV_ID, BackendFamily.DEDICATED)
registry.set_env_make_kwargs(_ENV_ID, CFG.make_kwargs)
if all(dataset.catalog_path(split).is_file() for split in dataset.RUNTIME_SPLITS):
    try:
        _register_tasks()
    except EnvDepsMissingError as exc:
        logger.warning("lite.scalecua catalog registration deferred: %s", exc.what)
