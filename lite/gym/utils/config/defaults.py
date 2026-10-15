"""Per-env config defaults: ``configs/default.yaml``, optionally swapped at startup.

Each env ships a ``configs/default.yaml`` declaring its tunable defaults in three
groups — ``env_kwargs`` (per-instance, passed to the env factory/bind),
``server_kwargs`` (per-deployment infra), and ``make_kwargs`` (env-wide
``gym.make`` defaults). This is the single, uniform reader.

To use a DIFFERENT config (works in both direct and server mode), set ONE env-var
at startup pointing to a complete replacement yaml; it is used WHOLE (no merge —
one file is the entire config, so what you read is what you get). NO per-key
env-vars.

    from lite.gym.utils import config
    ENV_DIR = str(Path(__file__).parent)
    CFG = config.load(ENV_DIR)
    ... CFG.env_kwargs["max_steps"] ...
    ... CFG.server_kwargs["memory_limit"] ...

Override at startup (direct shell, or the serve_env launch). ``<PREFIX>_CONFIG``
is either an ABSOLUTE path, or a RELATIVE value resolved under the env's
``configs/`` dir (a bare name gets a ``.yaml`` suffix):

    export LITE_OSWORLD_CONFIG=/abs/path/advanced.yaml  # full path
    export BROWSERGYM_CONFIG=isolation.yaml             # configs/isolation.yaml
    export BROWSERGYM_CONFIG=isolation                  # configs/isolation.yaml (bare name)

Precedence (defaults only — a rollout's env_kwargs still override per run):
    rollout/grpo yaml env_kwargs  >  <PREFIX>_CONFIG override  >  overrides[name]  >  default.yaml
(the ``overrides[name]`` layer applies only when a caller uses ``for_override(name)``;
single-variant envs skip it.)

Implementation owner: ``lite.gym.utils.config.defaults``. The package
``lite.gym.utils.config`` is a narrow facade so env modules can keep importing
``from lite.gym.utils import config``.

Run (test): ``uv run python -m pytest -n0 tests/gym/utils/config/test_config.py``
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from lite.utils.config import deep_merge


@dataclass(frozen=True)
class EnvConfig:
    """Resolved per-env defaults (default.yaml, or a <PREFIX>_CONFIG replacement)."""

    env_kwargs: dict
    server_kwargs: dict
    # Env-wide DEFAULTS for make() config. Distinct from env_kwargs (which go to
    # the env factory/bind): wrapper-owned keys such as step_timeout/reset_timeout
    # are consumed by gym.make(), while env-owned keys such as cursor are carried
    # to the env. An env declares them once here (the uniform case) via
    # registry.set_env_make_kwargs(); per-task register() and gym.make() still
    # override. Empty for envs that declare none. See registry.CARRIED_SPEC_KWARGS.
    make_kwargs: dict = field(default_factory=dict)
    # Per-name overrides for a multi-variant env (e.g. cua.bench: one entry per
    # dataset). Maps a name → {env_kwargs, server_kwargs, make_kwargs}, each merged
    # over the base groups above via ``for_override(name)`` (per-group deep merge —
    # see that method). Empty for single-variant envs. This keeps a growing env's
    # per-variant tuning fully config-driven (add a block per new name) instead of
    # hardcoded at registration.
    overrides: dict = field(default_factory=dict)
    # Readback/provenance for server-mode diagnostics. These fields are intentionally
    # separate from the three config groups above so callers cannot mistake config
    # source for a knob that should flow into gym.make().
    env_var_prefix: str = ""
    config_env_var: str = ""
    config_path: str = ""
    config_source: str = "default"
    config_env_var_set: bool = False

    def for_override(self, name: str) -> EnvConfig:
        """Return this config with ``overrides[name]`` **deep-merged** over the base
        (per-group deep merge — a nested override, e.g. ``env_kwargs.computer.image``,
        wins only at the named leaf and keeps its siblings, matching the rollout/train
        env_kwargs merge; a shallow ``{**base, **ov}`` would replace the whole sub-dict).
        Unknown name → the base config unchanged."""
        ov = self.overrides.get(name) or {}
        return EnvConfig(
            env_kwargs=deep_merge(self.env_kwargs, ov.get("env_kwargs") or {}),
            server_kwargs=deep_merge(self.server_kwargs, ov.get("server_kwargs") or {}),
            make_kwargs=deep_merge(self.make_kwargs, ov.get("make_kwargs") or {}),
            env_var_prefix=self.env_var_prefix,
            config_env_var=self.config_env_var,
            config_path=self.config_path,
            config_source=self.config_source,
            config_env_var_set=self.config_env_var_set,
        )

    def provenance(self) -> dict[str, Any]:
        """Machine-readable source metadata for this config file.

        This deliberately omits the override env-var's raw value. ``config_path`` is
        already the resolved path/name the process used; secret-bearing process
        environment belongs in the env-server's redacted process-env readback.
        """
        return {
            "env_var_prefix": self.env_var_prefix,
            "config_env_var": self.config_env_var,
            "config_env_var_set": self.config_env_var_set,
            "config_source": self.config_source,
            "config_path": self.config_path,
        }


@cache
def load(env_dir: str) -> EnvConfig:
    """Read ``<env_dir>/configs/default.yaml`` — unless ``<PREFIX>_CONFIG`` is set,
    in which case that file REPLACES it WHOLE (no merge). Cached per ``env_dir``
    (one read per process); the override env-var is read at first-call (env-module
    import) time, so it must be set before the env package is imported.
    """
    default_path = Path(env_dir) / "configs" / "default.yaml"
    raw = _read(default_path)
    assert "env_var_prefix" in raw, (
        f"{env_dir}/configs/default.yaml missing required key 'env_var_prefix'"
    )
    env_var_prefix = str(raw["env_var_prefix"])
    config_env_var = f"{env_var_prefix}_CONFIG"
    source_path = default_path
    source = "default"
    override = os.environ.get(config_env_var)
    if override:
        p = Path(override)
        if not p.is_absolute():
            # Relative value → resolve under the env's configs/ dir. A bare name
            # ("isolation") gets a .yaml suffix; "isolation.yaml" is used as-is.
            name = override if override.endswith((".yaml", ".yml")) else f"{override}.yaml"
            p = Path(env_dir) / "configs" / name
        source_path = p
        source = "override"
        raw = _read(p)   # complete replacement — the file IS the whole config
    return EnvConfig(
        env_kwargs=raw.get("env_kwargs") or {},
        server_kwargs=raw.get("server_kwargs") or {},
        make_kwargs=raw.get("make_kwargs") or {},
        overrides=raw.get("overrides") or {},
        env_var_prefix=env_var_prefix,
        config_env_var=config_env_var,
        config_path=str(source_path.resolve()),
        config_source=source,
        config_env_var_set=override is not None,
    )


def _read(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def finalize_env_kwargs(env_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize an ``env_kwargs`` dict for ``gym.make`` consumption.

    Two cleanups, both safe to apply idempotently anywhere along the
    merge chain (yaml / args / prompt_data row):

    1. **Strip top-level ``None`` values.** PyArrow's struct schema
       unification fills missing keys with ``None`` when nested dicts
       vary across rows (parquet roundtrip artifact). Those ``None``s
       are not user intent and would shadow downstream defaults during
       per-key dict-merge — drop them.
    2. **Coerce ``resolution`` to ``tuple``.** yaml / parquet store it
       as a list; downstream envs expect a hashable pair.

    Called once at the final composition step in both rollout
    (``lite.infer.rollout.run_rollout``) and train
    (``lite.train.rollout.core.engine``) so the two pipelines stay
    byte-identical at the gym.make boundary.
    """
    if not env_kwargs:
        return {}
    out = {k: v for k, v in env_kwargs.items() if v is not None}
    if "resolution" in out:
        out["resolution"] = tuple(out["resolution"])
    return out
