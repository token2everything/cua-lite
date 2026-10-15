"""cua.sandbox — open-ended CuaSandboxEnv wrapper (not a gym-registered env).

The env lives in ``env.py`` (NOT ``main.py``): ``main.py`` is the registry's
directory-scan marker for a *registerable* env (``registry.env_ids`` treats any
``<subdir>/main.py`` as a leaf env), and cua.sandbox is a direct wrapper — never
``gym.make``-d — so naming it ``main.py`` would falsely list it in the catalog.
Re-exported here so callers keep the flat ``from lite.gym.envs.cua.sandbox
import CuaSandboxEnv`` path.
"""

from __future__ import annotations

from lite.gym.envs.cua.sandbox.env import CuaSandboxEnv

__all__ = ["CuaSandboxEnv"]
