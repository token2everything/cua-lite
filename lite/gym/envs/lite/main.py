"""lite — umbrella env that imports all lite sub-environments.

Registry discovers envs via envs/{name}/main.py. This file imports
sub-env main modules so that "lite.demo" / "lite.osworld" tasks get registered when
the registry resolves "lite.{variant}" → base "lite" → envs/lite/main.py.
"""

from __future__ import annotations

import lite.gym.envs.lite.cuagym.main  # noqa: F401  (umbrella: browser + desktop backends)
import lite.gym.envs.lite.cuaworld.main  # noqa: F401  (umbrella: software variants)
import lite.gym.envs.lite.demo.main  # noqa: F401
import lite.gym.envs.lite.osworld.main  # noqa: F401
import lite.gym.envs.lite.scalecua.main  # noqa: F401  (umbrella: ScaleCUA OSWorld adapter)
