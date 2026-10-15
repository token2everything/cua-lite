"""webharbor — umbrella env namespace for offline-mirror web benchmarks.

These benchmarks run against shared static / offline-mirrored websites
(a WebHarbor mirror) rather than the live internet. The namespace makes
the "offline mirror" contract explicit at the directory level.

Registry discovers envs via envs/{name}/main.py. This file imports each
sub-env's main module so that "webharbor.webvoyager" tasks get registered
when the registry resolves "webharbor.{variant}" → base "webharbor" →
envs/webharbor/main.py.
"""

from __future__ import annotations

import lite.gym.envs.webharbor.webvoyager.main  # noqa: F401
