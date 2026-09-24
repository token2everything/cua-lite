#!/usr/bin/env python3
"""Public source-checkout launcher for the CUA-Lite environment server.

Usage::

    uv run python scripts/serve_env.py --port 30100

The packaged entrypoint accepts the same flags and also works from a wheel::

    python -m lite.gym.remote.serve --port 30100

CLI flags, authentication policy, and startup logic are owned by
:mod:`lite.gym.remote.serve`. See ``--help`` for the operator reference and
[docs/envs.md#env-server](/docs/envs.md#env-server) for deployment guidance.
"""

from __future__ import annotations

from lite.gym.remote.serve import _parse_args, main

if __name__ == "__main__":
    main(_parse_args())
