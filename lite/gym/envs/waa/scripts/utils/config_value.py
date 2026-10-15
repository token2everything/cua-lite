#!/usr/bin/env python3
"""Print one resolved WindowsAgentArena config value for lifecycle scripts.

Run: uv run python lite/gym/envs/waa/scripts/utils/config_value.py \\
       env_kwargs runner_image   # <section> <key>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lite.gym.utils import config

ENV_DIR = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("section", choices=("env_kwargs", "server_kwargs", "make_kwargs"))
    parser.add_argument("key")
    args = parser.parse_args()
    cfg = config.load(str(ENV_DIR))
    values = getattr(cfg, args.section)
    if args.key not in values:
        raise KeyError(f"{args.section}.{args.key} is not configured")
    value = values[args.key]
    print("" if value is None else value)


if __name__ == "__main__":
    main()
