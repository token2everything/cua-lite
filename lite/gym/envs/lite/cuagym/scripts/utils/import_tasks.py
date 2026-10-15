"""Unified CUA-Gym task importer entrypoint."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("web", "desktop"), required=True)
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    if args.backend == "web":
        from lite.gym.envs.lite.cuagym.scripts.utils.import_web_tasks import main as run
    else:
        from lite.gym.envs.lite.cuagym.scripts.utils.import_desktop_tasks import main as run
    run()


if __name__ == "__main__":
    main()
