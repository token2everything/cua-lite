"""Import ScaleCUA OSWorld tasks into the lite.scalecua task cache.

Run via the wrapper (sets PYTHONPATH to the repo root):
    uv run --no-sync bash lite/gym/envs/lite/scalecua/scripts/utils/tasks.sh generate
"""

from __future__ import annotations

from lite.gym.envs.lite.scalecua.src.utils.dataset import _main


if __name__ == "__main__":
    _main()
