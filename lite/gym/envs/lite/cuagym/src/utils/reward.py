"""CUA-Gym reward parser compatibility exports.

Every CUA-Gym task ships a ``reward.py`` that prints a final score line; both the
browser (subprocess) and desktop (in-container) backends run it verbatim and read the
score from stdout with the same rule. Upstream is inconsistent about casing
(``REWARD: x`` vs ``reward:x``) and sometimes prints diagnostics first, so the
match is case-insensitive and the LAST match wins.
"""

from __future__ import annotations

import math
import re

REWARD_RE = re.compile(
    r"^[ \t]*REWARD:[ \t]*"
    r"([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)"
    r"[ \t]*\r?$",
    re.IGNORECASE | re.MULTILINE,
)


def parse_reward(output: str) -> float:
    last = None
    for last in REWARD_RE.finditer(output):
        pass
    if last is None:
        raise ValueError("CUA-Gym reward output has no REWARD sentinel")
    value = float(last.group(1))
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"CUA-Gym reward must be within [0, 1], got {value!r}")
    return value


__all__ = ["REWARD_RE", "parse_reward"]
