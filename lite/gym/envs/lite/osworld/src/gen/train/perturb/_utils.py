"""Perturb-specific helpers (Track B only).

Holds the `KnobSpec` dataclass and the `make_perturb_row` row-builder.
Imported by every `perturb/<domain>.py` module.

Run-as-script: this module is import-only; it has no `__main__` entry.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from lite.gym.envs.lite.osworld.src.gen.common import (
    NOISE_CANDIDATES,
)


@dataclass
class KnobSpec:
    """Spec for one perturbable eval task (Track B)."""

    task_id: str
    knobs: list[dict]  # [{"name": "...", "kind": "enum", "values": [...], "eval_value": ...}]
    perturb_fn: str  # function name in the domain's perturb module
    perturbable: bool = True


def metadata_others(row: dict) -> dict:
    """Return task metadata.others, with a defensive empty-dict fallback."""
    metadata = row.get("metadata", {})
    others = metadata.get("others")
    return others if isinstance(others, dict) else {}


def oracle_actions_of(row: dict) -> list:
    """Return oracle actions from the Lite metadata-compatible location."""
    return metadata_others(row).get("oracle_actions", [])


def oracle_after_postconfig_of(row: dict) -> bool:
    return bool(metadata_others(row).get("oracle_after_postconfig", False))


def make_perturb_row(
    eval_row: dict,
    knob_assignment: dict,
    new_instruction: str,
    new_oracle: list,
    new_evaluator: dict,
    perturb_config_step: dict | None = None,
    pre_config_steps: list[dict] | None = None,
    oracle_after_postconfig: bool | None = None,
) -> dict:
    """Build one train.perturb.jsonl row from an eval row + perturbation."""
    knob_hash = hashlib.md5(
        json.dumps(knob_assignment, sort_keys=True).encode()
    ).hexdigest()[:8]
    task_id = f"perturb_{eval_row['task_id']}_{knob_hash}"

    eval_domain = eval_row["metadata"]["others"].get("domain", "unknown")

    # Build config: optional pre-steps come first (e.g. to fix upstream setup
    # bugs), then eval-row's config, then optional post-step.
    config = list(pre_config_steps or [])
    config.extend(eval_row["metadata"]["config"])
    if perturb_config_step:
        config.append(perturb_config_step)

    # validation-post: max_steps removed from JSONL output. yaml's
    # env_kwargs.max_steps owns runtime cap (mirrors make_synth_row).
    return {
        "task_id": task_id,
        "instruction": new_instruction,
        "metadata": {
            "others": {
                "domain": eval_domain,
                "source": f"perturb:{eval_row['task_id']}",
                "oracle_actions": new_oracle,
                "oracle_after_postconfig": (
                    oracle_after_postconfig if oracle_after_postconfig is not None
                    else oracle_after_postconfig_of(eval_row)
                ),
            },
            "config": config,
            "evaluator": new_evaluator,
            "noise": NOISE_CANDIDATES.get(eval_domain, {"candidates": ["mouse_jitter", "desktop_files"], "max_apply": 2}),
        },
    }
