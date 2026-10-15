"""Structural perturbation dispatch (Track B).

Each domain exports a single entry point with explicit budget args:
    perturb_<domain>_per_task(eval_row, rng, max_type1=N[, max_type2=M, ...]) -> list[dict]

Default budgets live in each fn's signature — that's the single source of truth.
Callers may override via `apply_structural_perturbation(..., max_type1=X)`; the
dispatcher uses `inspect.signature` to drop kwargs the domain doesn't accept.

Usage:
    from lite.gym.envs.lite.osworld.src.gen.train.perturb.dispatch import apply_structural_perturbation
    rows = apply_structural_perturbation(eval_row, rng)
"""

from __future__ import annotations

import inspect
import logging
import random

logger = logging.getLogger(__name__)

# Per-task instruction-only patches applied AFTER perturbation.
# Use when the upstream eval-task instruction is wrong (typo, stale ref)
# but we don't want to change eval.jsonl. Each entry: list of (old, new) tuples.
_INSTRUCTION_PATCHES: dict[str, list[tuple[str, str]]] = {
    # eval-task instruction says _config.yaml but the cloned academicpages.github.io
    # repo (and the evaluator's path) actually uses _config.yml.
    "osworld_multi_apps_e2392362": [("_config.yaml", "_config.yml")],
    # Instruction says "'Grammar test' folder" + "'answer doc' file" but the
    # config downloads "Grammer test 1/2/3.docx" (typo, flat on Desktop) and
    # "Answer.docx" (capital A).
    "osworld_multi_apps_1f18aa87": [
        ("'Grammar test' folder", "Desktop"),
        ("'answer doc' file", "'Answer.docx' file"),
    ],
}

_REGISTRY: dict[str, callable] | None = None


def _get_registry() -> dict[str, callable]:
    global _REGISTRY
    if _REGISTRY is None:
        from lite.gym.envs.lite.osworld.src.gen.train.perturb.chrome import perturb_chrome_per_task
        from lite.gym.envs.lite.osworld.src.gen.train.perturb.gimp import perturb_gimp_per_task
        from lite.gym.envs.lite.osworld.src.gen.train.perturb.libreoffice_calc import perturb_calc_per_task
        from lite.gym.envs.lite.osworld.src.gen.train.perturb.libreoffice_impress import perturb_impress_per_task
        from lite.gym.envs.lite.osworld.src.gen.train.perturb.libreoffice_writer import perturb_writer_per_task
        from lite.gym.envs.lite.osworld.src.gen.train.perturb.multi_apps import perturb_multi_apps_per_task
        from lite.gym.envs.lite.osworld.src.gen.train.perturb.os import perturb_os_per_task
        from lite.gym.envs.lite.osworld.src.gen.train.perturb.thunderbird import perturb_thunderbird_per_task
        from lite.gym.envs.lite.osworld.src.gen.train.perturb.vlc import perturb_vlc_per_task
        from lite.gym.envs.lite.osworld.src.gen.train.perturb.vs_code import perturb_vscode_per_task
        _REGISTRY = {
            "libreoffice_calc": perturb_calc_per_task,
            "libreoffice_writer": perturb_writer_per_task,
            "libreoffice_impress": perturb_impress_per_task,
            "chrome": perturb_chrome_per_task,
            "vlc": perturb_vlc_per_task,
            "vs_code": perturb_vscode_per_task,
            "os": perturb_os_per_task,
            "thunderbird": perturb_thunderbird_per_task,
            "gimp": perturb_gimp_per_task,
            "multi_apps": perturb_multi_apps_per_task,
        }
    return _REGISTRY


def apply_structural_perturbation(
    eval_row: dict,
    rng: random.Random,
    eval_instructions: set[str] | None = None,
    **budget_overrides,
) -> list[dict]:
    """Generate perturbed rows for one eval task.

    Looks up the domain entry, passes through `budget_overrides` (filtered to
    parameters the entry actually accepts via `inspect.signature`), then drops
    rows whose instruction matches any eval instruction (leakage), and applies
    per-task instruction patches.
    """
    # Skip eval tasks that are themselves infeasible. Perturbing an infeasible
    # eval task is unsafe because (a) the source state may have been crafted
    # around the infeasibility premise, and (b) per validation design,
    # train.perturb does NOT generate infeasible tasks (decline-with-reasoning
    # is an eval-only skill axis).
    ev = eval_row.get("metadata", {}).get("evaluator", {})
    fn_field = ev.get("func")
    if fn_field == "infeasible" or (isinstance(fn_field, list) and "infeasible" in fn_field):
        return []

    domain = eval_row["metadata"]["others"].get("domain", "")
    fn = _get_registry().get(domain)
    if fn is None:
        return []

    if budget_overrides:
        accepted = set(inspect.signature(fn).parameters)
        kwargs = {k: v for k, v in budget_overrides.items() if k in accepted}
    else:
        kwargs = {}

    try:
        rows = fn(eval_row, rng, **kwargs)
    except Exception:
        logger.exception("Perturb failed for %s/%s", domain, eval_row["task_id"])
        return []

    banned = eval_instructions if eval_instructions is not None else {eval_row.get("instruction", "")}
    rows = [r for r in rows if r.get("instruction", "") not in banned]

    patches = _INSTRUCTION_PATCHES.get(eval_row["task_id"], [])
    if patches:
        for r in rows:
            instr = r.get("instruction", "")
            for old, new in patches:
                instr = instr.replace(old, new)
            r["instruction"] = instr

    return rows
