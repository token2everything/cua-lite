"""CLI driver for generating training JSONL (Track A synth + Track B perturb).

Outputs separate files per track: ``train.synth.jsonl`` and
``train.perturb.jsonl`` (dot-separated to mirror the import paths
``...generate.train.synth`` / ``...generate.train.perturb``). Noise config
is stored declaratively in ``metadata.noise`` (per-domain candidates); the
env applies it at reset time only when ``noise=True`` is passed via
``env_kwargs``.

Usage:
    # Generate both tracks (default: data/train.synth.jsonl + data/train.perturb.jsonl)
    uv run python -m lite.gym.envs.lite.osworld.src.gen.train

    # Generate only synthetic track
    uv run python -m lite.gym.envs.lite.osworld.src.gen.train --track synth

    # Generate only perturbation track
    uv run python -m lite.gym.envs.lite.osworld.src.gen.train --track perturb

    # Specific domain
    uv run python -m lite.gym.envs.lite.osworld.src.gen.train --domain calc
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from lite.gym.envs.lite.osworld.src.gen.train.synth._utils import make_synth_row

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_OSWORLD_ROOT = Path(__file__).resolve().parents[3]  # train/__main__.py -> osworld/
_EVAL_PATH = _OSWORLD_ROOT / "data" / "eval.jsonl"
_DEFAULT_SYNTH_OUT = _OSWORLD_ROOT / "data" / "train.synth.jsonl"
_DEFAULT_PERTURB_OUT = _OSWORLD_ROOT / "data" / "train.perturb.jsonl"

_DOMAIN_ALIASES: dict[str, list[str]] = {
    "calc": ["libreoffice_calc"],
    "writer": ["libreoffice_writer"],
    "impress": ["libreoffice_impress"],
}


def _resolve_domains(domain_arg: str) -> list[str] | None:
    if domain_arg == "all":
        return None
    if domain_arg in _DOMAIN_ALIASES:
        return _DOMAIN_ALIASES[domain_arg]
    return [domain_arg]


def _generate_synth(domains: list[str] | None) -> list[dict]:
    """Generate synthetic training rows (Track A)."""
    from lite.gym.envs.lite.osworld.src.gen.train.synth.catalog import (
        ALL_TEMPLATES,
        TEMPLATES_BY_DOMAIN,
    )

    if domains is None:
        templates = ALL_TEMPLATES
    else:
        templates = []
        for d in domains:
            templates.extend(TEMPLATES_BY_DOMAIN.get(d, []))

    rows: list[dict] = []
    for template in templates:
        logger.info(
            "Generating %d rows for template %s (%s)",
            template.n_rows, template.template_id, template.domain,
        )
        for seed in range(1, template.n_rows + 1):
            params = template.param_fn(seed)
            if params.get("_skip"):
                logger.debug("Skipping seed %d for template %s", seed, template.template_id)
                continue
            row = make_synth_row(template, seed, params)
            rows.append(row)

    return rows


def _generate_perturb(domains: list[str] | None) -> list[dict]:
    """Generate perturbed training rows (Track B).

    Dispatches to per-domain structural perturbation functions. Each function
    returns [] when a task is not perturbable — domain functions are the source
    of truth on eligibility.
    """
    from lite.gym.envs.lite.osworld.src.gen.train.perturb.dispatch import (
        apply_structural_perturbation,
    )

    # Load eval rows
    eval_rows: dict[str, dict] = {}
    with _EVAL_PATH.open() as f:
        for line in f:
            row = json.loads(line.strip())
            eval_rows[row["task_id"]] = row

    eval_instructions: set[str] = {r["instruction"] for r in eval_rows.values()}

    rows: list[dict] = []
    rng = random.Random(42)
    seen_ids: set[str] = set()
    # Validation: cross-eval-source content-dedup. Two different eval
    # bases can produce structurally-identical perturb rows (e.g. vs_code
    # eval `0512bb38` and `4e60007a` both perturbed to "install ESLint" with
    # the same knob_hash because the chosen extension is independent of the
    # eval source state). task_id dedup alone misses this because eval_tid is
    # part of the task_id. Track (instruction, evaluator) tuples so the
    # second occurrence is dropped.
    seen_contents: set[tuple[str, str]] = set()
    dedup_dropped = 0

    structural_count = 0
    for task_id, eval_row in eval_rows.items():
        domain = eval_row["metadata"]["others"].get("domain", "unknown")
        if domains is not None and domain not in domains:
            continue

        new_rows = apply_structural_perturbation(eval_row, rng, eval_instructions=eval_instructions)
        for r in new_rows:
            if r["task_id"] in seen_ids:
                continue
            content_key = (
                r["instruction"],
                json.dumps(r["metadata"]["evaluator"], sort_keys=True),
            )
            if content_key in seen_contents:
                dedup_dropped += 1
                continue
            rows.append(r)
            seen_ids.add(r["task_id"])
            seen_contents.add(content_key)
            structural_count += 1

    if dedup_dropped:
        logger.info("Cross-source content-dedup dropped %d rows", dedup_dropped)
    logger.info("Structural perturbation: %d rows", structural_count)
    return rows


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rows -> %s", len(rows), path)


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        parser = argparse.ArgumentParser(
            description="Generate training JSONL for lite_osworld (Track A synth + Track B perturb)."
        )
        parser.add_argument(
            "--track", choices=["synth", "perturb", "all"], default="all",
            help="Which track to generate (default: all)",
        )
        parser.add_argument(
            "--domain", default="all",
            help="Domain to generate for (e.g. calc, chrome, all)",
        )
        parser.add_argument(
            "-o", "--output", default=None,
            help="Override output path (writes a single file instead of per-track files)",
        )
        parser.add_argument(
            "--seed", type=int, default=42,
            help="Random seed for reproducibility",
        )
        args = parser.parse_args()

    random.seed(args.seed)
    domains = _resolve_domains(args.domain)

    if args.track in ("synth", "all"):
        synth_rows = _generate_synth(domains)
        logger.info("Synthetic: %d rows", len(synth_rows))
        out = Path(args.output) if args.output else _DEFAULT_SYNTH_OUT
        _write_jsonl(synth_rows, out)

    if args.track in ("perturb", "all"):
        perturb_rows = _generate_perturb(domains)
        logger.info("Perturb: %d rows", len(perturb_rows))
        out = Path(args.output) if args.output else _DEFAULT_PERTURB_OUT
        _write_jsonl(perturb_rows, out)


if __name__ == "__main__":
    main()
