"""Generate eval.jsonl from OSWorld task JSONs + per-domain ORACLES.

Two sources of truth:

1. Upstream OSWorld task JSONs under ``osworld_evaluation_examples/examples/``
   — provide id, instruction, config, evaluator (the run-time eval contract).
2. Per-domain ``<domain>.py`` modules in this package — provide hand-curated
   oracle metadata (action replays, ``after_postconfig`` flag, and canonical
   environment-specific exclude reasons) that cannot be derived from upstream.

Together they regenerate ``data/eval.jsonl`` byte-identically. Two tests guard
that chain: ``test_osworld.py::TestOraclesReconstruction::
test_convert_all_byte_identical_to_eval_jsonl`` re-runs ``convert_all()`` and
asserts byte-equality (skipped when the upstream OSWorld task package is absent),
and ``test_osworld.py::test_catalog_lock_matches_generated_jsonl`` byte-locks the
committed file's sha256 against ``data/catalog.lock.json``.

Usage:
    uv run python -m lite.gym.envs.lite.osworld.src.gen.eval
    uv run python -m lite.gym.envs.lite.osworld.src.gen.eval --domain chrome
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
from pathlib import Path

from lite.gym.envs.lite.osworld import exclude_reasons
# Postconfig normalization lives in postconfig.py (shared with the
# lite.scalecua dataset import) — keep a single copy.
from lite.gym.envs.lite.osworld.src.gen.eval.postconfig import normalize_postconfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# `infeasible` is DERIVED, not hardcoded: a task is infeasible iff its own upstream
# ``evaluator.func == "infeasible"`` (OSWorld's ground truth = "the correct answer is
# that this task cannot be done"). This auto-tracks upstream exactly — including
# 480bcfea, whose evaluator says infeasible even though upstream's own
# ``test_infeasible.json`` manifest omits it (upstream lists 28; the real count by
# evaluator is 29). It also stops us from mislabeling env-difference tasks as
# "infeasible": tasks that merely need a live network fetch / a heavy pip install /
# a GUI-driven setup are FEASIBLE (upstream's evaluator is a real check), so they get
# authored oracle_actions in the per-domain ORACLES instead of an infeasible block.
def _is_infeasible(evaluator: dict) -> bool:
    func = evaluator.get("func")
    return func == "infeasible" or (isinstance(func, list) and "infeasible" in func)

# Tasks requiring Google Drive / gnome-online-accounts (not available in our VM).
_GOOGLE_AUTH_TASK_IDS: frozenset[str] = frozenset({
    "0c825995-5b70-4526-b663-113f4c999dd2",  # multi_apps
    "22a4636f-8179-4357-8e87-d1743ece1f81",  # multi_apps
    "78aed49a-a710-4321-a793-b611a7c5b56b",  # multi_apps
    "4e9f0faf-2ecc-4ae8-a804-28c9a75d1ddc",  # multi_apps
    "a0b9dc9c-fc07-4a88-8c5d-5e3ecad91bcb",  # multi_apps
    "46407397-a7d5-4c6b-92c6-dbe038b1457b",  # multi_apps
    "897e3b53-5d4d-444b-85cb-2cdc8a97d903",  # multi_apps
    "b52b40a5-ad70-4c53-b5b0-5650a8387052",  # multi_apps
})


def _load_oracles(domain: str) -> dict[str, dict]:
    """Import ``<domain>.py`` from this package and return its ORACLES dict."""
    mod = importlib.import_module(f"{__package__}.{domain}")
    return mod.ORACLES


def _find_examples_dir() -> Path:
    env_val = os.environ.get("OSWORLD_EXAMPLES")
    if env_val:
        return Path(env_val)
    try:
        import osworld_evaluation_examples
        pkg_dir = Path(osworld_evaluation_examples.__file__).parent / "examples"
        if pkg_dir.is_dir():
            return pkg_dir
    except ImportError:
        pass
    raise FileNotFoundError("Set OSWORLD_EXAMPLES env var or install osworld package.")


def _rewrite(obj):
    """Recursively substitute OSWorld template variables."""
    if isinstance(obj, str):
        s = obj.replace("{CLIENT_PASSWORD}", "user")
        s = s.replace("{SCREEN_WIDTH_HALF}", "960")
        s = s.replace("{SCREEN_HEIGHT_HALF}", "540")
        s = s.replace("{SCREEN_WIDTH}", "1920")
        s = s.replace("{SCREEN_HEIGHT}", "1080")
        return s
    if isinstance(obj, list):
        return [_rewrite(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _rewrite(v) for k, v in obj.items()}
    return obj


def convert_task(task_json: dict, domain: str, oracles: dict[str, dict]) -> dict:
    """Convert one OSWorld task JSON to our format.

    Args:
        task_json: upstream OSWorld task JSON.
        domain: OSWorld domain name (e.g. "chrome").
        oracles: per-domain ORACLES dict (UUID → curated entry).
    """
    evaluator = task_json.get("evaluator", {})
    task_id = task_json["id"]

    # Exclusion is driven by the two frozensets above (derivable rules) plus
    # any canonical reason recorded in the per-domain ORACLES (manual
    # env-specific task exclusions). Implementation notes for the frozensets:
    #   infeasible:  evaluator.func == "infeasible"
    #                (or "infeasible" in func when func is a list)
    #   google_auth: any config[].type in {"googledrive", "login"}
    #                OR any evaluator.result[].type == "googledrive_file"
    oracle_entry = oracles.get(task_id, {})
    exclude_reason = None
    if _is_infeasible(evaluator):
        exclude_reason = "infeasible"
    elif task_id in _GOOGLE_AUTH_TASK_IDS:
        exclude_reason = "google_auth"
    elif "exclude_reason" in oracle_entry:
        exclude_reason = oracle_entry["exclude_reason"]
    if exclude_reason:
        exclude_reason = exclude_reasons.validate(exclude_reason)

    # Apply per-task evaluator override (e.g. XFCE-vs-GNOME platform diffs,
    # missing files, broken upstream URLs). Stored in ORACLES, not in a
    # global frozenset, so each override is co-located with its domain.
    evaluator = _rewrite(evaluator)
    if "evaluator" in oracle_entry:
        evaluator = oracle_entry["evaluator"]

    # Inject evaluator OPTIONS without replacing the whole upstream evaluator
    # contract (func/result/expected stay verbatim). Use for a verifier-side
    # tweak like examine_shape=False on a task whose gold carries placeholder
    # positions LibreOffice re-anchors to layout defaults on save. For a
    # func-LIST evaluator the value must be
    # a parallel list (one options dict per func), since the runner pads a short
    # options list with {} (= upstream defaults) rather than broadcasting.
    if "evaluator_options" in oracle_entry and isinstance(evaluator, dict):
        evaluator = dict(evaluator)
        evaluator["options"] = _rewrite(oracle_entry["evaluator_options"])

    # Validation: normalize the postconfig save chain across LO / VS Code /
    # GIMP so eval inherits the same safety nets that synth+perturb use.
    # See `normalize_postconfig` for the per-app rules.
    if isinstance(evaluator, dict) and evaluator.get("postconfig"):
        evaluator = dict(evaluator)
        evaluator["postconfig"] = normalize_postconfig(
            evaluator["postconfig"], domain,
        )

    oracle_actions = oracle_entry.get("actions", [])
    others: dict = {
        "domain": domain,
        # No ``oracle_verified`` flag: ``bool(oracle_actions)`` is the same fact,
        # and every generator writes ``oracle_actions``.
        "oracle_actions": oracle_actions,
        "oracle_after_postconfig": bool(oracle_entry.get("after_postconfig", False)),
    }
    if exclude_reason:
        others["exclude_reason"] = exclude_reason
    # config_override replaces the upstream base config entirely (use when the
    # upstream task JSON has changed in a way that breaks prepend/append logic).
    if "config_override" in oracle_entry:
        config = _rewrite(oracle_entry["config_override"])
    else:
        config = _rewrite(task_json.get("config", []))
    config_prepend = _rewrite(oracle_entry.get("config_prepend", []))
    config_append = _rewrite(oracle_entry.get("config_append", []))
    return {
        "task_id": f"osworld_{domain}_{task_id[:8]}",
        "instruction": _rewrite(task_json.get("instruction", "")),
        "metadata": {
            "others": others,
            "osworld_id": task_id,
            "config": config_prepend + config + config_append,
            "evaluator": evaluator,
        },
    }


def convert_all(examples_dir: Path) -> list[dict]:
    tasks: list[dict] = []
    for domain_dir in sorted(examples_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
        domain = domain_dir.name
        oracles = _load_oracles(domain)
        before = len(tasks)
        for f in sorted(domain_dir.iterdir()):
            if f.suffix != ".json":
                continue
            task_json = json.loads(f.read_text())
            tasks.append(convert_task(task_json, domain, oracles))
        n_domain = len(tasks) - before
        n_excluded = sum(
            1 for t in tasks[before:]
            if t["metadata"]["others"].get("exclude_reason")
        )
        logger.info("Converted %d tasks from %s (%d excluded)", n_domain, domain, n_excluded)
    return tasks


def to_jsonl(tasks: list[dict], output: str) -> None:
    with open(output, "w") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    logger.info("Exported %d tasks → %s", len(tasks), output)


_OSWORLD_ROOT = Path(__file__).resolve().parents[3]  # eval/__main__.py -> osworld/
_DEFAULT_EVAL_OUT = str(_OSWORLD_ROOT / "data" / "eval.jsonl")


def main(args: argparse.Namespace | None = None):
    parser = argparse.ArgumentParser(description="Generate eval.jsonl from OSWorld task JSONs.")
    parser.add_argument("-o", "--output", default=_DEFAULT_EVAL_OUT)
    parser.add_argument("--domain", default=None)
    args = args or parser.parse_args()

    output = args.output or _DEFAULT_EVAL_OUT

    examples_dir = _find_examples_dir()
    if args.domain:
        oracles = _load_oracles(args.domain)
        tasks = []
        for f in sorted((examples_dir / args.domain).iterdir()):
            if f.suffix == ".json":
                tasks.append(convert_task(json.loads(f.read_text()), args.domain, oracles))
    else:
        tasks = convert_all(examples_dir)

    to_jsonl(tasks, output)


if __name__ == "__main__":
    main()
