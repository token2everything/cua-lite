"""Canonical TASK-LEVEL ``exclude_reason`` vocabulary for the lite.osworld task
family (shared base) and the lite.scalecua catalogs that import it.

An ``exclude_reason`` is a property of the *task* (env catalog row): the task is
not runnable / not oracle-verifiable in this env. It is written to
``metadata.others.exclude_reason`` by the catalog producers
(``osworld/src/gen/eval/__main__.py``, ``scalecua/src/utils/dataset.py``) and read
downstream via ``not m.others.get('exclude_reason')``.

FORMAT::

    entry    := category (":" detail)?
    category := [a-z][a-z0-9_]*     # snake_case, closed set (REGISTRY below)
    detail   := [a-z0-9][a-z0-9_]*  # optional snake_case sub-type

Rules: snake_case only, NO free prose in the value, closed vocabulary (no
free-text escape hatch), one-line human descriptions live only in ``REGISTRY``.
Trajectory-level exclusion is a SEPARATE namespace and does NOT belong here.

Run / self-check:
    uv run python -c "from lite.gym.envs.lite.osworld.exclude_reasons import selftest; selftest()"
"""

from __future__ import annotations

import re

#: category -> one-line description (the closed set; a value's category MUST be a key).
REGISTRY: dict[str, str] = {
    # --- not completable by any agent -------------------------------------
    "infeasible": "task cannot be completed by any agent (evaluator.func==infeasible or curated)",
    "google_auth": "requires Google/Drive login the env cannot provide",
    "proxy_required": "requires the residential/social-site proxy flag",
    # --- upstream / evaluator defects -------------------------------------
    "upstream_live_site_drift": "evaluator depends on live-site state that drifted / bot-protection / nondeterministic live fetch",
    "upstream_generated_eval_bug": "generated/upstream evaluator is broken or vacuous (no-op passes, impossible gold, code defect)",
    "instruction_eval_mismatch": "instruction and evaluator disagree on the target",
    "instruction_setup_mismatch": "instruction and setup/config disagree (incl. missing instruction asset URL)",
    "trivial_pass": "start state already satisfies the check (detail = which precheck, e.g. color_precheck)",
    "unverifiable": "task is valid but no oracle can verify it in this env (detail = what, e.g. pdf_generation)",
    # --- env / asset gaps -------------------------------------------------
    "missing_dependency": "instruction mandates a CLI tool vendored off the agent PATH (detail = tool, e.g. imagemagick)",
    "missing_reference_asset": "a referenced asset file is absent",
    "unsupported_asset_url": "an asset URL the importer cannot fetch/pin",
    "unsupported_setup": "a setup/config action the env does not support (detail = slug)",
    "unsupported_schema": "a task-JSON schema shape the importer cannot handle (detail = slug)",
    "flake": "nondeterministic env/runtime flake quarantined pending a code fix (detail = slug)",
}

#: Categories where a ``:detail`` is REQUIRED (a bare category is not meaningful).
DETAIL_REQUIRED: frozenset[str] = frozenset({
    "trivial_pass", "unverifiable",
    "missing_dependency", "unsupported_setup", "unsupported_schema", "flake",
})

#: Closed detail vocabulary for categories that require a ``:detail``. These are
#: current producer/catalog values plus documented canonical examples; details
#: not listed here are rejected, so ``category:any`` cannot silently become prose.
DETAIL_ALLOWED: dict[str, frozenset[str]] = {
    "trivial_pass": frozenset({"color_precheck"}),
    "unverifiable": frozenset({"pdf_generation", "ocr"}),
    "missing_dependency": frozenset({"imagemagick", "java"}),
    "unsupported_setup": frozenset({
        "close_all_libreoffice",
        "copyfile_from_guest_to_host",
        "navigate_to_chrome_extensions",
    }),
    "unsupported_schema": frozenset({
        "action_list",
        "action_object",
        "action_type_missing",
        "evaluator_postconfig_query_config",
    }),
    "flake": frozenset({
        "chrome_cdp_active_tab_nav",
        "chrome_gui_extension_load",
        "chrome_secure_prefs_mac",
    }),
}

_ENTRY_RE = re.compile(r"\A(?P<category>[a-z][a-z0-9_]*)(?::(?P<detail>[a-z0-9][a-z0-9_]*))?\Z")


def validate(reason: str) -> str:
    """Return ``reason`` if it is a well-formed, in-vocabulary task-level
    exclude_reason; raise ``ValueError`` otherwise. (Single value, not a list —
    task-level rows carry exactly one reason.)"""
    if not isinstance(reason, str) or not reason:
        raise ValueError(f"exclude_reason must be a non-empty string, got {reason!r}")
    m = _ENTRY_RE.match(reason)
    if not m:
        raise ValueError(
            f"exclude_reason {reason!r} does not match category(:detail)? "
            r"(^[a-z][a-z0-9_]*(:[a-z0-9][a-z0-9_]*)?$) — no prose, snake_case only"
        )
    category = m.group("category")
    if category not in REGISTRY:
        raise ValueError(
            f"exclude_reason category {category!r} is not in the closed vocabulary "
            f"(REGISTRY keys: {sorted(REGISTRY)})"
        )
    detail = m.group("detail")
    if category in DETAIL_REQUIRED and not detail:
        raise ValueError(f"exclude_reason category {category!r} requires a :detail")
    if category not in DETAIL_REQUIRED and detail:
        raise ValueError(f"exclude_reason category {category!r} does not accept a :detail")
    if detail and detail not in DETAIL_ALLOWED.get(category, frozenset()):
        raise ValueError(
            f"exclude_reason detail {detail!r} is not allowed for category {category!r} "
            f"(allowed: {sorted(DETAIL_ALLOWED.get(category, frozenset()))})"
        )
    return reason


def is_valid(reason: str) -> bool:
    try:
        validate(reason)
        return True
    except ValueError:
        return False


def selftest() -> None:
    for ok in ("infeasible", "upstream_live_site_drift",
               "missing_dependency:imagemagick", "unsupported_setup:close_all_libreoffice",
               "flake:chrome_cdp_active_tab_nav", "trivial_pass:color_precheck"):
        assert is_valid(ok), ok
    for bad in ("", "block: live-site redirect (x)", "trivial_pass: color check skipped",
                "Infeasible", "proxy_required:any",
                "missing_dependency:any", "unsupported_schema:legacy_config",
                "made_up_category", "infeasible\n", "infeasible\ninfeasible",
                "flake:chrome_cdp_active_tab_nav\n"):
        assert not is_valid(bad), bad
    print("exclude_reasons.selftest OK")


if __name__ == "__main__":
    selftest()
