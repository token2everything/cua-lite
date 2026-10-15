"""Synthetic template registry (Track A) — Design 5 (validation simplification).

Each domain module exports a TEMPLATES list of SynthTemplate instances.
The scaler in this file does ONE thing: cross-domain volume rebalance.

## Design 5 — Param downgrade for cross-domain volume

Global cap = `TARGET` (default `math.inf` → no cap, every template at natural
max). When finite, the cap is split per-domain proportional to that domain's
feasible-eval-row share: `target_per_domain = round(TARGET × eval[d] / Σeval)`.

For each domain:
  1. Every FileTask gets `n_rows = min(2, distinct_param_count)` (author intent).
     `distinct_param_count` is probed by running param_fn(0..3) and counting
     unique instructions.
  2. If sum(n_rows) > target → DOWNGRADE 2-Param FileTasks to 1 Param each
     until target is hit. (Sorted by template_id for determinism.)
  3. If sum(n_rows) > target even after all-1-Param → FLAG over-volume; the
     author should manually comment FileTasks in `<domain>.py` to trim further.
  4. If sum(n_rows) < target → FLAG under-volume; the author should add new
     FileTasks (cap-2×2 limits per-File contribution).

## What the scaler does NOT do (vs old design)

- NO `_TAXONOMY_EVAL_CLASS_K` dict
- NO `_EVAL_FUNC_TO_SKILL_CLASS` alias dict
- NO bucketing by skill_class
- NO silent zero (every FileTask emits ≥1 row)
- NO intra-domain skill ratio enforcement (author's responsibility per PD 4a)

Intra-domain skill ratio (PD 4a) and evaluator metric strictness (PD 4d) are
reviewed from generator metadata; authors adjust `<domain>.py` when a domain
drifts out of balance.

## Cap enforcement

cap-2×2 (≤2 FileTasks/File, ≤2 Params/FileTask) is enforced inside each
`<domain>.py:_emit_templates()` and `_to_synth_template()` — see those for
the per-domain implementations.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter

from lite.gym.envs.lite.osworld.src.gen.train.synth.chrome import TEMPLATES as CHROME_TEMPLATES
from lite.gym.envs.lite.osworld.src.gen.train.synth.gimp import TEMPLATES as GIMP_TEMPLATES
from lite.gym.envs.lite.osworld.src.gen.train.synth.libreoffice_calc import (
    TEMPLATES as CALC_TEMPLATES,
)
from lite.gym.envs.lite.osworld.src.gen.train.synth.libreoffice_impress import (
    TEMPLATES as IMPRESS_TEMPLATES,
)
from lite.gym.envs.lite.osworld.src.gen.train.synth.libreoffice_writer import (
    TEMPLATES as WRITER_TEMPLATES,
)
from lite.gym.envs.lite.osworld.src.gen.train.synth.multi_apps import (
    TEMPLATES as MULTI_APPS_TEMPLATES,
)
from lite.gym.envs.lite.osworld.src.gen.train.synth.os import TEMPLATES as OS_TEMPLATES
from lite.gym.envs.lite.osworld.src.gen.train.synth.thunderbird import (
    TEMPLATES as THUNDERBIRD_TEMPLATES,
)
from lite.gym.envs.lite.osworld.src.gen.train.synth.vlc import TEMPLATES as VLC_TEMPLATES
from lite.gym.envs.lite.osworld.src.gen.train.synth.vs_code import TEMPLATES as VSCODE_TEMPLATES
from lite.utils.path import project_root

# Global total-rows cap. `math.inf` (default) = no cap, emit every template at
# its natural max. A finite value (e.g. 1500) is split per-domain proportional
# to that domain's feasible-eval-row count; the per-domain Stage-B downgrade
# trims 2-Param FileTasks to 1 row until the per-domain cap is reached.
TARGET: float = math.inf

_EVAL_PATH = project_root() / "lite/gym/envs/lite/osworld/data/eval.jsonl"


def _probe_param_count(t, max_probe: int = 4) -> int:
    """Count distinct Params by probing `param_fn(seed)` for seed=0..max-1.

    Used to set initial `n_rows = min(2, distinct_count)` per cap-2×2.
    Single-Param FileTasks (e.g. configs that don't vary) get n_rows=1.
    """
    seen: set[str] = set()
    for seed in range(max_probe):
        try:
            params = t.param_fn(seed)
            instr = t.instruction_fn(params)
            seen.add(instr)
        except Exception:
            continue
    return min(len(seen) or 1, 2)


def _rescale_for_volume(templates: list) -> list:
    """Cross-domain volume rebalance via Param downgrade.

    See module docstring for the full algorithm. Mutates `template.n_rows`
    in-place; returns the same list.
    """
    if not _EVAL_PATH.exists():
        # No eval reference — leave each template at its natural max.
        for t in templates:
            t.n_rows = _probe_param_count(t)
        return templates

    # Per-domain eval row counts (skip infeasible — synth never has them).
    eval_per_domain: Counter = Counter()
    for line in _EVAL_PATH.open():
        r = json.loads(line)
        if r.get("metadata", {}).get("evaluator", {}).get("func") == "infeasible":
            continue
        eval_per_domain[r["metadata"]["others"].get("domain", "?")] += 1
    eval_total = sum(eval_per_domain.values())
    if eval_total == 0:
        return templates

    # Stage A: every template at author-intent max.
    for t in templates:
        t.n_rows = _probe_param_count(t)

    # Group by domain.
    from collections import defaultdict as _dd
    by_domain: dict[str, list] = _dd(list)
    for t in templates:
        by_domain[t.domain].append(t)

    # Per-domain Stage B: downgrade if over.
    flags: list[tuple] = []
    for domain, dom_ts in by_domain.items():
        eval_count = eval_per_domain.get(domain, 0)
        # Per-domain cap = TARGET * (this domain's eval share). When TARGET is
        # math.inf, the cap is also inf and Stage B never triggers — every
        # template emits at its natural Stage-A max.
        if math.isinf(TARGET):
            target = math.inf
        else:
            target = round(TARGET * eval_count / eval_total)
        max_pot = sum(t.n_rows for t in dom_ts)
        n_templates = len(dom_ts)

        if max_pot <= target:
            flags.append((domain, "UNDER", max_pot, target, n_templates))
            continue

        excess = max_pot - target
        # Downgrade 2-Param FileTasks (preserve 1-Param FileTasks at 1 row).
        downgrade_pool = sorted(
            [t for t in dom_ts if t.n_rows == 2],
            key=lambda t: t.template_id,
        )
        for t in downgrade_pool[:excess]:
            t.n_rows = 1

        post_rows = sum(t.n_rows for t in dom_ts)
        if post_rows > target:
            # Even all-1-Param overshoots — author has too many FileTasks.
            flags.append((domain, "OVER need_manual_comment",
                          post_rows, target, n_templates))

    # Print flags for author awareness.
    if flags:
        print("[synth scaler] Per-domain volume status:", file=sys.stderr)
        print(f"  {'DOMAIN':<22} {'STATUS':<26} {'rows':>6} / {'target':>6}  ({'N':>3}T)",
              file=sys.stderr)
        for domain, status, rows, target, n_t in flags:
            print(f"  {domain:<22} {status:<26} {rows:>6} / {target:>6}  ({n_t:>3}T)",
                  file=sys.stderr)

    return templates


# Drop rule:
#   ONLY drop a template when it is **structurally and definitively
#   unfixable** — eval contract impossible against the live env regardless
#   of agent skill, instruction wording, gold builder logic, or max_steps.
#
#   Valid (structural) drop reasons:
#     - chrome real-web (bot detect, redirect, login wall) on non-staged URLs
#     - extension marketplace network dependency
#     - schema / capability absent on target desktop image
#     - eval threshold tighter than ANY gold output can satisfy
#
#   NOT valid drop reasons (fixable, NOT drops):
#     - 'agent failed in rollout' → fix instruction / gold / max_steps
#     - 'capability ceiling' → not a structural environment defect
#     - '≥3 variants share failure' → MUST find source-side fix
#     - 'oracle failed' → fix gold builder or relax eval, don't drop
#
#   Filtering at this level avoids per-file source edits.
_DROPPED_TEMPLATE_IDS: set[tuple[str, str]] = {
    # ---- chrome ----
    # f_chrome_36__search_kayak_hotel: kayak.com aggressively bot-detects and redirects automated traffic to consent/captcha pages, stripping the sort= query param. chec
    ('chrome', 'f_chrome_36__search_kayak_hotel'),
    # f_chrome_53__search_spotify_track: open.spotify.com aggressively redirects unauthenticated/automated traffic to the login wall (accounts.spotify.com), losing the q=
    ('chrome', 'f_chrome_53__search_spotify_track'),
    # f_chrome_54__search_instacart_item: instacart.com requires geolocation/zip-code prompt and authentication; /store/search redirects to /store?q= or to onboarding flow.
    ('chrome', 'f_chrome_54__search_instacart_item'),
    # f_chrome_56__navigate_opentable_city: opentable.com /s with term= covers= is regional and often redirects to /<country>/s; term= is renamed to 'q' in some regions. Netw
    ('chrome', 'f_chrome_56__navigate_opentable_city'),
    # f_chrome_60__search_etsy_item: etsy.com /search?q= often gets canonicalized server-side to /search/<slug> for popular queries, losing the q= param. Bot detection
    ('chrome', 'f_chrome_60__search_etsy_item'),
    # f_chrome_83__search_marriott_hotel: marriott.com/search/findHotels.mi returns Akamai "Access Denied" challenge page on automated traffic — same bot-detect family as kayak/spotify/instacart/opentable/etsy/wayfair.
    ('chrome', 'f_chrome_83__search_marriott_hotel'),
    # f_chrome_86__search_costco_filter: costco.com search/listing returns Akamai "Access Denied" challenge page. Validation confirmed explicit report_infeasible(reason="Access Denied"). Same bot-detect family as kayak/spotify/instacart/opentable/etsy/wayfair/marriott. Joins existing 7-member cluster.
    ('chrome', 'f_chrome_86__search_costco_filter'),
    # f_chrome_57__search_glassdoor_job: glassdoor.com server-side canonicalizes /Job/jobs.htm?sc.keyword=…&locT=…&locId=… to /Job/<slug>-jobs-SRCH_IL.<n>,<m>_IC<id>_KO<r>,<s>.htm — query params dropped. cdjo eval on sc.keyword/locT/locId is unsatisfiable against the current site.
    ('chrome', 'f_chrome_57__search_glassdoor_job'),
    # f_chrome_97__search_indeed_compound: indeed.com's modern UI no longer exposes filter params (`remotejob=0`, `fromage=14`) in the URL — filters apply via JS-only state. cdjo eval on those keys is unsatisfiable against current Indeed.
    ('chrome', 'f_chrome_97__search_indeed_compound'),
    # f_chrome_78__compare_samsung_galaxy: samsung.com smartphone-compare page canonicalizes to /us/smartphones/<model>/compare/ (path-based; ?modelList= query param does not survive UI interaction). cdjo eval is unsatisfiable.
    ('chrome', 'f_chrome_78__compare_samsung_galaxy'),
    # f_chrome_87__search_wayfair_filter: wayfair.com search URLs use opaque session-bound paths and add tracking params; bot detection redirects automated traffic to chall
    ('chrome', 'f_chrome_87__search_wayfair_filter'),
    # ---- vs_code ----
    # vs_code_install_ext_eslint: Marketplace install — oracle runs `code --no-sandbox --install-extension dbaeumer.vscode-eslint --force` which requires live netwo
    ('vs_code', 'vs_code_install_ext_eslint'),
    # vs_code_install_ext_go: Same as eslint — `code --install-extension golang.go --force` requires marketplace.visualstudio.com network. Bundled extension ins
    ('vs_code', 'vs_code_install_ext_go'),
    # vs_code_install_ext_prettier: Same as eslint — marketplace install of esbenp.prettier-vscode requires network.
    ('vs_code', 'vs_code_install_ext_prettier'),
    # ---- libreoffice_impress ----
    # d_imp_19__title_to_bottom: eval-vs-gold contract broken — Param.examine_field='examine_text' but gold mutates shape.top only; oracle trivially passes. Upstream examine_title_bottom_position is hard-coded to text 'Product Comparison' (slides.py:317-325). No clean fix path.
    ('libreoffice_impress', 'd_imp_19__title_to_bottom'),
    # d_imp_55__pagenum_green: Eval requires modifying slideMaster1.xml sldNum srgbClr via View > Master Slide. Pagenum placeholders are not reliably recolorable through the LO Impress UI; oracle path itself is fragile.
    ('libreoffice_impress', 'd_imp_55__pagenum_green'),
    # d_imp_55__pagenum_red: Same as pagenum_green — slideMaster pagenum recolor impractical via LO Impress UI.
    ('libreoffice_impress', 'd_imp_55__pagenum_red'),
    # d_imp_67__left_panel: infeasible to verify deterministically. xdotool window-name probing fails because 'Slides View' is an a11y document-frame name, not an X11 window name; registrymodifications.xcu does not reliably persist panel visibility.
    ('libreoffice_impress', 'd_imp_67__left_panel'),
    # ---- chrome (bot-detect / URL-canonicalization cluster extension) ----
    # f_chrome_56__search_opentable_reservation: Akamai "Access Denied" challenge page. Same family as kayak/spotify/instacart/etsy/marriott/costco.
    ('chrome', 'f_chrome_56__search_opentable_reservation'),
    # f_chrome_58__search_monster_job: monster.com CAPTCHA on automated traffic.
    ('chrome', 'f_chrome_58__search_monster_job'),
    # f_chrome_79__compare_sony_headphones: sony.com URL canonicalizes to product-detail path; query-param compare is unsatisfiable. Same family as f_chrome_78__compare_samsung_galaxy.
    ('chrome', 'f_chrome_79__compare_sony_headphones'),
    # f_chrome_82__search_avis_rental: avis.com 404 on the synthesized base path + JS overlay loop. URL-canonicalization cluster.
    ('chrome', 'f_chrome_82__search_avis_rental'),
    # f_chrome_31__search_macys_filter: macys.com Akamai Access Denied. Same bot-detect family as kayak/spotify/instacart/opentable/etsy/marriott/costco.
    ('chrome', 'f_chrome_31__search_macys_filter'),
    # ---- chrome (bot-detect / URL-canonicalization) ----
    # f_chrome_40__search_yelp_restaurant: yelp.com anti-bot CAPTCHA verification on automated traffic. Same family as kayak/spotify/marriott bot-detect cluster.
    ('chrome', 'f_chrome_40__search_yelp_restaurant'),
    # f_chrome_50__search_google_query: google.com reCAPTCHA on automated traffic. Same family.
    ('chrome', 'f_chrome_50__search_google_query'),
    # f_chrome_35__search_jetblue_fare: jetblue.com URL uses adults=/redemPoint= params; cdjo eval reads from=/to=/depart=/pax. Page loads fully (no bot-block) but URL-canonicalization makes the eval contract unsatisfiable.
    ('chrome', 'f_chrome_35__search_jetblue_fare'),
    # f_chrome_63__navigate_twitter_profile: twitter.com domain renamed to x.com; agent lands on x.com/NASA but eval regex hard-codes ^https://(www\.)?twitter\.com/nasa. URL-canonicalization (domain-rename variant). Cannot satisfy without eval-regex update.
    ('chrome', 'f_chrome_63__navigate_twitter_profile'),
    # ---- chrome (bot-detect) ----
    # f_chrome_45__navigate_to_stackoverflow_tag: stackoverflow.com serves Cloudflare "Verify you are human" interstitial on container automated traffic. Same bot-detect class as yelp/google/marriott/aa/delta.
    ('chrome', 'f_chrome_45__navigate_to_stackoverflow_tag'),
    # f_chrome_74__search_american_flight: aa.com /booking/find-flights → Akamai Access Denied (edgesuite.net). Same Akamai family as marriott/costco/jetblue.
    ('chrome', 'f_chrome_74__search_american_flight'),
    # f_chrome_73__search_delta_flight: delta.com/flightsearch/search-results triggers Akamai Access Denied. Same family as aa.com/marriott/costco.
    ('chrome', 'f_chrome_73__search_delta_flight'),
    # ---- os ----
    # f_os_35__default_browser_firefox: Firefox not installed in the lite.osworld
    # Dockerfile (only google-chrome). Gold uses a stub firefox.desktop
    # sudo-tee workaround; agents reasonably reject this non-discoverable
    # shim path and report infeasible. Do not include until Firefox is actually
    # present in the image.
    ('os', 'f_os_35__default_browser_firefox'),
    # f_os_60__install_spotify: Spotify not in the lite.osworld Dockerfile; no
    # apt-installable path (no snap, no flatpak in image). Gold installs
    # `/usr/local/bin/spotify` shim — a non-discoverable workaround. Do not
    # include until Spotify is actually present in the image.
    ('os', 'f_os_60__install_spotify'),
}


# Hard tasks — documentation only, NOT filtered.
# Each entry is a template where the task is well-formed (oracle passes,
# eval is satisfiable) but the agent (GPT-5.4 reasoning_effort=low)
# reliably struggles. Family-cluster invariant has been applied: any cluster
# of variants sharing the same structural failure pattern should be treated as
# a source-side defect, not HARD. These entries WILL appear in train.synth.jsonl.
#
# Membership rules:
#   - Add when ≥2 sweeps fail without source-side bug evidence
#   - Remove when an upgraded agent unblocks the template, OR when
#     re-classification surfaces a hidden BUG
_HARD_TEMPLATE_IDS: set[tuple[str, str]] = {
    # ---- chrome ----
    # f_chrome_66__navigate_to_staged_wiki_yoga: Source stages 3 local HTML files (yoga, bicycle, volleyball) and opens yoga.html as the initial tab. Target is file:///home/user/D
    ('chrome', 'f_chrome_66__navigate_to_staged_wiki_yoga'),
    # f_chrome_67__navigate_to_staged_wiki_music: Same shape as f_chrome_66 — local HTML staging (beethoven, origami, paper-airplane), opens beethoven.html, asks to navigate to ori
    ('chrome', 'f_chrome_67__navigate_to_staged_wiki_music'),
    # f_chrome_73__search_delta_flight is omitted by _DROPPED_TEMPLATE_IDS: Akamai bot-block on /flightsearch/search-results.
    # f_chrome_24__search_rentalcars_zurich: cdjo (check_direct_json_object) eval on locationName/dropLocationName/filterCriteria_carCategory/filterCriteria_sortBy URL params. Historical failures were singleton agent-skill, not structural.
    ('chrome', 'f_chrome_24__search_rentalcars_zurich'),
    # f_chrome_11__set_profile_name: UI-state-not-persisted singleton — Chrome profile rename via Settings>Manage your account. Agent UI workflow doesn't flush new profile name to Local State / Preferences. Same shape family as vs_code kb_format_doc HARD.
    ('chrome', 'f_chrome_11__set_profile_name'),
    # f_chrome_29__navigate_to_dmv_eligibility: Singleton agent-skill issue around URL-path precision.
    ('chrome', 'f_chrome_29__navigate_to_dmv_eligibility'),
    # ---- gimp ----
    # gimp_config_tile_cache_2gb: GIMP Preferences > Environment dialog requires value entry plus unit dropdown switch (MiB→GiB). The instruction is well-formed (includes "Gibibyte" hint), but this remains an agent UI-skill ceiling.
    ('gimp', 'gimp_config_tile_cache_2gb'),
    # f_gimp_16__palette: F_GIMP_16 palette task — agent must open a JPG in GIMP, convert to Indexed mode (256 or 64 colors), export as PNG. Gold uses PIL `
    ('gimp', 'f_gimp_16__palette'),
    # f_gimp_29__fill_green_background: Task is structurally well-formed and gold should pass its own eval. Source: white canvas with pure-black circle. Gold rewrites eve
    ('gimp', 'f_gimp_29__fill_green_background'),
    # f_gimp_30__fill_green_background: Same shape as f_gimp_29 but with a black SQUARE on white. Distinct File (F_GIMP_30 != F_GIMP_29) with distinct basename `white_bac
    ('gimp', 'f_gimp_30__fill_green_background'),
    # f_gimp_2__crop_center: Image > Canvas Size or rectangle-select + crop with precise center. Multi-step GIMP UI agent-skill ceiling.
    ('gimp', 'f_gimp_2__crop_center'),
    # f_gimp_4__brightness_decrease: Colors > Brightness-Contrast dialog + slider. Same skill family as gimp_config_tile_cache_2gb (value-entry + dropdown). Singleton agent-skill.
    ('gimp', 'f_gimp_4__brightness_decrease'),
    # f_gimp_21__file_exists_rename: validation. terminated/n=13. Singleton gimp file-management op. Agent-skill, no source bug.
    ('gimp', 'f_gimp_21__file_exists_rename'),
    # f_gimp_15__contrast_increase: validation. terminated 0001/n=21 + 0002/n=30 (BOTH seeds fail). Colors > Brightness-Contrast dialog + slider — same skill as already-HARD f_gimp_4__brightness_decrease. ≥2 seed confirmation.
    ('gimp', 'f_gimp_15__contrast_increase'),
    # f_gimp_4__grayscale: Image > Mode > Grayscale + export. New skill but matches gimp color-mode cluster. Singleton agent-skill.
    ('gimp', 'f_gimp_4__grayscale'),
    # gimp_config_theme_dark: Edit > Preferences > Theme dialog — same UI cluster as gimp_config_tile_cache_2gb. Sibling singleton.
    ('gimp', 'gimp_config_theme_dark'),
    # ---- libreoffice_calc ----
    # f_calc_11__sort_by_userid: Sort by UserId (string) — _gold_sort with stable key (numbers-first, then strings); after LO Calc Sort agent should match.
    ('libreoffice_calc', 'f_calc_11__sort_by_userid'),
    # f_calc_14__groupby_educ_count: Agent-skill: agent created "By Education" sheet with all 5 levels showing Count=8 — uniform value indicates wrong COUNTIF aggregation. Also exhibits the "phantom Save-As dialog" mental-model gap (see f_calc_26 for full diagnosis): agent assumed Ctrl+S would pop a Save-As dialog (Windows/web prior), but lite.osworld LO 7.3 has the ODF/format warning disabled so Ctrl+S saves in-place silently. Agent then typed the filename expecting an open dialog → polluted the main spreadsheet window → later Ctrl+Shift+S really did open Save-As but the saved-out file was already corrupted. Cluster with f_calc_17 + f_calc_26 + f_calc_95.
    ('libreoffice_calc', 'f_calc_14__groupby_educ_count'),
    # f_calc_17__sort_by_tardy_min: Agent often reaches the correct-looking alphabetic sort by Student (Aiden→Beatriz→Caleb…), but eval `sheet_data sheet_idx0=0 sheet_idx1="EI0"` returns 0. Likely the sort included the header row OR the result was saved to a wrong path via the same phantom-Save-As-dialog mental-model gap (see f_calc_26). Sibling sort tasks (f_calc_11/12/18/20) all already HARD — same agent-skill family.
    ('libreoffice_calc', 'f_calc_17__sort_by_tardy_min'),
    # f_calc_26__summary_rollup: Agent has a mental-model gap, NOT a dialog-clearing skill gap. In lite.osworld LO 7.3, Ctrl+S on an existing .xlsx saves in-place silently (the image disables the "Warn when not saving in ODF" option, so no Use-ODF / Keep-Current-Format dialog ever appears). Agent's training prior expects a Save-As-style dialog after Ctrl+S → it presses Ctrl+S (which silently saves), then types "quarterly-rollup.xlsx" + Enter assuming a dialog is open → the type goes into the main Calc window (Name Box / cell), polluting the spreadsheet. A later Ctrl+Shift+S opens Save-As, but the working file state already diverged. Eval reads original `/Desktop/quarterly-rollup.xlsx` which never got the new sheet → 0. Cluster: f_calc_14 + f_calc_17 + f_calc_26 + f_calc_95.
    ('libreoffice_calc', 'f_calc_26__summary_rollup'),
    # f_calc_11__string_clean_email: string_clean op=lower or strip. _RULE_SHEET_DATA only (no style). Agent =LOWER / =TRIM matches values. One seed 0.0 (agent gave up
    ('libreoffice_calc', 'f_calc_11__string_clean_email'),
    # f_calc_12__sort_by_sku: Sort by SKU (string) variant of _gold_sort. validation oracle-fail; no validation sample.
    ('libreoffice_calc', 'f_calc_12__sort_by_sku'),
    # f_calc_15__filter_by_origin: Filter cluster: new sheet + copy header + matching rows in source order. Other filter_* tasks pass for some seeds — not a uniform
    ('libreoffice_calc', 'f_calc_15__filter_by_origin'),
    # f_calc_17__filter_by_status: Filter cluster. One seed 0.0 only.
    ('libreoffice_calc', 'f_calc_17__filter_by_status'),
    # f_calc_18__sort_by_onhand_asc: Sort numeric column ascending. _gold_sort + sheet_data eval. validation oracle-fail; no validation sample. Standard sort task.
    ('libreoffice_calc', 'f_calc_18__sort_by_onhand_asc'),
    # f_calc_26__sheet_rename_q1_summary: Rename Q1→Quarter1, add 'Quarter1 (Backup)' copy. Eval = sheet_name only. Doable via LO Sheet > Rename + Move/Copy. validation oracl
    ('libreoffice_calc', 'f_calc_26__sheet_rename_q1_summary'),
    # f_calc_31__derived_log_price: Derived-column task. Gold writes computed values via _gold_derived_col + number_format style sub-rule. Both seeds 0.0 with agent t
    ('libreoffice_calc', 'f_calc_31__derived_log_price'),
    # f_calc_32__derived_basis_points: Derived-column task — see f_calc_31__derived_log_price.
    ('libreoffice_calc', 'f_calc_32__derived_basis_points'),
    # f_calc_33__derived_thousand_units: Derived-column task.
    ('libreoffice_calc', 'f_calc_33__derived_thousand_units'),
    # f_calc_38__derived_days_since_open: Derived-column task — baseline date is the FIRST data row's date. Instruction says 'from the opening-balance row' — agent must inf
    ('libreoffice_calc', 'f_calc_38__derived_days_since_open'),
    # f_calc_3__filter_by_region: Filter cluster.
    ('libreoffice_calc', 'f_calc_3__filter_by_region'),
    # f_calc_43__derived_tempf: Derived-column task.
    ('libreoffice_calc', 'f_calc_43__derived_tempf'),
    # f_calc_44__derived_speed: Derived-column task.
    ('libreoffice_calc', 'f_calc_44__derived_speed'),
    # f_calc_45__derived_days_active: Derived-column + date arithmetic with hardcoded cutoff. Instruction explicitly states cutoff.
    ('libreoffice_calc', 'f_calc_45__derived_days_active'),
    # f_calc_46__derived_line_cost: Derived-column task.
    ('libreoffice_calc', 'f_calc_46__derived_line_cost'),
    # f_calc_47__derived_average: Derived-column = (Math+Science+English)/3. Agent =AVERAGE(C2:E2) matches value-wise.
    ('libreoffice_calc', 'f_calc_47__derived_average'),
    # f_calc_4__filter_to_new_sheet_by_genre: Filter cluster.
    ('libreoffice_calc', 'f_calc_4__filter_to_new_sheet_by_genre'),
    # f_calc_51__derived_temp_range: Derived-column task.
    ('libreoffice_calc', 'f_calc_51__derived_temp_range'),
    # f_calc_53__derived_amount_with_match: Derived-column task.
    ('libreoffice_calc', 'f_calc_53__derived_amount_with_match'),
    # f_calc_54__string_clean_author_upper: string_clean op=upper. Agent =UPPER(A2) matches.
    ('libreoffice_calc', 'f_calc_54__string_clean_author_upper'),
    # f_calc_56__derived_price_per_sqft: Derived-column task. validation: subagent proposed dropping style/number_format rule (per F_CALC_36 precedent) but FALSIFIED by host-side replay of f_calc_67_0002 — agent's saved xlsx is missing the derived column entirely (4 cols vs 5 expected), not a number_format-string mismatch. Genuine agent-skill failure on multi-step formula entry + column-add UI workflow.
    ('libreoffice_calc', 'f_calc_56__derived_price_per_sqft'),
    # f_calc_67__derived_annual_pledge: Derived-column task. validation falsified-fix replay-verify: agent never adds the AnnualValue column in 9 turns. Same agent-skill cluster as f_calc_56.
    ('libreoffice_calc', 'f_calc_67__derived_annual_pledge'),
    # f_calc_75__derived_per_device_fee: Derived-column task. validation falsified-fix cluster.
    ('libreoffice_calc', 'f_calc_75__derived_per_device_fee'),
    # f_calc_80__derived_late_fee: Derived-column task. validation falsified-fix cluster.
    ('libreoffice_calc', 'f_calc_80__derived_late_fee'),
    # f_calc_87__derived_cost_per_kw: Derived-column task. validation falsified-fix cluster.
    ('libreoffice_calc', 'f_calc_87__derived_cost_per_kw'),
    # f_calc_60__derived_weight_per_item: Derived-column task.
    ('libreoffice_calc', 'f_calc_60__derived_weight_per_item'),
    # f_calc_61__derived_fare_per_km: Derived-column task.
    ('libreoffice_calc', 'f_calc_61__derived_fare_per_km'),
    # f_calc_62__derived_calories_per_min: Derived-column task.
    ('libreoffice_calc', 'f_calc_62__derived_calories_per_min'),
    # f_calc_63__derived_annual_rent: Derived-column task.
    ('libreoffice_calc', 'f_calc_63__derived_annual_rent'),
    # f_calc_64__string_clean_title_lower: string_clean op=lower.
    ('libreoffice_calc', 'f_calc_64__string_clean_title_lower'),
    # f_calc_66__derived_runtime_hours: Derived-column task.
    ('libreoffice_calc', 'f_calc_66__derived_runtime_hours'),
    # f_calc_69__derived_nightly_rate: Derived-column task.
    ('libreoffice_calc', 'f_calc_69__derived_nightly_rate'),
    # f_calc_69__string_clean_guest_proper: string_clean op=proper_strip — gold writes ' '.join(v.split()).title() as VALUE. Agent =PROPER(TRIM(A2)) matches. validation oracle replay.
    ('libreoffice_calc', 'f_calc_69__string_clean_guest_proper'),
    # f_calc_6__derived_margin_col: Derived-column = Profit/Sales with format 0.0%. Both seeds 0.0; agent took thorough actions (turns 10-13 of 0001 added column + dr
    ('libreoffice_calc', 'f_calc_6__derived_margin_col'),
    # f_calc_6__filter_by_channel: Filter cluster.
    ('libreoffice_calc', 'f_calc_6__filter_by_channel'),
    # f_calc_70__derived_downloads_k: Derived-column task.
    ('libreoffice_calc', 'f_calc_70__derived_downloads_k'),
    # f_calc_71__string_clean_country_upper: string_clean op=upper.
    ('libreoffice_calc', 'f_calc_71__string_clean_country_upper'),
    # f_calc_73__string_clean_species_lower: string_clean op=lower — gold v.lower() values. Agent =LOWER(A2) matches. No validation sample.
    ('libreoffice_calc', 'f_calc_73__string_clean_species_lower'),
    # f_calc_81__derived_credit_hours: Derived-column task.
    ('libreoffice_calc', 'f_calc_81__derived_credit_hours'),
    # f_calc_85__derived_ratio: Derived-column task.
    ('libreoffice_calc', 'f_calc_85__derived_ratio'),
    # f_calc_88__derived_age: Derived-column task.
    ('libreoffice_calc', 'f_calc_88__derived_age'),
    # f_calc_89__derived_line_total: Derived-column task.
    ('libreoffice_calc', 'f_calc_89__derived_line_total'),
    # f_calc_92__column_reorder: Column reorder via cut + paste. sheet_data eval.
    ('libreoffice_calc', 'f_calc_92__column_reorder'),
    # f_calc_95__pad_zeros_seven_digits: Pad with leading zeros to 7 digits. Gold writes TEXT strings. Agent =TEXT(A2,'0000000') matches.
    ('libreoffice_calc', 'f_calc_95__pad_zeros_seven_digits'),
    # f_calc_29__color_recession_band — vacuous predicate verified: P[0] `>8.0` matches 0/36 (max=7.9), P[1] `>=10.0` matches 0/36 + `<3.5` matches 1/36 (fragile). See _BUG_TEMPLATE_IDS.
    # f_calc_38__derived_abs_or_direction: validation. Derived-column task. Same skill cluster as the 25+ derived_col HARD entries (multi-step formula + column-add UI).
    ('libreoffice_calc', 'f_calc_38__derived_abs_or_direction'),
    # f_calc_49__groupby_make_count: validation. groupby cluster sibling of already-HARD f_calc_14__groupby_educ_count. Same phantom-Save-As mental-model + COUNTIF aggregation agent-skill.
    ('libreoffice_calc', 'f_calc_49__groupby_make_count'),
    # f_calc_59__string_clean_name_upper: validation. string_clean cluster — agent-skill family with f_calc_11/_54/_64/_71/_73__string_clean_* (=UPPER/LOWER/PROPER + Ctrl+S save flow). Eval is sheet_data only; gold writes lowercased/uppercased values. Mismatched save path or formula not committed.
    ('libreoffice_calc', 'f_calc_59__string_clean_name_upper'),
    # f_calc_84__derived_miles: validation. Derived-column task (km→miles unit conversion). Same skill cluster as f_calc_38, derived_col HARDs.
    ('libreoffice_calc', 'f_calc_84__derived_miles'),
    # f_calc_72__derived_annual_fee: validation. derived_col cluster (25+ HARDs).
    ('libreoffice_calc', 'f_calc_72__derived_annual_fee'),
    # f_calc_39__groupby_status_count: validation. groupby cluster — sibling of f_calc_14 / f_calc_49 HARDs. Phantom-Save-As + COUNTIF aggregation agent-skill.
    ('libreoffice_calc', 'f_calc_39__groupby_status_count'),
    # f_calc_7__sort_by_revenue_desc: validation. sort cluster — siblings f_calc_11/_12/_17/_18 HARD.
    ('libreoffice_calc', 'f_calc_7__sort_by_revenue_desc'),
    # f_calc_45__filter_active_subs: validation. truncated/n=30. filter cluster — siblings f_calc_15/_17__filter_*/_3/_4/_6 HARD. Trigger N (turn-ceiling).
    ('libreoffice_calc', 'f_calc_45__filter_active_subs'),
    # f_calc_30__color_top_economies — Param[1] vacuous + Param[0] near-degenerate.
    # f_calc_1__copy_col_to_new_sheet: validation. terminated/n=28. Column ops cluster — sibling f_calc_92__column_reorder HARD. Multi-step UI: select column → cut → new sheet → paste → save.
    ('libreoffice_calc', 'f_calc_1__copy_col_to_new_sheet'),
    # f_calc_100__chart_revenue_expenses — joins f_calc_91 chart-eval-gap cluster: eval rules `[sheet_name, sheet_data sheet_idx0=0]` never inspect chart XML. Pass without creating a chart. See _BUG_TEMPLATE_IDS.
    # f_calc_96__date_duration_days: validation. terminated/n=15. Date-duration derived col — agent typed header as literal 'DaysActive=D2-C2' (formula text in header cell). Joins 25+ derived_col HARD cluster.
    ('libreoffice_calc', 'f_calc_96__date_duration_days'),
    # f_calc_12__string_clean_proper: validation. terminated/n=6. string_clean cluster — siblings f_calc_11/_54/_59/_64/_71/_73 HARD.
    ('libreoffice_calc', 'f_calc_12__string_clean_proper'),
    # f_calc_36__derived_income_thousands: validation. terminated/n=24. derived_col cluster (25+ HARDs).
    ('libreoffice_calc', 'f_calc_36__derived_income_thousands'),
    # f_calc_40__derived_price_with_tax: validation. terminated/n=18. derived_col cluster.
    ('libreoffice_calc', 'f_calc_40__derived_price_with_tax'),
    # f_calc_52__derived_revenue: validation. terminated/n=18. derived_col cluster.
    ('libreoffice_calc', 'f_calc_52__derived_revenue'),
    # f_calc_78__derived_pace: validation. terminated/n=14. derived_col cluster.
    ('libreoffice_calc', 'f_calc_78__derived_pace'),
    # f_calc_81__string_clean_title_lower: validation. terminated/n=14. string_clean cluster.
    ('libreoffice_calc', 'f_calc_81__string_clean_title_lower'),
    # f_calc_2__color_by_score: validation global validation. truncated/n=30. color_* cluster sibling — f_calc_29 / f_calc_30 HARD. Per-row conditional fill via Custom Color dialog. Trigger N.
    ('libreoffice_calc', 'f_calc_2__color_by_score'),
    # f_calc_22__color_top_share: validation 2026-05-12. Verified: validation predicate fix HOLDS (1 match — Nano 0.04 < 0.05). New failure mode is **Trigger F (LibreOffice GUI ceiling)**: agent stalls on Custom-Color hex picker (Format>Cells>Background>Custom Color hex entry). Replay 0001 turn_25 shows dialog open with no fill committed. Skill ceiling, NOT a generator bug.
    ('libreoffice_calc', 'f_calc_22__color_top_share'),
    # ---- libreoffice_impress ----
    # d_imp_05__compound_bold_and_bg: validation cluster B+D. Compound bold+bg task: agent applied background color correctly (#DCF0FF visible) but title text bold not applied or only partial selection. Agent never issued Ctrl+S (postconfig saves at exit). Agent-skill: compound styling + Ctrl+B selection management.
    ('libreoffice_impress', 'd_imp_05__compound_bold_and_bg'),
    # d_imp_11__title_color_1plus3: validation cluster B+D. Agent corrupted title text by typing color codes ("255", "140", "0") into title text box during RGB picker workflow; subsequent Delete deleted title content. Agent reported infeasible at turn_23 after self-corruption. RGB picker UI workflow capability gap.
    ('libreoffice_impress', 'd_imp_11__title_color_1plus3'),
    # d_imp_22__title_bold_to4: validation cluster B+D. Agent self-toggled bold off (Ctrl+A+Ctrl+B twice cancels itself), then only selected "Year" for final bold; self-terminated early. Title ends with partial bold. Standard title-bold agent-skill HARD.
    ('libreoffice_impress', 'd_imp_22__title_bold_to4'),
    # d_imp_33__swap_slides_strip5: validation cluster B+D. Swap_slides via drag in Slides panel is awkward; agent did multiple non-deterministic drags; final order doesn't match 1↔4 swap. Same family as already-HARD d_imp_40__swap_slides_p7 + d_imp_46__swap_slides_h7.
    ('libreoffice_impress', 'd_imp_33__swap_slides_strip5'),
    # d_imp_01__body_bold: Gold _gold_set_body_bold(0) correctly mutates only shape[1] (body textbox) on slide 0. Instruction 'Bold the body text on slide 1'
    ('libreoffice_impress', 'd_imp_01__body_bold'),
    # d_imp_01__title_color: Gold _gold_set_title_font_color(0,(255,0,0)) is title-only (shape[0] paragraphs[0].runs). Instruction names slide+title unambiguou
    ('libreoffice_impress', 'd_imp_01__title_color'),
    # d_imp_02__title_size: Title-only gold (_gold_set_title_font_size at synth/libreoffice_impress.py:352) matches title-only instruction. Eval examine_font_
    ('libreoffice_impress', 'd_imp_02__title_size'),
    # d_imp_03__body_color: Gold _gold_set_body_font_color(2,(0,128,0)) is body-only. Instruction 'body text on slide 3' unambiguous. Eval examine_color_rgb s
    ('libreoffice_impress', 'd_imp_03__body_color'),
    # d_imp_04__title_font_name: Gold _gold_set_title_font_name(0,'DejaVu Serif') (title-only, line 392). Source uses 'Liberation Sans' base font specifically beca
    ('libreoffice_impress', 'd_imp_04__title_font_name'),
    # d_imp_06__caption_font_name: Body-only gold _gold_set_body_font_name(0,'DejaVu Serif') matches 'caption' instruction (caption is shape[1] body). Eval examine_f
    ('libreoffice_impress', 'd_imp_06__caption_font_name'),
    # d_imp_06__title_color_on_photo: Gold _gold_set_title_font_color(0,(200,0,0)) title-only, correct. Hero-photo deck — title shape[0] sits above photo, agent must se
    ('libreoffice_impress', 'd_imp_06__title_color_on_photo'),
    # d_imp_07__title_size_banner: Title-only gold _gold_set_title_font_size(1,40) matches 'title on slide 2' instruction. Hero-photo banner deck — title shape[0] is
    ('libreoffice_impress', 'd_imp_07__title_size_banner'),
    # d_imp_09__title_color_2x2: Gold title-only color helper used correctly. Gallery 2x2 photo deck — title sits above 4-photo grid. Agent must select title shape
    ('libreoffice_impress', 'd_imp_09__title_color_2x2'),
    # d_imp_09__title_size_2x2: Title-only gold _gold_set_title_font_size(0,32). Same gallery deck as title_color_2x2. Instruction names slide+title clearly. Eval
    ('libreoffice_impress', 'd_imp_09__title_size_2x2'),
    # d_imp_13__body_bold_footer: Gold _gold_set_body_bold(1) is body-only (correctly scoped). Footer deck includes footer/pagenum shapes but body bold is the body-
    ('libreoffice_impress', 'd_imp_13__body_bold_footer'),
    # d_imp_14__bg_color_pagenum: Gold _gold_set_background(idx,rgb) is correct slide-wide mutator matching 'Give slide N a cream background' instruction. Eval exam
    ('libreoffice_impress', 'd_imp_14__bg_color_pagenum'),
    # d_imp_14__title_size_pagenum: Title-only gold _gold_set_title_font_size(0,36). Pagenum_5s deck with page-number footer placeholders — title is shape[0]. Instruc
    ('libreoffice_impress', 'd_imp_14__title_size_pagenum'),
    # d_imp_17__title_color_long_notes: Title-only gold _gold_set_title_font_color(2,(80,80,80)). Long-notes deck — title-only mutation correctly scoped. validation 0001 erro
    ('libreoffice_impress', 'd_imp_17__title_color_long_notes'),
    # d_imp_18__body_bold_portrait: Body-only gold _gold_set_body_bold(0). Portrait-orientation deck. Eval examine_font_bold satisfiable. validation 0001 SAID 'Done.' wit
    ('libreoffice_impress', 'd_imp_18__body_bold_portrait'),
    # d_imp_18__title_color_portrait: Title-only gold _gold_set_title_font_color correctly applied. Portrait deck — title shape[0]. validation 0001 SAID 'Done.' with 0 fina
    ('libreoffice_impress', 'd_imp_18__title_color_portrait'),
    # d_imp_21__title_color_long: Title-only gold (line 362) applied to long-text deck. Instruction unambiguous (slide 2, cranberry RGB 170,0,60). Eval examine_colo
    ('libreoffice_impress', 'd_imp_21__title_color_long'),
    # d_imp_25__body_color_subt6: Body-only gold _gold_set_body_font_color(2,(80,80,80)) on title_subtitle deck. The 'subtitle' IS the body shape[1] in this layout
    ('libreoffice_impress', 'd_imp_25__body_color_subt6'),
    # d_imp_26__title_size_h3nc: Title-only gold _gold_set_title_font_size(0,32) on hero-photo no-caption deck. Eval satisfiable. Same UI font-size skill pattern.
    ('libreoffice_impress', 'd_imp_26__title_size_h3nc'),
    # d_imp_30__title_color_g2x2_4: Title-only gold helper applied to gallery 2x2 4-slide deck. Same title_color cluster agent-skill pattern.
    ('libreoffice_impress', 'd_imp_30__title_color_g2x2_4'),
    # d_imp_30__title_color_g2x2_4_b: Two params: one uses _gold_set_title_font_color(0,(220,0,0)) (title-only), other uses _gold_set_body_font_color(2,(0,100,50)) (bod
    ('libreoffice_impress', 'd_imp_30__title_color_g2x2_4_b'),
    # d_imp_31__slide_bg_g3x3_4: Slide-wide background gold _gold_set_background(0,(245,235,220)). Gallery 3x3 4-slide deck. Eval examine_background_color satisfia
    ('libreoffice_impress', 'd_imp_31__slide_bg_g3x3_4'),
    # d_imp_40__swap_slides_p7: _gold_swap_slides reorders sldIdLst at XML level — equivalent to user drag in Slides panel. Eval examine_text on reordered slides
    ('libreoffice_impress', 'd_imp_40__swap_slides_p7'),
    # d_imp_41__body_color_serif: Body-only gold _gold_set_body_font_color(1,(60,60,60)) on serif deck. Instruction 'body text on slide 2' unambiguous, helper corre
    ('libreoffice_impress', 'd_imp_41__body_color_serif'),
    # d_imp_46__swap_slides_h7: Same as swap_slides_p7. 7-slide hero deck. Eval and gold correct (XML sldIdLst reorder). Other swap variants succeed so not a setu
    ('libreoffice_impress', 'd_imp_46__swap_slides_h7'),
    # d_imp_46__title_color_h7: Title-only gold _gold_set_title_font_color(2,(200,100,0)) on 7-slide hero deck. 0002 trajectory turn_12: agent on correct slide 3
    ('libreoffice_impress', 'd_imp_46__title_color_h7'),
    # d_imp_56__stretch_to_full_slide: Eval check_image_stretch_and_center compares modified vs snapshot of original; requires agent to resize+center image to within Inc
    ('libreoffice_impress', 'd_imp_56__stretch_to_full_slide'),
    # d_imp_21__body_align_long: validation. body_align styling cluster — sibling of body_color/body_bold/body_font HARD cluster (d_imp_01/_03/_06/_13/_18/_25/_41). Compound-styling UI skill ceiling.
    ('libreoffice_impress', 'd_imp_21__body_align_long'),
    # d_imp_22__title_size_to4: validation. title_size cluster — sibling of d_imp_02/_07/_09/_14/_26__title_size_* HARDs. Same Format>Character>Size UI agent-skill.
    ('libreoffice_impress', 'd_imp_22__title_size_to4'),
    # d_imp_24__title_color_subt3: validation. title_color cluster — sibling of d_imp_01/_06/_09/_17/_18/_21/_30/_46__title_color_* HARDs. RGB picker workflow agent-skill.
    ('libreoffice_impress', 'd_imp_24__title_color_subt3'),
    # d_imp_25__title_size_subt6: validation. title_size cluster, same as d_imp_22 entry above.
    ('libreoffice_impress', 'd_imp_25__title_size_subt6'),
    # d_imp_41__insert_table_features_5x2: validation. insert_table compound UI — Insert>Table dialog with 5×2 dims + slide placement + content population. NEW skill (insert_table not previously in HARD). 2-variant family with d_imp_51__insert_table_4x3_slide2.
    ('libreoffice_impress', 'd_imp_41__insert_table_features_5x2'),
    # d_imp_51__insert_table_4x3_slide2: validation. Same insert_table compound UI skill as d_imp_41. 2-variant family.
    ('libreoffice_impress', 'd_imp_51__insert_table_4x3_slide2'),
    # d_imp_58__add_image_position — strict examine_image_size without tolerance kwarg. See _BUG_TEMPLATE_IDS.
    # d_imp_08__compound_underline_and_align: validation. terminated/n=7. Compound styling cluster — siblings d_imp_05/_22/_33 HARD. Compound underline+align UI multi-step.
    ('libreoffice_impress', 'd_imp_08__compound_underline_and_align'),
    # d_imp_13__title_color_footer: validation. truncated/n=30. title_color cluster (8+ HARDs) — turn-ceiling N variant.
    ('libreoffice_impress', 'd_imp_13__title_color_footer'),
    # d_imp_61__compound_color_and_align: validation. truncated/n=30. Compound styling N. Sibling of d_imp_08 / d_imp_05.
    ('libreoffice_impress', 'd_imp_61__compound_color_and_align'),
    # ---- libreoffice_writer ----
    # f_writer_10__italic_para: F_WRITER_10 guide DejaVu, _src_genre(5). No title. idx=0='first', idx=3='fourth'. Correct mapping. Verified rollout: agent dragged
    ('libreoffice_writer', 'f_writer_10__italic_para'),
    # f_writer_11__find_replace: Find/Replace dialog is multi-step (Ctrl+H → focus Find → type → Tab → type → Replace All → close). Verified f_writer_11__find_repl
    ('libreoffice_writer', 'f_writer_11__find_replace'),
    # f_writer_13__page_break: mixed outcomes across variants — task is satisfiable, agent skill variance.
    ('libreoffice_writer', 'f_writer_13__page_break'),
    # f_writer_16__doc_spacing: Doc-wide line spacing: agent must Ctrl+A from BODY (not heading) then apply spacing.
    ('libreoffice_writer', 'f_writer_16__doc_spacing'),
    # f_writer_17__find_replace — `_0001` (Param[1]) is case-mismatch vacuous: instruction `'managers'` vs source para-3 `'Managers'`, case-sensitive replace → 0 substitutions → gold == source. (`_0002` = Param[0] uses `'training'` with 2 valid lowercase matches and is fine.) See _BUG_TEMPLATE_IDS.
    # f_writer_19__highlight_p0: Highlight via UI font-color dropdown is multi-step (drag-select para → click highlight). Even a visible yellow highlight on the opening paragraph can false-fail under compare_docx_strict + examine_highlight=True because the agent path gets a LibreOffice Ctrl+S round-trip while the gold is python-docx-only. Same shape as f_writer_4__highlight_para. HARD agent-skill / eval-strict-vs-UI.
    ('libreoffice_writer', 'f_writer_19__highlight_p0'),
    # f_writer_27__size20_p0: Size_para op (20pt opener) on image-host / structured / Gutenberg files. compare_docx_strict + examine_font_size demands every cha
    ('libreoffice_writer', 'f_writer_27__size20_p0'),
    # f_writer_29__italic_para: F_WRITER_29 image-host (Andromeda) — no title. idx=0='first', idx=2='third (the caption text)'. Instruction param idx=2 even uses
    ('libreoffice_writer', 'f_writer_29__italic_para'),
    # f_writer_2__doc_spacing: Doc-wide line spacing: agent must Ctrl+A from BODY (not heading) then apply spacing.
    ('libreoffice_writer', 'f_writer_2__doc_spacing'),
    # f_writer_31__size16_p1: Size_para op (16pt second para) on image-host / structured / Gutenberg files. compare_docx_strict + examine_font_size demands ever
    ('libreoffice_writer', 'f_writer_31__size16_p1'),
    # f_writer_32__find_replace: Find/Replace dialog is multi-step (Ctrl+H → focus Find → type → Tab → type → Replace All → close). Verified f_writer_11__find_repl
    ('libreoffice_writer', 'f_writer_32__find_replace'),
    # f_writer_34__doc_spacing: Doc-wide line spacing: agent must Ctrl+A from BODY (not heading) then apply spacing. Verified f_writer_34__doc_spacing_0001/turn_0
    ('libreoffice_writer', 'f_writer_34__doc_spacing'),
    # f_writer_3__find_replace: Find/Replace dialog is multi-step (Ctrl+H → focus Find → type → Tab → type → Replace All → close). Verified f_writer_11__find_repl
    ('libreoffice_writer', 'f_writer_3__find_replace'),
    # f_writer_40__italic_para: F_WRITER_40 has title at idx 0; body paras at idx 1..5. Param idx=2 → 'second paragraph of the essay' (where body[0]=idx1 is 'firs
    ('libreoffice_writer', 'f_writer_40__italic_para'),
    # f_writer_40__size14_body: Size_para op (14pt third body para) on image-host / structured / Gutenberg files. compare_docx_strict + examine_font_size demands
    ('libreoffice_writer', 'f_writer_40__size14_body'),
    # f_writer_43__doc_spacing: Doc-wide line spacing: agent must Ctrl+A from BODY (not heading) then apply spacing. Verified f_writer_34__doc_spacing_0001/turn_0
    ('libreoffice_writer', 'f_writer_43__doc_spacing'),
    # f_writer_48__page_break: Para_idx=4 ('Pre-Work Inspection' section). Single param so no variance data, but fails.
    ('libreoffice_writer', 'f_writer_48__page_break'),
    # f_writer_49__delete_rows_filter: Multi-step table-row deletion via UI is high-step-count; agent must visually identify P-tier rows. Likely fails on save+match. No
    ('libreoffice_writer', 'f_writer_49__delete_rows_filter'),
    # f_writer_4__highlight_para: F_WRITER_4 recipe short, no title. idx=0='first', idx=2='third'. Correct mapping. Highlight via UI font-color dropdown is multi-st
    ('libreoffice_writer', 'f_writer_4__highlight_para'),
    # f_writer_55__doc_spacing: Doc-wide line spacing: agent must Ctrl+A from BODY (not heading) then apply spacing. Verified f_writer_34__doc_spacing_0001/turn_0
    ('libreoffice_writer', 'f_writer_55__doc_spacing'),
    # f_writer_59__doc_spacing: Doc-wide line spacing: agent must Ctrl+A from BODY (not heading) then apply spacing. Verified f_writer_34__doc_spacing_0001/turn_0
    ('libreoffice_writer', 'f_writer_59__doc_spacing'),
    # f_writer_60__size18_opener: Size_para op (18pt opener Frankenstein) on image-host / structured / Gutenberg files. compare_docx_strict + examine_font_size dema
    ('libreoffice_writer', 'f_writer_60__size18_opener'),
    # f_writer_69__doc_font — eval-strict-vs-UI (compare_font_names rFonts strip).
    # f_writer_6__page_numbers_footer: Add page numbers via Insert > Page Number. Multi-step menu navigation. Standard HARD.
    ('libreoffice_writer', 'f_writer_6__page_numbers_footer'),
    # f_writer_71__page_numbers_footer: Add page numbers via Insert > Page Number. Multi-step menu navigation. Standard HARD.
    ('libreoffice_writer', 'f_writer_71__page_numbers_footer'),
    # f_writer_72__mixed_align_todo: Split-paragraph-then-align task: instruction asks agent to split first paragraph after 4 words (left-aligned first 4, right-aligned rest). Multi-step (place cursor, Enter, drag-select, apply alignments) and fragile under compare_docx_strict + paragraph-text matching. Agent skill / capability ceiling. HARD.
    ('libreoffice_writer', 'f_writer_72__mixed_align_todo'),
    # f_writer_73__doc_spacing: Doc-wide line spacing: agent must Ctrl+A from BODY (not heading) then apply spacing.
    ('libreoffice_writer', 'f_writer_73__doc_spacing'),
    # f_writer_78__italic_para: F_WRITER_78 image-host (portrait headshot) — no title. idx=2='third paragraph (the caption text)'. Instruction explicitly names an
    ('libreoffice_writer', 'f_writer_78__italic_para'),
    # f_writer_7__doc_font_arial — same eval-strict-vs-UI as f_writer_69.
    # f_writer_82__doc_spacing: Doc-wide line spacing: agent must Ctrl+A from BODY (not heading) then apply spacing. Verified f_writer_34__doc_spacing_0001/turn_0
    ('libreoffice_writer', 'f_writer_82__doc_spacing'),
    # f_writer_85__doc_spacing: Doc-wide line spacing: agent must Ctrl+A from BODY (not heading) then apply spacing. Verified f_writer_34__doc_spacing_0001/turn_0
    ('libreoffice_writer', 'f_writer_85__doc_spacing'),
    # f_writer_86__bold_body: F_WRITER_86 short memo: title at idx 0, single body para at idx 1. Param idx=1 → 'the body paragraph' (unambiguous since only one
    ('libreoffice_writer', 'f_writer_86__bold_body'),
    # f_writer_87__doc_spacing: Doc-wide line spacing: agent must Ctrl+A from BODY (not heading) then apply spacing. Verified f_writer_34__doc_spacing_0001/turn_0
    ('libreoffice_writer', 'f_writer_87__doc_spacing'),
    # f_writer_89__center_pizza_heading: Center-align paragraph. F_WRITER_89 (mixed pizza essay) — center first para. Standard UI op (Ctrl+E). Cluster of 2 — HARD agent sk
    ('libreoffice_writer', 'f_writer_89__center_pizza_heading'),
    # f_writer_8__italic_para: F_WRITER_8 essay 1.5-spaced, _src_genre(1). No title. idx=2='third', idx=5='sixth'. Correct mapping.
    ('libreoffice_writer', 'f_writer_8__italic_para'),
    # f_writer_94__center_heading: Center-align paragraph. F_WRITER_94 (chemistry notes) — center first/third para. Standard UI op (Ctrl+E). Cluster of 2 — HARD agen
    ('libreoffice_writer', 'f_writer_94__center_heading'),
    # f_writer_94__subscript_chem: Subscript every digit inside 4 chemical formulas in one paragraph — 4×1 or 4×2 digit selections, each requires precise drag-select
    ('libreoffice_writer', 'f_writer_94__subscript_chem'),
    # f_writer_33__append_para: Hypothesis 'asymmetric LO normalize on _gold_append_paragraph' was falsified by replay-verify 2026-05-11: oracle replay 1.0 with new symmetric LO-roundtrip, but validation trajectory FAIL — agent corrupted Heading 1 paragraph mid-doc (15 paras vs expected 12). Demoted from BUG → HARD. Genuine agent-execution failure; gold/eval contract is sound.
    ('libreoffice_writer', 'f_writer_33__append_para'),
    # f_writer_37__append_para: Hypothesis 'asymmetric LO normalize on _gold_append_paragraph' was falsified by replay-verify 2026-05-11: oracle replay 1.0 with new symmetric LO-roundtrip, but validation trajectory FAIL — agent corrupted Heading 1 paragraph mid-doc (15 paras vs expected 12). Demoted from BUG → HARD. Genuine agent-execution failure; gold/eval contract is sound.
    ('libreoffice_writer', 'f_writer_37__append_para'),
    # f_writer_39__append_para: Hypothesis 'asymmetric LO normalize on _gold_append_paragraph' was falsified by replay-verify 2026-05-11: oracle replay 1.0 with new symmetric LO-roundtrip, but validation trajectory FAIL — agent corrupted Heading 1 paragraph mid-doc (15 paras vs expected 12). Demoted from BUG → HARD. Genuine agent-execution failure; gold/eval contract is sound.
    ('libreoffice_writer', 'f_writer_39__append_para'),
    # f_writer_53__append_para: Hypothesis 'asymmetric LO normalize on _gold_append_paragraph' was falsified by replay-verify 2026-05-11: oracle replay 1.0 with new symmetric LO-roundtrip, but validation trajectory FAIL — agent corrupted Heading 1 paragraph mid-doc (15 paras vs expected 12). Demoted from BUG → HARD. Genuine agent-execution failure; gold/eval contract is sound.
    ('libreoffice_writer', 'f_writer_53__append_para'),
    # f_writer_67__append_para: Hypothesis 'asymmetric LO normalize on _gold_append_paragraph' was falsified by replay-verify 2026-05-11: oracle replay 1.0 with new symmetric LO-roundtrip, but validation trajectory FAIL — agent corrupted Heading 1 paragraph mid-doc (15 paras vs expected 12). Demoted from BUG → HARD. Genuine agent-execution failure; gold/eval contract is sound.
    ('libreoffice_writer', 'f_writer_67__append_para'),
    # f_writer_35__page_break: validation. page_break cluster — sibling of already-HARD f_writer_13/_48__page_break. Ctrl+Enter at correct paragraph; multi-step navigation. Family of 3 satisfied by HARD.
    ('libreoffice_writer', 'f_writer_35__page_break'),
    # f_writer_71__doc_case_lower: validation. doc_case 4-variant family (F_WRITER_67/_71/_80/_97) using _gold_doc_case_convert + compare_docx_files (text-only). Multi-step: Ctrl+A → Format>Text>UPPERCASE/lowercase. Eval is text-only files compare (NOT strict) — sound contract. Family-cluster invariant: ≥3 → must surface fix OR escalate; no clean fix path (genuine agent-skill ceiling per plan.md "real skill the agent currently can't do" exception). HARD.
    ('libreoffice_writer', 'f_writer_71__doc_case_lower'),
    # f_writer_95__color_table_words: validation. NEW eval bucket: evaluate_colored_words_in_tables per-word RGB inside table cells. Multi-step UI: drag-select words in cell + Format>Character>FontColor>Custom. Singleton agent-skill.
    ('libreoffice_writer', 'f_writer_95__color_table_words'),
    # f_writer_97__doc_case_convert: validation. doc_case 4-variant family with f_writer_71 above. Same root cause + same HARD justification.
    ('libreoffice_writer', 'f_writer_97__doc_case_convert'),
    # f_writer_22__color_red_p0: validation. terminated/n=13. Color first paragraph red — turn_12 screenshot visibly red 207-word paragraph, but compare_docx_strict returns 0. Same eval-strict-vs-UI shape as already-HARD f_writer_19__highlight_p0 / f_writer_95__color_table_words.
    ('libreoffice_writer', 'f_writer_22__color_red_p0'),
    # f_writer_26__insert_image: validation. terminated/n=11. Insert > Image dialog (Blue Marble into earth_brief docx). Image visibly embedded but compare_docx_strict fails on image bytes / packaging cardinality. NEW skill agent-skill + eval-strict-vs-UI.
    ('libreoffice_writer', 'f_writer_26__insert_image'),
    # f_writer_2__bold_para: validation. terminated/n=9. Bold first paragraph — sibling of already-HARD f_writer_86__bold_body. Standard Ctrl+B + paragraph-select agent-skill.
    ('libreoffice_writer', 'f_writer_2__bold_para'),
    # f_writer_31__insert_double_image: validation. terminated/n=23. Insert > Image cluster — siblings f_writer_26 + f_writer_81. ≥3-variant family-cluster. Same eval-strict-vs-UI (compare_docx_strict on image bytes/packaging) as f_writer_26. No clean source fix without deferred LO-normalize tightening.
    ('libreoffice_writer', 'f_writer_31__insert_double_image'),
    # f_writer_60__strike_p0: validation. terminated/n=7. Strikethrough first paragraph — Format>Character>Strikethrough multi-step UI. NEW skill, eval-strict-vs-UI suspect (similar shape to f_writer_19__highlight_p0 / _22__color_red_p0 HARD).
    ('libreoffice_writer', 'f_writer_60__strike_p0'),
    # f_writer_81__insert_image: validation. terminated/n=12. Insert > Image cluster — siblings f_writer_26 + f_writer_31. Family invariant satisfied → HARD per established eval-strict-vs-UI pattern.
    ('libreoffice_writer', 'f_writer_81__insert_image'),
    # f_writer_96__mixed_align_split: validation. terminated/n=23. Split-paragraph-then-align cluster — sibling of already-HARD f_writer_72__mixed_align_todo. Multi-step (place cursor, Enter, drag-select, apply alignments).
    ('libreoffice_writer', 'f_writer_96__mixed_align_split'),
    # f_writer_98__first_sentence_op: validation. terminated/n=17. first_sentence_op NEW pattern — find-first-sentence + apply formatting. Multi-step drag-select + Format dialog agent-skill.
    ('libreoffice_writer', 'f_writer_98__first_sentence_op'),
    # f_writer_30__append_para: validation global validation. Joins falsified-fix append_para HARD cluster (F_WRITER_33/_37/_39/_53/_67). Same genuine agent-execution failure pattern (Heading-1 corruption mid-doc).
    ('libreoffice_writer', 'f_writer_30__append_para'),
    # f_writer_9__size_para: size_para cluster — siblings F_WRITER_27/_31/_40/_60 HARD. Same compare_docx_strict + examine_font_size every-char-strict.
    ('libreoffice_writer', 'f_writer_9__size_para'),
    # f_writer_19__italic_p0: Agent tends to use SHIFT+ARROWDOWN+SHIFT+END line-select instead of triple-click paragraph-select; compare_docx_strict eval-strict-vs-UI fails when italic lands on a partial paragraph. Same shape as f_writer_19__highlight_p0 HARD.
    ('libreoffice_writer', 'f_writer_19__italic_p0'),
    # f_writer_50__doc_font — default-font task bug tracked in _BUG_TEMPLATE_IDS.
    # ---- multi_apps ----
    # multi_csv_to_docx_table_unemployment_recent12m: Multi-step CSV→docx table build with 12 data rows. This fundamentally requires more turns than the vision-only step budget; lower difficulty in the generator if needed.
    ('multi_apps', 'multi_csv_to_docx_table_unemployment_recent12m'),
    # multi_diff_text_strip_blanks — terminal-not-pre-opened generator bug, see _BUG_TEMPLATE_IDS for fix proposal.
    # multi_literal_match_last_row — missing `_terminal_preopen_steps()` in init (sibling shell tasks have it).
    # multi_topic_astronomy_photo_to_pptx: Init steps stage an astronomy photo asset and build a source pptx with titl
    ('multi_apps', 'multi_topic_astronomy_photo_to_pptx'),
    # multi_code_to_docx_requests_api: Multi-step: open .py + open .docx, append 40 lines preserving leading whitespace, save. Same shape as multi_csv_to_docx_table_unemployment_recent12m (over budget).
    ('multi_apps', 'multi_code_to_docx_requests_api'),
    # multi_compare_table_dept_subset: sheet-name mismatch cluster.
    # multi_csv_concat_rates_to_xlsx_2sheet: validation. truncated/n=30. Multi-step: open 2 CSVs in Calc, build 2-sheet xlsx, save. Cluster with multi_csv_to_docx_table_oil_wti_recent10 + already-HARD unemployment (≥3 trigger N family).
    ('multi_apps', 'multi_csv_concat_rates_to_xlsx_2sheet'),
    # multi_csv_to_docx_table_oil_wti_recent10: validation. truncated/n=30. CSV → docx-table 10-row populate over-budgeted at 30 turns. Verbatim match for already-HARD multi_csv_to_docx_table_unemployment_recent12m. Trigger N.
    ('multi_apps', 'multi_csv_to_docx_table_oil_wti_recent10'),
    # multi_topic_event/food/medical_cover/product_cover_photo_to_docx — compare_docx_strict examine_images byte-hash cluster, not agent skill.
    # multi_csv_to_docx_text_summary_us_state_income_top: validation. truncated/n=30. csv→docx text-summary cluster — sibling of already-HARD multi_csv_to_docx_table_oil_wti_recent10 / unemployment_recent12m. Trigger N.
    ('multi_apps', 'multi_csv_to_docx_text_summary_us_state_income_top'),
    # shell_awk_skip_header — terminal-not-pre-opened generator bug.
    # multi_compare_pdfs_extract_pages_bert_2to4 — terminal-not-pre-opened generator bug.
    # multi_topic_astronomy_photo_to_docx_cover — image byte-hash cluster in _BUG_TEMPLATE_IDS.
    # multi_topic_city_photo_to_pptx: validation. terminated/n=9. Sibling of already-HARD multi_topic_astronomy_photo_to_pptx. NEW topic but same skill.
    ('multi_apps', 'multi_topic_city_photo_to_pptx'),
    # multi_topic_office/wildlife_photo_to_docx_cover — image byte-hash cluster in _BUG_TEMPLATE_IDS.
    # multi_compare_table_filter_high_score: sheet-name mismatch cluster.
    # shell_cut_third_col — terminal-not-pre-opened generator bug.
    # ---- os ----
    # f_os_03__change_backup_time — terminal pre-open fix resolved keystroke race.
    # f_os_13__reschedule_db_snapshot — same terminal pre-open shape as f_os_03.
    # f_os_21__switch_to_utc: Native `is_utc_0` eval. Gold uses `sudo ln -sf /usr/share/zoneinfo/UTC /etc/localtime && echo UTC | sudo tee /etc/timezone`. Pre-c
    ('os', 'f_os_21__switch_to_utc'),
    # f_os_25__swap_localhost_alias — terminal pre-open fix.
    # ---- thunderbird ----
    # f_tb_33__set_attr_a — xulstore.json async-flush race (TB writes async on close; eval reads pre-flush).
    # f_tb_36__star_bills_b: Same eval + same gold as f_tb_36__star_bills (only instruction-paraphrased per validation single-anchor relaxation). Same gloda-DB-a
    ('thunderbird', 'f_tb_36__star_bills_b'),
    # f_tb_05__from_billing_to_receipts + f_tb_07__from_marketing_delete:
    # both templates dropped during thunderbird clone-reduction (redundant filter FileTasks).
    # ---- vlc ----
    # f_vlc_4c__clear_global_play_pause: Both Params legitimate: src writes global-key-play-pause=Ctrl+Space, Param 0 target='' (clear), Param 1 target='Space' (re-bind).
    ('vlc', 'f_vlc_4c__clear_global_play_pause'),
    # f_vlc_13__extract_frame_to_file: short-clip race; vlc.py duration 5→60 fix applied.
    # f_vlc_23__frame_to_wallpaper — compare_images on uniform-color frame is degenerate. See _BUG_TEMPLATE_IDS.
    # f_vlc_4b__set_slider_colours: validation. terminated/n=21. VLC preferences slider-colour edit — sibling of already-HARD f_vlc_4c__clear_global_play_pause (vlc UI cluster).
    ('vlc', 'f_vlc_4b__set_slider_colours'),
    # f_vlc_13__snapshot_t2: short-clip race; vlc.py duration 5→60 fix applied.
    # f_vlc_9__extract_audio_mp3: short-clip race; vlc.py duration 5→60 fix applied.
    # ---- vs_code ----
    # vs_code_settings_pylance_on_real_python_config: Eval-feasible: oracle merges {python.languageServer: 'Pylance'} into the pre-staged real settings.json via _string_aware_merge_set
    ('vs_code', 'vs_code_settings_pylance_on_real_python_config'),
    # vs_code_settings_strict_imports_real_config: Same shape as pylance_on_real_python_config — sets python.analysis.diagnosticMode='workspace'. Oracle is JSON merge, eval is check
    ('vs_code', 'vs_code_settings_strict_imports_real_config'),
    # vs_code_settings_typecheckmode_basic_real_config: Same shape — sets python.analysis.typeCheckingMode='basic'. Oracle = JSON merge; eval = check_json_settings.
    ('vs_code', 'vs_code_settings_typecheckmode_basic_real_config'),
    # kb_format_doc: validation. terminated/n=11. UI-set keybinding Ctrl+Shift+E → editor.action.formatDocument visibly applied in Keyboard Shortcuts view (Source=User), but ~/.config/Code/User/keybindings.json not flushed to disk → check_json_keybindings 0. NEW skill / VS Code UI-state-not-persisted singleton agent-skill.
    ('vs_code', 'kb_format_doc'),
    # ---- validation additions (verified agent-skill ceilings, no source-side fix) ----
    # libreoffice_calc (10): groupby/filter/derived/color/string_clean/dv clusters — phantom-Save-As + multi-step formula + Custom-Color dialog × N rows. Family-cluster invariant satisfied (each maps to documented HARD precedent).
    ('libreoffice_calc', 'f_calc_5__groupby_category_totals'),
    ('libreoffice_calc', 'f_calc_46__groupby_recipe_qty'),
    ('libreoffice_calc', 'f_calc_50__groupby_dept_salary'),
    ('libreoffice_calc', 'f_calc_50__derived_total_comp'),
    ('libreoffice_calc', 'f_calc_82__derived_price_per_gb'),
    ('libreoffice_calc', 'f_calc_24__color_low_uptime'),
    ('libreoffice_calc', 'f_calc_48__string_clean_book_proper'),
    ('libreoffice_calc', 'f_calc_77__dv_sold_yes_no'),
    ('libreoffice_calc', 'f_calc_2__filter_to_new_sheet'),
    ('libreoffice_calc', 'f_calc_99__filter_region_with_total'),
    ('libreoffice_calc', 'f_calc_39__filter_confirmed_to_sheet'),
    # libreoffice_impress (8): title_color RGB-rounding + caption-italic Ctrl+A partial-select toggle + swap_far save-pollution + duplicate-slide / table-insert. d_imp_59__title_move_position needs deferred Cm-tolerant eval helper.
    ('libreoffice_impress', 'd_imp_08__caption_italic'),
    ('libreoffice_impress', 'd_imp_20__swap_far'),
    ('libreoffice_impress', 'd_imp_28__duplicate_last_h5bottom'),
    ('libreoffice_impress', 'd_imp_32__title_color_g1p3_5'),
    ('libreoffice_impress', 'd_imp_39__title_color_p3'),
    ('libreoffice_impress', 'd_imp_44__title_color_h3top'),
    ('libreoffice_impress', 'd_imp_49__insert_table_footer_4x3'),
    # d_imp_59__title_move_position — strict examine_shape without tolerance, instruction "top-right corner" qualitative.
    # ('libreoffice_impress', 'd_imp_59__title_move_position'),  # see _BUG_TEMPLATE_IDS
    # gimp_config_undo_200
    # was dropped by gimp subagent (preferences down-weight pass — kept undo_100
    # as the representative). HARD entry would be a ghost now.
    # ---- validation additions (verified agent-skill ceilings, no source-side fix) ----
    # chrome: agent took GUI date-picker rabbit hole though gold URL was directly typeable. Single-seed skill.
    ('chrome', 'f_chrome_84__search_hilton_hotel'),
    # vlc: Qt QSpinBox Ctrl+A APPEND race (same root cause family as gimp GTK APPEND cluster, different toolkit).
    ('vlc', 'f_vlc_2__set_max_volume'),
    # libreoffice_calc (3): multi-step Shift+F11/sheet-rename + groupby aggregation + Move/Copy modal cluster.
    ('libreoffice_calc', 'f_calc_1__aggregate_to_new_sheet'),
    ('libreoffice_calc', 'f_calc_21__groupby_venue_totals'),
    ('libreoffice_calc', 'f_calc_93__sheet_rename_and_copy'),
    # libreoffice_writer (10): Release Notes infobar Trigger H victims + F&R/insert_image/append_para skill ceilings. f_writer_18__italic_p0 EXCLUDED — cluster escalation dependent (italic_p0 eval-strict-vs-UI deferred pending).
    ('libreoffice_writer', 'f_writer_1__bold_para'),
    # f_writer_38 / _44 / _93 __doc_font — eval-strict-vs-UI cluster (compare_font_names rFonts strip on LO 7.3 round-trip).
    ('libreoffice_writer', 'f_writer_13__find_replace'),
    ('libreoffice_writer', 'f_writer_28__find_replace'),
    ('libreoffice_writer', 'f_writer_1__append_signoff'),
    ('libreoffice_writer', 'f_writer_54__insert_image'),
    ('libreoffice_writer', 'f_writer_70__append_para'),
    ('libreoffice_writer', 'f_writer_93__append_para'),
    # libreoffice_impress (10): Ctrl+A partial-select toggle race (d_imp_16/_23) + title_color (d_imp_28 → joins cluster escalation) + skill ceilings (d_imp_31/_57/_63) + position-compare Cm-exact (d_imp_60 ×2 → joins d_imp_59 cluster escalation) + save-pollution (d_imp_62). d_imp_54__master_bg_blue flagged uncertain (distance-tolerant eval; validation second look).
    ('libreoffice_impress', 'd_imp_16__title_bold_notes'),
    ('libreoffice_impress', 'd_imp_23__title_italic_to8'),
    ('libreoffice_impress', 'd_imp_28__title_color_h5bottom'),
    ('libreoffice_impress', 'd_imp_31__add_summary_slide_g3x3'),
    ('libreoffice_impress', 'd_imp_54__master_bg_blue'),
    # d_imp_57__resize_picture_extra / d_imp_60__image_move_position / d_imp_60__image_move_extra — strict examine_modify_height / examine_shape without tolerance. See _BUG_TEMPLATE_IDS.
    # ('libreoffice_impress', 'd_imp_57__resize_picture_extra'),
    # ('libreoffice_impress', 'd_imp_60__image_move_position'),
    # ('libreoffice_impress', 'd_imp_60__image_move_extra'),
    ('libreoffice_impress', 'd_imp_62__multi_slide_subset_format'),
    ('libreoffice_impress', 'd_imp_63__audio_insert'),
    # multi_apps (2): 40+ line vision-only Go/Rust transcription with whitespace preservation. Siblings of HARD multi_code_to_docx_requests_api.
    ('multi_apps', 'multi_code_to_docx_gin_handler'),
    ('multi_apps', 'multi_code_to_docx_tokio_lib'),
    # ---- validation (2026-05-12): impress INFEAS cluster ----
    # 8 impress INFEAS verified as agent skill ceilings (Trigger N) — color-tolerant
    # evals already mitigate strictness, no I/J/K/L mismatch. Failure cause is
    # text-edit-vs-shape-select mode confusion + RGB custom-color dialog ceiling.
    # 4 of 8 already HARD (d_imp_01__body_bold, _30__title_color_g2x2_4,
    # _32__title_color_g1p3_5, _62__multi_slide_subset_format); these 4 are new.
    ('libreoffice_impress', 'd_imp_04__title_italic'),
    ('libreoffice_impress', 'd_imp_15__title_italic_logo'),
    ('libreoffice_impress', 'd_imp_54__master_bg_green'),
    ('libreoffice_impress', 'd_imp_62__title_color_subset_format'),
    # ---- validation (2026-05-12): writer + vs_code ceilings ----
    # f_writer_6__underline_para — distinct from already-HARD f_writer_6__page_numbers_footer.
    # Selection-cursor control failure (agent loses caret position, mangles doc).
    ('libreoffice_writer', 'f_writer_6__underline_para'),
    # vs_code: F&R whole-word + Save skill ceiling.
    ('vs_code', 'file_edit_rename_old_to_new'),
    # vs_code: workspace-trust dialog + nested JSON edit ceiling. The trust dialog
    # is a generator-wide gap (no pre-config dismissal in _vs_code_preopen_steps);
    # underlying JSON-edit skill remains hard regardless.
    ('vs_code', 'settings_minimap_multi'),
}


# Bug queue — templates with identified source-side fixes pending.
# Documentation only, NOT filtered. Each entry has a concrete fix proposal.
# Next validation should drain this queue: apply each fix, replay-verify,
# remove from this list. UNKNOWN entries here need replay-verify to confirm.
# Fixes already applied in the cycle that produced this list are NOT
# included.
_BUG_TEMPLATE_IDS: set[tuple[str, str]] = {
    # =====================================================================
    # validation (2026-05-12) → validation (2026-05-12) deferred verification:
    # 17 of 19 prior deferred entries verified PASS via docker oracle replay
    # + direct container probes and removed from this set:
    #   - f_calc_55 / f_calc_77 (chart_type via compare_calc_chart_type)
    #   - f_os_32 / f_os_33 / f_os_36 (gsettings with DBUS export)
    #   - 6 multi_topic_*_photo_to_{pptx,docx}/gallery_4 (LO_SAVE_POSTCONFIG
    #     ^Save$ matcher in common.py — modal WM_NAME confirmed = "Save")
    #   - 6 multi_epub_md_* (compare_epub override in eval/metrics.py now
    #     matches pandoc EPUB3 layout `EPUB/content.opf` + `*.xhtml`)
    # =====================================================================
    # [verified and removed 2026-05-12]
    # f_calc_88__freeze_header_row — eval/metrics.py::check_xlsx_freeze_pane
    # helper added (reads openpyxl freeze_panes directly, bypasses LO normalize
    # strip). _gold_freeze_panes drops `_LO_NORMALIZE_TAIL`; _eval_compare_table
    # splits freeze rules into `func=['check_xlsx_freeze_pane','compare_table']`.
    # Oracle replay 0.0→1.0.
    # ---- thunderbird (no-action, pending validation) ----
    # f_tb_36__star_bills: 2-template family; agent UI flow correct but
    # eval reads gloda.sqlite which may not reflect star state until
    # TB indexer flushes. Leave in BUG; validation will reconfirm.
    # Validation simple pkill+sync regressed to trivial_pass (preopen TB writes
    # its own defaults that match eval target); needs deeper redesign — likely
    # re-order pre-config so src writes happen AFTER preopen+pkill, not before.
    # gloda.sqlite path is `_gloda_py` (separate from the `_xulstore_attr_py`
    # heredoc bug noted in f_tb_33 below).
    ('thunderbird', 'f_tb_36__star_bills'),
    # =====================================================================
    # Validation promotions (2026-05-12): 7 entries moved
    # from _HARD_TEMPLATE_IDS after per-domain trajectory review identified
    # them as task bugs (not agent-skill ceilings).
    # =====================================================================
    # [verified and removed 2026-05-12]
    # 5 validation doc_font entries (f_writer_7_arial / 38 / 44 / 69 / 93) cleared
    # by B7 fix: writer/_docx_body_py:431 + _docx_structured_py:1219 + _src_qa_format:1466
    # now emit `doc.styles['Normal'].font.name = font_name`. styles.xml inspection
    # confirms target font appears in Normal-style rPr (LO toolbar reads up the
    # cascade before reaching theme fallback `minorHAnsi → Cambria`).
    # [verified and removed 2026-05-12]
    # multi_literal_match_last_row — root cause was NOT missing preopen
    # (that was already wired). Real defect: upstream `literal_match` does
    # `str(result) == str(expected)`, but lite_osworld getter returns the
    # inner `rules` dict for `{"type":"rule","rules":{"expected":...}}`, so
    # comparison stringified the dict and never matched. Fixed via deferred
    # `literal_match` override in eval/metrics.py that unwraps `{"expected":...}`
    # before delegating to upstream. Oracle replay 0.0→1.0; max-turns 0 = 0.0.
    # ---- thunderbird (1) — _src_xulstore_attr_clean heredoc bug + flush race ----
    # f_tb_33__set_attr_a: validation identified the actual
    # root cause is a SHELL HEREDOC BUG in `_src_xulstore_attr_clean`
    # (thunderbird.py near line 587): `cmd_parts` are joined with ` && ` —
    # producing a line `PYEOF && python3 << 'PYEOF'` that is NOT a bare
    # delimiter, so bash never closes the first heredoc. First python3 swallows
    # both scripts as stdin, fails with SyntaxError, returns 1, and **src writes
    # nothing**. F_TB_33 then trivial-passes because TB's tarball-default
    # `sizemode="maximized"` happens to equal the eval target. F_TB_31 hid the
    # same bug (init==target → broken src irrelevant).
    #   proposed fix (subagent had implemented but not verified before kill):
    #   emit ONE execute step per heredoc instead of `&& `-joining. Reverted
    #   pending end-to-end validate.py confirmation. See validation review
    #   notes for the original async-flush hypothesis (still possible secondary
    #   cause once src actually executes).
    ('thunderbird', 'f_tb_33__set_attr_a'),
    # Dropped-template evidence (validation 2026-05-12) moved out of this data file:
    # see /devs/envs/lite.osworld/synth/dropped_templates.md. The drops themselves
    # are the entries above; this dict stays their single owner.
}


def _apply_drop_filter(templates: list) -> list:
    return [t for t in templates if (t.domain, t.template_id) not in _DROPPED_TEMPLATE_IDS]


ALL_TEMPLATES: list = _apply_drop_filter(_rescale_for_volume(
    CALC_TEMPLATES
    + WRITER_TEMPLATES
    + IMPRESS_TEMPLATES
    + GIMP_TEMPLATES
    + CHROME_TEMPLATES
    + VLC_TEMPLATES
    + VSCODE_TEMPLATES
    + THUNDERBIRD_TEMPLATES
    + OS_TEMPLATES
    + MULTI_APPS_TEMPLATES
))

TEMPLATES_BY_DOMAIN: dict[str, list] = {}
for _t in ALL_TEMPLATES:
    TEMPLATES_BY_DOMAIN.setdefault(_t.domain, []).append(_t)
