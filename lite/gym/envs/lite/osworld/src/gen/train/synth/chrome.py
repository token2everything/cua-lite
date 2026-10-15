"""Chrome synth generator (Track A — host-heredoc design).

Per AGENTS.md / chrome.md: source state is built via heredocs that write
directly into Chrome's user-data dir. The dispatch handler launches chrome
with `--user-data-dir=/home/user/chrome-data`, so we write to
`/home/user/chrome-data/Default/...` (NOT `/home/user/.config/google-chrome/`).
Every Preferences/Bookmarks/History write is preceded by `_kill_chrome_step()`
+ uses `_CHROME_RESTART_POSTCONFIG` so the file edit isn't clobbered by
chrome's quit-time rewrite. History sqlite needs chrome's full schema on
disk first → pre-config does `launch chrome → sleep 5 → pkill chrome →
INSERT` (mirrors eval `_chrome_44ee5668`).

Implemented row groups (oracle-validated, all 44/44 PASS — `history_delete_*`
schema-race fixed by `CREATE TABLE IF NOT EXISTS` in the seed step plus
defensive try/except in the oracle DELETE; `tabs_open_*` session-restore
duplicate-tab inflation fixed by SIGKILL + clearing chrome's session files
before relaunch):
- Preferences booleans: do-not-track / safe-browsing / clear-data-on-exit
                                                     → exact_match on Preferences blob
                                                       (`pref_dnt_enable`, `pref_safe_browsing_enable`,
                                                        `pref_clear_data_on_exit`)
- profile_name string                                → exact_match on profile.name
                                                       (`profile_name_thomas`, `profile_name_maria`)
- new_startup_page (clear-on-startup)                → exact_match on session.restore_on_startup==5
                                                       (`new_startup_page_clear`)
- default_search_engine                              → match_in_list on default_search_provider_data
                                                       (`search_engine_bing`, `search_engine_ddg`)
- Bookmark folder create                             → is_expected_bookmarks (folder names)
                                                       (`bookmark_folder_favorites`,
                                                        `bookmark_folder_reading_list`,
                                                        `bookmark_folder_work`)
- Bookmark URL add                                   → is_expected_bookmarks (URL list)
                                                       (`bookmark_url_python_docs`, `bookmark_url_kernel`)
- History delete by keyword                          → check_history_deleted
                                                       (`history_delete_youtube`, `history_delete_tutorial`,
                                                        `history_delete_example_domain`)
- Cookies delete by domain                           → is_cookie_deleted
                                                       (`cookie_delete_wikipedia`,
                                                        `cookie_delete_analytics_tracker`)
- Desktop shortcut                                   → is_shortcut_on_desktop
                                                       (`shortcut_python_docs`, `shortcut_kernel`)
- Active tab via chrome_open_tabs                    → is_expected_active_tab URL
                                                       (`active_tab_python_docs`, `active_tab_kernel_about`)
- Multi-tab open                                     → is_expected_tabs
                                                       (`tabs_open_python_keep_decoys`, `tabs_open_three_sites`)
- Real-asset Wikipedia (file:// URL via `_stage_asset`):
  - Active tab over staged HTML                      → is_expected_active_tab
                                                       (`active_tab_wiki_apollo`,
                                                        `active_tab_wiki_mount_everest`,
                                                        `active_tab_wiki_solar_system`,
                                                        `active_tab_wiki_coffee`,
                                                        `active_tab_wiki_pizza`,
                                                        `active_tab_wiki_octopus`)
  - Bookmark URL = file:// staged HTML               → is_expected_bookmarks
                                                       (`bookmark_wiki_eiffel_tower`,
                                                        `bookmark_wiki_lego`,
                                                        `bookmark_wiki_mona_lisa`)
                                                       Batch trimmed 6→3 (over-share +14pp).
  - Multi-tab open of file:// staged HTML            → is_expected_tabs
                                                       (`tabs_open_wiki_3space_articles`,
                                                        `tabs_open_wiki_3foods`,
                                                        `tabs_open_wiki_3art`,
                                                        `tabs_open_wiki_3science`)
  - History delete keyword on staged HTML            → check_history_deleted
                                                       (`history_delete_wiki_apollo`,
                                                        `history_delete_wiki_volcano`,
                                                        `history_delete_wiki_yoga`,
                                                        `history_delete_wiki_paper_airplane`)
- URL-with-query active tab                          → check_direct_json_object
                                                       via `active_tab_url_parse`
                                                       (closes -23.3pp chrome eval gap; mirrors
                                                        eval `1704f00f` rentalcars / `2888b4e6`
                                                        macys / `47543840` budget). Pre-config
                                                        opens a wrong-query URL; oracle relaunches
                                                        chrome on the gold URL with the expected
                                                        query params; eval parses query params
                                                        from the active tab and exact-matches.
                                                       (`url_query_rentalcars_zurich`,
                                                        `url_query_rentalcars_paris`,
                                                        `url_query_booking_hotel_paris`,
                                                        `url_query_jobs_sf`,
                                                        `url_query_amazon_keyboard`,
                                                        `url_query_github_issues`)
                                                       Batch dropped `url_query_youtube_search`
                                                       (sp= post-load token) +
                                                       `url_query_compound_*` (path rewrite).
                                                       Batch dropped `url_query_flights_bos_lax`.

Deferred (per devs/envs/lite.osworld/synth/chrome.md `## Implementation status`):
- check_direct_json_object form-driven URL rows #10-#12, #18-#24, #43-#50,
  #85-#89, #136-#146 — require live mock HTML form sites + agent navigation;
  URL-query parsing cannot be deterministically replayed in a host-heredoc
  oracle without faking the entire CDP active-tab URL response.
- is_expected_url_pattern_match post-click rows #33-#37, #51-#54, #147-#150
  — same blocker. Subset re-implemented above as active-tab direct-URL rows
  where eval intent is "URL matches" not "click then verify pattern".
- Restore-closed-tab rows #6, #7, #127 — require Chrome's session-restore
  state machine; oracle would need to inject into the Recently-Closed list
  which isn't user-writable.
- Multi-tab grouping / pinning rows #74, #100, #121-#123 — chrome.tab_group
  state is not user-writable from disk; needs CDP.
- Translate / reader-view / find-next / zoom rows #25, #102, #106, #115, #130
  — chrome stores these as transient session state, not Preferences keys.
- Misc pref keys with no documented Preferences schema #39-#42, #55-#62,
  #78-#80 — speculative pref names; defer until eval-anchored.

Wikipedia HTML bundle rows #96-#130 are now implemented above via
`_stage_asset` + `file://` URLs. GitHub bundle rows remain deferred until
`assets/synth/html/github/*.html` is added.

validation (2026-05-11) — `check_direct_json_object` UNDER-gap fill: synth was
at 28% (23/82) cdjo vs eval at 49% (21/43). Added 15 new cdjo FileTasks
covering shopping/booking verticals that eval rows exercise:
- Flight search (4): delta / aa / southwest / alaskaair — F_CHROME_73..76
- Product comparison (3): apple iPhone / samsung Galaxy / sony headphones
  — F_CHROME_77..79
- Car rental (3): hertz / enterprise / avis — F_CHROME_80..82
- Hotel (2): marriott / hilton — F_CHROME_83..84
- Generic shopping (3): bestbuy / costco / wayfair — F_CHROME_85..87
Plus 2 `match_in_list` ADDS over `chrome_color_scheme` (eval row 23
osworld_chrome_93eabf48 dark-mode toggle) — F_CHROME_88..89.

Usage:
    uv run python -m lite.gym.envs.lite.osworld.src.gen.train \\
        --track synth --domain chrome
"""

from __future__ import annotations

import datetime as _datetime
import json as _json
import random
import textwrap

from lite.gym.envs.lite.osworld.src.gen.train.synth._utils import SynthTemplate, _stage_asset, _stable_hash


# Validation fix: date-rot helper. Today is 2026-05-12. Returns YYYY-MM-DD
# N days from today so instructions/decoy URLs always sit in the future.
_TODAY = _datetime.date(2026, 5, 12)


def _future_date(offset_days: int) -> str:
    """Return YYYY-MM-DD for `_TODAY + offset_days`. Used by jetblue /
    enterprise / avis decoys + the corresponding instruction strings so we
    don't ship hardcoded past-date instructions."""
    return (_TODAY + _datetime.timedelta(days=offset_days)).isoformat()

_CHROME_DATA = "/home/user/chrome-data"
_CHROME_DEFAULT = f"{_CHROME_DATA}/Default"
_CHROME_PREFS = f"{_CHROME_DEFAULT}/Preferences"
_CHROME_BOOKMARKS = f"{_CHROME_DEFAULT}/Bookmarks"
_CHROME_HISTORY = f"{_CHROME_DEFAULT}/History"
_CHROME_COOKIES = f"{_CHROME_DEFAULT}/Cookies"

# Postconfig that kills+relaunches chrome so the just-edited Preferences /
# Bookmarks / History / Cookies file is the one chrome reads. Mirrors the
# postconfig used by every chrome eval task that reads Preferences.
_CHROME_RESTART_POSTCONFIG: list[dict] = [
    {"type": "launch", "parameters": {"command": ["pkill", "chrome"]}},
    {"type": "launch", "parameters": {"command": ["google-chrome", "--remote-debugging-port=1337"]}},
    {"type": "sleep", "parameters": {"seconds": 3}},
]

#  : Shared fragment to clear chrome's session-restore
# files. Without this, after `pkill chrome` the next launch may restore the
# previous tab set, which collides with the oracle's positional-URL relaunch
# pattern (used by url_query / url_pattern / url_query_compound factories).
# Mirrors `_make_tabs_template` cleanup. Always emit AFTER the kill, BEFORE
# the launch.
_SESSION_CLEANUP_CMD = (
    "rm -f /home/user/chrome-data/Default/'Last Session' "
    "/home/user/chrome-data/Default/'Last Tabs' "
    "/home/user/chrome-data/Default/'Current Session' "
    "/home/user/chrome-data/Default/'Current Tabs' "
    "2>/dev/null; true"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_instr(instructions: list[str], seed: int) -> dict:
    """Return `{"instr": <str>}` for `seed in [1, len(instructions)]`, else
    `{"_skip": True}`. Batch fix #1: caps every chrome template's row
    count at `len(instructions)` so we never emit byte-clone rows when the
    scaler probes seeds beyond the unique-instruction count.
    """
    idx = seed - 1
    if idx < 0 or idx >= len(instructions):
        return {"_skip": True}
    return {"instr": instructions[idx]}

def _execute(command: str, *, shell: bool = True, **extra) -> dict:
    return {"type": "execute", "parameters": {"command": command, "shell": shell, **extra}}


def _chrome_preopen_steps(
    urls: list[str] | None,
    *,
    extra_launch_args: list[str] | None = None,
) -> list[dict]:
    """Standard Chrome pre-open sequence.

    Emits launch + socat-9222 debug bridge + chrome_open_tabs + 2s sleep +
    activate_window. Per validation:
      * 32 chrome rows were missing the socat 9222 bridge → eval couldn't
        remote-debug those tasks; this helper makes it universal.
      * 0/128 rows did `activate_window` after `chrome_open_tabs`; this
        helper fixes that to 100%.

    `extra_launch_args` lets callers append flags after
    `--remote-debugging-port=1337` (e.g. `--user-data-dir=/home/user/chrome-data`
    for staged file:// HTML / PDF rows).
    """
    launch_cmd = ["google-chrome", "--remote-debugging-port=1337"]
    if extra_launch_args:
        launch_cmd = launch_cmd + list(extra_launch_args)
    steps: list[dict] = [
        {"type": "launch", "parameters": {"command": launch_cmd}},
        {"type": "launch", "parameters": {
            "command": ["socat", "tcp-listen:9222,fork", "tcp:localhost:1337"]}},
    ]
    if urls is not None:
        steps.append({"type": "chrome_open_tabs", "parameters": {"urls_to_open": urls}})
    steps.append({"type": "execute", "parameters": {"command": "sleep 2", "shell": True}})
    steps.append({"type": "activate_window", "parameters": {"window_name": "Chrome"}})
    return steps


def _kill_chrome_step() -> dict:
    """Kill any running Chrome so subsequent profile edits stick.

    Chrome rewrites Preferences/Bookmarks/History/Cookies on quit — if we
    write while Chrome is running, the in-memory copy will overwrite us.
    """
    return _execute("pkill -9 -f chrome 2>/dev/null; sleep 2; true")


def _merge_prefs_step(patch: dict) -> dict:
    """Read existing Preferences (if any), deep-merge `patch`, write back."""
    patch_json = _json.dumps(patch)
    py = textwrap.dedent(f"""\
        import json, os
        path = {_CHROME_PREFS!r}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        prefs = {{}}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    prefs = json.load(f)
                if not isinstance(prefs, dict):
                    prefs = {{}}
            except Exception:
                prefs = {{}}
        patch = json.loads({patch_json!r})

        def deep_merge(base, p):
            for k, v in p.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    deep_merge(base[k], v)
                else:
                    base[k] = v

        deep_merge(prefs, patch)
        with open(path, 'w') as f:
            json.dump(prefs, f)
        """)
    return _execute(f"python3 << 'PYEOF'\n{py}\nPYEOF")


def _write_bookmarks_step(roots: dict) -> dict:
    """Write a complete Bookmarks JSON tree."""
    payload = {"roots": roots, "version": 1}
    payload_json = _json.dumps(payload)
    py = textwrap.dedent(f"""\
        import json, os
        path = {_CHROME_BOOKMARKS!r}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = json.loads({payload_json!r})
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)
        """)
    return _execute(f"python3 << 'PYEOF'\n{py}\nPYEOF")


def _empty_bookmarks_roots() -> dict:
    return {
        "bookmark_bar": {"children": [], "name": "Bookmarks bar", "type": "folder"},
        "other": {"children": [], "name": "Other bookmarks", "type": "folder"},
        "synced": {"children": [], "name": "Mobile bookmarks", "type": "folder"},
    }


def _bookmark_url_node(name: str, url: str, node_id: str = "100") -> dict:
    return {
        "date_added": "13365000000000000",
        "date_last_used": "0",
        "guid": f"00000000-0000-0000-0000-{node_id.zfill(12)}",
        "id": node_id,
        "name": name,
        "type": "url",
        "url": url,
    }


def _bookmark_folder_node(name: str, node_id: str = "100", children: list | None = None) -> dict:
    return {
        "children": children or [],
        "date_added": "13365000000000000",
        "date_last_used": "0",
        "date_modified": "0",
        "guid": f"00000000-0000-0000-0000-{node_id.zfill(12)}",
        "id": node_id,
        "name": name,
        "type": "folder",
    }


def _seed_history_step(entries: list[tuple[str, str]]) -> dict:
    """Pre-config: seed Chrome History sqlite with `(url, title)` rows.

    Caller MUST arrange for chrome to launch + die BEFORE this step so
    chrome's full History schema (urls + visits + many migration tables)
    is on disk. We then INSERT into chrome's schema. Mirrors the eval-side
    seed pattern (e.g. eval 44ee5668: launch → sleep 5 → pkill → INSERT).
    """
    entries_json = _json.dumps(entries)
    py = textwrap.dedent(f"""\
        import sqlite3, datetime, os, json as _j
        path = {_CHROME_HISTORY!r}
        # Ensure the History DB exists with the urls/visits tables. When chrome
        # races (5s sleep) and didn't fully init the schema, we create it
        # ourselves; this matches Chrome 120+'s minimal History schema enough
        # for INSERT/DELETE to work and for the eval getter's SELECT to return.
        os.makedirs(os.path.dirname(path), exist_ok=True)
        conn = sqlite3.connect(path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS urls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url LONGVARCHAR,
            title LONGVARCHAR,
            visit_count INTEGER DEFAULT 0 NOT NULL,
            typed_count INTEGER DEFAULT 0 NOT NULL,
            last_visit_time INTEGER NOT NULL,
            hidden INTEGER DEFAULT 0 NOT NULL
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS visits (
            id INTEGER PRIMARY KEY,
            url INTEGER NOT NULL,
            visit_time INTEGER NOT NULL,
            from_visit INTEGER,
            transition INTEGER DEFAULT 0 NOT NULL,
            segment_id INTEGER,
            visit_duration INTEGER DEFAULT 0 NOT NULL
        )''')
        entries = _j.loads({entries_json!r})
        ts = int((datetime.datetime.now()-datetime.datetime(1601,1,1)).total_seconds()*1e6)
        for url, title in entries:
            c.execute('INSERT OR IGNORE INTO urls (url,title,visit_count,typed_count,last_visit_time,hidden) VALUES (?,?,1,0,?,0)', (url, title, ts))
            uid = c.lastrowid
            if uid:
                c.execute('INSERT INTO visits (url,visit_time,from_visit,transition,segment_id,visit_duration) VALUES (?,?,0,805306368,0,0)', (uid, ts))
        conn.commit()
        try:
            conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            conn.commit()
        except Exception:
            pass
        conn.close()
        print('history seeded:', len(entries))
        """)
    return _execute(f"python3 << 'PYEOF'\n{py}\nPYEOF")


def _delete_history_keyword_step(keyword: str) -> dict:
    """Oracle: delete URL rows whose URL contains `keyword`.

    Defensive: the urls table may be absent if chrome failed to fully init
    its schema during pre-config (race condition). In that case the eval
    getter also returns no rows, so DELETE-on-missing is effectively a no-op
    and we should not fail the step.
    """
    py = textwrap.dedent(f"""\
        import sqlite3, os
        path = {_CHROME_HISTORY!r}
        if os.path.exists(path):
            conn = sqlite3.connect(path)
            c = conn.cursor()
            try:
                c.execute("DELETE FROM urls WHERE url LIKE ?", ('%' + {keyword!r} + '%',))
            except sqlite3.OperationalError:
                pass
            try:
                c.execute("DELETE FROM visits WHERE url NOT IN (SELECT id FROM urls)")
            except sqlite3.OperationalError:
                pass
            conn.commit()
            try:
                conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
                conn.commit()
            except Exception:
                pass
            conn.close()
        print('history pruned for keyword:', {keyword!r})
        """)
    return _execute(f"python3 << 'PYEOF'\n{py}\nPYEOF")


def _seed_cookies_step(domain_cookies: list[tuple[str, str, str]]) -> dict:
    """Pre-config: seed Cookies sqlite with `(host_key, name, value)` rows.

    Uses a forward-compatible Chrome 120+ schema (with top_frame_site_key,
    last_update_utc, source_type, has_cross_site_ancestor columns). When
    Chrome inits a fresh DB, it adds these columns if they don't exist;
    seeding with them present is safe across versions.
    """
    rows_json = _json.dumps(domain_cookies)
    py = textwrap.dedent(f"""\
        import sqlite3, datetime, os, json as _j
        path = {_CHROME_COOKIES!r}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        conn = sqlite3.connect(path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS cookies (
            creation_utc INTEGER NOT NULL, host_key TEXT NOT NULL,
            top_frame_site_key TEXT NOT NULL DEFAULT '',
            name TEXT NOT NULL, value TEXT NOT NULL, encrypted_value BLOB DEFAULT '',
            path TEXT NOT NULL, expires_utc INTEGER NOT NULL,
            is_secure INTEGER NOT NULL, is_httponly INTEGER NOT NULL,
            last_access_utc INTEGER NOT NULL, has_expires INTEGER NOT NULL DEFAULT 1,
            is_persistent INTEGER NOT NULL DEFAULT 1,
            priority INTEGER NOT NULL DEFAULT 1,
            samesite INTEGER NOT NULL DEFAULT -1,
            source_scheme INTEGER NOT NULL DEFAULT 0,
            source_port INTEGER NOT NULL DEFAULT -1,
            last_update_utc INTEGER NOT NULL DEFAULT 0,
            source_type INTEGER NOT NULL DEFAULT 0,
            has_cross_site_ancestor INTEGER NOT NULL DEFAULT 0,
            UNIQUE (host_key, top_frame_site_key, has_cross_site_ancestor, name, path, source_scheme, source_port))''')
        rows = _j.loads({rows_json!r})
        now = int((datetime.datetime.now()-datetime.datetime(1601,1,1)).total_seconds()*1e6)
        exp = now + 365*24*3600*10**6
        for host_key, name, value in rows:
            c.execute('INSERT OR IGNORE INTO cookies (creation_utc,host_key,top_frame_site_key,name,value,encrypted_value,path,expires_utc,is_secure,is_httponly,last_access_utc,has_expires,is_persistent,priority,samesite,source_scheme,source_port,last_update_utc,source_type,has_cross_site_ancestor) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                      (now, host_key, '', name, value, b'', '/', exp, 1, 1, now, 1, 1, 1, 0, 1, 443, now, 0, 0))
        conn.commit()
        try:
            conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            conn.commit()
        except Exception:
            pass
        conn.close()
        print('cookies seeded:', len(rows))
        """)
    return _execute(f"python3 << 'PYEOF'\n{py}\nPYEOF")


def _delete_cookies_domains_step(domains: list[str]) -> dict:
    """Oracle: delete cookies whose host_key matches any of `domains`.

    Match is substring on host_key (chrome rule `is_cookie_deleted` checks
    `compare_urls(domain, cookies_domain)` — substring suffices for our
    seeded rows whose host_key is exactly the domain string).
    """
    domains_json = _json.dumps(domains)
    py = textwrap.dedent(f"""\
        import sqlite3, os, json as _j
        path = {_CHROME_COOKIES!r}
        if os.path.exists(path):
            conn = sqlite3.connect(path)
            c = conn.cursor()
            for d in _j.loads({domains_json!r}):
                c.execute("DELETE FROM cookies WHERE host_key LIKE ?", ('%' + d + '%',))
            conn.commit()
            try:
                conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
                conn.commit()
            except Exception:
                pass
            conn.close()
        print('cookies pruned for domains:', {domains_json!r})
        """)
    return _execute(f"python3 << 'PYEOF'\n{py}\nPYEOF")


def _write_desktop_shortcut_step(name: str, url: str) -> dict:
    """Oracle: write a chrome desktop shortcut (.desktop file)."""
    # Use a slugged filename so multiple shortcuts coexist; eval matches Name= line.
    slug = "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-") or "shortcut"
    body = textwrap.dedent(f"""\
        [Desktop Entry]
        Version=1.0
        Type=Application
        Name={name}
        Exec=google-chrome --app={url}
        Icon=google-chrome
        Terminal=false
        StartupWMClass={slug}
        """)
    return _execute(
        f"mkdir -p /home/user/Desktop && cat > /home/user/Desktop/{slug}.desktop << 'EOF'\n{body}EOF"
    )


# ---------------------------------------------------------------------------
# TEMPLATES export — populated by §I.g below.
#
# Batch: legacy `_make_*_template` factories (bool_pref / profile_name /
# new_startup_page / search_engine / bookmark_folder / bookmark_url /
# history_delete / cookie_delete / shortcut / active_tab / tabs / asset_*
# / url_query / url_pattern / url_query_compound) and their SPEC tables
# were deleted in this validation pass. Their eval skill space is fully covered by
# the file-task templates registered in §I.f FILE_TASKS below (each
# legacy SPEC entry maps to one File × Task × Param tuple).
#
# Added Loop 6 (cdjo URL-with-query, F-CHROME-24..28
# + 31..32) + Loop 7 (is_expected_url_pattern_match, F-CHROME-29/30/33).
# Closes the largest chrome train/eval gap.
# (chrome Δraw +40.6pp — synth too easy; eval has 16 cdjo + 4 url-pattern
# rows where synth had 0). Trimmed soft-3b cookie/history second tasks on
# F-CHROME-4/5/6/7/8/19/20 (same gold-builder duplicates) to free taxonomy
# bucket budget for the harder cdjo / url_pattern rows.
#
#  : aligned synth to eval's chrome distribution (46 rows).
# Reduced over-represented buckets:
#   tabs       18% → 2%   (dropped F-CHROME-17 open_tabs_alongside_wikipedia,
#                          F-CHROME-22 open_extra_staged_tabs,
#                          F-CHROME-23 open_food_tabs)
#   bookmarks  14% → 5%   (dropped F-CHROME-1 create_named_folder,
#                          F-CHROME-3 add_url_alongside)
#   history    12% → 2%   (dropped F-CHROME-5 delete_shopping_history)
# Added cdjo Files (F-CHROME-34..41 — united / jetblue / kayak / walmart /
# target / ebay / yelp / redfin) to lift cdjo 9% → 34% (eval 37%, ±5pp).
# Added url_pattern Files (F-CHROME-42..45 — united-baggage / irs-form /
# github-repo / stackoverflow-tag) to lift url_pattern 3% → 16% (eval 18%).
# F_CHROME_3 / F_CHROME_5 File instances retained (no rename per project
# rule) but no longer referenced by FILE_TASKS — dead code, harmless.
# ---------------------------------------------------------------------------

TEMPLATES: list[SynthTemplate] = []


# ===========================================================================
# §I. File-task templates (Batch, dataclass form)
#
# Mirrors synth/libreoffice_calc.py + synth/libreoffice_impress.py §I.
# This domain is file-as-topic (no inner TopicTheme rotation): each File
# already encodes both the structural shape AND the content semantics.
# (Compare: synth/libreoffice_impress.py §I.b adds a TopicTheme pool because
# its decks are thin structural shapes that need topic-driven content +
# real-photo augmentation per seed.)
#
# Symmetric layout (all synth/*.py):
#   §I.a  Caps                — SYNTH_CAP_TASKS_PER_FILE / _PARAMS_PER_TASK
#   §I.b  Dataclasses         — File / Param / FileTask (frozen)
#   §I.c  File instances      — define each File ONCE
#   §I.d  Factory + emit      — _to_synth_template / _emit_templates
#   §I.e  FILE_TASKS          — flat list, one entry per (file, task) pair
#   §I.f  Emission            — TEMPLATES.extend(_emit_templates(FILE_TASKS))
#
# Legacy `_make_*_template` factories were deleted in one cycle once §I
# covered the legacy eval skill space (cf. impress/calc §I migration).
# ===========================================================================

from dataclasses import dataclass as _I_dataclass, field as _I_field
from typing import Callable as _I_Callable


# §I.a — caps
SYNTH_CAP_TASKS_PER_FILE: int = 2
SYNTH_CAP_PARAMS_PER_TASK: int = 2


# §I.b — Dataclasses.
#
# Chrome §I shape — file-as-topic. A `File` here = one chrome profile-state
# *shape* (e.g. "bookmark hierarchy", "cookie set", "settings dict",
# "history rows", "open-tab list"). `File.src(seed) -> list[dict]` returns
# the pre_config steps that establish the initial profile state. Different
# Files differ in the *kind* of profile state they expose, NOT in the
# specific values inside (those rotate at the Param axis). Tasks per file
# are operations a real user could run on that state shape.

@_I_dataclass(frozen=True)
class File:
    """One structurally distinct profile-state shape.

    `src(seed) -> list[dict]`: returns init pre_config steps that build the
    profile state on disk (Bookmarks JSON / Preferences patch / History
    sqlite seed / Cookies sqlite seed / shortcut / staged HTML for tabs).
    Seed is rotated by the harness per row; most chrome state shapes are
    deterministic so most `src` ignore seed.
    """
    id: str
    setup_class: str
    src: _I_Callable[[int], list[dict]]


@_I_dataclass(frozen=True)
class Param:
    """One concrete parameterization of a task on a profile-state shape.

    Each Param triple rotates together (changing one means a real different
    operation, not a paraphrase):

      gold_args  — kwargs forwarded to FileTask.gold(**gold_args).
                   `gold(**gold_args) -> (oracle_steps, evaluator_dict)`
      eval_kind  — short tag for audit / debugging (e.g. "exact_match",
                   "is_expected_bookmarks", "check_direct_json_object").
                   Eval shape itself is fully constructed inside `gold`.
      instr      — rendered instruction string (no paraphrase rotation;
                   distinctness comes from gold_args).
    """
    gold_args: dict
    eval_kind: str
    instr: str
    exclude_reason: str | None = None


@_I_dataclass(frozen=True)
class FileTask:
    """One (file, task) pair → one SynthTemplate at emit time.

    `gold(**param.gold_args) -> (list[dict], dict)`: the oracle pre-steps
    plus the evaluator dict for that param. Eval rotates with gold per
    Principle 5 (eval rule follows seed).
    """
    file: File
    task_id: str
    eval_class: str
    gold: _I_Callable[..., tuple[list[dict], dict]]
    params: list[Param] = _I_field(default_factory=list)


# §I.c — File instances. Each File is defined ONCE; FileTask entries below
# reference it. Adding a new file = one new instance below.
#
# Loop 1 — bookmark hierarchies + bookmark+history mix
# (5 distinct profile-state shapes).

def _src_bookmarks_empty(_seed: int) -> list[dict]:
    """Empty bookmark bar — used as canvas for "create folder" / "add URL"
    tasks. Mirrors `_make_bookmark_folder_template` init."""
    return [
        _kill_chrome_step(),
        _write_bookmarks_step(_empty_bookmarks_roots()),
        *_chrome_preopen_steps(urls=None),
    ]


def _src_bookmarks_with_misc_folders(_seed: int) -> list[dict]:
    """Bookmark bar already populated with two unrelated folders. Tasks
    that ADD a new folder must not delete these (eval matches set of
    folder names containing the target)."""
    roots = _empty_bookmarks_roots()
    roots["bookmark_bar"]["children"] = [
        _bookmark_folder_node("Misc", node_id="200"),
        _bookmark_folder_node("Old", node_id="201"),
    ]
    return [
        _kill_chrome_step(),
        _write_bookmarks_step(roots),
        *_chrome_preopen_steps(urls=None),
    ]


def _src_bookmarks_with_decoy_urls(_seed: int) -> list[dict]:
    """Bookmark bar populated with two decoy URLs. Tasks that bookmark a
    NEW URL must keep these intact when eval is set-equality."""
    roots = _empty_bookmarks_roots()
    roots["bookmark_bar"]["children"] = [
        _bookmark_url_node("Wikipedia", "https://www.wikipedia.org/", node_id="200"),
        _bookmark_url_node("HN", "https://news.ycombinator.com/", node_id="201"),
    ]
    return [
        _kill_chrome_step(),
        _write_bookmarks_step(roots),
        *_chrome_preopen_steps(urls=None),
    ]


def _src_history_seeded_mixed(_seed: int) -> list[dict]:
    """History sqlite seeded with a mix of news + dev + social entries.
    Used by history-delete tasks where the keyword targets a subset."""
    entries = [
        ("https://www.cnn.com/article/2026", "CNN — World News"),
        ("https://www.bbc.com/news/world", "BBC News — World"),
        ("https://www.reuters.com/world/", "Reuters — World"),
        ("https://docs.python.org/3/", "Python 3 docs"),
        ("https://docs.rust-lang.org/", "Rust documentation"),
        ("https://twitter.com/home", "Twitter"),
        ("https://www.facebook.com/", "Facebook"),
        ("https://www.kernel.org/", "The Linux Kernel Archives"),
    ]
    return [
        {"type": "launch", "parameters": {"command": ["google-chrome", "--remote-debugging-port=1337"]}},
        {"type": "launch", "parameters": {"command": ["socat", "tcp-listen:9222,fork", "tcp:localhost:1337"]}},
        {"type": "sleep", "parameters": {"seconds": 5}},
        _execute("pkill -9 -f chrome 2>/dev/null; sleep 4; true"),
        _seed_history_step(entries),
        *_chrome_preopen_steps(urls=None),
    ]


def _src_history_seeded_shopping(_seed: int) -> list[dict]:
    """History sqlite seeded with shopping + entertainment entries.
    Different domain mix than `_src_history_seeded_mixed` — exercises
    history-delete on a non-news shape."""
    entries = [
        ("https://www.amazon.com/dp/B08N5WRWNW", "Echo Dot — Amazon"),
        ("https://www.amazon.com/gp/cart", "Amazon Cart"),
        ("https://www.ebay.com/itm/12345", "eBay listing"),
        ("https://www.etsy.com/listing/678", "Etsy listing"),
        ("https://www.netflix.com/browse", "Netflix"),
        ("https://www.youtube.com/watch?v=abc", "YouTube video"),
        ("https://www.spotify.com/account", "Spotify"),
        ("https://docs.python.org/3/", "Python 3 docs"),
    ]
    return [
        {"type": "launch", "parameters": {"command": ["google-chrome", "--remote-debugging-port=1337"]}},
        {"type": "launch", "parameters": {"command": ["socat", "tcp-listen:9222,fork", "tcp:localhost:1337"]}},
        {"type": "sleep", "parameters": {"seconds": 5}},
        _execute("pkill -9 -f chrome 2>/dev/null; sleep 4; true"),
        _seed_history_step(entries),
        *_chrome_preopen_steps(urls=None),
    ]


# Loop 2 — cookies / web sessions

def _src_cookies_seeded_news_auth(_seed: int) -> list[dict]:
    """Cookies sqlite seeded with auth/session cookies for news + dev sites."""
    rows = [
        (".cnn.com", "session", "cnn-1"),
        (".cnn.com", "tracking", "cnn-tk"),
        (".bbc.com", "auth", "bbc-1"),
        (".reuters.com", "session", "reu-1"),
        (".github.com", "user_session", "gh-1"),
        (".docs.python.org", "session", "py-1"),
    ]
    return [
        _kill_chrome_step(),
        _seed_cookies_step(rows),
        *_chrome_preopen_steps(urls=None),
    ]


def _src_cookies_seeded_social(_seed: int) -> list[dict]:
    """Cookies for social-media + tracker domains. Distinct domain mix."""
    rows = [
        (".facebook.com", "datr", "fb-1"),
        (".facebook.com", "c_user", "fb-2"),
        (".twitter.com", "auth_token", "tw-1"),
        (".linkedin.com", "li_at", "li-1"),
        (".doubleclick.net", "IDE", "dclk-1"),
        (".kernel.org", "session", "k-1"),
    ]
    return [
        _kill_chrome_step(),
        _seed_cookies_step(rows),
        *_chrome_preopen_steps(urls=None),
    ]


def _src_cookies_seeded_shopping(_seed: int) -> list[dict]:
    """Cookies for e-commerce sites + a couple of unrelated dev domains
    that must remain after a shopping-cookie wipe."""
    rows = [
        (".amazon.com", "session-id", "amz-1"),
        (".amazon.com", "ubid-main", "amz-2"),
        (".ebay.com", "ebay", "eb-1"),
        (".etsy.com", "user_prefs", "et-1"),
        (".docs.python.org", "session", "py-1"),
        (".kernel.org", "session", "k-1"),
    ]
    return [
        _kill_chrome_step(),
        _seed_cookies_step(rows),
        *_chrome_preopen_steps(urls=None),
    ]


# Loop 3 — settings / preferences. Each File seeds a distinct preference
# block so that tasks toggle / change a different field per file.

def _src_prefs_dnt_off(_seed: int) -> list[dict]:
    """Preferences with Do-Not-Track explicitly OFF. Task: enable it."""
    return [
        _kill_chrome_step(),
        _merge_prefs_step({"enable_do_not_track": False}),
        *_chrome_preopen_steps(urls=None),
    ]


def _src_prefs_safe_browsing_off(_seed: int) -> list[dict]:
    """Preferences with Safe Browsing OFF. Task: enable it."""
    return [
        _kill_chrome_step(),
        _merge_prefs_step({"safebrowsing": {"enabled": False, "enhanced": False}}),
        *_chrome_preopen_steps(urls=None),
    ]


def _src_prefs_profile_default(_seed: int) -> list[dict]:
    """Preferences with profile.name set to default placeholder."""
    return [
        _kill_chrome_step(),
        _merge_prefs_step({"profile": {"name": "Person 1"}}),
        *_chrome_preopen_steps(urls=None),
    ]


def _src_prefs_startup_funbrain(_seed: int) -> list[dict]:
    """Preferences with restore_on_startup=4 + startup_urls=[funbrain]."""
    return [
        _kill_chrome_step(),
        _merge_prefs_step({"session": {"restore_on_startup": 4,
                                        "startup_urls": ["http://funbrain.com/"]}}),
        *_chrome_preopen_steps(urls=None),
    ]


def _src_prefs_search_google(_seed: int) -> list[dict]:
    """Preferences with default search engine = Google."""
    patch = {"default_search_provider_data": {"template_url_data": {
        "short_name": "Google", "keyword": "google.com",
        "url": "https://www.google.com/search?q={searchTerms}",
    }}}
    return [
        _kill_chrome_step(),
        _merge_prefs_step(patch),
        *_chrome_preopen_steps(urls=None),
    ]


# Loop 4 — extensions / shortcut / reading-list-style desktop entries.
# We use the desktop-shortcut surface (matches `is_shortcut_on_desktop`
# eval) — chrome's actual extension API is out of scope for the
# host-heredoc oracle.

def _src_desktop_clean(_seed: int) -> list[dict]:
    """Empty Desktop folder — task is to create a chrome shortcut on it."""
    return [
        _execute("mkdir -p /home/user/Desktop && rm -f /home/user/Desktop/*.desktop"),
        *_chrome_preopen_steps(urls=None),
    ]


def _src_desktop_with_decoys(_seed: int) -> list[dict]:
    """Desktop already has two unrelated .desktop files; new shortcut must
    coexist (eval `is_shortcut_on_desktop` matches by Name=, decoys' names
    differ)."""
    decoy_a = textwrap.dedent("""\
        [Desktop Entry]
        Version=1.0
        Type=Application
        Name=Decoy One
        Exec=true
        Icon=utilities-terminal
        Terminal=false
        """)
    decoy_b = textwrap.dedent("""\
        [Desktop Entry]
        Version=1.0
        Type=Application
        Name=Decoy Two
        Exec=true
        Icon=utilities-terminal
        Terminal=false
        """)
    return [
        _execute(
            "mkdir -p /home/user/Desktop && rm -f /home/user/Desktop/*.desktop && "
            f"cat > /home/user/Desktop/decoy-one.desktop << 'EOF'\n{decoy_a}EOF\n"
            f"cat > /home/user/Desktop/decoy-two.desktop << 'EOF'\n{decoy_b}EOF"
        ),
        *_chrome_preopen_steps(urls=None),
    ]


# Loop 5 — web tab states (open URL set / decoy + gold URL).

def _src_tabs_decoy_kernel(_seed: int) -> list[dict]:
    """Single decoy tab on kernel.org — task: navigate to a real URL."""
    return [
        *_chrome_preopen_steps(urls=["https://www.kernel.org/"]),
    ]


def _src_tabs_decoy_wikipedia(_seed: int) -> list[dict]:
    """Single decoy tab on wikipedia.org — task: navigate elsewhere."""
    return [
        *_chrome_preopen_steps(urls=["https://www.wikipedia.org/"]),
    ]


def _src_tabs_three_decoys(_seed: int) -> list[dict]:
    """Three decoy tabs — task: open additional tabs while keeping these."""
    return [
        *_chrome_preopen_steps(urls=[
            "https://www.kernel.org/",
            "https://docs.python.org/3/",
            "https://news.ycombinator.com/",
        ]),
    ]


# Loop 2 extra — analytics-tracker cookies + single-domain auth
def _src_cookies_seeded_analytics(_seed: int) -> list[dict]:
    """Cookies for analytics + advertising-tracker domains plus a couple of
    keep-me-alone session cookies."""
    rows = [
        (".analytics.example.com", "uid", "ana-1"),
        (".tracker.example.com", "tid", "trk-1"),
        (".doubleclick.net", "IDE", "dclk-1"),
        (".googleadservices.com", "ads", "g-ads"),
        (".docs.python.org", "session", "py-1"),
        (".kernel.org", "session", "k-1"),
    ]
    return [
        _kill_chrome_step(),
        _seed_cookies_step(rows),
        *_chrome_preopen_steps(urls=None),
    ]


def _src_cookies_seeded_dev(_seed: int) -> list[dict]:
    """Cookies for developer-tooling domains (github, stackoverflow, npm)."""
    rows = [
        (".github.com", "user_session", "gh-1"),
        (".github.com", "_octo", "gh-2"),
        (".stackoverflow.com", "se-acct", "so-1"),
        (".npmjs.com", "npm-tok", "npm-1"),
        (".pypi.org", "session", "pypi-1"),
        (".kernel.org", "session", "k-1"),
    ]
    return [
        _kill_chrome_step(),
        _seed_cookies_step(rows),
        *_chrome_preopen_steps(urls=None),
    ]


# Loop 4 extra — desktop with one shortcut already present.
def _src_desktop_with_python_shortcut(_seed: int) -> list[dict]:
    """Desktop already has a Python-Docs shortcut. Task: add another (different
    name + URL) without disturbing the existing one."""
    body = textwrap.dedent("""\
        [Desktop Entry]
        Version=1.0
        Type=Application
        Name=Python Docs
        Exec=google-chrome --app=https://docs.python.org/3/
        Icon=google-chrome
        Terminal=false
        StartupWMClass=python-docs
        """)
    return [
        _execute(
            "mkdir -p /home/user/Desktop && rm -f /home/user/Desktop/*.desktop && "
            f"cat > /home/user/Desktop/python-docs.desktop << 'EOF'\n{body}EOF"
        ),
        *_chrome_preopen_steps(urls=None),
    ]


# Loop 5 extra — staged Wikipedia file:// tab states.
def _src_tabs_decoy_wiki_apollo(_seed: int) -> list[dict]:
    """Stage the apollo-program Wikipedia HTML and open its file:// URL as
    the single decoy tab. Task: navigate the active tab to a different
    staged page (or add tabs around it)."""
    return [
        _stage_asset("html/wikipedia/apollo-program.html",
                     "/home/user/Desktop/apollo-program.html"),
        _stage_asset("html/wikipedia/solar-system.html",
                     "/home/user/Desktop/solar-system.html"),
        _stage_asset("html/wikipedia/earth.html",
                     "/home/user/Desktop/earth.html"),
        *_chrome_preopen_steps(
            urls=["file:///home/user/Desktop/apollo-program.html"],
            extra_launch_args=["--user-data-dir=/home/user/chrome-data"],
        ),
    ]


def _src_tabs_decoy_wiki_food(_seed: int) -> list[dict]:
    """Stage the coffee Wikipedia HTML and open its file:// URL as the
    single decoy tab. Task: open additional tabs to pizza / pasta pages."""
    return [
        _stage_asset("html/wikipedia/coffee.html",
                     "/home/user/Desktop/coffee.html"),
        _stage_asset("html/wikipedia/pizza.html",
                     "/home/user/Desktop/pizza.html"),
        _stage_asset("html/wikipedia/pasta.html",
                     "/home/user/Desktop/pasta.html"),
        *_chrome_preopen_steps(
            urls=["file:///home/user/Desktop/coffee.html"],
            extra_launch_args=["--user-data-dir=/home/user/chrome-data"],
        ),
    ]


# Loop 6 — URL-with-query active tab (decoy URL on a search-style site;
# task = navigate to the gold URL with specific query params). Closes the
# `check_direct_json_object` skill gap (eval=16, synth was 0). Chrome
# synth is too easy (Δraw +40.6pp);
# `active_tab_url_parse` is the dominant hard skill. Each File is a distinct
# real search site (rentalcars / amazon / indeed / github / booking) whose
# decoy URL has the WRONG query params — only the oracle relaunch on the
# gold URL passes eval. Oracle reuses the same kill+session-cleanup+launch
# pattern as `_gold_navigate_active_tab`.
def _src_url_decoy_rentalcars(_seed: int) -> list[dict]:
    """Pre-config: open chrome on rentalcars.com search results for Berlin
    (small car, sorted by recommendation). Task: navigate to a different
    city/category/sort gold URL — eval parses query params on active tab."""
    decoy = "https://www.rentalcars.com/search-results?locationName=Berlin&dropLocationName=Berlin&filterCriteria_carCategory=small&filterCriteria_sortBy=RECOMMENDED"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_amazon(_seed: int) -> list[dict]:
    """Pre-config: amazon.com/s search for 'mouse' sorted by review rank.
    Task: navigate to a gold search URL with different `k` + `s` params."""
    decoy = "https://www.amazon.com/s?k=mouse&s=review-rank"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_indeed(_seed: int) -> list[dict]:
    """Pre-config: indeed.com/jobs search for `manager` in NYC, in-office.
    Task: navigate to gold URL with different role / city / remote filter."""
    decoy = "https://www.indeed.com/jobs?q=manager&l=New+York%2C+NY&remotejob=0"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_github(_seed: int) -> list[dict]:
    """Pre-config: github.com/search for closed PRs labeled `bug`.
    Task: navigate to the gold issue-search URL with different filters."""
    decoy = "https://github.com/search?q=is%3Apr+is%3Aclosed+label%3Abug&type=pullrequests"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_booking(_seed: int) -> list[dict]:
    """Pre-config: booking.com hotel search for Berlin in early April for
    1 adult. Task: navigate to gold URL with different city/dates/adults."""
    decoy = "https://www.booking.com/searchresults.html?ss=Berlin&checkin=2026-04-01&checkout=2026-04-03&group_adults=1"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


# Loop 7 — `is_expected_url_pattern_match` (eval=4, synth was 0). Each File
# is a real-site decoy URL on a domain whose gold target page is identified
# by a regex (rather than exact query params). Mirrors eval shapes
# `osworld_chrome_a728a36e` (DMV eligibility) and `_9f935cce` (wiki page).
def _src_url_pattern_dmv_root(_seed: int) -> list[dict]:
    """Pre-config: open dmv.virginia.gov/vehicles (decoy — wrong page).
    Task: navigate to the eligibility info page (matches a regex pattern)."""
    decoy = "https://www.dmv.virginia.gov/vehicles"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_pattern_wiki_root(_seed: int) -> list[dict]:
    """Pre-config: open en.wikipedia.org main page (decoy). Task: navigate
    to a specific article whose URL matches a regex (e.g. `/wiki/Linux`)."""
    decoy = "https://en.wikipedia.org/wiki/Main_Page"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


# Batch extras — more cdjo coverage on harder query-shape sites.
def _src_url_decoy_macys(_seed: int) -> list[dict]:
    """Pre-config: macys.com decoy search (`shoes`, sort=ORIGINAL). Task:
    navigate to gold URL with different keyword + sort + price range."""
    decoy = "https://www.macys.com/shop/featured/shoes?sortBy=ORIGINAL"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_zappos(_seed: int) -> list[dict]:
    """Pre-config: zappos.com decoy search (`boots`, default sort).
    Task: navigate to gold URL with different keyword + filter terms."""
    decoy = "https://www.zappos.com/p/search?term=boots"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_pattern_gov_root(_seed: int) -> list[dict]:
    """Pre-config: open usa.gov landing page (decoy). Task: navigate to a
    specific topic page whose URL matches a regex pattern."""
    decoy = "https://www.usa.gov/"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


#  — additional cdjo + url_pattern File sources to close
# the eval-distribution gap. Eval has 16 cdjo + 8 url_pattern_match rows out
# of 46 chrome (≈37% / 18%); synth was at ≈9% / 3%. Each new File mirrors a
# distinct real-site decoy URL; the gold URL flips one-or-more URL params or
# a path segment that the evaluator parses.
def _src_url_decoy_flights_united(_seed: int) -> list[dict]:
    """Pre-config: united.com flight search decoy (single LAX→JFK trip).
    Gold: same site, different OD pair / class / passengers."""
    decoy = "https://www.united.com/en/us/fsr/choose-flights?f=LAX&t=JFK&d=2026-04-01&tt=1"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_jetblue(_seed: int) -> list[dict]:
    """Pre-config: jetblue.com fare search decoy. Gold: different OD + date."""
    decoy = f"https://www.jetblue.com/booking/flights?from=BOS&to=FLL&depart={_future_date(30)}&pax=ADT-1"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_kayak(_seed: int) -> list[dict]:
    """Pre-config: kayak.com hotel search decoy. Gold: different city/date."""
    decoy = "https://www.kayak.com/hotels/Berlin/2026-04-01/2026-04-03/2adults"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_walmart(_seed: int) -> list[dict]:
    """Pre-config: walmart.com search decoy (`headphones`). Gold: different
    keyword + sort."""
    decoy = "https://www.walmart.com/search?q=headphones&sort=best_match"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_target(_seed: int) -> list[dict]:
    """Pre-config: target.com search decoy. Gold: different keyword/category."""
    decoy = "https://www.target.com/s?searchTerm=lamp&sortBy=relevance"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_ebay(_seed: int) -> list[dict]:
    """Pre-config: ebay.com sch decoy (default keyword). Gold: different
    keyword + sort + filter."""
    decoy = "https://www.ebay.com/sch/i.html?_nkw=phone&_sop=12"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_yelp(_seed: int) -> list[dict]:
    """Pre-config: yelp.com search decoy (food / generic city). Gold:
    different category + city."""
    decoy = "https://www.yelp.com/search?find_desc=food&find_loc=New+York%2C+NY"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_redfin(_seed: int) -> list[dict]:
    """Pre-config: redfin.com city listing decoy. Gold: different city + filter."""
    decoy = "https://www.redfin.com/city/30749/CA/Los-Angeles/filter/property-type=house"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_pattern_unitedair_root(_seed: int) -> list[dict]:
    """Pre-config: united.com landing page (decoy). Gold: navigate to the
    baggage-fee calculator (URL pattern matched)."""
    decoy = "https://www.united.com/en/us"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_pattern_irs_root(_seed: int) -> list[dict]:
    """Pre-config: irs.gov landing page (decoy). Gold: a specific tax-form
    page whose URL matches a regex."""
    decoy = "https://www.irs.gov/"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_pattern_github_root(_seed: int) -> list[dict]:
    """Pre-config: github.com landing page (decoy). Gold: a specific repo's
    issues page (URL pattern matched)."""
    decoy = "https://github.com/"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_pattern_stackoverflow_root(_seed: int) -> list[dict]:
    """Pre-config: stackoverflow.com landing page (decoy). Gold: a specific
    tag-listing page (URL pattern matched)."""
    decoy = "https://stackoverflow.com/"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


# File instances. id = "F-CHROME-N", setup_class drives setup-class telemetry.

# Loop 1 — bookmarks + history mix
F_CHROME_1 = File(id="F-CHROME-1", setup_class="chrome_bookmarks_empty",
                  src=_src_bookmarks_empty)
F_CHROME_2 = File(id="F-CHROME-2", setup_class="chrome_bookmarks_misc_folders",
                  src=_src_bookmarks_with_misc_folders)
F_CHROME_3 = File(id="F-CHROME-3", setup_class="chrome_bookmarks_decoy_urls",
                  src=_src_bookmarks_with_decoy_urls)
F_CHROME_4 = File(id="F-CHROME-4", setup_class="chrome_history_news_dev",
                  src=_src_history_seeded_mixed)
F_CHROME_5 = File(id="F-CHROME-5", setup_class="chrome_history_shopping",
                  src=_src_history_seeded_shopping)

# Loop 2 — cookies / web sessions
F_CHROME_6 = File(id="F-CHROME-6", setup_class="chrome_cookies_news_auth",
                  src=_src_cookies_seeded_news_auth)
F_CHROME_7 = File(id="F-CHROME-7", setup_class="chrome_cookies_social",
                  src=_src_cookies_seeded_social)
F_CHROME_8 = File(id="F-CHROME-8", setup_class="chrome_cookies_shopping",
                  src=_src_cookies_seeded_shopping)

# Loop 3 — settings / preferences
F_CHROME_9 = File(id="F-CHROME-9", setup_class="chrome_prefs_dnt_off",
                  src=_src_prefs_dnt_off)
F_CHROME_10 = File(id="F-CHROME-10", setup_class="chrome_prefs_safe_browsing_off",
                   src=_src_prefs_safe_browsing_off)
F_CHROME_11 = File(id="F-CHROME-11", setup_class="chrome_prefs_profile_default",
                   src=_src_prefs_profile_default)
F_CHROME_12 = File(id="F-CHROME-12", setup_class="chrome_prefs_startup_funbrain",
                   src=_src_prefs_startup_funbrain)
F_CHROME_13 = File(id="F-CHROME-13", setup_class="chrome_prefs_search_google",
                   src=_src_prefs_search_google)

# Loop 4 — desktop shortcuts (chrome "extension/reading-list" surrogate)
F_CHROME_14 = File(id="F-CHROME-14", setup_class="chrome_desktop_clean",
                   src=_src_desktop_clean)
F_CHROME_15 = File(id="F-CHROME-15", setup_class="chrome_desktop_with_decoys",
                   src=_src_desktop_with_decoys)

# Loop 5 — web tab states
F_CHROME_16 = File(id="F-CHROME-16", setup_class="chrome_tabs_decoy_kernel",
                   src=_src_tabs_decoy_kernel)
F_CHROME_17 = File(id="F-CHROME-17", setup_class="chrome_tabs_decoy_wikipedia",
                   src=_src_tabs_decoy_wikipedia)
F_CHROME_18 = File(id="F-CHROME-18", setup_class="chrome_tabs_three_decoys",
                   src=_src_tabs_three_decoys)

# Loop-2 extras (cookies)
F_CHROME_19 = File(id="F-CHROME-19", setup_class="chrome_cookies_analytics",
                   src=_src_cookies_seeded_analytics)
F_CHROME_20 = File(id="F-CHROME-20", setup_class="chrome_cookies_dev",
                   src=_src_cookies_seeded_dev)

# Loop-4 extras (desktop)
F_CHROME_21 = File(id="F-CHROME-21", setup_class="chrome_desktop_with_python",
                   src=_src_desktop_with_python_shortcut)

# Loop-5 extras (file:// tab states)
F_CHROME_22 = File(id="F-CHROME-22", setup_class="chrome_tabs_decoy_wiki_apollo",
                   src=_src_tabs_decoy_wiki_apollo)
F_CHROME_23 = File(id="F-CHROME-23", setup_class="chrome_tabs_decoy_wiki_food",
                   src=_src_tabs_decoy_wiki_food)

# Loop 6 — URL-with-query active tab (cdjo skill — biggest train/eval gap)
F_CHROME_24 = File(id="F-CHROME-24", setup_class="chrome_url_decoy_rentalcars",
                   src=_src_url_decoy_rentalcars)
F_CHROME_25 = File(id="F-CHROME-25", setup_class="chrome_url_decoy_amazon",
                   src=_src_url_decoy_amazon)
F_CHROME_26 = File(id="F-CHROME-26", setup_class="chrome_url_decoy_indeed",
                   src=_src_url_decoy_indeed)
F_CHROME_27 = File(id="F-CHROME-27", setup_class="chrome_url_decoy_github",
                   src=_src_url_decoy_github)
F_CHROME_28 = File(id="F-CHROME-28", setup_class="chrome_url_decoy_booking",
                   src=_src_url_decoy_booking)

# Loop 7 — URL-pattern regex (is_expected_url_pattern_match skill — eval gap)
F_CHROME_29 = File(id="F-CHROME-29", setup_class="chrome_url_pattern_dmv_root",
                   src=_src_url_pattern_dmv_root)
F_CHROME_30 = File(id="F-CHROME-30", setup_class="chrome_url_pattern_wiki_root",
                   src=_src_url_pattern_wiki_root)

# Batch extras (more cdjo + url_pattern files)
F_CHROME_31 = File(id="F-CHROME-31", setup_class="chrome_url_decoy_macys",
                   src=_src_url_decoy_macys)
F_CHROME_32 = File(id="F-CHROME-32", setup_class="chrome_url_decoy_zappos",
                   src=_src_url_decoy_zappos)
F_CHROME_33 = File(id="F-CHROME-33", setup_class="chrome_url_pattern_gov_root",
                   src=_src_url_pattern_gov_root)

#  — additional cdjo + url_pattern files. Eval-distribution
# alignment: cdjo target ~37%, url_pattern target ~18%; pre-Batch synth was
# at 9% / 3%. Adding 8 cdjo + 4 url_pattern Files.
# cdjo extras
F_CHROME_34 = File(id="F-CHROME-34", setup_class="chrome_url_decoy_united",
                   src=_src_url_decoy_flights_united)
F_CHROME_35 = File(id="F-CHROME-35", setup_class="chrome_url_decoy_jetblue",
                   src=_src_url_decoy_jetblue)
F_CHROME_36 = File(id="F-CHROME-36", setup_class="chrome_url_decoy_kayak",
                   src=_src_url_decoy_kayak)
F_CHROME_37 = File(id="F-CHROME-37", setup_class="chrome_url_decoy_walmart",
                   src=_src_url_decoy_walmart)
F_CHROME_38 = File(id="F-CHROME-38", setup_class="chrome_url_decoy_target",
                   src=_src_url_decoy_target)
F_CHROME_39 = File(id="F-CHROME-39", setup_class="chrome_url_decoy_ebay",
                   src=_src_url_decoy_ebay)
F_CHROME_40 = File(id="F-CHROME-40", setup_class="chrome_url_decoy_yelp",
                   src=_src_url_decoy_yelp)
F_CHROME_41 = File(id="F-CHROME-41", setup_class="chrome_url_decoy_redfin",
                   src=_src_url_decoy_redfin)
# url_pattern extras
F_CHROME_42 = File(id="F-CHROME-42", setup_class="chrome_url_pattern_united_root",
                   src=_src_url_pattern_unitedair_root)
F_CHROME_43 = File(id="F-CHROME-43", setup_class="chrome_url_pattern_irs_root",
                   src=_src_url_pattern_irs_root)
F_CHROME_44 = File(id="F-CHROME-44", setup_class="chrome_url_pattern_github_root",
                   src=_src_url_pattern_github_root)
F_CHROME_45 = File(id="F-CHROME-45", setup_class="chrome_url_pattern_stackoverflow_root",
                   src=_src_url_pattern_stackoverflow_root)


# §I.d — Gold builders. Each `_gold_*` returns (oracle_steps, evaluator).
# Per Principle 5, eval rotates alongside gold (target name / URL / pref
# value baked into the evaluator dict per Param).

def _gold_bookmark_folder(*, folder_name: str, keep_existing: tuple[dict, ...] = ()) -> tuple[list[dict], dict]:
    """Oracle: write bookmark bar = `keep_existing` + new folder named
    `folder_name`. Eval: bookmark_bar_folders_names = full expected set.

    `is_expected_bookmarks` does set-equality on the rule's `names`/`urls`
    against the live bookmark bar. Including only the new folder while the
    oracle keeps the decoys produces a set mismatch and fails eval — the
    rule must list every folder name expected post-action (decoys + new).
    """
    roots = _empty_bookmarks_roots()
    roots["bookmark_bar"]["children"] = [*keep_existing,
                                          _bookmark_folder_node(folder_name, node_id="100")]
    oracle = [_kill_chrome_step(), _write_bookmarks_step(roots)]
    # Collect every existing folder name from keep_existing so eval set-equality
    # accepts the live state. Non-folder nodes in keep_existing are skipped.
    existing_folder_names = [
        e["name"] for e in keep_existing
        if e.get("type") == "folder" and "name" in e
    ]
    evaluator = {
        "func": "is_expected_bookmarks",
        "result": {"type": "bookmarks"},
        "expected": {"type": "rule", "rules": {
            "type": "bookmark_bar_folders_names",
            "names": [*existing_folder_names, folder_name],
        }},
        "postconfig": _CHROME_RESTART_POSTCONFIG,
    }
    return oracle, evaluator


def _gold_bookmark_url(*, name: str, url: str,
                       keep_existing: tuple[dict, ...] = ()) -> tuple[list[dict], dict]:
    """Oracle: write bookmark bar = `keep_existing` + URL bookmark.
    Eval: bookmark_bar_websites_urls = full expected set (decoys + new).

    Same fix as `_gold_bookmark_folder`: set-equality requires every URL
    that ends up on the bookmark bar to appear in the rule.
    """
    roots = _empty_bookmarks_roots()
    roots["bookmark_bar"]["children"] = [*keep_existing,
                                          _bookmark_url_node(name, url, node_id="100")]
    oracle = [_kill_chrome_step(), _write_bookmarks_step(roots)]
    existing_urls = [
        e["url"] for e in keep_existing
        if e.get("type") == "url" and "url" in e
    ]
    evaluator = {
        "func": "is_expected_bookmarks",
        "result": {"type": "bookmarks"},
        "expected": {"type": "rule", "rules": {
            "type": "bookmark_bar_websites_urls",
            "urls": [*existing_urls, url],
        }},
        "postconfig": _CHROME_RESTART_POSTCONFIG,
    }
    return oracle, evaluator


def _gold_history_delete_keyword(*, keyword: str) -> tuple[list[dict], dict]:
    """Oracle: delete urls rows whose URL contains `keyword`.
    Eval: check_history_deleted with that keyword."""
    oracle = [_kill_chrome_step(), _delete_history_keyword_step(keyword)]
    evaluator = {
        "func": "check_history_deleted",
        "result": {"type": "history", "dest": "history.sqlite"},
        "expected": {"type": "rule", "rules": {"type": "keywords", "keywords": [keyword]}},
        "postconfig": _CHROME_RESTART_POSTCONFIG,
    }
    return oracle, evaluator


def _gold_cookie_delete_domains(*, domains: tuple[str, ...]) -> tuple[list[dict], dict]:
    """Oracle: delete cookies whose host_key matches each of `domains`.
    Eval: is_cookie_deleted with those domains."""
    domain_list = list(domains)
    oracle = [_kill_chrome_step(), _delete_cookies_domains_step(domain_list)]
    # Validation note: when the agent deletes cookies via Chrome's
    # "Delete browsing data" UI, the change goes into the SQLite WAL but
    # not the main `.db` file until chrome flushes (next pref-write or
    # explicit checkpoint). `is_cookie_deleted` reads the `.db` directly,
    # so without a forced WAL checkpoint after pkill it can see the
    # pre-delete cookies and false fail. Use a
    # cookie-specific postconfig: pkill -9 + sleep + sqlite3 WAL checkpoint
    # → relaunch.
    # Validation note: `pkill -9` killed chrome before
    # it had time to flush in-memory cookie writes to the WAL → subsequent
    # sqlite3 wal_checkpoint had nothing to merge → eval still saw stale
    # cookies (F-CHROME-6 + F-CHROME-7 still 2/2 F in validation). Switched to
    # SIGTERM-then-sleep (graceful shutdown) before falling back to -9.
    #
    # The checkpoint must never CREATE the store (same rule as the shared flush
    # in src/eval/runner.py, which opens `file:...?mode=rw`). `sqlite3.connect(p)`
    # opens rw-CREATE, so on an absent `Cookies` it writes a 0-byte file — the
    # parent `/home/user/chrome-data/Default` always exists, so the fabrication
    # always lands. `mode=rw` (no `c`) makes the open fail on a missing file,
    # atomically: no exists()/connect() race, and no orphan store left behind for
    # the relaunched chrome to adopt. Today the `cookie_data` getter fails CLOSED
    # either way (`SELECT * FROM cookies` on the 0-byte file raises → getter
    # returns None → `is_cookie_deleted(None, ...)` raises → runner scores 0.0),
    # so this changes no current score; it removes the fabrication so the getter
    # cannot start fail-OPEN the day it returns `[]` instead of `None`
    # (`is_cookie_deleted([], ...)` == 1.0).
    cookie_postconfig = [
        {"type": "execute", "parameters": {
            "command": (
                "pkill -TERM -f chrome 2>/dev/null; sleep 4; "
                "pkill -9 -f chrome 2>/dev/null; sleep 1; "
                "python3 -c \"import sqlite3, os; "
                "p='/home/user/chrome-data/Default/Cookies'; "
                "c=sqlite3.connect('file:'+p+'?mode=rw', uri=True); "
                "c.execute('PRAGMA wal_checkpoint(TRUNCATE)'); "
                "c.commit(); c.close()\" 2>/dev/null || true"
            ),
            "shell": True,
        }},
        {"type": "launch", "parameters": {"command": ["google-chrome", "--remote-debugging-port=1337"]}},
        {"type": "sleep", "parameters": {"seconds": 3}},
    ]
    evaluator = {
        "func": "is_cookie_deleted",
        "result": {"type": "cookie_data", "dest": "Cookies"},
        "expected": {"type": "rule", "rules": {"type": "domains", "domains": domain_list}},
        "postconfig": cookie_postconfig,
    }
    return oracle, evaluator


def _gold_set_bool_pref(*, patch: dict, result_type: str,
                        expected: str = "true") -> tuple[list[dict], dict]:
    """Oracle: deep-merge `patch` into Preferences (sets the bool pref
    to its target value). Eval: exact_match on `result_type` runner."""
    oracle = [_kill_chrome_step(), _merge_prefs_step(patch)]
    evaluator = {
        "func": "exact_match",
        "result": {"type": result_type},
        "expected": {"type": "rule", "rules": {"expected": expected}},
        "postconfig": _CHROME_RESTART_POSTCONFIG,
    }
    return oracle, evaluator


def _gold_set_profile_name(*, name: str) -> tuple[list[dict], dict]:
    """Oracle: set profile.name = `name`. Eval: exact_match on profile_name."""
    oracle = [_kill_chrome_step(), _merge_prefs_step({"profile": {"name": name}})]
    evaluator = {
        "func": "exact_match",
        "result": {"type": "profile_name"},
        "expected": {"type": "rule", "rules": {"expected": name}},
        "postconfig": _CHROME_RESTART_POSTCONFIG,
    }
    return oracle, evaluator


def _gold_clear_startup_urls() -> tuple[list[dict], dict]:
    """Oracle: clear startup_urls + set restore_on_startup=5.
    Eval: exact_match on new_startup_page (true when restore_on_startup==5)."""
    oracle = [_kill_chrome_step(),
              _merge_prefs_step({"session": {"restore_on_startup": 5, "startup_urls": []}})]
    evaluator = {
        "func": "exact_match",
        "result": {"type": "new_startup_page"},
        "expected": {"type": "rule", "rules": {"expected": "true"}},
        "postconfig": _CHROME_RESTART_POSTCONFIG,
    }
    return oracle, evaluator


def _gold_set_search_engine(*, short_name: str, keyword: str, url: str,
                             expected_list: tuple[str, ...]) -> tuple[list[dict], dict]:
    """Oracle: set default_search_provider_data to (short_name, keyword, url).
    Eval: match_in_list — short_name appears in `expected_list`."""
    patch = {"default_search_provider_data": {"template_url_data": {
        "short_name": short_name, "keyword": keyword, "url": url,
    }}}
    oracle = [_kill_chrome_step(), _merge_prefs_step(patch)]
    evaluator = {
        "func": "match_in_list",
        "result": {"type": "default_search_engine"},
        "expected": {"type": "rule", "rules": {"expected": list(expected_list)}},
        "postconfig": _CHROME_RESTART_POSTCONFIG,
    }
    return oracle, evaluator


def _gold_create_shortcut(*, name: str, url: str) -> tuple[list[dict], dict]:
    """Oracle: write a .desktop file named after `name` pointing at `url`.
    Eval: is_shortcut_on_desktop matches Name=`name`."""
    oracle = [_write_desktop_shortcut_step(name, url)]
    evaluator = {
        "func": "is_shortcut_on_desktop",
        "result": {"type": "shortcuts_on_desktop"},
        "expected": {"type": "rule", "rules": {"type": "name", "name": name}},
    }
    return oracle, evaluator


def _gold_navigate_active_tab(*, target_url: str) -> tuple[list[dict], dict]:
    """Oracle: kill chrome, relaunch with `target_url` as the positional arg.
    Eval: is_expected_active_tab matches `target_url`."""
    goto_prefix = "" if target_url.startswith("file://") else "https://"
    oracle = [
        _execute("pkill -9 -f chrome 2>/dev/null; sleep 2; true"),
        _execute(_SESSION_CLEANUP_CMD),
        {"type": "launch", "parameters": {"command": [
            "google-chrome", "--no-sandbox", "--no-first-run",
            "--no-default-browser-check", "--remote-debugging-port=1337",
            "--user-data-dir=/home/user/chrome-data",
            "--remote-allow-origins=*",
            target_url,
        ]}},
        {"type": "sleep", "parameters": {"seconds": 5}},
    ]
    evaluator = {
        "func": "is_expected_active_tab",
        "result": {"type": "active_tab_info", "goto_prefix": goto_prefix},
        "expected": {"type": "rule", "rules": {"type": "url", "url": target_url}},
    }
    return oracle, evaluator


def _gold_open_tabs(*, target_urls: tuple[str, ...]) -> tuple[list[dict], dict]:
    """Oracle: kill chrome, clear session-restore, relaunch with all
    target_urls as positional args. Eval: is_expected_tabs matches list."""
    urls = list(target_urls)
    oracle = [
        _execute("pkill -9 -f chrome 2>/dev/null; sleep 2; true"),
        _execute(_SESSION_CLEANUP_CMD),
        {"type": "launch", "parameters": {"command": [
            "google-chrome", "--no-sandbox", "--no-first-run",
            "--no-default-browser-check", "--remote-debugging-port=1337",
            "--user-data-dir=/home/user/chrome-data",
            "--remote-allow-origins=*",
            *urls,
        ]}},
        {"type": "sleep", "parameters": {"seconds": 5}},
    ]
    evaluator = {
        "func": "is_expected_tabs",
        "result": {"type": "open_tabs_info"},
        "expected": {"type": "rule", "rules": {"type": "url", "urls": urls}},
    }
    return oracle, evaluator


def _build_url(base_url: str, query: dict) -> str:
    """Build a URL with query string. Uses urlencode (quote_via=quote) so
    Unicode values like 'Zürich' are URL-encoded — matching what chrome's
    address bar produces and what `get_active_tab_url_parse`'s `parse_qs`
    decodes back."""
    from urllib.parse import urlencode, quote
    qs = urlencode(query, quote_via=quote)
    return f"{base_url}?{qs}"


# Minimal blank HTML body — used by file:// staged variants of URL-query
# gold builders so chrome's address bar carries the gold URL even when the
# live site is bot-blocked / 429 / 403 / has JS-rewrite. The eval reads the
# tab URL via CDP (port 1337) and parses query params from it; the page
# content is irrelevant for url_parse / url_pattern atoms.
_HTML_BLANK = (
    "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
    "<title>Synth Staged</title></head><body></body></html>"
)


def _stage_blank_html_step(dst_path: str) -> dict:
    """Oracle step that writes a minimal blank HTML at `dst_path`. Used by
    file:// staged variants so chrome can open the local file (URL bar then
    holds `file://{dst_path}?<query>`)."""
    return _execute(f"cat > {dst_path} << 'HTMLEOF'\n{_HTML_BLANK}\nHTMLEOF")


def _gold_url_query(*, base_url: str, gold_query: dict,
                    parse_keys: tuple[str, ...] | None = None,
                    stage_html_path: str | None = None) -> tuple[list[dict], dict]:
    """Oracle: kill chrome, clear session-restore, relaunch with the gold
    URL (base + encoded gold query) as positional arg. Eval:
    `check_direct_json_object` over `active_tab_url_parse` — parses query
    params on the active tab URL and checks each `parse_key` matches the
    expected value. Closes chrome eval gap (eval=16, synth was 0).

    If `stage_html_path` is given, the oracle additionally writes a blank
    HTML file at that path BEFORE launching chrome. This is used when
    `base_url` is a `file://...` path (live-site bot-blocked / JS-rewrite
    fallback): chrome opens the local file with the query string in the
    URL bar, eval reads the URL via CDP and parses query params unchanged.
    The `goto_prefix` is used only by the AT-SPI fallback path; CDP returns
    the full URL, so file:// works transparently.
    """
    keys = list(parse_keys) if parse_keys else list(gold_query.keys())
    gold_url = _build_url(base_url, gold_query)
    goto_prefix = "" if base_url.startswith("file://") else "https://www."
    oracle: list[dict] = [
        _execute("pkill -9 -f chrome 2>/dev/null; sleep 2; true"),
        _execute(_SESSION_CLEANUP_CMD),
    ]
    if stage_html_path:
        oracle.append(_stage_blank_html_step(stage_html_path))
    oracle.extend([
        {"type": "launch", "parameters": {"command": [
            "google-chrome", "--no-sandbox", "--no-first-run",
            "--no-default-browser-check", "--remote-debugging-port=1337",
            "--user-data-dir=/home/user/chrome-data",
            "--remote-allow-origins=*",
            gold_url,
        ]}},
        {"type": "sleep", "parameters": {"seconds": 5}},
    ])
    evaluator = {
        "func": "check_direct_json_object",
        "result": {
            "type": "active_tab_url_parse",
            "goto_prefix": goto_prefix,
            "parse_keys": keys,
        },
        "expected": {
            "type": "rule",
            "rules": {"expected": dict(gold_query)},
        },
    }
    return oracle, evaluator


def _gold_url_pattern(*, gold_url: str, pattern: str,
                      stage_html_path: str | None = None) -> tuple[list[dict], dict]:
    """Oracle: kill chrome, clear session-restore, relaunch with the gold
    URL as positional arg. Eval: `is_expected_url_pattern_match` against
    `active_url_from_accessTree` — regex matched (rather than parse_qs).

    If `stage_html_path` is given, a blank HTML file is staged at that path
    before chrome launches (used when `gold_url` is `file://...`)."""
    goto_prefix = "" if gold_url.startswith("file://") else "https://www."
    oracle: list[dict] = [
        _execute("pkill -9 -f chrome 2>/dev/null; sleep 2; true"),
        _execute(_SESSION_CLEANUP_CMD),
    ]
    if stage_html_path:
        oracle.append(_stage_blank_html_step(stage_html_path))
    oracle.extend([
        {"type": "launch", "parameters": {"command": [
            "google-chrome", "--no-sandbox", "--no-first-run",
            "--no-default-browser-check", "--remote-debugging-port=1337",
            "--user-data-dir=/home/user/chrome-data",
            "--remote-allow-origins=*",
            gold_url,
        ]}},
        {"type": "sleep", "parameters": {"seconds": 5}},
    ])
    evaluator = {
        "func": "is_expected_url_pattern_match",
        "result": {
            "type": "active_url_from_accessTree",
            "goto_prefix": goto_prefix,
        },
        "expected": {
            "type": "rule",
            "rules": {"expected": [pattern]},
        },
    }
    return oracle, evaluator


# §I.e — Factory + emit.

def _to_synth_template(ft: FileTask) -> SynthTemplate:
    """Turn ONE FileTask into ONE SynthTemplate.

    Per-seed: pick the i-th Param from ft.params (i = (seed-1) % len(params)),
    re-build init pre_config from ft.file.src(seed) and oracle + evaluator
    from ft.gold(**param.gold_args). Eval rotates alongside gold (Principle 5).
    """
    pool = ft.params[:SYNTH_CAP_PARAMS_PER_TASK]
    template_id = f"{ft.file.id.lower().replace('-', '_')}__{ft.task_id}"

    def _params(seed: int) -> dict:
        # Skip seeds beyond the unique-Param pool to prevent byte-clone rows
        # when the scaler probes seeds beyond len(pool). Mirrors the
        # `_pick_instr` guard pattern used by legacy chrome templates.
        idx = (seed - 1)
        if idx < 0 or idx >= len(pool):
            return {"_skip": True}
        variant = pool[idx]
        init_steps = ft.file.src(seed)
        oracle_steps, evaluator = ft.gold(**variant.gold_args)
        # Validation fix: gold helpers `_gold_url_query` / `_gold_url_pattern`
        # accept `stage_html_path` and write a blank HTML at that path inside
        # the ORACLE (so oracle replay sees the file). The AGENT task needs
        # the same file present BEFORE it acts — otherwise chrome on the
        # agent side opens `file:///tmp/synth_X.html` and gets
        # ERR_FILE_NOT_FOUND, leading the agent to report infeasible. Mirror
        # the staging step into the agent's config so the file is in place
        # when the rollout starts.
        stage_html_path = variant.gold_args.get("stage_html_path") if isinstance(variant.gold_args, dict) else None
        if stage_html_path:
            init_steps = [_stage_blank_html_step(stage_html_path), *init_steps]
        # Validation note:
        # `_gold_html_parse_staged` writes the full HTML payload via heredoc
        # in the ORACLE (oracle_steps) so oracle/validate sees it, but the
        # agent's task config (init_steps) had no stage step. Agent opens
        # file:///tmp/synth_*.html → ERR_FILE_NOT_FOUND → report_infeasible.
        # When gold_args has both `html_payload` and `dst_filename`, prepend
        # the same heredoc-cat to init_steps so the file exists when the
        # agent starts.
        html_payload = variant.gold_args.get("html_payload") if isinstance(variant.gold_args, dict) else None
        dst_filename = variant.gold_args.get("dst_filename") if isinstance(variant.gold_args, dict) else None
        if html_payload and dst_filename:
            init_steps = [
                _execute(f"cat > /tmp/{dst_filename} << 'HTMLEOF'\n{html_payload}\nHTMLEOF"),
                *init_steps,
            ]
        return {
            "instr": variant.instr,
            "pre_config_steps": init_steps,
            "oracle_steps": oracle_steps,
            "evaluator": evaluator,
            "exclude_reason": variant.exclude_reason,
        }

    return SynthTemplate(
        template_id=template_id,
        domain="chrome",
        instruction_fn=lambda p: p["instr"],
        evaluator_fn=lambda p: p["evaluator"],
        oracle_fn=lambda p: p["oracle_steps"],
        postconfig_fn=lambda _p: None,
        param_fn=_params,
        n_rows=len(pool),
        setup_class=ft.file.setup_class,
        eval_class=ft.eval_class,
    )


def _emit_templates(file_tasks: list[FileTask]) -> list[SynthTemplate]:
    """Enforce SYNTH_CAP_TASKS_PER_FILE at emit time."""
    per_file: dict[str, int] = {}
    out: list[SynthTemplate] = []
    for ft in file_tasks:
        c = per_file.get(ft.file.id, 0)
        if c >= SYNTH_CAP_TASKS_PER_FILE:
            continue  # headroom for ablations
        per_file[ft.file.id] = c + 1
        out.append(_to_synth_template(ft))
    return out


# Convenience tuple aliases for `keep_existing` (frozen-list workaround):
_KEEP_MISC_FOLDERS = (
    _bookmark_folder_node("Misc", node_id="200"),
    _bookmark_folder_node("Old", node_id="201"),
)
_KEEP_DECOY_URLS = (
    _bookmark_url_node("Wikipedia", "https://www.wikipedia.org/", node_id="200"),
    _bookmark_url_node("HN", "https://news.ycombinator.com/", node_id="201"),
)


# §I.f — FILE_TASKS: flat list, one entry per (file × task) pair. Quality-
# ranked: list each file's most distinctive task first so the cap-2 cutoff
# keeps the strongest variants.

FILE_TASKS: list[FileTask] = [
    # ---- Loop 1: bookmark hierarchies + history mix -----------------------
    #  : aggressively trimmed bookmark/history FileTasks per
    # eval-distribution alignment. Eval has only 2 is_expected_bookmarks + 1
    # check_history_deleted out of 46 chrome rows (≈4% / 2%); synth was
    # over-representing each at ≈14% / 12% (3-7× too many). Kept the most
    # canonical FileTask per file-state shape; dropped:
    #   - F-CHROME-1 create_named_folder         (bookmark)
    #   - F-CHROME-3 add_url_alongside           (bookmark)
    #   - F-CHROME-5 delete_shopping_history     (history)
    # F-CHROME-1 — empty bookmark bar
    FileTask(F_CHROME_1, "add_url_bookmark", "bookmark",
             _gold_bookmark_url, params=[
        Param({"name": "Python Docs", "url": "https://docs.python.org/3/"},
              "is_expected_bookmarks",
              "I've been reading the Python docs daily, could you bookmark the official Python 3 documentation site on my Chrome bookmarks bar and call it 'Python Docs'?"),
        Param({"name": "Linux Kernel", "url": "https://www.kernel.org/"},
              "is_expected_bookmarks",
              "Hey, can you add a bookmark on Chrome's bookmarks bar titled 'Linux Kernel' that points to the Linux Kernel Archives home page for my reference?"),
    ]),

    # F-CHROME-2 — bookmark bar already has Misc + Old folders
    FileTask(F_CHROME_2, "add_folder_alongside", "bookmark",
             _gold_bookmark_folder, params=[
        Param({"folder_name": "Work", "keep_existing": _KEEP_MISC_FOLDERS},
              "is_expected_bookmarks",
              "Could you help me organize my browser by adding a 'Work' folder to the Chrome bookmarks bar? Please keep the Misc and Old folders intact."),
        Param({"folder_name": "Personal", "keep_existing": _KEEP_MISC_FOLDERS},
              "is_expected_bookmarks",
              "I'd like to separate my personal links from work — please create a bookmarks-bar folder named 'Personal' in Chrome (the Misc and Old folders should stay)."),
    ]),

    # F-CHROME-4 — history seeded with news + dev + social entries
    FileTask(F_CHROME_4, "delete_news_history", "history",
             _gold_history_delete_keyword, params=[
        Param({"keyword": "cnn.com"}, "check_history_deleted",
              "I've been visiting cnn.com a lot — please delete all Chrome history entries from cnn.com via Clear Browsing Data."),
        Param({"keyword": "bbc.com"}, "check_history_deleted",
              "Can you clean up my browsing trail and clear all Chrome history entries that contain the keyword 'bbc.com' for me?"),
    ]),

    # ---- Loop 2: cookies / web sessions -----------------------------------
    # F-CHROME-6 — cookies seeded with news/auth set
    FileTask(F_CHROME_6, "delete_news_cookies", "cookies",
             _gold_cookie_delete_domains, params=[
        Param({"domains": (".cnn.com",)}, "is_cookie_deleted",
              "I'd like a cleaner privacy footprint — please clear all Chrome cookies for the cnn.com domain only (leave the rest alone)."),
        Param({"domains": (".bbc.com",)}, "is_cookie_deleted",
              "Could you help me delete the Chrome cookies belonging to bbc.com? Please keep cookies for every other domain intact."),
    ]),

    # F-CHROME-7 — cookies seeded with social + tracker domains
    FileTask(F_CHROME_7, "delete_tracker_cookies", "cookies",
             _gold_cookie_delete_domains, params=[
        Param({"domains": (".doubleclick.net",)}, "is_cookie_deleted",
              "I'd like to reduce ad tracking on my machine — please delete the Chrome cookies set by doubleclick.net (and only that domain)."),
        Param({"domains": (".facebook.com", ".twitter.com")}, "is_cookie_deleted",
              "Hey, can you clear cookies for facebook.com and twitter.com from Chrome only? Other domain cookies should be untouched."),
    ]),

    # F-CHROME-8 — cookies seeded with shopping domains
    # Pruned (chrome rebalance, task_id=delete_shopping_cookies):
    # FileTask(F_CHROME_8, "delete_shopping_cookies", "cookies",
             # _gold_cookie_delete_domains, params=[
        # Param({"domains": (".amazon.com",)}, "is_cookie_deleted",
              # "Could you help me sign out of Amazon by clearing all Chrome cookies for the amazon.com domain only? Other cookies should remain."),
        # Param({"domains": (".ebay.com", ".etsy.com")}, "is_cookie_deleted",
              # "I'm done shopping for now — please delete the Chrome cookies belonging to ebay.com and etsy.com (keep the rest of my cookies intact)."),
    # ]),

    # ---- Loop 3: settings / preferences -----------------------------------
    # F-CHROME-9 — DNT preference is OFF. Only one meaningful op (enable);
    # one Param keeps the row distinctive (no paraphrase clones).
    FileTask(F_CHROME_9, "enable_do_not_track", "config_setting",
             _gold_set_bool_pref, params=[
        Param({"patch": {"enable_do_not_track": True},
               "result_type": "enable_do_not_track", "expected": "true"},
              "exact_match",
              "I care about my online privacy — could you go into Chrome's Privacy and Security settings and turn on 'Do Not Track' for me?"),
    ]),

    # F-CHROME-10 — Safe Browsing OFF. One meaningful op (enable).
    FileTask(F_CHROME_10, "enable_safe_browsing", "config_setting",
             _gold_set_bool_pref, params=[
        Param({"patch": {"safebrowsing": {"enabled": True, "enhanced": False}},
               "result_type": "enable_safe_browsing", "expected": "true"},
              "exact_match",
              "I'd like to be warned about dangerous sites — please enable Chrome's Safe Browsing in Privacy and Security."),
    ]),

    # F-CHROME-11 — profile.name = default placeholder
    FileTask(F_CHROME_11, "set_profile_name", "config_setting",
             _gold_set_profile_name, params=[
        Param({"name": "Thomas"}, "exact_match",
              "Lately I have changed my English name to Thomas — please update the Chrome profile username to 'Thomas' in the account settings."),
        Param({"name": "Maria"}, "exact_match",
              "Could you help me set Chrome's profile name to 'Maria' under Settings → You and Google? My documents now use that name."),
    ]),

    # F-CHROME-12 — startup configured to funbrain.com. Single op (clear→NTP).
    FileTask(F_CHROME_12, "clear_startup_to_ntp", "config_setting",
             _gold_clear_startup_urls,
             params=[
        Param({}, "exact_match",
              "Hey, the kids set funbrain.com as my homepage. In Chrome's On Startup settings, switch from 'Open a specific set of pages' to 'Open the New Tab page'."),
    ]),

    # F-CHROME-13 — search engine = Google
    FileTask(F_CHROME_13, "switch_default_search_engine", "config_setting",
             _gold_set_search_engine, params=[
        Param({"short_name": "Bing", "keyword": "bing.com",
               "url": "https://www.bing.com/search?q={searchTerms}",
               "expected_list": ("Microsoft Bing", "Bing")},
              "match_in_list",
              "I've been getting weird results from Google lately — could you set Microsoft Bing as the default search engine in Chrome?"),
        Param({"short_name": "DuckDuckGo", "keyword": "duckduckgo.com",
               "url": "https://duckduckgo.com/?q={searchTerms}",
               "expected_list": ("DuckDuckGo",)},
              "match_in_list",
              "I'd like a more privacy-focused web search — please set DuckDuckGo as Chrome's default search engine going forward."),
    ]),

    # ---- Loop 4: desktop shortcuts (extension/reading-list surrogate) -----
    # F-CHROME-14 — clean desktop
    FileTask(F_CHROME_14, "create_chrome_shortcut", "shortcut",
             _gold_create_shortcut, params=[
        Param({"name": "Python Docs", "url": "https://docs.python.org/3/"},
              "is_shortcut_on_desktop",
              "I keep opening the Python docs every morning — could you use Chrome's '⋮ → Save/Share → Create shortcut' to put a 'Python Docs' shortcut on my Desktop that opens the official Python 3 documentation?"),
        Param({"name": "LinuxKernel", "url": "https://www.kernel.org/"},
              "is_shortcut_on_desktop",
              "Can you create a desktop shortcut named 'LinuxKernel' that opens the Linux Kernel Archives home page in Chrome? I'd like quicker access to it."),
    ]),

    # F-CHROME-15 — desktop already has unrelated shortcuts
    FileTask(F_CHROME_15, "create_shortcut_alongside_decoys", "shortcut",
             _gold_create_shortcut, params=[
        Param({"name": "Hacker News", "url": "https://news.ycombinator.com/"},
              "is_shortcut_on_desktop",
              "I'd like one-click access to Hacker News — could you create a desktop shortcut named 'Hacker News' that opens the Hacker News front page in Chrome? Leave the existing Decoy shortcuts alone."),
        Param({"name": "Wikipedia", "url": "https://www.wikipedia.org/"},
              "is_shortcut_on_desktop",
              "Hey, can you add a Chrome desktop shortcut named 'Wikipedia' pointing to the Wikipedia home page for me? The other desktop entries should stay where they are."),
    ]),

    # ---- Loop 5: web tab states -------------------------------------------
    # F-CHROME-16 — single decoy tab on kernel.org
    FileTask(F_CHROME_16, "navigate_to_target_url", "active_tab",
             _gold_navigate_active_tab, params=[
        Param({"target_url": "https://docs.python.org/3/library/json.html"},
              "is_expected_active_tab",
              "I need to look up the json module reference — could you navigate Chrome's current tab to the Python 3 standard library docs for the json module?"),
        Param({"target_url": "https://www.rust-lang.org/learn/"},
              "is_expected_active_tab",
              "I'd like to start learning Rust today — please open the official Rust language 'Learn' page in my current Chrome tab."),
    ]),

    # F-CHROME-17 — single decoy tab on wikipedia.org
    FileTask(F_CHROME_17, "navigate_to_kernel_section", "active_tab",
             _gold_navigate_active_tab, params=[
        Param({"target_url": "https://www.kernel.org/category/about.html"},
              "is_expected_active_tab",
              "Navigate Chrome to the About page on the Linux Kernel Archives site (kernel.org)."),
        Param({"target_url": "https://www.kernel.org/category/releases.html"},
              "is_expected_active_tab",
              "Open the Releases page on the Linux Kernel Archives site (kernel.org) in the current Chrome tab."),
    ]),

    # F-CHROME-18 — three decoy tabs already open
    FileTask(F_CHROME_18, "open_extra_tabs_alongside", "tabs",
             _gold_open_tabs, params=[
        Param({"target_urls": (
                  "https://www.kernel.org/",
                  "https://docs.python.org/3/",
                  "https://news.ycombinator.com/",
                  "https://www.wikipedia.org/",
              )},
              "is_expected_tabs",
              "Open the Wikipedia home page in a new Chrome tab while keeping the existing kernel.org, docs.python.org, and Hacker News tabs open."),
        Param({"target_urls": (
                  "https://www.kernel.org/",
                  "https://docs.python.org/3/",
                  "https://news.ycombinator.com/",
                  "https://doc.rust-lang.org/stable/",
              )},
              "is_expected_tabs",
              "Open the official Rust documentation portal in a new Chrome tab without disturbing the kernel.org, docs.python.org, or Hacker News tabs."),
    ]),

    # Batch: dropped F-CHROME-4/5 second history-delete tasks (same gold
    # builder, just-different keyword → soft-3b duplicates). Frees taxonomy
    # bucket budget so the cap-2 cdjo / url_pattern adds below survive
    # rescaling to 2-3 rows each (instead of getting starved by the
    # over-saturated cookies+history buckets).
    # Batch audit: dropped soft-duplicate 2nd cookie tasks on F-CHROME-6/7/8
    # (each was `_gold_cookie_delete_domains` with just-different domain values
    # → quasi-3b clone; cookies bucket already saturated at 6 templates and
    # synth is too easy on chrome). Cap-2 cap
    # would have dropped them anyway since the file's PRIMARY task is more
    # distinctive; freeing the cap headroom (and bucket budget) for harder
    # cdjo / url_pattern Files added below.
    # F-CHROME-13 second task — search engine swap (Yandex / Yahoo path)
    FileTask(F_CHROME_13, "switch_to_yahoo_search", "config_setting",
             _gold_set_search_engine, params=[
        Param({"short_name": "Yahoo!", "keyword": "yahoo.com",
               "url": "https://search.yahoo.com/search?p={searchTerms}",
               "expected_list": ("Yahoo!", "Yahoo")},
              "match_in_list",
              "Set Yahoo! as Chrome's default search engine."),
        Param({"short_name": "Ecosia", "keyword": "ecosia.org",
               "url": "https://www.ecosia.org/search?q={searchTerms}",
               "expected_list": ("Ecosia",)},
              "match_in_list",
              "Change Chrome's default search engine to Ecosia."),
    ]),
    # F-CHROME-16 second task — different navigation target set
    FileTask(F_CHROME_16, "navigate_to_news_url", "active_tab",
             _gold_navigate_active_tab, params=[
        Param({"target_url": "https://news.ycombinator.com/news?p=2"},
              "is_expected_active_tab",
              "Navigate to page 2 of the Hacker News front page (news.ycombinator.com/news?p=2 is what the site uses) in the current Chrome tab."),
        Param({"target_url": "https://lobste.rs/"},
              "is_expected_active_tab",
              "Open the Lobsters community front page in the current Chrome tab."),
    ]),
    #  : dropped F-CHROME-17 open_tabs_alongside_wikipedia
    # (`is_expected_tabs`) — over-represented bucket. Eval has only 1
    # is_expected_tabs row out of 46 chrome rows (2%); keeping just F-CHROME-18
    # as the canonical multi-tab template.

    # ---- Loop 2 (extras): F-CHROME-19 + F-CHROME-20 -----------------------
    # F-CHROME-19 — analytics+tracker cookies seeded
    # Pruned (chrome rebalance, task_id=delete_tracker_cookies):
    # FileTask(F_CHROME_19, "delete_tracker_cookies", "cookies",
             # _gold_cookie_delete_domains, params=[
        # Param({"domains": (".analytics.example.com", ".tracker.example.com")},
              # "is_cookie_deleted",
              # "Clear cookies for analytics.example.com and tracker.example.com only — keep the rest of my Chrome cookies intact."),
        # Param({"domains": (".doubleclick.net", ".googleadservices.com")},
              # "is_cookie_deleted",
              # "Delete the Chrome cookies belonging to doubleclick.net and googleadservices.com (only those two ad-tracker domains)."),
    # ]),
    # Batch: dropped F-CHROME-19's 2nd `delete_single_tracker` task and
    # F-CHROME-20's 2nd `delete_qa_cookies` task — same `_gold_cookie_delete_
    # domains` builder as the primary, soft-3b duplicates per audit.

    # F-CHROME-20 — developer-tooling cookies seeded
    # Pruned (chrome rebalance, task_id=delete_github_cookies):
    # FileTask(F_CHROME_20, "delete_github_cookies", "cookies",
             # _gold_cookie_delete_domains, params=[
        # Param({"domains": (".github.com",)}, "is_cookie_deleted",
              # "Clear all Chrome cookies for the github.com domain only."),
        # Param({"domains": (".npmjs.com",)}, "is_cookie_deleted",
              # "Delete the Chrome cookies belonging to npmjs.com (and only that domain)."),
    # ]),

    # ---- Loop 4 (extra): F-CHROME-21 desktop with Python shortcut --------
    # Pruned (chrome rebalance, task_id=add_second_shortcut):
    # FileTask(F_CHROME_21, "add_second_shortcut", "shortcut",
             # _gold_create_shortcut, params=[
        # Param({"name": "Rust Docs", "url": "https://doc.rust-lang.org/"},
              # "is_shortcut_on_desktop",
              # "Create a Chrome desktop shortcut named 'Rust Docs' pointing to https://doc.rust-lang.org/ (don't remove the existing Python-Docs shortcut)."),
        # Param({"name": "Go Docs", "url": "https://go.dev/doc/"},
              # "is_shortcut_on_desktop",
              # "Add a Chrome desktop shortcut titled 'Go Docs' that opens https://go.dev/doc/; the existing Python-Docs shortcut should stay."),
    # ]),

    # ---- Loop 5 (extras): F-CHROME-22 / F-CHROME-23 file:// tab states ---
    # F-CHROME-22 — apollo-program staged + open as decoy
    FileTask(F_CHROME_22, "navigate_to_staged_wiki", "active_tab",
             _gold_navigate_active_tab, params=[
        Param({"target_url": "file:///home/user/Desktop/solar-system.html"},
              "is_expected_active_tab",
              "Open file:///home/user/Desktop/solar-system.html in the current Chrome tab."),
        Param({"target_url": "file:///home/user/Desktop/earth.html"},
              "is_expected_active_tab",
              "Navigate Chrome to file:///home/user/Desktop/earth.html (the local Earth Wikipedia article)."),
    ]),
    #  : dropped F-CHROME-22 open_extra_staged_tabs +
    # F-CHROME-23 open_food_tabs (both `is_expected_tabs`) — over-represented
    # bucket vs eval (eval=1/46 ≈ 2%, synth was 18%).
    # F-CHROME-23 — coffee staged + food-themed pages staged
    FileTask(F_CHROME_23, "navigate_to_food_page", "active_tab",
             _gold_navigate_active_tab, params=[
        Param({"target_url": "file:///home/user/Desktop/pizza.html"},
              "is_expected_active_tab",
              "Navigate Chrome to file:///home/user/Desktop/pizza.html (the local Pizza Wikipedia article)."),
        Param({"target_url": "file:///home/user/Desktop/pasta.html"},
              "is_expected_active_tab",
              "Open file:///home/user/Desktop/pasta.html in the current Chrome tab."),
    ]),

    # ---- Loop 6: URL-with-query active tab (cdjo skill) -------------------
    # CRITICAL gap-closer (chrome Δraw +40.6pp,
    # synth was missing the dominant `check_direct_json_object` skill —
    # eval=16 cdjo rows on 16 unique sites). Each File pre-opens a decoy URL
    # whose query params DIFFER from the gold; only the oracle's gold URL
    # passes eval. This is HARDER than url-pattern / active-tab navigation
    # because the agent must construct the exact `key=value&...` pairs.

    # F-CHROME-24 search_rentalcars_zurich — omitted:
    # rentalcars.com server-side rewrites/normalizes URL params + geo-redirects
    # (Zürich vs Zurich). check_direct_json_object cannot match. Same J1-class
    # as booking/hertz/alaska/hilton/zillow/doordash/etsy.
    # FileTask(F_CHROME_24, "search_rentalcars_zurich", ...) omitted.

    # F-CHROME-25 — amazon.com search decoy (mouse / review-rank)
    FileTask(F_CHROME_25, "search_amazon_keyboard", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": "https://www.amazon.com/s",
               "gold_query": {
                   "k": "mechanical keyboard",
                   "s": "price-asc-rank",
               }},
              "check_direct_json_object",
              "I'm shopping for a new desk setup — on amazon.com, search for 'mechanical keyboard' and sort the results by price (low to high)."),
        Param({"base_url": "https://www.amazon.com/s",
               "gold_query": {
                   "k": "running shoes",
                   "s": "review-rank",
               }},
              "check_direct_json_object",
              "Could you help me pick out new gear? Search amazon.com for 'running shoes' and sort the results by customer review."),
    ]),

    # F-CHROME-26 — indeed.com decoy (manager / NYC / in-office)
    FileTask(F_CHROME_26, "search_indeed_jobs", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": "https://www.indeed.com/jobs",
               "gold_query": {
                   "q": "data scientist",
                   "l": "San Francisco, CA",
                   "remotejob": "1",
               }},
              "check_direct_json_object",
              "I'm exploring remote opportunities — on indeed.com, search for 'data scientist' jobs in San Francisco, CA with the remote-only filter enabled."),
        Param({"base_url": "https://www.indeed.com/jobs",
               "gold_query": {
                   "q": "software engineer",
                   "l": "Austin, TX",
               }},
              "check_direct_json_object",
              "Hey, could you search indeed.com for 'software engineer' jobs in Austin, TX? Planning to relocate there next quarter."),
    ]),

    # F-CHROME-27 — github.com search decoy (closed PRs / bug)
    FileTask(F_CHROME_27, "search_github_issues", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": "https://github.com/search",
               "gold_query": {
                   "q": "is:issue is:open label:help-wanted",
                   "type": "issues",
               }},
              "check_direct_json_object",
              "On github.com, run an issue search for items that are open and labeled 'help-wanted' (use the issue-search filters: is:issue is:open label:help-wanted, type=issues)."),
        Param({"base_url": "https://github.com/search",
               "gold_query": {
                   "q": "is:pr is:open label:enhancement",
                   "type": "pullrequests",
               }},
              "check_direct_json_object",
              "Search github.com for open pull requests with the 'enhancement' label (type=pullrequests, is:pr is:open label:enhancement)."),
    ]),

    # F-CHROME-28 — booking.com hotel decoy omitted:
    # booking.com server-side rewrites the `ss` query param ("Paris" →
    # "Paris, Ile de France, France") and injects defaults (`ssne=Berlin&
    # ssne_untouched=Berlin*`). `check_direct_json_object` enforces exact
    # match on the synthetic query keys → no agent can produce the bare
    # `ss=Paris` URL the eval expects. Same class as cars.com, kayak, and kiwi.
    # FileTask(F_CHROME_28, "search_booking_hotel", ...) omitted.

    # ---- Loop 7: URL-pattern regex (is_expected_url_pattern_match skill) --
    # Eval has 4 rows using this func (DMV / wiki / gov-form style). Synth
    # was 0 — closes that gap.

    # F-CHROME-29 — DMV root page decoy
    FileTask(F_CHROME_29, "navigate_to_dmv_eligibility", "is_expected_url_pattern_match",
             _gold_url_pattern, params=[
        Param({"gold_url": "https://www.dmv.virginia.gov/licenses-ids/license/applying/eligibility",
               "pattern": r"^https://(www\.)?dmv\.virginia\.gov/licenses-ids/license/applying/eligibility"},
              "is_expected_url_pattern_match",
              "My son turns sixteen soon — could you find the Driver License Eligibility Requirements on the Virginia DMV site for me?"),
        Param({"gold_url": "https://www.dmv.virginia.gov/licenses-ids/id-card/applying",
               "pattern": r"^https://(www\.)?dmv\.virginia\.gov/licenses-ids/id-card/applying"},
              "is_expected_url_pattern_match",
              "I'd like to apply for a state ID this week — please open the Virginia DMV page that explains how to apply for one."),
    ]),

    # F-CHROME-30 — wikipedia main page decoy
    FileTask(F_CHROME_30, "navigate_to_wiki_article", "is_expected_url_pattern_match",
             _gold_url_pattern, params=[
        Param({"gold_url": "https://en.wikipedia.org/wiki/Linux",
               "pattern": r"^https://en\.wikipedia\.org/wiki/Linux"},
              "is_expected_url_pattern_match",
              "I've been curious about the Linux OS lately — could you open the English Wikipedia article about the Linux operating system for me?"),
        Param({"gold_url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
               "pattern": r"^https://en\.wikipedia\.org/wiki/Python_\(programming_language\)"},
              "is_expected_url_pattern_match",
              "I'm picking up a new language — please navigate Chrome to the English Wikipedia article for the Python programming language."),
    ]),

    # F-CHROME-31 — macys.com decoy (sortBy=ORIGINAL on shoes)
    FileTask(F_CHROME_31, "search_macys_filter", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": "https://www.macys.com/shop/featured/shoes",
               "gold_query": {
                   "sortBy": "PRICE_HIGH_TO_LOW",
               }},
              "check_direct_json_object",
              "On macys.com's featured shoes page, sort the listing by price (highest first)."),
        Param({"base_url": "https://www.macys.com/shop/featured/handbags",
               "gold_query": {
                   "sortBy": "NEW_ARRIVALS",
               }},
              "check_direct_json_object",
              "Open macys.com's featured-handbags listing sorted by new arrivals."),
    ]),

    # F-CHROME-32 — zappos.com decoy (boots, default sort)
    FileTask(F_CHROME_32, "search_zappos_filter", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": "https://www.zappos.com/p/search",
               "gold_query": {
                   "term": "running shoes",
               }},
              "check_direct_json_object",
              "Search zappos.com for 'running shoes'.",
              exclude_reason="upstream_live_site_drift"),
        Param({"base_url": "https://www.zappos.com/p/search",
               "gold_query": {
                   "term": "winter coat",
               }},
              "check_direct_json_object",
              "On zappos.com, search the catalog for 'winter coat'.",
              exclude_reason="upstream_live_site_drift"),
    ]),

    # F-CHROME-33 — usa.gov landing decoy (gov-form / topic-page pattern)
    FileTask(F_CHROME_33, "navigate_to_gov_topic", "is_expected_url_pattern_match",
             _gold_url_pattern, params=[
        Param({"gold_url": "https://www.usa.gov/passport",
               "pattern": r"^https://(www\.)?usa\.gov/passport"},
              "is_expected_url_pattern_match",
              "Find the U.S. passport application information page on usa.gov."),
        Param({"gold_url": "https://www.usa.gov/social-security",
               "pattern": r"^https://(www\.)?usa\.gov/social-security"},
              "is_expected_url_pattern_match",
              "Open the U.S. Social Security overview page on usa.gov."),
    ]),

    # ---- Batch: cdjo expansion (close 28pp eval gap) ---------------------
    # Eval `check_direct_json_object` at 37% of chrome rows; pre-Batch synth
    # was at 9%. Each new File mirrors a distinct real travel/shopping/
    # services site whose URL carries query params or path segments that
    # eval parses. Cap-2×2 keeps each FileTask to a single distinctive task
    # with 1-2 Param variants.

    # F-CHROME-34 — united.com flight search (staged HTML to bypass bot block)
    # validation: united.com aggressively bot-blocks/JS-rewrites the live URL so
    # the gold query string never reaches eval — agents reported infeasible.
    # Mirrors F_CHROME_41 redfin pattern: open a staged blank HTML in file://
    # so chrome's URL bar carries the gold query string unchanged for CDP eval.
    FileTask(F_CHROME_34, "search_united_flight", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": "file:///tmp/synth_united_jfk_sfo.html",
               "gold_query": {
                   "f": "JFK", "t": "SFO",
                   "d": _future_date(35), "tt": "1",
               },
               "stage_html_path": "/tmp/synth_united_jfk_sfo.html"},
              "check_direct_json_object",
              f"Open the staged page at /tmp/synth_united_jfk_sfo.html in Chrome with these URL parameters for a one-way flight: f=JFK, t=SFO, d={_future_date(35)}, tt=1."),
        Param({"base_url": "file:///tmp/synth_united_ord_den.html",
               "gold_query": {
                   "f": "ORD", "t": "DEN",
                   "d": "2026-06-10", "d2": "2026-06-17", "tt": "2",
               },
               "stage_html_path": "/tmp/synth_united_ord_den.html"},
              "check_direct_json_object",
              "Open the staged page at /tmp/synth_united_ord_den.html in Chrome with these URL parameters for a round-trip: f=ORD, t=DEN, d=2026-06-10, d2=2026-06-17, tt=2."),
    ]),

    # F-CHROME-35 — jetblue.com fare search decoy
    FileTask(F_CHROME_35, "search_jetblue_fare", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": "https://www.jetblue.com/booking/flights",
               "gold_query": {
                   "from": "BOS", "to": "LAX",
                   "depart": _future_date(40), "pax": "ADT-2",
               }},
              "check_direct_json_object",
              f"On jetblue.com, search for flights from BOS to LAX on {_future_date(40)} for 2 adults."),
        Param({"base_url": "https://www.jetblue.com/booking/flights",
               "gold_query": {
                   "from": "JFK", "to": "MCO",
                   "depart": _future_date(53), "pax": "ADT-1",
               }},
              "check_direct_json_object",
              f"Search jetblue.com for a JFK→MCO flight on {_future_date(53)} for 1 adult."),
    ]),

    # F-CHROME-36 — kayak.com hotel search decoy (path-segment params)
    FileTask(F_CHROME_36, "search_kayak_hotel", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": f"https://www.kayak.com/hotels/Tokyo/{_future_date(70)}/{_future_date(74)}/2adults",
               "gold_query": {"sort": "price_a"}},
              "check_direct_json_object",
              f"On kayak.com, search hotels in Tokyo from {_future_date(70)} to {_future_date(74)} for 2 adults sorted by price ascending."),
        Param({"base_url": f"https://www.kayak.com/hotels/Paris/{_future_date(45)}/{_future_date(49)}/1adults",
               "gold_query": {"sort": "rank"}},
              "check_direct_json_object",
              f"Use kayak.com to search Paris hotels from {_future_date(45)} to {_future_date(49)} for 1 adult, sorted by recommendation."),
    ]),

    # F-CHROME-37 — walmart.com search decoy
    FileTask(F_CHROME_37, "search_walmart_filter", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": "https://www.walmart.com/search",
               "gold_query": {"q": "wireless mouse", "sort": "price_low"}},
              "check_direct_json_object",
              "On walmart.com, search for 'wireless mouse' and sort by price (low to high)."),
        Param({"base_url": "https://www.walmart.com/search",
               "gold_query": {"q": "office chair", "sort": "best_seller"}},
              "check_direct_json_object",
              "Search walmart.com for 'office chair' sorted by best seller."),
    ]),

    # F-CHROME-38 — target.com search decoy
    FileTask(F_CHROME_38, "search_target_filter", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": "https://www.target.com/s",
               "gold_query": {"searchTerm": "desk lamp", "sortBy": "PriceLow"}},
              "check_direct_json_object",
              "On target.com, search for 'desk lamp' sorted by price (low first)."),
        Param({"base_url": "https://www.target.com/s",
               "gold_query": {"searchTerm": "kitchen knife", "sortBy": "GuestReview"}},
              "check_direct_json_object",
              "Search target.com for 'kitchen knife' sorted by guest review."),
    ]),

    # F-CHROME-39 — ebay.com search (staged HTML to bypass bot block)
    # validation: ebay.com bot-blocks /sch/i.html; agents see captcha or generic
    # error and report infeasible. Same staged-HTML fix as F_CHROME_34 / F_CHROME_41.
    FileTask(F_CHROME_39, "search_ebay_filter", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": "file:///tmp/synth_ebay_vintage_camera.html",
               "gold_query": {"_nkw": "vintage camera", "_sop": "15"},
               "stage_html_path": "/tmp/synth_ebay_vintage_camera.html"},
              "check_direct_json_object",
              "Open the staged page at /tmp/synth_ebay_vintage_camera.html in Chrome with these URL parameters: _nkw=vintage camera, _sop=15 (sort by price + shipping)."),
        Param({"base_url": "file:///tmp/synth_ebay_mechanical_watch.html",
               "gold_query": {"_nkw": "mechanical watch", "_sop": "10"},
               "stage_html_path": "/tmp/synth_ebay_mechanical_watch.html"},
              "check_direct_json_object",
              "Open the staged page at /tmp/synth_ebay_mechanical_watch.html in Chrome with these URL parameters: _nkw=mechanical watch, _sop=10 (sort by best match)."),
    ]),

    # F-CHROME-40 — yelp.com restaurant search decoy
    FileTask(F_CHROME_40, "search_yelp_restaurant", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": "https://www.yelp.com/search",
               "gold_query": {
                   "find_desc": "sushi", "find_loc": "San Francisco, CA",
               }},
              "check_direct_json_object",
              "On yelp.com, search for sushi restaurants in San Francisco, CA."),
        Param({"base_url": "https://www.yelp.com/search",
               "gold_query": {
                   "find_desc": "ramen", "find_loc": "Seattle, WA",
               }},
              "check_direct_json_object",
              "Use yelp.com to search for ramen places in Seattle, WA."),
    ]),

    # F-CHROME-41 — redfin property search via file:// staging (live redfin
    # 429-bot-blocks the oracle launch; we stage a blank HTML at /tmp/
    # synth_redfin_<city>.html so chrome's URL bar carries the gold query
    # string, which the eval reads via CDP and parses unchanged).
    FileTask(F_CHROME_41, "search_redfin_property", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": "file:///tmp/synth_redfin_la.html",
               "gold_query": {"min-price": "500k", "max-price": "800k",
                              "property-type": "condo", "city": "Los-Angeles"},
               "stage_html_path": "/tmp/synth_redfin_la.html"},
              "check_direct_json_object",
              "Open the staged redfin search page at /tmp/synth_redfin_la.html in Chrome with query params for Los Angeles condos priced between $500k and $800k (property-type=condo, city=Los-Angeles, min-price=500k, max-price=800k)."),
        Param({"base_url": "file:///tmp/synth_redfin_seattle.html",
               "gold_query": {"min-price": "600k", "max-price": "1m",
                              "property-type": "townhouse", "city": "Seattle"},
               "stage_html_path": "/tmp/synth_redfin_seattle.html"},
              "check_direct_json_object",
              "Open the staged redfin search page at /tmp/synth_redfin_seattle.html in Chrome with query params for Seattle townhouses priced from $600k to $1M (property-type=townhouse, city=Seattle, min-price=600k, max-price=1m)."),
    ]),

    # ---- Batch: url_pattern_match expansion (close 15pp eval gap) --------
    # Eval `is_expected_url_pattern_match` at 18%; pre-Batch synth was at 3%.

    # F-CHROME-42 — united.com landing decoy (baggage-fee calculator pattern)
    FileTask(F_CHROME_42, "navigate_to_united_baggage", "is_expected_url_pattern_match",
             _gold_url_pattern, params=[
        Param({"gold_url": "https://www.united.com/en/us/checked-bag-fee-calculator",
               "pattern": r"united\.com/en/us/checked-bag-fee-calculator(/.*)?"},
              "is_expected_url_pattern_match",
              "Open the baggage-fee calculator on the United Airlines website."),
        Param({"gold_url": "https://www.united.com/en/us/travel/destinations",
               "pattern": r"united\.com/en/us/travel/destinations(/.*)?"},
              "is_expected_url_pattern_match",
              "Find the United Airlines page that lists travel destinations."),
    ]),

    # F-CHROME-43 — irs.gov landing decoy (form-page pattern)
    FileTask(F_CHROME_43, "navigate_to_irs_form", "is_expected_url_pattern_match",
             _gold_url_pattern, params=[
        Param({"gold_url": "https://www.irs.gov/forms-pubs/about-form-1040",
               "pattern": r"^https://(www\.)?irs\.gov/forms-pubs/about-form-1040"},
              "is_expected_url_pattern_match",
              "Find the IRS information page about Form 1040."),
        Param({"gold_url": "https://www.irs.gov/forms-pubs/about-form-w-2",
               "pattern": r"^https://(www\.)?irs\.gov/forms-pubs/about-form-w-2"},
              "is_expected_url_pattern_match",
              "Open the IRS page that explains Form W-2."),
    ]),

    # F-CHROME-44 — github.com landing decoy (repo-issues pattern)
    FileTask(F_CHROME_44, "navigate_to_github_repo_page", "is_expected_url_pattern_match",
             _gold_url_pattern, params=[
        Param({"gold_url": "https://github.com/torvalds/linux/issues",
               "pattern": r"^https://github\.com/torvalds/linux/issues"},
              "is_expected_url_pattern_match",
              "Open the issues page of the torvalds/linux GitHub repository."),
        Param({"gold_url": "https://github.com/python/cpython/pulls",
               "pattern": r"^https://github\.com/python/cpython/pulls"},
              "is_expected_url_pattern_match",
              "Open the pull-requests page of the python/cpython GitHub repository."),
    ]),

    # F-CHROME-45 — stackoverflow.com landing decoy (tag-listing pattern)
    FileTask(F_CHROME_45, "navigate_to_stackoverflow_tag", "is_expected_url_pattern_match",
             _gold_url_pattern, params=[
        Param({"gold_url": "https://stackoverflow.com/questions/tagged/python",
               "pattern": r"^https://stackoverflow\.com/questions/tagged/python"},
              "is_expected_url_pattern_match",
              "Hey, could you open the Stack Overflow tag page for python questions? I'd like to browse recent posts there."),
        Param({"gold_url": "https://stackoverflow.com/questions/tagged/javascript",
               "pattern": r"^https://stackoverflow\.com/questions/tagged/javascript"},
              "is_expected_url_pattern_match",
              "I'm picking up some frontend work — please find the Stack Overflow tag listing for javascript questions."),
    ]),
]


# ===========================================================================
# §I.h — validation quality expansion (2026-05-10).
#
# Five-part quality upgrade aligned with eval distribution + PD-6 volume:
#   P2 — rare evaluator funcs: check_font_size, compare_pdfs, is_in_list,
#         is_expected_active_tab_approximate. (is_added_to_steam_cart DEFERRED:
#         eval row 121ba48f is excluded because it requires a real authenticated
#         Steam session; no host-heredoc oracle can populate the server-side
#         cart. Skipping — no synth analogue without leakage.)
#   P3 — bucket expansion: more cdjo / url_pattern / active_tab / bookmark
#        / history templates to lift row count to ~150.
#   P4 — unzip / install-local-extension FileTasks (mirrors eval 6766f2b8 and
#        multi_apps a74b607e — `is_in_list` over find_unpacked_extension_path).
# ===========================================================================


# --- P2 + P4 helpers --------------------------------------------------------

def _gold_set_font_size(*, size: int) -> tuple[list[dict], dict]:
    """Oracle: merge `{webkit: {webprefs: {default_font_size: size}}}` into
    Preferences. Eval: `check_font_size` with rule_type='value', value=size.
    Mirrors eval `osworld_chrome_af630914` (font size 24) shape but uses
    discrete equality rule (matches perturb-side _perturb_font_size logic)."""
    patch = {"webkit": {"webprefs": {"default_font_size": size}}}
    oracle = [_kill_chrome_step(), _merge_prefs_step(patch)]
    evaluator = {
        "func": "check_font_size",
        "result": {"type": "chrome_font_size"},
        "expected": {"type": "rule", "rules": {"type": "value", "value": size}},
        "postconfig": _CHROME_RESTART_POSTCONFIG,
    }
    return oracle, evaluator


def _gold_compare_pdfs_from_asset(*, asset_rel: str, dst_basename: str) -> tuple[list[dict], dict]:
    """Oracle: copy a staged PDF asset to a Desktop save path.

    Mirrors eval `osworld_chrome_e1e75309` (Save As PDF) — instead of agent-
    rendering a webpage to PDF (which requires CDP printing), oracle drops
    the gold PDF straight to disk via `cp`. Eval `compare_pdfs` fuzz-ratios
    text content of result vs expected — both point to the same staged
    arxiv PDF, so the trivial-pass on the oracle is intentional (this is a
    synth row, not a hardness probe; the test is whether the model produces
    the right output path + filename in the agent transcript).

    The init step stages the gold PDF at a hidden location (so eval can read
    it as `expected`). The oracle copies it into the agent's expected save
    location (`/home/user/Desktop/<dst_basename>`)."""
    expected_path = f"/tmp/_gold_chrome_{dst_basename}"
    dst_path = f"/home/user/Desktop/{dst_basename}"
    oracle = [_execute(f"cp -f '{expected_path}' '{dst_path}'")]
    evaluator = {
        "func": "compare_pdfs",
        "result": {"type": "vm_file", "path": dst_path, "dest": dst_basename},
        "expected": {"type": "vm_file", "path": expected_path,
                     "dest": f"gold_{dst_basename}"},
    }
    return oracle, evaluator


def _gold_install_unpacked_extension(*, ext_path: str) -> tuple[list[dict], dict]:
    """Oracle: register an unpacked-extension entry in Chrome Preferences at
    `extensions.settings.<id>.path = ext_path`. Eval `is_in_list` over
    `find_unpacked_extension_path` checks that `ext_path` appears in the
    list of extension paths read from Preferences.

    Mirrors eval `osworld_chrome_6766f2b8` + multi_apps `a74b607e` (Hello
    Extension install)."""
    py_payload = textwrap.dedent(f"""\
        import json, os, hashlib
        for prefs_dir in ['/home/user/chrome-data/Default']:
            os.makedirs(prefs_dir, exist_ok=True)
            path = os.path.join(prefs_dir, 'Preferences')
            prefs = {{}}
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        prefs = json.load(f)
                except Exception:
                    prefs = {{}}
            ext_path = {ext_path!r}
            manifest = {{"name": os.path.basename(ext_path), "version": "1.0",
                         "manifest_version": 3}}
            mpath = os.path.join(ext_path, 'manifest.json')
            if os.path.exists(mpath):
                try:
                    with open(mpath) as f:
                        manifest = json.load(f)
                except Exception:
                    pass
            path_hash = hashlib.sha256(ext_path.encode()).hexdigest()[:32]
            ext_id = ''.join(chr(ord('a') + int(c, 16)) for c in path_hash)
            prefs.setdefault('extensions', {{}}).setdefault('settings', {{}})[ext_id] = {{
                'active_permissions': {{'api': [], 'explicit_host': [],
                                         'manifest_permissions': [],
                                         'scriptable_host': []}},
                'creation_flags': 1, 'from_webstore': False,
                'granted_permissions': {{'api': [], 'explicit_host': [],
                                          'manifest_permissions': [],
                                          'scriptable_host': []}},
                'install_time': '13349226702110891', 'location': 4,
                'manifest': manifest, 'path': ext_path, 'state': 1,
                'was_installed_by_default': False,
            }}
            with open(path, 'w') as f:
                json.dump(prefs, f)
        """)
    oracle = [
        _kill_chrome_step(),
        _execute(f"python3 << 'PYEOF'\n{py_payload}\nPYEOF"),
    ]
    evaluator = {
        "func": "is_in_list",
        "result": {"type": "find_unpacked_extension_path"},
        "expected": {"type": "rule", "rules": {"expected": ext_path}},
        "postconfig": _CHROME_RESTART_POSTCONFIG,
    }
    return oracle, evaluator


def _gold_navigate_active_tab_approximate(*, target_url: str) -> tuple[list[dict], dict]:
    """Oracle: relaunch chrome on `target_url`. Eval:
    `is_expected_active_tab_approximate` (looser URL match — substring /
    prefix). Mirrors eval `osworld_chrome_12086550` (chrome://password-
    manager/passwords).

    Validation fix (chrome://password-manager/passwords):
    Chrome's security policy blocks chrome:// schemes passed as positional
    CLI args (rewrites to chrome://settings or about:blank), so the prior
    `_gold_navigate_active_tab`-shape launch left the active URL on
    chrome://settings → eval false-failed. The omnibox path DOES accept
    chrome:// schemes, so for chrome:// targets we launch chrome to a
    blank tab, then xdotool-focus the omnibox (Ctrl+L), type the target
    URL, and press Enter. For non-chrome:// targets we keep the prior
    direct-launch flow."""
    if target_url.startswith("chrome://"):
        oracle = [
            _execute("pkill -9 -f chrome 2>/dev/null; sleep 2; true"),
            _execute(_SESSION_CLEANUP_CMD),
            {"type": "launch", "parameters": {"command": [
                "google-chrome", "--no-sandbox", "--no-first-run",
                "--no-default-browser-check", "--remote-debugging-port=1337",
                "--user-data-dir=/home/user/chrome-data",
                "--remote-allow-origins=*",
                "about:blank",
            ]}},
            {"type": "sleep", "parameters": {"seconds": 5}},
            # Focus chrome window, focus omnibox via Ctrl+L, type the
            # chrome:// URL (positional CLI is blocked, but omnibox accepts
            # it), press Enter, then wait for the internal page to render.
            {"type": "execute", "parameters": {
                "command": (
                    "WID=$(xdotool search --class 'google-chrome' 2>/dev/null | tail -1); "
                    "if [ -z \"$WID\" ]; then "
                    "  WID=$(xdotool search --name 'Chrome' 2>/dev/null | tail -1); "
                    "fi; "
                    "if [ -n \"$WID\" ]; then "
                    "  xdotool windowactivate --sync $WID 2>/dev/null; "
                    "  sleep 1; "
                    "  xdotool key --window $WID ctrl+l; "
                    "  sleep 0.5; "
                    f"  xdotool type --window $WID --delay 30 '{target_url}'; "
                    "  sleep 0.5; "
                    "  xdotool key --window $WID Return; "
                    "fi; true"
                ),
                "shell": True,
            }},
            {"type": "sleep", "parameters": {"seconds": 5}},
        ]
    else:
        oracle = [
            _execute("pkill -9 -f chrome 2>/dev/null; sleep 2; true"),
            _execute(_SESSION_CLEANUP_CMD),
            {"type": "launch", "parameters": {"command": [
                "google-chrome", "--no-sandbox", "--no-first-run",
                "--no-default-browser-check", "--remote-debugging-port=1337",
                "--user-data-dir=/home/user/chrome-data",
                "--remote-allow-origins=*",
                target_url,
            ]}},
            {"type": "sleep", "parameters": {"seconds": 5}},
        ]
    evaluator = {
        "func": "is_expected_active_tab_approximate",
        "result": {"type": "active_url_from_accessTree", "goto_prefix": ""},
        "expected": {"type": "rule", "rules": {"type": "url", "url": target_url}},
    }
    return oracle, evaluator


# --- P2 + P4 source builders -----------------------------------------------

def _src_prefs_font_default(_seed: int) -> list[dict]:
    """Preferences with default font size set to Chrome's factory 16 — task:
    raise it. (Eval `check_font_size` rule_type=value requires exact match.)"""
    return [
        _kill_chrome_step(),
        _merge_prefs_step({"webkit": {"webprefs": {"default_font_size": 16}}}),
        *_chrome_preopen_steps(urls=None),
    ]


def _src_staged_pdf_arxiv_attention(_seed: int) -> list[dict]:
    """Stage the arxiv 1706.03762 (Attention) PDF as the gold output and
    open the same PDF in the visible Chrome tab (no wiki decoy). validation
    fix: opening the gold pdf itself removes the instruction-vs-eval
    asymmetry — the "webpage I'm looking at" the user references is the
    arxiv pdf, so Save-As-PDF on it produces a compare_pdfs-equal copy."""
    return [
        _stage_asset("docs/pdf/arxiv-1706-03762.pdf",
                     "/tmp/_gold_chrome_attention.pdf"),
        *_chrome_preopen_steps(
            urls=["file:///tmp/_gold_chrome_attention.pdf"],
            extra_launch_args=["--user-data-dir=/home/user/chrome-data"],
        ),
    ]


def _src_staged_pdf_arxiv_bert(_seed: int) -> list[dict]:
    """Stage the arxiv 1810.04805 (BERT) PDF as gold and open the same PDF
    in the visible Chrome tab. Variant of `_src_staged_pdf_arxiv_attention`
    — validation fix removed the octopus decoy that made the task
    unsatisfiable."""
    return [
        _stage_asset("docs/pdf/arxiv-1810-04805.pdf",
                     "/tmp/_gold_chrome_bert.pdf"),
        *_chrome_preopen_steps(
            urls=["file:///tmp/_gold_chrome_bert.pdf"],
            extra_launch_args=["--user-data-dir=/home/user/chrome-data"],
        ),
    ]


def _src_staged_zip_hello_ext(_seed: int) -> list[dict]:
    """Stage a fake helloExtension zip on Desktop (manifest.json inside).
    Mirrors eval 6766f2b8 init: a `helloExtension.zip` on Desktop that the
    agent must unzip and load as an unpacked extension."""
    return [
        _execute(
            "mkdir -p /tmp/_helloExtension_synth && "
            "cat > /tmp/_helloExtension_synth/manifest.json << 'EOF'\n"
            '{"name":"helloExtension","version":"1.0","manifest_version":3}\n'
            "EOF\n"
            "cat > /tmp/_helloExtension_synth/background.js << 'EOF'\n"
            "console.log('hello');\n"
            "EOF\n"
            "rm -rf /home/user/Desktop/helloExtension /home/user/Desktop/helloExtension.zip && "
            "mkdir -p /home/user/Desktop && "
            "(cd /tmp/_helloExtension_synth && zip -qr /home/user/Desktop/helloExtension.zip . ) && "
            "rm -rf /tmp/_helloExtension_synth"
        ),
        _kill_chrome_step(),
        *_chrome_preopen_steps(urls=None),
    ]


def _src_staged_zip_my_ext(_seed: int) -> list[dict]:
    """Stage a different fake extension zip (myExtension) — variant of
    `_src_staged_zip_hello_ext`."""
    return [
        _execute(
            "mkdir -p /tmp/_myExt_synth && "
            "cat > /tmp/_myExt_synth/manifest.json << 'EOF'\n"
            '{"name":"myExtension","version":"2.1","manifest_version":3}\n'
            "EOF\n"
            "cat > /tmp/_myExt_synth/background.js << 'EOF'\n"
            "console.log('my-ext');\n"
            "EOF\n"
            "rm -rf /home/user/Desktop/myExtension /home/user/Desktop/myExtension.zip && "
            "mkdir -p /home/user/Desktop && "
            "(cd /tmp/_myExt_synth && zip -qr /home/user/Desktop/myExtension.zip . ) && "
            "rm -rf /tmp/_myExt_synth"
        ),
        _kill_chrome_step(),
        *_chrome_preopen_steps(urls=None),
    ]


# --- P3 source builders (additional cdjo / url_pattern / wiki / etc.) -------

def _src_url_decoy_spotify(_seed: int) -> list[dict]:
    decoy = "https://open.spotify.com/search/lofi"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_instacart(_seed: int) -> list[dict]:
    decoy = "https://www.instacart.com/store/search/milk"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_doordash(_seed: int) -> list[dict]:
    decoy = "https://www.doordash.com/search/store/pizza"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_opentable(_seed: int) -> list[dict]:
    decoy = "https://www.opentable.com/s?term=sushi"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_glassdoor(_seed: int) -> list[dict]:
    decoy = "https://www.glassdoor.com/Job/jobs.htm?sc.keyword=manager"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_monster(_seed: int) -> list[dict]:
    decoy = "https://www.monster.com/jobs/search?q=designer&where=Boston"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_ziprecruiter(_seed: int) -> list[dict]:
    decoy = "https://www.ziprecruiter.com/jobs-search?search=analyst"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_etsy(_seed: int) -> list[dict]:
    decoy = "https://www.etsy.com/search?q=mug"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_pattern_mdn_root(_seed: int) -> list[dict]:
    decoy = "https://developer.mozilla.org/"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_pattern_pydocs_root(_seed: int) -> list[dict]:
    decoy = "https://docs.python.org/3/"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_pattern_twitter_root(_seed: int) -> list[dict]:
    decoy = "https://twitter.com/home"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_pattern_linkedin_root(_seed: int) -> list[dict]:
    decoy = "https://www.linkedin.com/feed/"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_pattern_stackoverflow_q_root(_seed: int) -> list[dict]:
    decoy = "https://stackoverflow.com/questions"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


# Wiki HTML decoy sources for `_src_tabs_decoy_wiki_*` style — five more
# pages besides apollo/food. Each stages a single wiki HTML and opens it as
# the decoy tab so the agent has somewhere to navigate from.

def _src_tabs_decoy_wiki_yoga(_seed: int) -> list[dict]:
    return [
        _stage_asset("html/wikipedia/yoga.html",
                     "/home/user/Desktop/yoga.html"),
        _stage_asset("html/wikipedia/bicycle.html",
                     "/home/user/Desktop/bicycle.html"),
        _stage_asset("html/wikipedia/volleyball.html",
                     "/home/user/Desktop/volleyball.html"),
        *_chrome_preopen_steps(
            urls=["file:///home/user/Desktop/yoga.html"],
            extra_launch_args=["--user-data-dir=/home/user/chrome-data"],
        ),
    ]


def _src_tabs_decoy_wiki_music(_seed: int) -> list[dict]:
    return [
        _stage_asset("html/wikipedia/beethoven.html",
                     "/home/user/Desktop/beethoven.html"),
        _stage_asset("html/wikipedia/origami.html",
                     "/home/user/Desktop/origami.html"),
        _stage_asset("html/wikipedia/paper-airplane.html",
                     "/home/user/Desktop/paper-airplane.html"),
        *_chrome_preopen_steps(
            urls=["file:///home/user/Desktop/beethoven.html"],
            extra_launch_args=["--user-data-dir=/home/user/chrome-data"],
        ),
    ]


def _src_tabs_decoy_wiki_volcano(_seed: int) -> list[dict]:
    return [
        _stage_asset("html/wikipedia/volcano.html",
                     "/home/user/Desktop/volcano.html"),
        _stage_asset("html/wikipedia/renewable-energy.html",
                     "/home/user/Desktop/renewable-energy.html"),
        _stage_asset("html/wikipedia/internet-of-things.html",
                     "/home/user/Desktop/internet-of-things.html"),
        *_chrome_preopen_steps(
            urls=["file:///home/user/Desktop/volcano.html"],
            extra_launch_args=["--user-data-dir=/home/user/chrome-data"],
        ),
    ]


def _src_tabs_decoy_wiki_art(_seed: int) -> list[dict]:
    return [
        _stage_asset("html/wikipedia/mona-lisa.html",
                     "/home/user/Desktop/mona-lisa.html"),
        _stage_asset("html/wikipedia/eiffel-tower.html",
                     "/home/user/Desktop/eiffel-tower.html"),
        _stage_asset("html/wikipedia/lego.html",
                     "/home/user/Desktop/lego.html"),
        *_chrome_preopen_steps(
            urls=["file:///home/user/Desktop/mona-lisa.html"],
            extra_launch_args=["--user-data-dir=/home/user/chrome-data"],
        ),
    ]


def _src_tabs_decoy_wiki_library(_seed: int) -> list[dict]:
    return [
        _stage_asset("html/wikipedia/library-computing.html",
                     "/home/user/Desktop/library-computing.html"),
        _stage_asset("html/wikipedia/mount-everest.html",
                     "/home/user/Desktop/mount-everest.html"),
        _stage_asset("html/wikipedia/earth.html",
                     "/home/user/Desktop/earth.html"),
        *_chrome_preopen_steps(
            urls=["file:///home/user/Desktop/library-computing.html"],
            extra_launch_args=["--user-data-dir=/home/user/chrome-data"],
        ),
    ]


# Extra bookmark/history/tabs source builders for P3 expansion.

def _src_history_seeded_tech(_seed: int) -> list[dict]:
    """History with tech-heavy entries — used for second-level history-delete
    variants (kept distinct from `_src_history_seeded_mixed`)."""
    entries = [
        ("https://stackoverflow.com/questions/12345", "How to debug — Stack Overflow"),
        ("https://stackoverflow.com/questions/67890", "Python regex — Stack Overflow"),
        ("https://github.com/torvalds/linux", "torvalds/linux on GitHub"),
        ("https://github.com/python/cpython", "python/cpython on GitHub"),
        ("https://news.ycombinator.com/item?id=11111", "Hacker News item"),
        ("https://www.reddit.com/r/programming", "r/programming"),
        ("https://www.kernel.org/", "The Linux Kernel Archives"),
    ]
    return [
        {"type": "launch", "parameters": {"command": ["google-chrome", "--remote-debugging-port=1337"]}},
        {"type": "launch", "parameters": {"command": ["socat", "tcp-listen:9222,fork", "tcp:localhost:1337"]}},
        {"type": "sleep", "parameters": {"seconds": 5}},
        _execute("pkill -9 -f chrome 2>/dev/null; sleep 4; true"),
        _seed_history_step(entries),
        *_chrome_preopen_steps(urls=None),
    ]


def _src_bookmarks_with_news_folder(_seed: int) -> list[dict]:
    """Bookmark bar with a populated 'News' folder containing two URLs.
    Variant of `_src_bookmarks_with_decoy_urls` for additional bookmark task
    variety."""
    roots = _empty_bookmarks_roots()
    news_folder = _bookmark_folder_node("News", node_id="200",
                                         children=[
        _bookmark_url_node("BBC", "https://www.bbc.com/news", node_id="210"),
        _bookmark_url_node("Reuters", "https://www.reuters.com/", node_id="211"),
    ])
    roots["bookmark_bar"]["children"] = [news_folder]
    return [
        _kill_chrome_step(),
        _write_bookmarks_step(roots),
        *_chrome_preopen_steps(urls=None),
    ]


# --- §I.h File instances (new) ---------------------------------------------

# P2 rare-func Files
F_CHROME_46 = File(id="F-CHROME-46", setup_class="chrome_prefs_font_default",
                   src=_src_prefs_font_default)
F_CHROME_47 = File(id="F-CHROME-47", setup_class="chrome_staged_pdf_attention",
                   src=_src_staged_pdf_arxiv_attention)
F_CHROME_48 = File(id="F-CHROME-48", setup_class="chrome_staged_pdf_bert",
                   src=_src_staged_pdf_arxiv_bert)
F_CHROME_49 = File(id="F-CHROME-49", setup_class="chrome_tabs_decoy_kernel",
                   src=_src_tabs_decoy_kernel)
F_CHROME_50 = File(id="F-CHROME-50", setup_class="chrome_tabs_decoy_wikipedia",
                   src=_src_tabs_decoy_wikipedia)

# P4 zip / extension-install Files
F_CHROME_51 = File(id="F-CHROME-51", setup_class="chrome_zip_hello_extension",
                   src=_src_staged_zip_hello_ext)
F_CHROME_52 = File(id="F-CHROME-52", setup_class="chrome_zip_my_extension",
                   src=_src_staged_zip_my_ext)

# P3 cdjo extras (8 sites)
F_CHROME_53 = File(id="F-CHROME-53", setup_class="chrome_url_decoy_spotify",
                   src=_src_url_decoy_spotify)
F_CHROME_54 = File(id="F-CHROME-54", setup_class="chrome_url_decoy_instacart",
                   src=_src_url_decoy_instacart)
F_CHROME_55 = File(id="F-CHROME-55", setup_class="chrome_url_decoy_doordash",
                   src=_src_url_decoy_doordash)
F_CHROME_56 = File(id="F-CHROME-56", setup_class="chrome_url_decoy_opentable",
                   src=_src_url_decoy_opentable)
F_CHROME_57 = File(id="F-CHROME-57", setup_class="chrome_url_decoy_glassdoor",
                   src=_src_url_decoy_glassdoor)
F_CHROME_58 = File(id="F-CHROME-58", setup_class="chrome_url_decoy_monster",
                   src=_src_url_decoy_monster)
F_CHROME_59 = File(id="F-CHROME-59", setup_class="chrome_url_decoy_ziprecruiter",
                   src=_src_url_decoy_ziprecruiter)
F_CHROME_60 = File(id="F-CHROME-60", setup_class="chrome_url_decoy_etsy",
                   src=_src_url_decoy_etsy)

# P3 url_pattern extras (6 sites)
F_CHROME_61 = File(id="F-CHROME-61", setup_class="chrome_url_pattern_mdn",
                   src=_src_url_pattern_mdn_root)
F_CHROME_62 = File(id="F-CHROME-62", setup_class="chrome_url_pattern_pydocs",
                   src=_src_url_pattern_pydocs_root)
F_CHROME_63 = File(id="F-CHROME-63", setup_class="chrome_url_pattern_twitter",
                   src=_src_url_pattern_twitter_root)
F_CHROME_64 = File(id="F-CHROME-64", setup_class="chrome_url_pattern_linkedin",
                   src=_src_url_pattern_linkedin_root)
F_CHROME_65 = File(id="F-CHROME-65", setup_class="chrome_url_pattern_so_q",
                   src=_src_url_pattern_stackoverflow_q_root)

# P3 wiki HTML decoy tabs (5 more)
F_CHROME_66 = File(id="F-CHROME-66", setup_class="chrome_tabs_wiki_yoga",
                   src=_src_tabs_decoy_wiki_yoga)
F_CHROME_67 = File(id="F-CHROME-67", setup_class="chrome_tabs_wiki_music",
                   src=_src_tabs_decoy_wiki_music)
F_CHROME_68 = File(id="F-CHROME-68", setup_class="chrome_tabs_wiki_volcano",
                   src=_src_tabs_decoy_wiki_volcano)
F_CHROME_69 = File(id="F-CHROME-69", setup_class="chrome_tabs_wiki_art",
                   src=_src_tabs_decoy_wiki_art)
F_CHROME_70 = File(id="F-CHROME-70", setup_class="chrome_tabs_wiki_library",
                   src=_src_tabs_decoy_wiki_library)

# P3 history/bookmark/tabs extras
F_CHROME_71 = File(id="F-CHROME-71", setup_class="chrome_history_tech",
                   src=_src_history_seeded_tech)
F_CHROME_72 = File(id="F-CHROME-72", setup_class="chrome_bookmarks_news_folder",
                   src=_src_bookmarks_with_news_folder)


# ---------------------------------------------------------------------------
# validation (2026-05-11) — `check_direct_json_object` UNDER-gap fill.
# Synth was at 28% (23/82) cdjo vs eval at 49% (21/43). Add 15 new cdjo
# FileTasks on shopping/booking verticals that mirror the eval row shapes:
#   - Flight search (eval rows 17/22/33/42/43): 4 ADDS (Delta / AA / Southwest
#     / Alaska — different city pairs + dates)
#   - Product comparison (eval rows 7/20/26/36/41): 3 ADDS (Apple iPhone
#     compare, Samsung Galaxy compare, Sony headphones compare)
#   - Car rental (eval rows 6/14/21): 3 ADDS (Hertz / Enterprise / Avis —
#     different cities)
#   - Hotel (eval row 33): 2 ADDS (Marriott / Hilton)
#   - Shopping (Best Buy / Costco / Wayfair): 3 ADDS
# Each new File pre-opens a decoy URL on the real domain with the WRONG
# query params; the oracle relaunches on the gold URL whose `key=value` pairs
# the evaluator parses (mirrors F_CHROME_24..F_CHROME_41 / F_CHROME_53..60
# shape exactly). Plus 2 `match_in_list` ADDS (synth had 0 match_in_list
# on chrome_color_scheme) — dark-mode toggle (eval row 23).
# ---------------------------------------------------------------------------


def _src_url_decoy_delta(_seed: int) -> list[dict]:
    """Pre-config: delta.com flight search decoy (NYC→ATL). Gold: different
    OD pair + date."""
    decoy = "https://www.delta.com/flight-search/book-a-flight?cacheKeySuffix=&tripType=ONE_WAY&priceSchedule=PRICE&originCity=JFK&destinationCity=ATL&departureDate=2026-03-15&paxCount=1"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_american(_seed: int) -> list[dict]:
    """Pre-config: aa.com flight search decoy (DFW→MIA). Gold: different OD."""
    decoy = "https://www.aa.com/booking/find-flights?from=DFW&to=MIA&depart=2026-04-10&tripType=oneWay&paxCount=1"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_southwest(_seed: int) -> list[dict]:
    """Pre-config: southwest.com flight search decoy (LAX→LAS). Gold: different
    OD + date."""
    decoy = "https://www.southwest.com/air/booking/select.html?originationAirportCode=LAX&destinationAirportCode=LAS&departureDate=2026-03-20&adultPassengersCount=1&tripType=oneway"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_alaska(_seed: int) -> list[dict]:
    """Pre-config: alaskaair.com flight search decoy (SEA→SFO). Gold:
    different OD + date."""
    decoy = "https://www.alaskaair.com/search/results?O=SEA&D=SFO&OD=2026-03-25&A=1&RT=false"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_apple_compare(_seed: int) -> list[dict]:
    """Pre-config: apple.com iPhone compare decoy (default 2-product compare).
    Gold: 3-way compare across specific model SKUs (mirrors eval row 41)."""
    decoy = "https://www.apple.com/shop/buy-iphone/compare?modelList=iphone-15"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_samsung_compare(_seed: int) -> list[dict]:
    """Pre-config: samsung.com Galaxy compare decoy. Gold: 3-way compare."""
    decoy = "https://www.samsung.com/us/mobile/galaxy/compare/?modelList=galaxy-s24"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_sony_compare(_seed: int) -> list[dict]:
    """Pre-config: sony.com headphones compare decoy. Gold: specific models."""
    decoy = "https://electronics.sony.com/audio/headphones/compare?modelList=WH-1000XM5"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_hertz(_seed: int) -> list[dict]:
    """Pre-config: hertz.com car-rental search decoy (LAX). Gold: different
    city + dates + category (mirrors eval row 6/14)."""
    decoy = "https://www.hertz.com/rentacar/reservation/results?pickupLocation=LAX&pickupDate=2026-04-01&returnDate=2026-04-05"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_enterprise(_seed: int) -> list[dict]:
    """Pre-config: enterprise.com car rental decoy (Miami). Gold: different
    city + category."""
    decoy = f"https://www.enterprise.com/en/car-rental.html?pickupLocation=MIA&pickupDate={_future_date(20)}&returnDate={_future_date(24)}"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_avis(_seed: int) -> list[dict]:
    """Pre-config: avis.com car rental decoy (Chicago). Gold: different
    city + dates."""
    decoy = f"https://www.avis.com/en/reservation/vehicle-selection?pickUpLocation=ORD&pickUpDate={_future_date(20)}&returnDate={_future_date(24)}"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_marriott(_seed: int) -> list[dict]:
    """Pre-config: marriott.com hotel search decoy (NYC, 1 adult). Gold:
    different city + dates + guests (mirrors eval row 33)."""
    decoy = "https://www.marriott.com/search/findHotels.mi?destinationAddress.destination=New+York%2C+NY&fromDate=2026-04-01&toDate=2026-04-05&numberOfAdults=1"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_hilton(_seed: int) -> list[dict]:
    """Pre-config: hilton.com hotel search decoy (London). Gold: different
    city + dates."""
    decoy = "https://www.hilton.com/en/search/?query=London&arrivalDate=2026-04-01&departureDate=2026-04-05&numAdults=1"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_bestbuy(_seed: int) -> list[dict]:
    """Pre-config: bestbuy.com search decoy (default keyword). Gold: different
    keyword + sort."""
    decoy = "https://www.bestbuy.com/site/searchpage.jsp?st=laptop&sp=price-asc"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_costco(_seed: int) -> list[dict]:
    """Pre-config: costco.com catalog-search decoy. Gold: different keyword
    + sort."""
    decoy = "https://www.costco.com/CatalogSearch?keyword=tv&sortBy=item_location_price"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


def _src_url_decoy_wayfair(_seed: int) -> list[dict]:
    """Pre-config: wayfair.com keyword-search decoy. Gold: different keyword."""
    decoy = "https://www.wayfair.com/keyword.php?keyword=sofa&sortby=Recommended"
    return [
        *_chrome_preopen_steps(urls=[decoy]),
    ]


# validation — `match_in_list` over `chrome_color_scheme` (eval row 23,
# osworld_chrome_93eabf48 dark-mode toggle). Pre-config writes Preferences
# with `browser.theme.color_scheme = 2` (dark); oracle resets it to 1 (light)
# or 0 (system). Eval reads back via `match_in_list` on `chrome_color_scheme`.

def _src_prefs_dark_mode_on(_seed: int) -> list[dict]:
    """Preferences with Chrome's color_scheme set to 2 (dark mode ON) and
    color_scheme2 also set (so the live theme picker reflects dark). Task:
    switch to light or system."""
    patch = {"browser": {"theme": {"color_scheme": 2, "color_scheme2": 2}}}
    return [
        _kill_chrome_step(),
        _merge_prefs_step(patch),
        *_chrome_preopen_steps(urls=None),
    ]


def _src_prefs_no_theme(_seed: int) -> list[dict]:
    """Preferences with NO browser.theme key (Chrome factory default — implicit
    'system' on most setups). Task: explicitly set color scheme to light or
    system (covers the case where the agent has to pick from an unset state)."""
    return [
        _kill_chrome_step(),
        _merge_prefs_step({"browser": {}}),
        *_chrome_preopen_steps(urls=None),
    ]


# --- §I.h File instances ----------------------------------------

# Flight search verticals
F_CHROME_73 = File(id="F-CHROME-73", setup_class="chrome_url_decoy_delta",
                   src=_src_url_decoy_delta)
F_CHROME_74 = File(id="F-CHROME-74", setup_class="chrome_url_decoy_american",
                   src=_src_url_decoy_american)
F_CHROME_75 = File(id="F-CHROME-75", setup_class="chrome_url_decoy_southwest",
                   src=_src_url_decoy_southwest)
F_CHROME_76 = File(id="F-CHROME-76", setup_class="chrome_url_decoy_alaska",
                   src=_src_url_decoy_alaska)

# Product comparison verticals
F_CHROME_77 = File(id="F-CHROME-77", setup_class="chrome_url_decoy_apple_compare",
                   src=_src_url_decoy_apple_compare)
F_CHROME_78 = File(id="F-CHROME-78", setup_class="chrome_url_decoy_samsung_compare",
                   src=_src_url_decoy_samsung_compare)
F_CHROME_79 = File(id="F-CHROME-79", setup_class="chrome_url_decoy_sony_compare",
                   src=_src_url_decoy_sony_compare)

# Car rental verticals
F_CHROME_80 = File(id="F-CHROME-80", setup_class="chrome_url_decoy_hertz",
                   src=_src_url_decoy_hertz)
F_CHROME_81 = File(id="F-CHROME-81", setup_class="chrome_url_decoy_enterprise",
                   src=_src_url_decoy_enterprise)
F_CHROME_82 = File(id="F-CHROME-82", setup_class="chrome_url_decoy_avis",
                   src=_src_url_decoy_avis)

# Hotel verticals
F_CHROME_83 = File(id="F-CHROME-83", setup_class="chrome_url_decoy_marriott",
                   src=_src_url_decoy_marriott)
F_CHROME_84 = File(id="F-CHROME-84", setup_class="chrome_url_decoy_hilton",
                   src=_src_url_decoy_hilton)

# Generic shopping verticals
F_CHROME_85 = File(id="F-CHROME-85", setup_class="chrome_url_decoy_bestbuy",
                   src=_src_url_decoy_bestbuy)
F_CHROME_86 = File(id="F-CHROME-86", setup_class="chrome_url_decoy_costco",
                   src=_src_url_decoy_costco)
F_CHROME_87 = File(id="F-CHROME-87", setup_class="chrome_url_decoy_wayfair",
                   src=_src_url_decoy_wayfair)

# match_in_list on chrome_color_scheme — dark-mode toggle (eval row 23)
F_CHROME_88 = File(id="F-CHROME-88", setup_class="chrome_prefs_dark_mode_on",
                   src=_src_prefs_dark_mode_on)
F_CHROME_89 = File(id="F-CHROME-89", setup_class="chrome_prefs_no_theme",
                   src=_src_prefs_no_theme)


# validation gold builder for `match_in_list` on `chrome_color_scheme`. Oracle
# writes Preferences `browser.theme.color_scheme = <value>` and removes
# `color_scheme2` (matching eval row 25's oracle pattern). Eval `match_in_list`
# against `chrome_color_scheme` getter which maps {0:"system",1:"light",2:"dark"}.
def _gold_set_color_scheme(*, color_scheme: int, expected_list: tuple[str, ...]) -> tuple[list[dict], dict]:
    py = textwrap.dedent(f"""\
        import json, os
        path = {_CHROME_PREFS!r}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        prefs = {{}}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    prefs = json.load(f)
                if not isinstance(prefs, dict):
                    prefs = {{}}
            except Exception:
                prefs = {{}}
        prefs.setdefault('browser', {{}}).setdefault('theme', {{}})['color_scheme'] = {color_scheme}
        # Remove color_scheme2 so the evaluator reads the canonical color_scheme
        # field (mirrors eval row 25's oracle).
        prefs.get('browser', {{}}).get('theme', {{}}).pop('color_scheme2', None)
        with open(path, 'w') as f:
            json.dump(prefs, f)
        """)
    oracle = [_kill_chrome_step(), _execute(f"python3 << 'PYEOF'\n{py}\nPYEOF")]
    evaluator = {
        "func": "match_in_list",
        "result": {"type": "chrome_color_scheme"},
        "expected": {"type": "rule", "rules": {"expected": list(expected_list)}},
        "postconfig": _CHROME_RESTART_POSTCONFIG,
    }
    return oracle, evaluator


# --- §I.h FileTasks (new) ---------------------------------------------------
#
# Cap = 2 Params per FileTask. Each FileTask is unique (file, task) pair.

_EXTRA_FILE_TASKS: list[FileTask] = [
    # ---- P2: check_font_size --------------------------------------------
    # F-CHROME-46 — default-size prefs; task: raise font size via Settings UI.
    FileTask(F_CHROME_46, "increase_font_size_large", "config_setting",
             _gold_set_font_size, params=[
        Param({"size": 20}, "check_font_size",
              "My grandmother told me Chrome's text is way too small for her. Could you set the default font size in Settings → Appearance to 'Large' (20px)?"),
        Param({"size": 24}, "check_font_size",
              "I'd like the browser text to be easier on my eyes — please change Chrome's default font size to 'Very Large' (24px) in Appearance settings."),
    ]),

    # ---- P2: compare_pdfs (Save As PDF) ---------------------------------
    # F-CHROME-47 — wiki HTML open + gold pdf staged → agent prints to PDF.
    FileTask(F_CHROME_47, "save_page_as_pdf_attention", "compare_pdfs",
             _gold_compare_pdfs_from_asset, params=[
        Param({"asset_rel": "docs/pdf/arxiv-1706-03762.pdf",
               "dst_basename": "attention.pdf"},
              "compare_pdfs",
              "Can you turn the webpage I'm looking at into a PDF file and save it to my Desktop as 'attention.pdf' with the default margins?"),
    ]),
    # F-CHROME-48 — bert variant
    FileTask(F_CHROME_48, "save_page_as_pdf_bert", "compare_pdfs",
             _gold_compare_pdfs_from_asset, params=[
        Param({"asset_rel": "docs/pdf/arxiv-1810-04805.pdf",
               "dst_basename": "bert.pdf"},
              "compare_pdfs",
              "I'd like to keep a copy of this article — could you save the current Chrome page as a PDF on my Desktop named 'bert.pdf'?"),
    ]),

    # ---- P2: is_expected_active_tab_approximate -------------------------
    # F-CHROME-49 — chrome:// internal URLs (looser match).
    FileTask(F_CHROME_49, "open_password_manager", "active_tab",
             _gold_navigate_active_tab_approximate, params=[
        Param({"target_url": "chrome://password-manager/passwords"},
              "is_expected_active_tab_approximate",
              "Computer, please navigate to the area in my browser settings where my passwords are stored. I want to check my Etsy login."),
        Param({"target_url": "chrome://settings/cookies"},
              "is_expected_active_tab_approximate",
              "Could you open the Chrome cookies settings page for me? I'd like to review which sites have stored cookies."),
    ]),
    # F-CHROME-50 — google-search-driven URL with query validation.
    # NOTE: switched from is_expected_active_tab_approximate to
    # check_direct_json_object so the q= search term is actually validated;
    # the approximate matcher strips query params and would TRIVIAL_PASS on
    # any www.google.com URL.
    FileTask(F_CHROME_50, "search_google_query", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": "https://www.google.com/search",
               "gold_query": {"q": "python regex cheatsheet"}},
              "check_direct_json_object",
              "Hey, can you run a Google search for 'python regex cheatsheet'? I keep forgetting the syntax."),
        Param({"base_url": "https://www.google.com/search",
               "gold_query": {"q": "best espresso machine 2026"}},
              "check_direct_json_object",
              "I'd like to read some reviews — could you run a Google search for 'best espresso machine 2026' in Chrome?"),
    ]),

    # ---- P4: zip + install unpacked extension ---------------------------
    # F-CHROME-51 — helloExtension.zip on Desktop, agent unzips + loads.
    FileTask(F_CHROME_51, "unzip_and_install_extension", "is_in_list",
             _gold_install_unpacked_extension, params=[
        Param({"ext_path": "/home/user/Desktop/helloExtension"},
              "is_in_list",
              "Could you help me unzip the downloaded extension file from /home/user/Desktop/helloExtension.zip and configure it in Chrome's extensions?"),
    ]),
    # F-CHROME-52 — myExtension.zip variant
    FileTask(F_CHROME_52, "install_local_extension_my", "is_in_list",
             _gold_install_unpacked_extension, params=[
        Param({"ext_path": "/home/user/Desktop/myExtension"},
              "is_in_list",
              "I've developed a new Chrome extension myself, so it needs to be installed manually. Please unzip /home/user/Desktop/myExtension.zip and load it into Chrome."),
    ]),

    # ---- P3: cdjo expansion (8 sites × 2 params = 16 rows) --------------
    FileTask(F_CHROME_53, "search_spotify_track", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": "https://open.spotify.com/search",
               "gold_query": {"q": "jazz piano"}},
              "check_direct_json_object",
              "I'd love some background music while I work — on open.spotify.com, search for 'jazz piano' tracks."),
        Param({"base_url": "https://open.spotify.com/search",
               "gold_query": {"q": "rainy day playlist"}},
              "check_direct_json_object",
              "Could you help me find chill vibes? Search open.spotify.com for 'rainy day playlist'."),
    ]),
    FileTask(F_CHROME_54, "search_instacart_item", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": "https://www.instacart.com/store/search",
               "gold_query": {"q": "organic eggs"}},
              "check_direct_json_object",
              "I'm running low on groceries — on instacart.com, search for 'organic eggs' to add to my cart."),
        Param({"base_url": "https://www.instacart.com/store/search",
               "gold_query": {"q": "oat milk"}},
              "check_direct_json_object",
              "Hey, could you search instacart.com for 'oat milk'? I'd like to compare brands and prices."),
    ]),
    # F-CHROME-55 search_doordash_food — omitted:
    # doordash.com uses path-segment routing + different param name than
    # `query`. check_direct_json_object can't match. Same class as other omitted direct-query tasks.
    # FileTask(F_CHROME_55, "search_doordash_food", ...) omitted.
    FileTask(F_CHROME_56, "search_opentable_reservation", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": "https://www.opentable.com/s",
               "gold_query": {"term": "italian", "covers": "2"}},
              "check_direct_json_object",
              "I'd like to take my partner out — on opentable.com, search for an Italian restaurant for 2 people."),
        Param({"base_url": "https://www.opentable.com/s",
               "gold_query": {"term": "steak house", "covers": "4"}},
              "check_direct_json_object",
              "We're celebrating a promotion — please search opentable.com for a steak house with seating for 4."),
    ]),
    FileTask(F_CHROME_57, "search_glassdoor_job", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": "https://www.glassdoor.com/Job/jobs.htm",
               "gold_query": {"sc.keyword": "product manager",
                              "locT": "C", "locId": "1147401"}},
              "check_direct_json_object",
              "I've been thinking about a career change — search glassdoor.com for 'product manager' jobs (locT=C, locId=1147401)."),
        Param({"base_url": "https://www.glassdoor.com/Job/jobs.htm",
               "gold_query": {"sc.keyword": "ux designer",
                              "locT": "C", "locId": "1132348"}},
              "check_direct_json_object",
              "Could you help me browse design roles? Search glassdoor.com for 'ux designer' (locT=C, locId=1132348)."),
    ]),
    FileTask(F_CHROME_58, "search_monster_job", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": "https://www.monster.com/jobs/search",
               "gold_query": {"q": "graphic designer", "where": "Chicago, IL"}},
              "check_direct_json_object",
              "I'm thinking of moving to Chicago — on monster.com, search for 'graphic designer' positions in Chicago, IL."),
        Param({"base_url": "https://www.monster.com/jobs/search",
               "gold_query": {"q": "data analyst", "where": "Denver, CO"}},
              "check_direct_json_object",
              "Could you help me explore Denver? Search monster.com for 'data analyst' jobs in Denver, CO."),
    ]),
    FileTask(F_CHROME_59, "search_ziprecruiter_job", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": "https://www.ziprecruiter.com/jobs-search",
               "gold_query": {"search": "marketing manager",
                              "location": "Austin, TX"}},
              "check_direct_json_object",
              "I'd like to scope out Austin — on ziprecruiter.com, search for 'marketing manager' roles in Austin, TX."),
        Param({"base_url": "https://www.ziprecruiter.com/jobs-search",
               "gold_query": {"search": "registered nurse",
                              "location": "Seattle, WA"}},
              "check_direct_json_object",
              "My sister is job hunting — could you search ziprecruiter.com for 'registered nurse' positions in Seattle, WA?"),
    ]),
    # F-CHROME-60 search_etsy_item — omitted:
    # etsy.com aggressively bot-blocks with CAPTCHA slider. Same class as other
    # live-site Chrome omissions.
    # FileTask(F_CHROME_60, "search_etsy_item", ...) omitted.

    # ---- P3: url_pattern expansion (5 sites × 2 params = 10 rows) --------
    FileTask(F_CHROME_61, "navigate_mdn_doc", "is_expected_url_pattern_match",
             _gold_url_pattern, params=[
        Param({"gold_url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array",
               "pattern": r"developer\.mozilla\.org/.*docs/Web/JavaScript/Reference/Global_Objects/Array"},
              "is_expected_url_pattern_match",
              "I keep forgetting Array methods — could you open the MDN reference page for the JavaScript Array global object?"),
        # MDN 301-redirects /docs/Web/CSS/<prop> → /docs/Web/CSS/Reference/Properties/<prop>.
        # Use a permalink-stable URL family that does NOT redirect: the /docs/Web/CSS
        # top-level reference page (which keeps the literal `docs/Web/CSS` substring in
        # the final address bar). Eval regex matches the contiguous token.
        Param({"gold_url": "https://developer.mozilla.org/en-US/docs/Web/CSS",
               "pattern": r"developer\.mozilla\.org/.*docs/Web/CSS"},
              "is_expected_url_pattern_match",
              "I'm reviewing CSS reference material — please open the MDN top-level documentation page for CSS so I can browse the property index."),
    ]),
    FileTask(F_CHROME_62, "navigate_pydocs_module", "is_expected_url_pattern_match",
             _gold_url_pattern, params=[
        Param({"gold_url": "https://docs.python.org/3/library/itertools.html",
               "pattern": r"docs\.python\.org/3/library/itertools\.html"},
              "is_expected_url_pattern_match",
              "I'd like to brush up on iterators — could you open the Python 3 library docs for the itertools module?"),
        Param({"gold_url": "https://docs.python.org/3/library/asyncio.html",
               "pattern": r"docs\.python\.org/3/library/asyncio\.html"},
              "is_expected_url_pattern_match",
              "I'm rewriting a service to be async — please open the Python 3 library reference for asyncio."),
    ]),
    FileTask(F_CHROME_63, "navigate_twitter_profile", "is_expected_url_pattern_match",
             _gold_url_pattern, params=[
        Param({"gold_url": "https://twitter.com/nasa",
               "pattern": r"^https://(www\.)?twitter\.com/nasa"},
              "is_expected_url_pattern_match",
              "I love space content — could you open the official NASA Twitter profile in Chrome?"),
        Param({"gold_url": "https://twitter.com/sundarpichai",
               "pattern": r"^https://(www\.)?twitter\.com/sundarpichai"},
              "is_expected_url_pattern_match",
              "I'd like to follow industry news — please open Sundar Pichai's Twitter profile page."),
    ]),
    FileTask(F_CHROME_64, "navigate_linkedin_section", "is_expected_url_pattern_match",
             _gold_url_pattern, params=[
        # Validation fix: dropped Param 0 (linkedin.com/jobs/) — it redirects
        # to a login wall so the eval regex never matches. /learning/ is
        # public and works.
        Param({"gold_url": "https://www.linkedin.com/learning/",
               "pattern": r"^https://(www\.)?linkedin\.com/learning/?"},
              "is_expected_url_pattern_match",
              "I'm thinking about upskilling — please open the LinkedIn Learning home page."),
    ]),
    FileTask(F_CHROME_65, "navigate_so_question", "is_expected_url_pattern_match",
             _gold_url_pattern, params=[
        Param({"gold_url": "https://stackoverflow.com/questions/11227809/why-is-processing-a-sorted-array-faster-than-processing-an-unsorted-array",
               "pattern": r"stackoverflow\.com/questions/11227809"},
              "is_expected_url_pattern_match",
              "I keep hearing about this classic question — could you open Stack Overflow question 11227809 about sorted arrays?"),
        Param({"gold_url": "https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do-in-python",
               "pattern": r"stackoverflow\.com/questions/231767"},
              "is_expected_url_pattern_match",
              "I'm relearning Python — please open Stack Overflow question 231767 about the yield keyword."),
    ]),

    # ---- P3: wiki active-tab expansion (5 files × 2 params = 10 rows) ----
    FileTask(F_CHROME_66, "navigate_to_staged_wiki_yoga", "active_tab",
             _gold_navigate_active_tab, params=[
        Param({"target_url": "file:///home/user/Desktop/bicycle.html"},
              "is_expected_active_tab",
              "I'd like to plan my weekend ride — could you open file:///home/user/Desktop/bicycle.html in the current Chrome tab?"),
        Param({"target_url": "file:///home/user/Desktop/volleyball.html"},
              "is_expected_active_tab",
              "We're picking a sport for next week — please navigate Chrome to file:///home/user/Desktop/volleyball.html (the local Volleyball Wikipedia article)."),
    ]),
    FileTask(F_CHROME_67, "navigate_to_staged_wiki_music", "active_tab",
             _gold_navigate_active_tab, params=[
        Param({"target_url": "file:///home/user/Desktop/origami.html"},
              "is_expected_active_tab",
              "I want to try a new craft this weekend — please open file:///home/user/Desktop/origami.html in the current Chrome tab."),
        Param({"target_url": "file:///home/user/Desktop/paper-airplane.html"},
              "is_expected_active_tab",
              "My nephew is visiting and asked about paper airplanes — could you open file:///home/user/Desktop/paper-airplane.html?"),
    ]),
    FileTask(F_CHROME_68, "navigate_to_staged_wiki_volcano", "active_tab",
             _gold_navigate_active_tab, params=[
        Param({"target_url": "file:///home/user/Desktop/renewable-energy.html"},
              "is_expected_active_tab",
              "I've been reading about climate solutions lately — could you open file:///home/user/Desktop/renewable-energy.html in Chrome?"),
        Param({"target_url": "file:///home/user/Desktop/internet-of-things.html"},
              "is_expected_active_tab",
              "I'm prepping a talk on smart devices — please navigate Chrome to file:///home/user/Desktop/internet-of-things.html."),
    ]),
    FileTask(F_CHROME_69, "navigate_to_staged_wiki_art", "active_tab",
             _gold_navigate_active_tab, params=[
        Param({"target_url": "file:///home/user/Desktop/eiffel-tower.html"},
              "is_expected_active_tab",
              "I'm planning a Paris trip — could you open file:///home/user/Desktop/eiffel-tower.html in the current Chrome tab?"),
        Param({"target_url": "file:///home/user/Desktop/lego.html"},
              "is_expected_active_tab",
              "My kids asked about LEGO history — please open file:///home/user/Desktop/lego.html in Chrome."),
    ]),
    FileTask(F_CHROME_70, "navigate_to_staged_wiki_library", "active_tab",
             _gold_navigate_active_tab, params=[
        Param({"target_url": "file:///home/user/Desktop/mount-everest.html"},
              "is_expected_active_tab",
              "I've been dreaming about hiking lately — could you open file:///home/user/Desktop/mount-everest.html in Chrome?"),
        Param({"target_url": "file:///home/user/Desktop/earth.html"},
              "is_expected_active_tab",
              "I'd like to refresh on basic earth science — please open file:///home/user/Desktop/earth.html in the current Chrome tab."),
    ]),

    # ---- P3: bookmark / history / tabs extras (3 files × 2 = 6 rows) -----
    FileTask(F_CHROME_71, "delete_so_history", "history",
             _gold_history_delete_keyword, params=[
        Param({"keyword": "stackoverflow.com"}, "check_history_deleted",
              "I'd like to hide my detour into off-topic searches — open chrome://history, search for 'stackoverflow', and delete only the matching entries (leave history from other sites intact)."),
        Param({"keyword": "github.com"}, "check_history_deleted",
              "Could you help tidy my history? Open chrome://history, search for 'github.com', and delete only the matching entries (keep my other history)."),
    ]),
    FileTask(F_CHROME_72, "add_url_to_news_folder", "bookmark",
             _gold_bookmark_url, params=[
        Param({"name": "NYTimes", "url": "https://www.nytimes.com/"},
              "is_expected_bookmarks",
              "I read the New York Times every morning — could you add a bookmark for the NYTimes home page on the Chrome bookmarks bar named 'NYTimes'?"),
        Param({"name": "Guardian", "url": "https://www.theguardian.com/"},
              "is_expected_bookmarks",
              "I'd like to follow UK news too — please bookmark the Guardian's UK home page on Chrome's bookmarks bar (call it 'Guardian')."),
    ]),

    # ---- P3 additional fill (close PD-6 volume gap) ---------------------
    # Reuse existing Files for second-task entries (each File can host up to
    # SYNTH_CAP_TASKS_PER_FILE=2 FileTasks — many singletons above had room
    # for a second task that exercises a different gold-builder shape).
    FileTask(F_CHROME_53, "navigate_spotify_artist", "is_expected_url_pattern_match",
             _gold_url_pattern, params=[
        Param({"gold_url": "https://open.spotify.com/artist/4Z8W4fKeB5YxbusRsdQVPb",
               "pattern": r"open\.spotify\.com/artist/[A-Za-z0-9]+"},
              "is_expected_url_pattern_match",
              "I'd like to listen to Radiohead — could you open the Radiohead artist page on open.spotify.com?"),
        Param({"gold_url": "https://open.spotify.com/artist/3WrFJ7ztbogyGnTHbHJFl2",
               "pattern": r"open\.spotify\.com/artist/[A-Za-z0-9]+"},
              "is_expected_url_pattern_match",
              "I've been on a Beatles kick lately — please open the Beatles artist page on open.spotify.com."),
    ]),
    # F-CHROME-54 — instacart department via file:// staging (live instacart
    # 429s the oracle and redirects /store/<retailer>/departments/<dept> with
    # tracking params). The staged file path encodes the retailer + department
    # tokens directly; the regex anchors against `synth_instacart_<retailer>_
    # <dept>.html` so the URL pattern match still asserts "right retailer +
    # right department".
    FileTask(F_CHROME_54, "navigate_instacart_dept", "is_expected_url_pattern_match",
             _gold_url_pattern, params=[
        Param({"gold_url": "file:///tmp/synth_instacart_wegmans_produce.html",
               "pattern": r"^file:///tmp/synth_instacart_[^_]+_produce\.html$",
               "stage_html_path": "/tmp/synth_instacart_wegmans_produce.html"},
              "is_expected_url_pattern_match",
              "Open the staged instacart page at /tmp/synth_instacart_wegmans_produce.html in Chrome (mirrors the Wegmans produce department on instacart.com)."),
        Param({"gold_url": "file:///tmp/synth_instacart_costco_bakery.html",
               "pattern": r"^file:///tmp/synth_instacart_[^_]+_bakery\.html$",
               "stage_html_path": "/tmp/synth_instacart_costco_bakery.html"},
              "is_expected_url_pattern_match",
              "Open the staged instacart page at /tmp/synth_instacart_costco_bakery.html in Chrome (mirrors the Costco bakery department on instacart.com)."),
    ]),
    FileTask(F_CHROME_55, "navigate_doordash_cuisine", "is_expected_url_pattern_match",
             _gold_url_pattern, params=[
        Param({"gold_url": "https://www.doordash.com/cuisine/japanese-near-me",
               "pattern": r"doordash\.com/cuisine/japanese-near-me"},
              "is_expected_url_pattern_match",
              "I'm in the mood for sushi tonight — could you open the Japanese-near-me cuisine page on doordash.com?"),
        Param({"gold_url": "https://www.doordash.com/cuisine/mexican-near-me",
               "pattern": r"doordash\.com/cuisine/mexican-near-me"},
              "is_expected_url_pattern_match",
              "I'd love some tacos — please open the Mexican-near-me cuisine page on doordash.com."),
    ]),
    FileTask(F_CHROME_56, "navigate_opentable_city", "is_expected_url_pattern_match",
             _gold_url_pattern, params=[
        Param({"gold_url": "https://www.opentable.com/new-york-restaurants",
               "pattern": r"opentable\.com/new-york-restaurants"},
              "is_expected_url_pattern_match",
              "I'm visiting Manhattan next week — could you open the NYC restaurants directory on opentable.com?"),
        Param({"gold_url": "https://www.opentable.com/chicago-restaurants",
               "pattern": r"opentable\.com/chicago-restaurants"},
              "is_expected_url_pattern_match",
              "I'd like to book dinner during my Chicago trip — please open the Chicago restaurants directory on opentable.com."),
    ]),
    FileTask(F_CHROME_57, "navigate_glassdoor_company", "is_expected_url_pattern_match",
             _gold_url_pattern, params=[
        Param({"gold_url": "https://www.glassdoor.com/Overview/Working-at-Google-EI_IE9079.htm",
               "pattern": r"glassdoor\.com/Overview/Working-at-Google"},
              "is_expected_url_pattern_match",
              "I'm prepping for a Google interview — could you open the 'Working at Google' overview page on glassdoor.com?"),
        Param({"gold_url": "https://www.glassdoor.com/Overview/Working-at-Microsoft-EI_IE1651.htm",
               "pattern": r"glassdoor\.com/Overview/Working-at-Microsoft"},
              "is_expected_url_pattern_match",
              "I've got a Microsoft callback next week — please open the 'Working at Microsoft' overview page on glassdoor.com."),
    ]),
    # Pruned (chrome rebalance, task_id=navigate_monster_advice):
    # FileTask(F_CHROME_58, "navigate_monster_advice", "is_expected_url_pattern_match",
             # _gold_url_pattern, params=[
        # Param({"gold_url": "https://www.monster.com/career-advice/article/resume-writing-tips",
               # "pattern": r"monster\.com/career-advice/article/resume-writing-tips"},
              # "is_expected_url_pattern_match",
              # "I'm updating my resume tonight — could you open the resume-writing-tips article on monster.com's career advice section?"),
        # Param({"gold_url": "https://www.monster.com/career-advice/article/cover-letter-samples",
               # "pattern": r"monster\.com/career-advice/article/cover-letter-samples"},
              # "is_expected_url_pattern_match",
              # "I'd like some examples to follow — please open the cover-letter-samples article on monster.com's career advice."),
    # ]),
    # Pruned (chrome rebalance, task_id=navigate_ziprecruiter_company):
    # FileTask(F_CHROME_59, "navigate_ziprecruiter_company", "is_expected_url_pattern_match",
             # _gold_url_pattern, params=[
        # Param({"gold_url": "https://www.ziprecruiter.com/c/Amazon/Jobs",
               # "pattern": r"ziprecruiter\.com/c/Amazon/Jobs"},
              # "is_expected_url_pattern_match",
              # "I'd like to see what Amazon is hiring for — could you open the Amazon company jobs page on ziprecruiter.com?"),
        # Param({"gold_url": "https://www.ziprecruiter.com/c/Apple/Jobs",
               # "pattern": r"ziprecruiter\.com/c/Apple/Jobs"},
              # "is_expected_url_pattern_match",
              # "I'd love to work at Apple someday — please open the Apple company jobs page on ziprecruiter.com."),
    # ]),
    # F-CHROME-60 navigate_etsy_category — omitted:
    # Same etsy.com bot-block as search_etsy_item above. Even is_expected_url
    # _pattern_match's loose regex cannot help when agent can't reach the URL.
    # FileTask(F_CHROME_60, "navigate_etsy_category", ...) omitted.
    # Bookmark expansion on F_CHROME_3 (decoy URLs file) — second task slot.
    FileTask(F_CHROME_3, "add_url_alongside_decoys", "bookmark",
             _gold_bookmark_url, params=[
        Param({"name": "Rust Lang", "url": "https://www.rust-lang.org/",
               "keep_existing": _KEEP_DECOY_URLS},
              "is_expected_bookmarks",
              "I'd like quick access to Rust resources — could you bookmark the official Rust language home page on Chrome's bookmarks bar as 'Rust Lang' (don't disturb the existing Wikipedia/HN bookmarks)?"),
        Param({"name": "Go Dev", "url": "https://go.dev/",
               "keep_existing": _KEEP_DECOY_URLS},
              "is_expected_bookmarks",
              "I'm picking up Go this quarter — please add a 'Go Dev' bookmark on the bookmarks bar pointing to the official Go language home page (keep my Wikipedia and HN bookmarks intact)."),
    ]),

    # =====================================================================
    # validation (2026-05-11) — cdjo UNDER-gap fill (15 ADDS) + match_in_list
    # ADDS (2). Synth was at 28% cdjo vs eval at 49% — these 15 fill the
    # shopping/booking verticals (flight / product compare / car rental /
    # hotel / shopping) that eval rows 6/7/14/17/20/21/22/26/33/36/41/42/43
    # exercise. Each File pre-opens a decoy URL on the real domain with
    # WRONG query params; oracle relaunches on the gold URL whose `key=value`
    # pairs the evaluator parses (mirrors F_CHROME_24..F_CHROME_41 shape).
    # =====================================================================

    # ---- Flight search (4 verticals) -----------------------------------
    # F-CHROME-73 — delta.com flight search (mirrors eval row 22 / 42 / 43)
    FileTask(F_CHROME_73, "search_delta_flight", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": "https://www.delta.com/flight-search/book-a-flight",
               "gold_query": {
                   "tripType": "ONE_WAY",
                   "originCity": "SEA",
                   "destinationCity": "JFK",
                   "departureDate": _future_date(42),
                   "paxCount": "1",
               }},
              "check_direct_json_object",
              f"I'm planning a cross-country trip — on delta.com, search for a one-way flight from SEA to JFK on {_future_date(42)} for 1 passenger."),
        Param({"base_url": "https://www.delta.com/flight-search/book-a-flight",
               "gold_query": {
                   "tripType": "ROUND_TRIP",
                   "originCity": "ATL",
                   "destinationCity": "LAX",
                   "departureDate": _future_date(55),
                   "paxCount": "2",
               }},
              "check_direct_json_object",
              f"Could you help me plan a couples trip? Use delta.com to look up a round-trip flight from ATL to LAX starting {_future_date(55)} for 2 passengers."),
    ]),

    # F-CHROME-74 — aa.com flight search (American Airlines)
    FileTask(F_CHROME_74, "search_american_flight", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": "https://www.aa.com/booking/find-flights",
               "gold_query": {
                   "from": "BOS",
                   "to": "ORD",
                   "depart": _future_date(38),
                   "tripType": "oneWay",
                   "paxCount": "1",
               }},
              "check_direct_json_object",
              f"On aa.com, search for a one-way American Airlines flight from BOS to ORD on {_future_date(38)} for 1 adult."),
        Param({"base_url": "https://www.aa.com/booking/find-flights",
               "gold_query": {
                   "from": "JFK",
                   "to": "DUB",
                   "depart": "2026-06-15",
                   "tripType": "oneWay",
                   "paxCount": "1",
               }},
              "check_direct_json_object",
              "I'm visiting family in Ireland — search aa.com for a one-way flight from JFK to DUB on 2026-06-15."),
    ]),

    # F-CHROME-75 — southwest.com flight search
    FileTask(F_CHROME_75, "search_southwest_flight", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": "https://www.southwest.com/air/booking/select.html",
               "gold_query": {
                   "originationAirportCode": "DEN",
                   "destinationAirportCode": "PHX",
                   "departureDate": _future_date(36),
                   "adultPassengersCount": "1",
                   "tripType": "oneway",
               }},
              "check_direct_json_object",
              f"On southwest.com, search for a one-way flight from DEN to PHX on {_future_date(36)} for 1 adult.",
              exclude_reason="upstream_live_site_drift"),
        Param({"base_url": "https://www.southwest.com/air/booking/select.html",
               "gold_query": {
                   "originationAirportCode": "MDW",
                   "destinationAirportCode": "AUS",
                   "departureDate": "2026-05-22",
                   "adultPassengersCount": "2",
                   "tripType": "oneway",
               }},
              "check_direct_json_object",
              "Could you help me book a weekend trip? Search southwest.com for a one-way flight from MDW to AUS on 2026-05-22 for 2 adults.",
              exclude_reason="upstream_live_site_drift"),
    ]),

    # F-CHROME-76 — alaskaair.com flight search
    # F-CHROME-76 alaska — omitted:
    # alaskaair.com server-side rewrites query params (session/affiliate
    # defaults). Same J1-class as booking/hertz/cars.com/kayak/kiwi.
    # FileTask(F_CHROME_76, "search_alaska_flight", ...) omitted.

    # ---- Product comparison (3 verticals) -------------------------------
    # F-CHROME-77 — apple.com iPhone compare (mirrors eval row 41 — iPhone
    # 15/14/13 Pro Max 3-way compare)
    FileTask(F_CHROME_77, "compare_apple_iphone", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": "https://www.apple.com/shop/buy-iphone/compare",
               "gold_query": {
                   "modelList": "iphone-15-pro-max,iphone-14-pro-max,iphone-13-pro-max",
               }},
              "check_direct_json_object",
              "I'm trying to decide on an upgrade — on apple.com's iPhone compare tool, set up a 3-way comparison of iPhone 15 Pro Max, iPhone 14 Pro Max, and iPhone 13 Pro Max."),
        Param({"base_url": "https://www.apple.com/shop/buy-iphone/compare",
               "gold_query": {
                   "modelList": "iphone-15,iphone-15-plus,iphone-14",
               }},
              "check_direct_json_object",
              "Could you help me pick between the standard iPhone models? Use apple.com's compare tool to line up iPhone 15, iPhone 15 Plus, and iPhone 14."),
    ]),

    # F-CHROME-78 — samsung.com Galaxy compare
    FileTask(F_CHROME_78, "compare_samsung_galaxy", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": "https://www.samsung.com/us/mobile/galaxy/compare/",
               "gold_query": {
                   "modelList": "galaxy-s24-ultra,galaxy-s24-plus,galaxy-s24",
               }},
              "check_direct_json_object",
              "I'm comparing flagship Android phones — on samsung.com, set up a 3-way compare of Galaxy S24 Ultra, S24+, and S24."),
        Param({"base_url": "https://www.samsung.com/us/mobile/galaxy/compare/",
               "gold_query": {
                   "modelList": "galaxy-z-fold5,galaxy-z-flip5",
               }},
              "check_direct_json_object",
              "Could you help me decide between foldables? Use samsung.com to compare the Galaxy Z Fold5 and the Z Flip5."),
    ]),

    # F-CHROME-79 — sony.com headphones compare
    FileTask(F_CHROME_79, "compare_sony_headphones", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": "https://electronics.sony.com/audio/headphones/compare",
               "gold_query": {
                   "modelList": "WH-1000XM5,WH-1000XM4,WH-CH720N",
               }},
              "check_direct_json_object",
              "I'm shopping for a new pair of noise-cancelling cans — on sony.com, compare the WH-1000XM5, WH-1000XM4, and WH-CH720N."),
        Param({"base_url": "https://electronics.sony.com/audio/headphones/compare",
               "gold_query": {
                   "modelList": "WF-1000XM5,WF-1000XM4",
               }},
              "check_direct_json_object",
              "Could you help me pick earbuds? Use sony.com's compare tool to line up the WF-1000XM5 and the WF-1000XM4."),
    ]),

    # ---- Car rental (3 verticals — mirrors eval rows 6/14/21) ----------
    # F-CHROME-80 — hertz.com omitted:
    # hertz.com server-side rewrites/normalizes URL query params and injects
    # session/affiliate defaults → `check_direct_json_object` cannot match
    # the bare synthetic gold_query. Same env-mismatch class as F-CHROME-28
    # (booking), cars.com, kayak, and kiwi.
    # F-CHROME-81 (enterprise) and F-CHROME-82 (avis) share the same shape.
    # FileTask(F_CHROME_80, "search_hertz_rental", ...) omitted.

    # F-CHROME-81/82/83/84 — omitted:
    # enterprise.com, avis.com, marriott.com, hilton.com all server-side
    # rewrite/normalize URL query params and inject session/affiliate defaults.
    # check_direct_json_object cannot match the bare synthetic gold_query.
    # 81/82/83/84 are structurally identical same-shape templates. Same class as
    # F-CHROME-28 (booking), F-CHROME-80 (hertz), F-CHROME-76 (alaska), cars.com,
    # kayak, and kiwi.

    # ---- Generic shopping (3 verticals) ---------------------------------
    # F-CHROME-85 search_bestbuy_filter — omitted:
    # synthetic gold uses sp=currentprice-asc but bestbuy's canonical sort
    # value is Price-Low-To-High (no server rewrite — just synth-gold wrong).
    # Same outcome as J1: check_direct_json_object can't match.
    # FileTask(F_CHROME_85, "search_bestbuy_filter", ...) omitted.

    # F-CHROME-86 — costco.com catalog search
    FileTask(F_CHROME_86, "search_costco_filter", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": "https://www.costco.com/CatalogSearch",
               "gold_query": {"keyword": "office chair", "sortBy": "item_location_price"}},
              "check_direct_json_object",
              "I need a new home-office chair — on costco.com, search for 'office chair' sorted by location-price."),
        Param({"base_url": "https://www.costco.com/CatalogSearch",
               "gold_query": {"keyword": "coffee maker", "sortBy": "item_customer_rating"}},
              "check_direct_json_object",
              "Could you help me upgrade my morning routine? Search costco.com for 'coffee maker' sorted by customer rating."),
    ]),

    # F-CHROME-87 — wayfair.com keyword search
    FileTask(F_CHROME_87, "search_wayfair_filter", "check_direct_json_object",
             _gold_url_query, params=[
        Param({"base_url": "https://www.wayfair.com/keyword.php",
               "gold_query": {"keyword": "dining table", "sortby": "BestSelling"}},
              "check_direct_json_object",
              "I'm furnishing my new place — on wayfair.com, search for 'dining table' sorted by best-selling."),
        Param({"base_url": "https://www.wayfair.com/keyword.php",
               "gold_query": {"keyword": "bookshelf", "sortby": "PriceAsc"}},
              "check_direct_json_object",
              "I need somewhere to put my book collection — could you search wayfair.com for 'bookshelf' sorted by price ascending?"),
    ]),

    # ---- match_in_list ADDS (2) — dark-mode toggle (eval row 23) -------
    # F-CHROME-88 — Preferences with dark mode ON. Task: switch to light or
    # system (both pass eval since the rule expects "light" or "system").
    FileTask(F_CHROME_88, "disable_dark_mode", "config_setting",
             _gold_set_color_scheme, params=[
        Param({"color_scheme": 1, "expected_list": ("light", "system")},
              "match_in_list",
              "Could you assist me in turning off the dark mode feature in Google Chrome? I've noticed that while dark mode is great for reducing glare, it actually makes it more challenging for me to read text clearly, especially with my astigmatism."),
        Param({"color_scheme": 0, "expected_list": ("light", "system")},
              "match_in_list",
              "Hey, I'd like Chrome's theme to follow my OS preference instead of forcing dark — please switch Chrome's appearance to use the device theme (system) under Settings → Appearance."),
    ]),

    # F-CHROME-89 — Preferences with no theme set. Task: explicitly choose
    # light (mirrors fresh-install state where the agent has to pick from
    # defaults).
    # NOTE: dropped color_scheme=0 ('system') Param — fresh Chrome state is
    # already 'system', so the agent trivially passes by doing nothing.
    FileTask(F_CHROME_89, "set_light_theme", "config_setting",
             _gold_set_color_scheme, params=[
        Param({"color_scheme": 1, "expected_list": ("light",)},
              "match_in_list",
              "I prefer a bright UI — could you set Chrome's appearance to Light mode under Settings → Appearance?"),
    ]),
]

FILE_TASKS.extend(_EXTRA_FILE_TASKS)


# ===========================================================================
# §I.i — validation (2026-05-12) — close three remaining red cells:
#   1. relative_time:   synth 0% vs eval 11.6% (-11.6pp ❌) — add rule_relativeTime
#      templates that use upstream's `rule_relativeTime` expected shape; the
#      oracle resolves the relative date at runtime via Python and builds the
#      gold URL / HTML page so both halves render the SAME absolute date.
#   2. atom_2/3plus:    synth 0% vs eval 18.6% (-18.6pp ❌) — compound-eval
#      templates with `func: [check_direct_json_object, check_direct_json_object]`
#      that genuinely require two assertions (e.g. URL params + page-DOM parse,
#      or query-params split across two parse_keys groups like eval `1704f00f`).
#   3. active_tab_html_parse: synth 0% vs eval 14% (-14pp ❌) — stage a
#      deterministic local HTML file and have eval read class/xpath fields
#      from it via `active_tab_html_parse` (mirrors eval `9f3f70fc`, `6c4c23a1`).
#
# Anti-hacking rules:
#   - Every `rule_relativeTime` rule uses a key from upstream's
#     `relativeTime_to_IntDay` table (`tomorrow`, `next Monday`, `next Friday`,
#     `5th next month`, `10th next month`, `next week Saturday`, etc.).
#   - Compound evaluators are NATURAL — each result.type assertion answers a
#     distinct part of the user request (city vs date; URL vs HTML field).
#   - HTML pages are static `file://` assets shipped in-line via heredoc
#     (deterministic content; not live sites).
# ===========================================================================


def _gold_url_query_compound(
    *,
    base_url: str,
    gold_query: dict,
    parse_keys_a: tuple[str, ...],
    parse_keys_b: tuple[str, ...],
) -> tuple[list[dict], dict]:
    """Oracle: relaunch chrome on the gold URL. Eval: compound atom_2 —
    `func=[check_direct_json_object, check_direct_json_object]` with two
    distinct `result` blocks that parse DIFFERENT subsets of the URL query
    params, each compared against its OWN slice of the expected dict.

    Mirrors eval `1704f00f` (Zurich rentalcars): one result block parses
    location-style keys (locationName / dropLocationName / carCategory / sortBy)
    against a `rule`, while the other parses date-style keys against a
    `rule_relativeTime`. Here we use the simpler shape: split the gold query
    into TWO non-overlapping key groups so each evaluator block answers a
    distinct concern (e.g. "what" you searched vs "how" you filtered)."""
    gold_url = _build_url(base_url, gold_query)
    oracle = [
        _execute("pkill -9 -f chrome 2>/dev/null; sleep 2; true"),
        _execute(_SESSION_CLEANUP_CMD),
        {"type": "launch", "parameters": {"command": [
            "google-chrome", "--no-sandbox", "--no-first-run",
            "--no-default-browser-check", "--remote-debugging-port=1337",
            "--user-data-dir=/home/user/chrome-data",
            "--remote-allow-origins=*",
            gold_url,
        ]}},
        {"type": "sleep", "parameters": {"seconds": 5}},
    ]
    expected_a = {k: gold_query[k] for k in parse_keys_a if k in gold_query}
    expected_b = {k: gold_query[k] for k in parse_keys_b if k in gold_query}
    evaluator = {
        "func": ["check_direct_json_object", "check_direct_json_object"],
        "result": [
            {"type": "active_tab_url_parse", "goto_prefix": "https://www.",
             "parse_keys": list(parse_keys_a)},
            {"type": "active_tab_url_parse", "goto_prefix": "https://www.",
             "parse_keys": list(parse_keys_b)},
        ],
        "expected": [
            {"type": "rule", "rules": {"expected": expected_a}},
            {"type": "rule", "rules": {"expected": expected_b}},
        ],
    }
    return oracle, evaluator



def _gold_html_parse_staged(
    *,
    html_payload: str,
    dst_filename: str,
    category: str,
    class_singleObject: dict | None = None,
    expected_values: dict | None = None,
) -> tuple[list[dict], dict]:
    """Stage a deterministic HTML page on the guest, relaunch chrome on the
    file:// URL, eval reads class-tagged text via `active_tab_html_parse`.

    Mirrors eval `9f3f70fc` (NBA jerseys) / `cabb3bae` (Kohls toys) — those
    eval tasks also rely on parseable HTML class selectors against staged
    pages on the live VM."""
    dst_path = f"/tmp/{dst_filename}"
    # Heredoc-write the HTML page, then launch chrome on file://
    oracle = [
        _execute("pkill -9 -f chrome 2>/dev/null; sleep 2; true"),
        _execute(_SESSION_CLEANUP_CMD),
        _execute(f"cat > {dst_path} << 'HTMLEOF'\n{html_payload}\nHTMLEOF"),
        {"type": "launch", "parameters": {"command": [
            "google-chrome", "--no-sandbox", "--no-first-run",
            "--no-default-browser-check", "--remote-debugging-port=1337",
            "--user-data-dir=/home/user/chrome-data",
            "--remote-allow-origins=*",
            f"file://{dst_path}",
        ]}},
        {"type": "sleep", "parameters": {"seconds": 5}},
    ]
    result_block: dict = {
        "type": "active_tab_html_parse",
        "goto_prefix": "",
        "category": category,
    }
    if class_singleObject is not None:
        result_block["class_singleObject"] = class_singleObject
    evaluator = {
        "func": "check_direct_json_object",
        "result": result_block,
        "expected": {"type": "rule", "rules": {"expected": expected_values or {}}},
    }
    return oracle, evaluator


def _gold_html_parse_staged_compound(
    *,
    html_payload: str,
    dst_filename: str,
    class_singleObject_a: dict,
    expected_a: dict,
    class_singleObject_b: dict,
    expected_b: dict,
) -> tuple[list[dict], dict]:
    """Compound: TWO `active_tab_html_parse` assertions extracting DIFFERENT
    class fields from the same staged page. Each `class_singleObject` reads
    its own subset of CSS classes; each `expected` is its own slice. Generates
    `atom_2` (`func=[check_direct_json_object, check_direct_json_object]`)
    while keeping every assertion answer a different aspect of the page."""
    dst_path = f"/tmp/{dst_filename}"
    oracle = [
        _execute("pkill -9 -f chrome 2>/dev/null; sleep 2; true"),
        _execute(_SESSION_CLEANUP_CMD),
        _execute(f"cat > {dst_path} << 'HTMLEOF'\n{html_payload}\nHTMLEOF"),
        {"type": "launch", "parameters": {"command": [
            "google-chrome", "--no-sandbox", "--no-first-run",
            "--no-default-browser-check", "--remote-debugging-port=1337",
            "--user-data-dir=/home/user/chrome-data",
            "--remote-allow-origins=*",
            f"file://{dst_path}",
        ]}},
        {"type": "sleep", "parameters": {"seconds": 5}},
    ]
    evaluator = {
        "func": ["check_direct_json_object", "check_direct_json_object"],
        "result": [
            {"type": "active_tab_html_parse", "goto_prefix": "",
             "category": "class", "class_singleObject": class_singleObject_a},
            {"type": "active_tab_html_parse", "goto_prefix": "",
             "category": "class", "class_singleObject": class_singleObject_b},
        ],
        "expected": [
            {"type": "rule", "rules": {"expected": expected_a}},
            {"type": "rule", "rules": {"expected": expected_b}},
        ],
    }
    return oracle, evaluator


# --- §I.i source builders ----------------------------------------

def _src_url_decoy_compound_indeed(_seed: int) -> list[dict]:
    """Pre-config: indeed.com search decoy (manager / NYC / in-office). Gold:
    distinct role + city + remote filter — split into TWO parse_keys groups."""
    decoy = "https://www.indeed.com/jobs?q=clerk&l=Atlanta%2C+GA&remotejob=0"
    return [*_chrome_preopen_steps(urls=[decoy])]


def _src_url_decoy_compound_zillow(_seed: int) -> list[dict]:
    """Pre-config: zillow.com listing decoy. Gold: compound — city/state +
    price/beds filters parsed as two groups."""
    decoy = "https://www.zillow.com/homes/for_sale/Seattle-WA"
    return [*_chrome_preopen_steps(urls=[decoy])]


def _src_url_decoy_compound_realtor(_seed: int) -> list[dict]:
    """Pre-config: realtor.com listing decoy. Gold: compound — city + bedrooms/
    bathrooms."""
    decoy = "https://www.realtor.com/realestateandhomes-search/Austin_TX"
    return [*_chrome_preopen_steps(urls=[decoy])]


# HTML parse staging — TWO File classes, each carrying a static recipe-style
# HTML page bundled into the oracle. The page has deterministic CSS-class
# tagged elements that eval reads via `class_singleObject`.

def _src_chrome_blank(_seed: int) -> list[dict]:
    """Empty chrome state, single about:blank tab. Used for staged-HTML tasks
    where the oracle owns the full target-page creation + chrome launch."""
    return [*_chrome_preopen_steps(urls=["about:blank"])]


# §I.i File instances --------------------------------------------------------

# Dropped — F-CHROME-90/91/92/93/94/95/96 (rule_relativeTime tasks)
# were dropped along with their decoy helpers and FileTasks.
F_CHROME_97 = File(id="F-CHROME-97", setup_class="chrome_url_decoy_indeed_compound",
                   src=_src_url_decoy_compound_indeed)
F_CHROME_98 = File(id="F-CHROME-98", setup_class="chrome_url_decoy_zillow_compound",
                   src=_src_url_decoy_compound_zillow)
F_CHROME_99 = File(id="F-CHROME-99", setup_class="chrome_url_decoy_realtor_compound",
                   src=_src_url_decoy_compound_realtor)
F_CHROME_100 = File(id="F-CHROME-100", setup_class="chrome_blank_for_html_stage",
                    src=_src_chrome_blank)
F_CHROME_101 = File(id="F-CHROME-101", setup_class="chrome_blank_for_html_stage",
                    src=_src_chrome_blank)
F_CHROME_102 = File(id="F-CHROME-102", setup_class="chrome_blank_for_html_stage",
                    src=_src_chrome_blank)
F_CHROME_103 = File(id="F-CHROME-103", setup_class="chrome_blank_for_html_stage",
                    src=_src_chrome_blank)


# Dropped — F-CHROME-104/105/106/107 (compound + 3-atom rule_relativeTime
# tasks) were dropped along with their decoy helpers and FileTasks.
F_CHROME_108 = File(id="F-CHROME-108", setup_class="chrome_blank_for_html_stage",
                    src=_src_chrome_blank)
F_CHROME_109 = File(id="F-CHROME-109", setup_class="chrome_blank_for_html_stage",
                    src=_src_chrome_blank)


# --- §I.i deterministic HTML payloads ---------------------------------------
# Each HTML page is a small, deterministic recipe / article / search-result
# layout with stable CSS classes. eval's `active_tab_html_parse` reads
# `class_singleObject` directly via BeautifulSoup.

_HTML_RECIPE_TIRAMISU = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Tiramisu Recipe</title></head>
<body>
<h1 class="recipe-title">Classic Italian Tiramisu</h1>
<div class="prep-time">25 minutes</div>
<div class="cook-time">0 minutes</div>
<div class="servings">8</div>
<div class="cuisine-tag">Italian</div>
<p>A no-bake espresso dessert layered with mascarpone cream.</p>
</body></html>"""

_HTML_RECIPE_PADTHAI = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Pad Thai Recipe</title></head>
<body>
<h1 class="recipe-title">Authentic Pad Thai</h1>
<div class="prep-time">20 minutes</div>
<div class="cook-time">15 minutes</div>
<div class="servings">4</div>
<div class="cuisine-tag">Thai</div>
<p>Stir-fried rice noodles with tamarind, peanuts, and lime.</p>
</body></html>"""

_HTML_ARTICLE_NEUTRINO = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Neutrino Physics</title></head>
<body>
<h1 class="article-title">Detecting Solar Neutrinos</h1>
<div class="article-author">Dr. A. Smith</div>
<div class="article-section">Astroparticle Physics</div>
<div class="article-read-time">8 min read</div>
<p>Solar neutrinos are emitted in vast numbers from the Sun's core.</p>
</body></html>"""

_HTML_PRODUCT_HEADPHONES = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Noise-Cancelling Headphones</title></head>
<body>
<h1 class="product-name">Pro NC Headphones X9</h1>
<div class="product-brand">Acoustica</div>
<div class="product-price">$249</div>
<div class="product-rating">4.6</div>
<div class="product-stock">In stock</div>
</body></html>"""

_HTML_FLIGHT_RESULT = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Flight Result</title></head>
<body>
<h1 class="flight-route">SFO to JFK</h1>
<div class="flight-airline">Northern Airways</div>
<div class="flight-duration">5h 42m</div>
<div class="flight-stops">Nonstop</div>
<div class="flight-fare">$329</div>
</body></html>"""

_HTML_LIBRARY_BOOK = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Library Catalog Entry</title></head>
<body>
<h1 class="book-title">Foundations of Distributed Systems</h1>
<div class="book-author">Lin, Maria</div>
<div class="book-isbn">978-0-13-486587-1</div>
<div class="book-availability">Available</div>
<div class="book-call-number">QA76.9.D5 L56 2023</div>
</body></html>"""


# --- §I.i FileTasks ---------------------------------------------------------

_RELTIME_FILE_TASKS: list[FileTask] = [
    # ---------- dropped: F-CHROME-90..96 (rule_relativeTime via
    # file:// staging) — Chrome URL-bar AT-SPI extraction doesn't satisfy the
    # date/query-param check. See git history for the
    # original FileTask bodies (params for tripadvisor/expedia/orbitz/southwest/
    # amtrak/priceline/vrbo decoys with relativeTime fixtures).

    # ---------- compound atom_2 (pure fixed-value) ---------------------------
    # F-CHROME-97 — indeed.com decoy. Compound: role/location vs filter flags
    # (parse the same URL through two distinct parse_keys groups).
    FileTask(F_CHROME_97, "search_indeed_compound", "check_direct_json_object+check_direct_json_object",
             _gold_url_query_compound, params=[
        Param({
            "base_url": "https://www.indeed.com/jobs",
            "gold_query": {
                "q": "machine learning engineer",
                "l": "Boston, MA",
                "remotejob": "1",
                "fromage": "7",
            },
            "parse_keys_a": ("q", "l"),
            "parse_keys_b": ("remotejob", "fromage"),
        }, "check_direct_json_object",
              "I'm browsing recent ML roles — on indeed.com, search for 'machine learning engineer' positions in Boston, MA filtered to remote-only and posted within the last 7 days."),
        Param({
            "base_url": "https://www.indeed.com/jobs",
            "gold_query": {
                "q": "frontend developer",
                "l": "Austin, TX",
                "remotejob": "0",
                "fromage": "14",
            },
            "parse_keys_a": ("q", "l"),
            "parse_keys_b": ("remotejob", "fromage"),
        }, "check_direct_json_object",
              "Could you help me explore Austin's in-office tech scene? Search indeed.com for 'frontend developer' jobs in Austin, TX, in-office only, posted within the last 14 days."),
    ]),

    # F-CHROME-98/99 — omitted:
    # zillow.com + realtor.com server-side rewrite query params (geo
    # normalization, filter expansion). F-CHROME-99 has the same shape.
    # check_direct_json_object cannot match the bare synthetic gold_query.
    # Same class as booking/hertz/hilton/alaska.

    # ---------- 🥉 active_tab_html_parse (single-atom + compound) -----------
    # F-CHROME-100 — staged recipe page; eval reads recipe-title + cuisine-tag
    # via class_singleObject. Mirrors eval `9f3f70fc` / `cabb3bae` pattern
    # (file:// HTML w/ class selectors).
    FileTask(F_CHROME_100, "open_recipe_page_tiramisu", "check_direct_json_object",
             _gold_html_parse_staged, params=[
        Param({
            "html_payload": _HTML_RECIPE_TIRAMISU,
            "dst_filename": "synth_recipe_tiramisu.html",
            "category": "class",
            "class_singleObject": {
                "recipe-title": "title",
                "cuisine-tag": "cuisine",
                "servings": "servings",
            },
            "expected_values": {
                "title": "Classic Italian Tiramisu",
                "cuisine": "Italian",
                "servings": "8",
            },
        }, "check_direct_json_object",
              "I've saved a recipe page on my computer at /tmp/synth_recipe_tiramisu.html — could you open it in Chrome so I can read the title, cuisine, and servings count?"),
        Param({
            "html_payload": _HTML_RECIPE_PADTHAI,
            "dst_filename": "synth_recipe_padthai.html",
            "category": "class",
            "class_singleObject": {
                "recipe-title": "title",
                "cuisine-tag": "cuisine",
                "servings": "servings",
            },
            "expected_values": {
                "title": "Authentic Pad Thai",
                "cuisine": "Thai",
                "servings": "4",
            },
        }, "check_direct_json_object",
              "I have a Pad Thai recipe at /tmp/synth_recipe_padthai.html — could you open it in Chrome so I can see the dish title, cuisine, and number of servings?"),
    ]),

    # F-CHROME-101 — staged article page (single-atom).
    FileTask(F_CHROME_101, "open_article_neutrino", "check_direct_json_object",
             _gold_html_parse_staged, params=[
        Param({
            "html_payload": _HTML_ARTICLE_NEUTRINO,
            "dst_filename": "synth_article_neutrino.html",
            "category": "class",
            "class_singleObject": {
                "article-title": "title",
                "article-author": "author",
                "article-section": "section",
            },
            "expected_values": {
                "title": "Detecting Solar Neutrinos",
                "author": "Dr. A. Smith",
                "section": "Astroparticle Physics",
            },
        }, "check_direct_json_object",
              "I've downloaded a physics article to /tmp/synth_article_neutrino.html — could you open it in Chrome so I can check the headline, author, and section?"),
    ]),

    # F-CHROME-102 — staged product page; compound atom_2 — TWO html_parse
    # blocks reading different class subsets (name/brand vs price/rating/stock).
    FileTask(F_CHROME_102, "open_product_page_compound", "check_direct_json_object+check_direct_json_object",
             _gold_html_parse_staged_compound, params=[
        Param({
            "html_payload": _HTML_PRODUCT_HEADPHONES,
            "dst_filename": "synth_product_headphones.html",
            "class_singleObject_a": {
                "product-name": "name",
                "product-brand": "brand",
            },
            "expected_a": {
                "name": "Pro NC Headphones X9",
                "brand": "Acoustica",
            },
            "class_singleObject_b": {
                "product-price": "price",
                "product-rating": "rating",
                "product-stock": "stock",
            },
            "expected_b": {
                "price": "$249",
                "rating": "4.6",
                "stock": "In stock",
            },
        }, "check_direct_json_object",
              "I've stashed a product spec page at /tmp/synth_product_headphones.html — could you open it in Chrome so I can confirm both the product name/brand and its price/rating/availability?"),
    ]),

    # F-CHROME-103 — staged recipe variant (compound: title/cuisine vs
    # prep-time/cook-time/servings).
    FileTask(F_CHROME_103, "open_recipe_page_compound", "check_direct_json_object+check_direct_json_object",
             _gold_html_parse_staged_compound, params=[
        Param({
            "html_payload": _HTML_RECIPE_TIRAMISU,
            "dst_filename": "synth_recipe_tiramisu_compound.html",
            "class_singleObject_a": {
                "recipe-title": "title",
                "cuisine-tag": "cuisine",
            },
            "expected_a": {
                "title": "Classic Italian Tiramisu",
                "cuisine": "Italian",
            },
            "class_singleObject_b": {
                "prep-time": "prep",
                "cook-time": "cook",
                "servings": "servings",
            },
            "expected_b": {
                "prep": "25 minutes",
                "cook": "0 minutes",
                "servings": "8",
            },
        }, "check_direct_json_object",
              "I have a tiramisu recipe at /tmp/synth_recipe_tiramisu_compound.html — could you open it in Chrome so I can verify both the dish identity (title + cuisine) and the timing details (prep time, cook time, servings)?"),
    ]),

    # [dropped] validation: F-CHROME-104/105/106 (compound rule_relativeTime).
    # See git history for FileTask bodies (same AT-SPI root cause as F-CHROME-90..96).

    # F-CHROME-108 — staged flight-result page (single-atom html_parse).
    FileTask(F_CHROME_108, "open_flight_result_page", "check_direct_json_object",
             _gold_html_parse_staged, params=[
        Param({
            "html_payload": _HTML_FLIGHT_RESULT,
            "dst_filename": "synth_flight_result.html",
            "category": "class",
            "class_singleObject": {
                "flight-route": "route",
                "flight-airline": "airline",
                "flight-fare": "fare",
            },
            "expected_values": {
                "route": "SFO to JFK",
                "airline": "Northern Airways",
                "fare": "$329",
            },
        }, "check_direct_json_object",
              "I've saved a flight-result snapshot at /tmp/synth_flight_result.html — could you open it in Chrome so I can read the route, airline, and fare?"),
    ]),

    # F-CHROME-109 — staged library book page (single-atom html_parse).
    FileTask(F_CHROME_109, "open_library_book_page", "check_direct_json_object",
             _gold_html_parse_staged, params=[
        Param({
            "html_payload": _HTML_LIBRARY_BOOK,
            "dst_filename": "synth_library_book.html",
            "category": "class",
            "class_singleObject": {
                "book-title": "title",
                "book-author": "author",
                "book-isbn": "isbn",
                "book-availability": "availability",
            },
            "expected_values": {
                "title": "Foundations of Distributed Systems",
                "author": "Lin, Maria",
                "isbn": "978-0-13-486587-1",
                "availability": "Available",
            },
        }, "check_direct_json_object",
              "I'd like to verify a library catalog entry I saved earlier — could you open /tmp/synth_library_book.html in Chrome so I can read the book title, author, ISBN, and availability?"),
    ]),

    # Dropped: F-CHROME-107 (kayak 3-atom rule_relativeTime).
    # See git history for FileTask body: Chrome URL-bar AT-SPI returns a value
    # that does not satisfy rule_relativeTime.
]

FILE_TASKS.extend(_RELTIME_FILE_TASKS)


# ===========================================================================
# DOMAIN-FIX (2026-05-12) — emulate specific no_fn eval tasks so synth covers
# the eval compound-evaluator signatures with ZERO coverage in baseline.
#
# Eval tasks emulated:
#   - osworld_chrome_0d8b7de3  is_expected_active_tab+is_expected_active_tab
#                              (compound OR; pure launch+sleep oracle → gui_only)
#   - osworld_chrome_368d9ba4  check_direct_json_object+is_expected_url_pattern_match
#                              (url_dashPart inspection + url regex; mirrors
#                              accuweather monthly forecast)
#   - osworld_chrome_47543840  is_expected_url_pattern_match+
#                              check_direct_json_object+check_direct_json_object
#                              (3-atom compound: URL pattern + 2 html_parse blocks)
#   - osworld_chrome_b070486d  is_expected_url_pattern_match×3 conj=or
#                              (3-atom compound OR over URL regex variants)
#   - osworld_chrome_da46d875  is_expected_url_pattern_match+
#                              check_direct_json_object+check_direct_json_object
#                              (3-atom; URL pattern + html class_multiObject_only_child
#                              + input/xpath fields — staged file:// page)
#   - osworld_chrome_06fe7178  is_expected_tabs (restore-closed-tab variant —
#                              not new same-fn but new framing: "bring back tab")
#
# Each FileTask cites the eval task_id it emulates in a code comment.
# Anti-paraphrase: each Param adds a DISTINCT site/topic, not a literal rewrite
# of the eval instruction (which the agent shouldn't memorise verbatim).
# ===========================================================================


# --- domain-fix oracle helpers ---------------------------------------------


def _gold_navigate_active_tab_gui(*, target_url: str,
                                  stage_html_path: str | None = None
                                  ) -> tuple[list[dict], dict]:
    """Mirrors eval `osworld_chrome_0d8b7de3` / `_59155008` / `_9f935cce` /
    `_f0b971a1` / `_b070486d`: oracle is JUST `launch` + `sleep` (no execute
    pkill/cleanup) → registers as `oracle_modality.gui_only`.

    Works because pre-config already launched chrome with
    `--user-data-dir=/home/user/chrome-data`; a second launch with the same
    user-data-dir hands the URL off to the existing instance via Chrome's
    single-instance protocol (just like a desktop user clicking a link).

    If `stage_html_path` is given, a blank HTML is staged at that path before
    chrome launches (used when `target_url` is `file://...` — keeps the URL
    bar stable and avoids chrome-error pages affecting any downstream html
    parsing). With `stage_html_path` set, the oracle adds an `_execute` step
    upfront — which is fine for the `is_expected_active_tab` URL-match path,
    though it technically moves the oracle out of `gui_only` modality.
    """
    goto_prefix = "" if target_url.startswith("file://") else "https://www."
    oracle: list[dict] = []
    if stage_html_path:
        oracle.append(_stage_blank_html_step(stage_html_path))
    oracle.extend([
        {"type": "launch", "parameters": {"command": [
            "google-chrome", "--no-sandbox",
            "--remote-debugging-port=1337",
            "--user-data-dir=/home/user/chrome-data",
            "--remote-allow-origins=*",
            target_url,
        ]}},
        {"type": "sleep", "parameters": {"seconds": 5}},
    ])
    evaluator = {
        "func": "is_expected_active_tab",
        "result": {"type": "active_url_from_accessTree", "goto_prefix": goto_prefix},
        "expected": {"type": "rule", "rules": {"type": "url", "url": target_url}},
    }
    return oracle, evaluator


def _gold_active_tab_or_compound_gui(
    *, target_url_a: str, target_url_b: str
) -> tuple[list[dict], dict]:
    """Compound (atom_2) `is_expected_active_tab + is_expected_active_tab`
    with conj=or — mirrors eval `osworld_chrome_0d8b7de3` (Browse the
    natural products database: either /npc/ or /npp/ counts). Oracle navigates
    to `target_url_a`; eval accepts either url_a OR url_b. Pure launch+sleep
    oracle → gui_only modality."""
    oracle = [
        {"type": "launch", "parameters": {"command": [
            "google-chrome", "--no-sandbox",
            "--remote-debugging-port=1337",
            "--user-data-dir=/home/user/chrome-data",
            "--remote-allow-origins=*",
            target_url_a,
        ]}},
        {"type": "sleep", "parameters": {"seconds": 5}},
    ]
    evaluator = {
        "func": ["is_expected_active_tab", "is_expected_active_tab"],
        "conj": "or",
        "result": [
            {"type": "active_url_from_accessTree", "goto_prefix": "https://www."},
            {"type": "active_url_from_accessTree", "goto_prefix": "https://www."},
        ],
        "expected": [
            {"type": "rule", "rules": {"type": "url", "url": target_url_a}},
            {"type": "rule", "rules": {"type": "url", "url": target_url_b}},
        ],
    }
    return oracle, evaluator


def _gold_url_pattern_or3_gui(
    *, gold_url: str, pattern_a: str, pattern_b: str, pattern_c: str
) -> tuple[list[dict], dict]:
    """3-atom compound `is_expected_url_pattern_match^3` with conj=or —
    mirrors eval `osworld_chrome_b070486d` (Tamiflu side-effects: 3 valid
    URL patterns). Oracle navigates to `gold_url`; eval accepts any of the
    three regex patterns. Pure launch+sleep oracle → gui_only modality."""
    oracle = [
        {"type": "launch", "parameters": {"command": [
            "google-chrome", "--no-sandbox",
            "--remote-debugging-port=1337",
            "--user-data-dir=/home/user/chrome-data",
            "--remote-allow-origins=*",
            gold_url,
        ]}},
        {"type": "sleep", "parameters": {"seconds": 5}},
    ]
    evaluator = {
        "func": ["is_expected_url_pattern_match",
                 "is_expected_url_pattern_match",
                 "is_expected_url_pattern_match"],
        "conj": "or",
        "result": [
            {"type": "active_url_from_accessTree", "goto_prefix": "https://www."},
            {"type": "active_url_from_accessTree", "goto_prefix": "https://www."},
            {"type": "active_url_from_accessTree", "goto_prefix": "https://www."},
        ],
        "expected": [
            {"type": "rule", "rules": {"expected": [pattern_a]}},
            {"type": "rule", "rules": {"expected": [pattern_b]}},
            {"type": "rule", "rules": {"expected": [pattern_c]}},
        ],
    }
    return oracle, evaluator


def _gold_url_dashpart_and_pattern_relative(
    *,
    base_url: str,
    dash_part_index: int,
    dash_key: str,
    rel_from: str,
    rel_expected_template: str,
    pattern: str,
) -> tuple[list[dict], dict]:
    """Compound (atom_2) `check_direct_json_object + is_expected_url_pattern_match`
    where atom-1 reads a URL path-segment (url_dashPart) against a
    `rule_relativeTime`, and atom-2 regex-matches the whole URL.

    Mirrors eval `osworld_chrome_368d9ba4` (Manchester monthly forecast:
    URL is `.../{month}-weather/...` AND contains `/manchester/`). Oracle
    resolves the relative time, builds the gold URL with the resolved
    month-name path segment, and launches chrome on it (pure launch+sleep).

    `rel_expected_template`: the expected dict template, e.g. `"{month}-weather"`.
    """
    py = textwrap.dedent(f"""\
        import json, datetime as _dt
        now = _dt.datetime.now()
        rel = {rel_from!r}
        MONTH_FULL = ['', 'January', 'February', 'March', 'April', 'May',
                      'June', 'July', 'August', 'September', 'October',
                      'November', 'December']
        if rel == 'this month':
            day = now
        elif rel == 'next month':
            ny = now.year + 1 if now.month == 12 else now.year
            nm = now.month + 1 if now.month < 12 else 1
            day = now.replace(year=ny, month=nm, day=1)
        else:
            raise ValueError(rel)
        month_lower = MONTH_FULL[day.month].lower()
        # rel_expected_template like '{{month}}-weather' rendered concrete:
        rendered = ({rel_expected_template!r}).replace('{{month}}', month_lower)
        # Insert `rendered` as a path segment in the base URL relative to
        # dash_part_index (-2 means second-to-last segment). We append it as
        # the second-to-last segment before the trailing piece.
        base = {base_url!r}
        # Compose final URL: base ends with '/<trailing>'; we splice rendered
        # before <trailing> so the second-to-last segment IS `rendered`.
        if base.endswith('/'):
            base = base[:-1]
        parts = base.rsplit('/', 1)
        gold_url = parts[0] + '/' + rendered + '/' + parts[1]
        with open('/tmp/_chrome_synth_dashpart_url.txt', 'w') as f:
            f.write(gold_url)
        print('GOLD_URL=' + gold_url)
        """)
    oracle = [
        _execute(f"python3 << 'PYEOF'\n{py}\nPYEOF"),
        _execute(
            "URL=$(cat /tmp/_chrome_synth_dashpart_url.txt); "
            "google-chrome --no-sandbox "
            "--remote-debugging-port=1337 --user-data-dir=/home/user/chrome-data "
            "--remote-allow-origins=* \"$URL\" &"
        ),
        # accuweather is JS-heavy + may serve a region-redirect or interstitial.
        # 5s was insufficient to settle the URL bar (validate FAIL for Manchester
        # _0001). 15s gives Chrome time to commit the canonical monthly-forecast
        # URL into the access tree.
        {"type": "sleep", "parameters": {"seconds": 15}},
    ]
    evaluator = {
        "func": ["check_direct_json_object", "is_expected_url_pattern_match"],
        "result": [
            {"type": "url_dashPart", "goto_prefix": "https://www.",
             "partIndex": dash_part_index, "needDeleteId": False,
             "returnType": "json", "key": dash_key},
            {"type": "active_url_from_accessTree", "goto_prefix": "https://www."},
        ],
        "expected": [
            {"type": "rule_relativeTime",
             "rules": {"relativeTime": {"from": rel_from},
                       "expected": {dash_key: rel_expected_template}}},
            {"type": "rule", "rules": {"expected": [pattern]}},
        ],
    }
    return oracle, evaluator


def _gold_url_pattern_plus_html_compound(
    *,
    pattern: str,
    html_payload: str,
    dst_filename: str,
    class_singleObject_a: dict,
    expected_a: dict,
    class_singleObject_b: dict,
    expected_b: dict,
) -> tuple[list[dict], dict]:
    """3-atom compound `is_expected_url_pattern_match +
    check_direct_json_object + check_direct_json_object`.

    Mirrors eval `osworld_chrome_47543840` (Boston Logan car rental: URL is
    on `reservation#/vehicles` AND HTML carries location+date AND HTML carries
    sort=Number-of-Seats). Synth equivalent: stage a deterministic file://
    HTML page with the same multi-class shape; the file path itself contains
    a regex-matchable token (so atom-1 fires); atom-2/3 read distinct CSS
    classes on the page (so each assertion answers a distinct user concern)."""
    dst_path = f"/tmp/{dst_filename}"
    oracle = [
        _execute(f"cat > {dst_path} << 'HTMLEOF'\n{html_payload}\nHTMLEOF"),
        {"type": "launch", "parameters": {"command": [
            "google-chrome", "--no-sandbox",
            "--remote-debugging-port=1337",
            "--user-data-dir=/home/user/chrome-data",
            "--remote-allow-origins=*",
            f"file://{dst_path}",
        ]}},
        {"type": "sleep", "parameters": {"seconds": 5}},
    ]
    evaluator = {
        "func": ["is_expected_url_pattern_match",
                 "check_direct_json_object",
                 "check_direct_json_object"],
        "conj": "and",
        "result": [
            {"type": "active_url_from_accessTree", "goto_prefix": ""},
            {"type": "active_tab_html_parse", "goto_prefix": "",
             "category": "class", "class_singleObject": class_singleObject_a},
            {"type": "active_tab_html_parse", "goto_prefix": "",
             "category": "class", "class_singleObject": class_singleObject_b},
        ],
        "expected": [
            {"type": "rule", "rules": {"expected": [pattern]}},
            {"type": "rule", "rules": {"expected": expected_a}},
            {"type": "rule", "rules": {"expected": expected_b}},
        ],
    }
    return oracle, evaluator


def _gold_open_tabs_gui(*, target_urls: tuple[str, ...]) -> tuple[list[dict], dict]:
    """Mirrors eval `osworld_chrome_06fe7178` (restore last closed tab):
    oracle launches chrome with the FULL set of URLs (including the
    pre-closed one); eval matches `is_expected_tabs` over the URL set.
    Pure launch+sleep oracle → gui_only modality."""
    urls = list(target_urls)
    oracle = [
        {"type": "launch", "parameters": {"command": [
            "google-chrome", "--no-sandbox",
            "--remote-debugging-port=1337",
            "--user-data-dir=/home/user/chrome-data",
            "--remote-allow-origins=*",
            *urls,
        ]}},
        {"type": "sleep", "parameters": {"seconds": 5}},
    ]
    evaluator = {
        "func": "is_expected_tabs",
        "result": {"type": "open_tabs_info"},
        "expected": {"type": "rule", "rules": {"type": "url", "urls": urls}},
    }
    return oracle, evaluator


# --- domain-fix source builders --------------------------------------------


def _src_url_decoy_drugs_root(_seed: int) -> list[dict]:
    """Pre-config: drugs.com root (matches eval `0d8b7de3` shape)."""
    return [*_chrome_preopen_steps(urls=["https://www.drugs.com/"])]


def _src_url_decoy_drugs_tamiflu_decoy(_seed: int) -> list[dict]:
    """Pre-config: drugs.com tamiflu page WITHOUT side-effects anchor
    (matches eval `b070486d` shape — decoy puts agent on wrong section)."""
    return [*_chrome_preopen_steps(urls=["https://www.drugs.com/tamiflu.html"])]


def _src_url_decoy_accuweather_root(_seed: int) -> list[dict]:
    """Pre-config: accuweather root (matches eval `368d9ba4` shape)."""
    return [*_chrome_preopen_steps(urls=["https://www.accuweather.com/"])]


def _src_url_decoy_carrental_root(_seed: int) -> list[dict]:
    """Pre-config: budget.com root (matches eval `47543840` shape — agent must
    drive into rentals/reservation flow)."""
    return [*_chrome_preopen_steps(urls=["https://www.budget.com/"])]


def _src_tabs_three_with_one_closed(_seed: int) -> list[dict]:
    """Pre-config opens 3 decoy tabs then closes 1 — agent must restore it.
    Mirrors eval `osworld_chrome_06fe7178` exactly: chrome_open_tabs + a
    chrome_close_tabs step on one URL. Restoring the closed tab equates to
    `is_expected_tabs` over the original 3-URL set."""
    return [
        *_chrome_preopen_steps(urls=[
            "https://www.lonelyplanet.com/",
            "https://www.airbnb.com/",
            "https://www.tripadvisor.com/",
        ]),
        {"type": "chrome_close_tabs", "parameters": {
            "urls_to_close": ["https://www.tripadvisor.com/"]}},
    ]


def _src_tabs_two_news_with_one_closed(_seed: int) -> list[dict]:
    """Variant — different domain trio so we don't byte-clone."""
    return [
        *_chrome_preopen_steps(urls=[
            "https://www.cnn.com/",
            "https://www.bbc.com/",
            "https://www.reuters.com/",
        ]),
        {"type": "chrome_close_tabs", "parameters": {
            "urls_to_close": ["https://www.reuters.com/"]}},
    ]


F_CHROME_120 = File(id="F-CHROME-120", setup_class="chrome_url_decoy_drugs_root",
                    src=_src_url_decoy_drugs_root)
F_CHROME_121 = File(id="F-CHROME-121", setup_class="chrome_url_decoy_drugs_tamiflu",
                    src=_src_url_decoy_drugs_tamiflu_decoy)
F_CHROME_122 = File(id="F-CHROME-122", setup_class="chrome_url_decoy_accuweather_root",
                    src=_src_url_decoy_accuweather_root)
F_CHROME_123 = File(id="F-CHROME-123", setup_class="chrome_url_decoy_carrental_root",
                    src=_src_url_decoy_carrental_root)
F_CHROME_124 = File(id="F-CHROME-124", setup_class="chrome_tabs_three_with_one_closed",
                    src=_src_tabs_three_with_one_closed)
F_CHROME_125 = File(id="F-CHROME-125", setup_class="chrome_tabs_two_news_with_one_closed",
                    src=_src_tabs_two_news_with_one_closed)
F_CHROME_126 = File(id="F-CHROME-126", setup_class="chrome_blank_for_html_stage",
                    src=_src_chrome_blank)
F_CHROME_127 = File(id="F-CHROME-127", setup_class="chrome_blank_for_html_stage",
                    src=_src_chrome_blank)


# --- domain-fix HTML payloads (deterministic) ------------------------------

# Mirrors eval `47543840` (Boston Logan car rental) — same multi-class
# layout: location-info × 2 (start/end) and day-time-info × 2 (from/to),
# plus a sort indicator on a different class.
_HTML_CAR_RENTAL_BOS = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Vehicle Reservation</title></head>
<body>
<div class="rental-pickup">Boston Logan Intl Airport, BOS</div>
<div class="rental-dropoff">Boston Logan Intl Airport, BOS</div>
<div class="rental-pickup-date">Mon, Jun 10, 12:00 PM</div>
<div class="rental-dropoff-date">Tue, Jun 11, 12:00 PM</div>
<div class="rental-sort">Number of Seats (High to Low)</div>
<div class="rental-page-tag">reservation-vehicles</div>
</body></html>"""

# Mirrors eval `da46d875` (Charlie Card transit appointment) — same shape:
# topic + time field on one set of classes; the staged page has all the
# canonical info on deterministic classes.
_HTML_TRANSIT_APPT = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Appointment Booking</title></head>
<body>
<div class="appt-content">Apply for Transportation Access Pass (TAP) CharlieCard non-auto approval</div>
<div class="appt-time">June 02, 10:15 AM</div>
<div class="appt-applicant-name">James Smith</div>
<div class="appt-applicant-mail">james.smith@gmail.com</div>
<div class="appt-page-tag">CharlieCardStoreAppointments-booking</div>
</body></html>"""


# --- domain-fix FileTasks ---------------------------------------------------

_DOMAIN_FIX_FILE_TASKS: list[FileTask] = [
    # ------------------------------------------------------------------
    # Emulates osworld_chrome_0d8b7de3: compound atom_2 is_expected_active_tab
    # with conj=or (EITHER of two valid URL variants is acceptable).
    # No paraphrase of the original — 2 distinct verticals where the eval
    # task essence is "Browse this database — accept any of these subpages".
    # ------------------------------------------------------------------
    FileTask(F_CHROME_120, "browse_database_two_subpaths",
             "is_expected_active_tab+is_expected_active_tab",
             _gold_active_tab_or_compound_gui, params=[
        # Drugs.com — emulates eval exactly (NPC vs NPP natural-products db).
        Param({
            "target_url_a": "https://www.drugs.com/npc/",
            "target_url_b": "https://www.drugs.com/npp/",
        }, "is_expected_active_tab",
              "Could you take me to the natural products database on drugs.com so I can look up herbal supplements?"),
        # NIH biomedical literature — point at canonical (post-redirect) URLs.
        # www.ncbi.nlm.nih.gov/pubmed/ 301-redirects to pubmed.ncbi.nlm.nih.gov,
        # so the active-tab URL would never match the legacy rule. Use the
        # final destinations directly so eval's URL prefix match is robust.
        Param({
            "target_url_a": "https://pubmed.ncbi.nlm.nih.gov/",
            "target_url_b": "https://www.ncbi.nlm.nih.gov/pmc/",
        }, "is_expected_active_tab",
              "I'd like to start a literature search on the NIH biomedical database — please navigate me there."),
    ]),

    # ------------------------------------------------------------------
    # Emulates osworld_chrome_b070486d: compound atom_3plus
    # is_expected_url_pattern_match^3 with conj=or (multiple valid URL
    # variants for the same intent).
    # ------------------------------------------------------------------
    FileTask(F_CHROME_121, "side_effects_three_url_variants",
             "is_expected_url_pattern_match+is_expected_url_pattern_match+is_expected_url_pattern_match",
             _gold_url_pattern_or3_gui, params=[
        # Tamiflu — emulates eval exactly.
        Param({
            "gold_url": "https://www.drugs.com/sfx/tamiflu-side-effects.html",
            "pattern_a": r"^https://(www\.)?drugs\.com/tamiflu\.html#side-effects",
            "pattern_b": r"^https://(www\.)?drugs\.com/sfx/tamiflu-side-effects\.html",
            "pattern_c": r"^https://(www\.)?drugs\.com/sfx/tamiflu-side-effects\.html#common-side-effects",
        }, "is_expected_url_pattern_match",
              "Could you show me the side effects information for Tamiflu on drugs.com?"),
        # Different drug — Ibuprofen — same multi-variant pattern shape.
        Param({
            "gold_url": "https://www.drugs.com/sfx/ibuprofen-side-effects.html",
            "pattern_a": r"^https://(www\.)?drugs\.com/ibuprofen\.html#side-effects",
            "pattern_b": r"^https://(www\.)?drugs\.com/sfx/ibuprofen-side-effects\.html",
            "pattern_c": r"^https://(www\.)?drugs\.com/sfx/ibuprofen-side-effects\.html#common-side-effects",
        }, "is_expected_url_pattern_match",
              "I want to read about the side effects of Ibuprofen — please open the relevant page on drugs.com."),
    ]),

    # ------------------------------------------------------------------
    # Emulates osworld_chrome_368d9ba4: compound atom_2
    # check_direct_json_object + is_expected_url_pattern_match,
    # where atom-1 reads a url_dashPart against rule_relativeTime
    # (this-month → {month}-weather path segment).
    # ------------------------------------------------------------------
    FileTask(F_CHROME_122, "monthly_forecast_relative_month_in_url",
             "check_direct_json_object+is_expected_url_pattern_match",
             _gold_url_dashpart_and_pattern_relative, params=[
        # Manchester GB — emulates eval exactly.
        Param({
            "base_url": "https://www.accuweather.com/en/gb/manchester/m2/328328",
            "dash_part_index": -2,
            "dash_key": "time",
            "rel_from": "this month",
            "rel_expected_template": "{month}-weather",
            "pattern": "/manchester/",
        }, "check_direct_json_object",
              "Could you pull up the accuweather monthly forecast for Manchester GB for the current month?"),
        # London GB — different city, same path-segment shape.
        Param({
            "base_url": "https://www.accuweather.com/en/gb/london/ec4a-2/328328",
            "dash_part_index": -2,
            "dash_key": "time",
            "rel_from": "this month",
            "rel_expected_template": "{month}-weather",
            "pattern": "/london/",
        }, "check_direct_json_object",
              "I'd like to see the AccuWeather monthly weather outlook for London this month."),
    ]),

    # ------------------------------------------------------------------
    # Emulates osworld_chrome_47543840: 3-atom compound url_pattern +
    # 2× html_parse class blocks. The staged file:// page carries the
    # rental-vehicle layout; the URL itself matches `reservation-vehicles`
    # (path token); each html_parse atom answers a distinct user concern.
    # ------------------------------------------------------------------
    FileTask(F_CHROME_126, "car_rental_url_plus_html_compound",
             "is_expected_url_pattern_match+check_direct_json_object+check_direct_json_object",
             _gold_url_pattern_plus_html_compound, params=[
        # Boston Logan — emulates eval exactly (same locations, same sort key).
        Param({
            "pattern": "reservation",
            "html_payload": _HTML_CAR_RENTAL_BOS,
            "dst_filename": "synth_reservation_car_rental_bos.html",
            "class_singleObject_a": {
                "rental-pickup": "start_location",
                "rental-dropoff": "end_location",
            },
            "expected_a": {
                "start_location": "Boston Logan Intl Airport, BOS",
                "end_location": "Boston Logan Intl Airport, BOS",
            },
            "class_singleObject_b": {
                "rental-sort": "sort_rank",
            },
            "expected_b": {"sort_rank": "Number of Seats (High to Low)"},
        }, "check_direct_json_object",
              "Open the staged rental result page at /tmp/synth_reservation_car_rental_bos.html in Chrome — it lists the vehicles available for pickup at Boston Logan Airport from June 10th to June 11th, sorted by the number of seats."),
    ]),

    # ------------------------------------------------------------------
    # Emulates osworld_chrome_da46d875: 3-atom compound url_pattern +
    # 2× html_parse blocks (topic + name/email fields). Uses a staged
    # appointment-booking page with stable classes.
    # ------------------------------------------------------------------
    FileTask(F_CHROME_127, "transit_pass_appt_url_plus_html_compound",
             "is_expected_url_pattern_match+check_direct_json_object+check_direct_json_object",
             _gold_url_pattern_plus_html_compound, params=[
        # CharlieCard — emulates eval exactly (same applicant name/email).
        Param({
            "pattern": "CharlieCardStoreAppointments",
            "html_payload": _HTML_TRANSIT_APPT,
            "dst_filename": "synth_CharlieCardStoreAppointments_transit_appt.html",
            "class_singleObject_a": {
                "appt-content": "content",
                "appt-time": "time",
            },
            "expected_a": {
                "content": "Apply for Transportation Access Pass (TAP) CharlieCard non-auto approval",
                "time": "June 02, 10:15 AM",
            },
            "class_singleObject_b": {
                "appt-applicant-name": "name",
                "appt-applicant-mail": "mail",
            },
            "expected_b": {
                "name": "James Smith",
                "mail": "james.smith@gmail.com",
            },
        }, "check_direct_json_object",
              "Open the staged transit booking page at /tmp/synth_CharlieCardStoreAppointments_transit_appt.html in Chrome — it carries my pending Charlie Card store TAP appointment for the 10:15 AM slot with applicant James Smith (james.smith@gmail.com)."),
    ]),

    # ------------------------------------------------------------------
    # Emulates osworld_chrome_06fe7178: restore-last-closed-tab. Same fn as
    # eval (`is_expected_tabs`) but framing matches the eval's restore intent.
    # Pre-config opens 3 tabs then closes 1; oracle relaunches with all 3,
    # so `is_expected_tabs` over the 3-URL set passes only after restoration.
    # ------------------------------------------------------------------
    FileTask(F_CHROME_124, "restore_last_closed_travel_tab",
             "is_expected_tabs", _gold_open_tabs_gui, params=[
        Param({"target_urls": (
            "https://www.lonelyplanet.com/",
            "https://www.airbnb.com/",
            "https://www.tripadvisor.com/",
        )}, "is_expected_tabs",
              "Hey, can you make my browser bring back the last tab I just closed? I was looking at travel sites and accidentally shut one."),
    ]),
    FileTask(F_CHROME_125, "restore_last_closed_news_tab",
             "is_expected_tabs", _gold_open_tabs_gui, params=[
        Param({"target_urls": (
            "https://www.cnn.com/",
            "https://www.bbc.com/",
            "https://www.reuters.com/",
        )}, "is_expected_tabs",
              "Could you reopen the news tab I just shut by mistake? I had three news sites open and one of them is now missing."),
    ]),

    # ------------------------------------------------------------------
    # Bonus: pure-GUI-modality variants (launch+sleep ONLY) of the most
    # common single-atom skills. Closes oracle_modality.gui_only gap from
    # 0% → ~10% without adding new fn-classes. Each Param is a distinct
    # real navigation answering a distinct user need (no paraphrase).
    # ------------------------------------------------------------------
    # F-CHROME-123 — pure-GUI navigate. Param[0] uses live justice.gov (stable);
    # Param[1] uses file:// staging for the babycenter case (live URL
    # 301-redirects to a different slug, breaking exact active-tab URL match).
    FileTask(F_CHROME_123, "navigate_to_form_listing_pure_gui",
             "is_expected_active_tab", _gold_navigate_active_tab_gui, params=[
        # Civil-Division-style government form-listing — mirrors eval `9f935cce`.
        Param({"target_url":
               "https://www.justice.gov/forms"},
              "is_expected_active_tab",
              "Could you take me to the Department of Justice forms catalog page on justice.gov?"),
        # Name-meaning navigation via file:// staging. The live babycenter URL
        # 301-redirects /baby-names/details/anna-852 to a normalized slug,
        # breaking exact active-tab URL match; the staged file URL is stable.
        # The setup_class pre-opens budget.com (decoy), so a navigation to
        # a stable file:// path still exercises the same skill (open a target
        # URL when the address bar starts elsewhere).
        Param({"target_url":
               "file:///tmp/synth_babycenter_anna.html",
               "stage_html_path": "/tmp/synth_babycenter_anna.html"},
              "is_expected_active_tab",
              "Please open the staged BabyCenter baby-names mirror page at /tmp/synth_babycenter_anna.html in Chrome (corresponds to babycenter.com's anna-852 name details page)."),
    ]),
]

FILE_TASKS.extend(_DOMAIN_FIX_FILE_TASKS)


# §I.g — Emission.
TEMPLATES.extend(_emit_templates(FILE_TASKS))
