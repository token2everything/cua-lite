"""Render the canonical HuggingFace dataset card for a cua-lite repo.

One template; one configs-YAML emitter. Every dataset published to
``cua-lite/<Name>`` runs through here so every repo on the Hub looks
identical (front matter, sections, stats table format).

Inputs:

* ``name`` — repo name (e.g. ``"ScaleCUA"``).
* ``repo`` — dict loaded from a dataset's ``repo.json``; see
  :func:`load_repo_json`. Carries ``description``, ``license``,
  ``citation``, ``original_urls``, ``extra_notes``.
* ``stats`` — :class:`lite.data.staging.DatasetStats` with
  ``by_partition``, ``unique_images``, ``image_store_bytes``.

Outputs the README markdown string. :func:`render_card` and
:func:`load_repo_json` are the surface; ``configs_yaml`` and ``stats_table``
are section builders that only :func:`render_card` calls. ``configs_yaml``
stays exported because its glob matrix is unit-tested directly
(``tests/data/hf/test_card.py``); ``stats_table`` is not exported.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from lite.data.staging import ORG, DatasetStats

# ---------------------------------------------------------------------------
# repo.json
# ---------------------------------------------------------------------------

def load_repo_json(preproc_dir: Path | str) -> dict:
    """Read ``repo.json`` from a directory that owns one.

    Three producers write this file: ``lite/data/preproc/<dataset>/`` (checked in,
    preproc datasets), ``devs/data/<route>/`` (checked in, published rollout routes,
    passed to ``lite.data.hf.stage --repo-dir``), and a rollout staging dir, which
    ``stage`` writes by merging the route file with this run's provenance.

    Required keys: ``description``, ``original_urls`` (list[str]), ``license``,
    ``citation``. Optional: ``extra_notes`` (default ``""``).
    """
    p = Path(preproc_dir) / "repo.json"
    data = json.loads(p.read_text())
    for k in ("description", "original_urls", "license", "citation"):
        if k not in data:
            raise KeyError(f"{p} missing required key: {k!r}")
    data.setdefault("extra_notes", "")
    return data


# ---------------------------------------------------------------------------
# Configs YAML
# ---------------------------------------------------------------------------

def _data_files_globs(
    platform: str | None,
    task_type: str | None,
    splits: set[str],
    *,
    config_label: str | None = None,
) -> list[dict]:
    """Globs covering every layout permutation HF push could emit, for the
    splits that ACTUALLY have files (``splits``).

    HF's fsspec glob accepts ``**`` only as a path component, not fused
    with a suffix, so we enumerate exact depths.

    Only declare a split when it has data: the HF dataset-viewer rejects a
    config whose declared split resolves to ZERO files ("the splits use
    different data file formats"), which is what happens for a small dataset
    that hash-splits to an empty ``validation`` (e.g. a rollout-staged cohort).

    ``config_label`` (the ``--config-names`` override path) scopes the globs to
    a single variant FILENAME (``<label>.parquet``) — that label lives under the
    split folder as the variant filename — gathered across every cohort. It
    overrides the ``platform`` / ``task_type`` args.
    """
    out: list[dict] = []
    for sp in ("train", "validation"):
        if sp not in splits:
            continue
        if config_label is not None:
            paths = [
                f"*/*/{sp}/{config_label}.parquet",
                f"*/*/{sp}/{config_label}/*.parquet",
            ]
        else:
            pat_plat = platform or "*"
            pat_tt = task_type if task_type else "*"
            paths = [
                f"{pat_plat}/{pat_tt}/{sp}*parquet",
                f"{pat_plat}/{pat_tt}/{sp}/*.parquet",
                f"{pat_plat}/{pat_tt}/{sp}/*/*.parquet",
            ]
        out.append({"split": sp, "path": paths})
    return out


def _build_configs(stats: DatasetStats, *, config_name_override: bool = False) -> list[dict]:
    """Emit ``default``, optionally per-platform, and one per cohort.

    ``config_name_override`` (the ``--config-names`` staging path) instead emits
    ``default`` + one config per DISTINCT non-empty variant in ``by_partition``,
    naming each config VERBATIM after the variant label and globbing that label's
    files across all cohorts. The derived ``<platform>.<task_type>`` cohort
    configs are NOT emitted. See :func:`_build_configs_override`.
    """
    if config_name_override:
        return _build_configs_override(stats)
    return _build_configs_default(stats)


def _build_configs_override(stats: DatasetStats) -> list[dict]:
    """``--config-names`` override path: ``default`` + one config per distinct
    variant label, named VERBATIM, with label-scoped (cross-cohort) globs."""
    all_splits = {sp for (_p, _tt, sp, _v) in stats.by_partition}
    configs: list[dict] = [
        {"config_name": "default", "data_files": _data_files_globs(None, None, all_splits)},
    ]
    labels = sorted({v for (_p, _tt, _sp, v) in stats.by_partition if v})
    for label in labels:
        ls = {sp for (_p, _tt, sp, v) in stats.by_partition if v == label}
        configs.append({
            "config_name": label,
            "data_files": _data_files_globs(None, None, ls, config_label=label),
        })
    return configs


def _build_configs_default(stats: DatasetStats) -> list[dict]:
    """Emit ``default``, optionally per-platform, and one per cohort.

    Cohort config names use all-period spelling (``mobile.grounding.action``).
    The HF dataset-viewer percent-encodes ``@`` in cohort asset URLs while
    the S3 signed-URL signature is computed over the unencoded path — the
    mismatch returns 403 on every image fetch and the cohort preview shows
    broken-image icons. ``.`` is unreserved per RFC 3986, never URL-encoded
    by clients, no signature mismatch.

    The agent registry lookup key in code remains
    ``<agent>@<platform>@<task_type>`` (e.g. ``qwen3_vl@mobile@grounding.action``);
    only the user-facing HF ``config_name`` token uses ``.`` between platform
    and task_type.

    Per-platform configs are only emitted when the dataset spans more than
    one platform — for single-platform datasets, ``default`` already
    covers it.
    """
    platforms = sorted({p for (p, _tt, _sp, _v) in stats.by_partition})
    cohorts = sorted({(p, tt) for (p, tt, _sp, _v) in stats.by_partition})
    all_splits = {sp for (_p, _tt, sp, _v) in stats.by_partition}
    configs: list[dict] = [
        {"config_name": "default", "data_files": _data_files_globs(None, None, all_splits)},
    ]
    if len(platforms) > 1:
        for p in platforms:
            ps = {sp for (pp, _tt, sp, _v) in stats.by_partition if pp == p}
            configs.append({"config_name": p, "data_files": _data_files_globs(p, None, ps)})
    for (p, tt) in cohorts:
        cs = {sp for (pp, tt2, sp, _v) in stats.by_partition if pp == p and tt2 == tt}
        configs.append({
            "config_name": f"{p}.{tt}",
            "data_files": _data_files_globs(p, tt, cs),
        })
    return configs


def _yaml_dump_configs(configs: list[dict]) -> str:
    """Minimal YAML emitter sufficient for our shape — avoids a PyYAML dep."""
    lines = ["configs:"]
    for cfg in configs:
        lines.append(f"- config_name: {cfg['config_name']}")
        lines.append("  data_files:")
        for df in cfg["data_files"]:
            lines.append(f"  - split: {df['split']}")
            lines.append("    path:")
            for p in df["path"]:
                lines.append(f"    - \"{p}\"")
    return "\n".join(lines)


def configs_yaml(stats: DatasetStats, *, config_name_override: bool = False) -> str:
    return _yaml_dump_configs(_build_configs(stats, config_name_override=config_name_override))


# ---------------------------------------------------------------------------
# Stats table
# ---------------------------------------------------------------------------

def stats_table(stats: DatasetStats) -> str:
    by_part: dict[tuple, dict[str, int]] = defaultdict(lambda: {"train": 0, "validation": 0})
    for (platform, task_type, split, variant), n in stats.by_partition.items():
        by_part[(platform, task_type, variant)][split] = n
    lines = [
        "| platform | task_type | variant | train | validation |",
        "|---|---|---|---:|---:|",
    ]
    for (platform, task_type, variant), counts in sorted(by_part.items()):
        lines.append(
            f"| {platform} | {task_type} | {variant} | "
            f"{counts['train']:,} | {counts['validation']:,} |"
        )
    return "\n".join(lines)


def _config_examples(
    name: str, stats: DatasetStats, *, org: str = ORG, config_name_override: bool = False
) -> str:
    # In override mode the configs are the verbatim variant labels (see
    # _build_configs_override), NOT the derived <platform>.<task_type> cohorts —
    # so the example must load an existing label, else it'd raise at load time.
    if config_name_override:
        labels = sorted({v for (_p, _tt, _sp, v) in stats.by_partition if v})
        if not labels:
            return "# (no sub-configs for this dataset)"
        return (
            f"# just one named subset (config)\n"
            f'ds = load_dataset("{org}/{name}", "{labels[0]}")'
        )
    platforms = sorted({p for (p, _tt, _sp, _v) in stats.by_partition})
    cohorts = sorted({(p, tt) for (p, tt, _sp, _v) in stats.by_partition})
    samples: list[str] = []
    if len(platforms) > 1:
        samples.append(
            f"# just one platform\n"
            f'ds = load_dataset("{org}/{name}", "{platforms[0]}")'
        )
    if cohorts:
        p, tt = cohorts[0]
        samples.append(
            f"# just one (platform, task_type) cohort\n"
            f'ds = load_dataset("{org}/{name}", "{p}.{tt}")'
        )
    return "\n\n".join(samples) if samples else "# (no sub-configs for this dataset)"


# ---------------------------------------------------------------------------
# Card template
# ---------------------------------------------------------------------------

_TEMPLATE = """---
license: other
tags:
- cua-lite
- gui
- sft
task_categories:
- image-text-to-text
{configs_yaml}
---

# {org}/{name}

{description}

## Origin

{origin_list}

## Load via `datasets`

```python
from datasets import load_dataset

# entire dataset
ds = load_dataset("{org}/{name}")

{config_examples}
```

After loading, parse `metadata` as JSON before filtering by `metadata_kind`,
`dims`, or `others.*`; every row carries a rich metadata object inside that JSON
string (see schema below). CUA rows use `dims == [platform, task_type]`.

## Schema

Published parquet columns:

| column | type | notes |
|---|---|---|
| `images` | list[Image] | embedded PNG/JPEG bytes; HF viewer renders thumbnails |
| `messages` | string (JSON array) | parse as JSON to OpenAI-style turns with `role`, structured `content`, nested `tool_calls`, and `role:"tool"` results |
| `metadata` | string (JSON object) | parse as JSON to fields `metadata_kind`, `dims`, `extra_tool_schemas`, CUA-only `valid_actions`, and `others` |
| `_folded` | string (JSON array, optional) | folded grounding/understanding rows only; authoritative per-instruction `messages` / `metadata` members |

Coordinate values in `messages` are normalized to `[0, 1000]` integers.
The JSON examples below show the decoded shape, not the raw string cell.

`metadata.extra_tool_schemas[*]` uses the nested Chat Completions function-tool
declaration shape:

```json
{{
  "metadata_kind": "cua",
  "dims": ["desktop", "use"],
  "extra_tool_schemas": [
    {{
      "type": "function",
      "function": {{
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {{
          "type": "object",
          "properties": {{"cmd": {{"type": "string"}}}},
          "required": ["cmd"]
        }}
      }}
    }}
  ],
  "valid_actions": ["click", "type"],
  "others": {{}}
}}
```

`messages[].tool_calls[*]` uses the matching nested invocation shape. Tool
results pair `tool_calls[].id` with `role:"tool"` `tool_call_id`:

```json
[
  {{
    "role": "user",
    "content": [
      {{"type": "image", "index": 0}},
      {{"type": "text", "text": "Click the OK button."}}
    ]
  }},
  {{
    "role": "assistant",
    "tool_calls": [
      {{
        "id": "call_0000",
        "type": "function",
        "function": {{
          "name": "computer",
          "arguments": {{
            "actions": [
              {{"action": "click", "coordinate": [640, 400]}},
              {{"action": "type", "text": "hello"}}
            ]
          }}
        }}
      }}
    ]
  }},
  {{
    "role": "tool",
    "tool_call_id": "call_0000",
    "content": [
      {{"type": "image", "index": 1}},
      {{"type": "text", "text": "clicked; typed"}}
    ]
  }}
]
```

**Image-dedup (`grounding.*` / `understanding` cohorts).** These cohorts are
single-image-per-row and many rows share the same screenshot, so to avoid
re-embedding identical image bytes once per instruction they are stored
*folded*: one row per unique screenshot (image embedded once), carrying an
extra **`_folded`** column — a JSON string with the authoritative list of
per-instruction members for that screenshot. Each member's `messages` and
`metadata` values are the same opaque JSON strings described above. The row's
top-level `messages` is a JSON string containing the members concatenated for
viewer convenience. `use` cohorts are not folded. **Use
`lite.data.hf.download` to consume this repo** — it unfolds automatically back
to one row per instruction; reading the parquet directly yields the folded form.

## Layout

```
<platform>/<task_type>/<split>/<variant>/shard-NNNNN-of-NNNNN.parquet
```

- `platform` ∈ {{desktop, browser, mobile}}
- `task_type` ∈ {{understanding, grounding.action, grounding.point, grounding.bbox, use}} — used verbatim as the dir component
- HF config names are `<platform>.<task_type>` by default (e.g. `mobile.grounding.action`) — UNLESS the dataset was staged with `--config-names`, which sets verbatim, explicitly-chosen config names (see the `configs:` block above for the authoritative list). The agent registry lookup key in code is `<agent>@<platform>@<task_type>` (e.g. `qwen3_vl@mobile@grounding.action`); only this user-facing token uses `.` between platform and task_type, because `@` triggers a 403 on the dataset-viewer's signed image URLs.
- HF split names stay `train` / `validation` (the `datasets` library blacklists `<>:/\\|?*` in split names; everything else is fine in config_name)
- `validation` is an in-distribution held-out slice: no validation **sample** also appears in `train` — content-identical rows (same `images` + same `messages`, differing only in their ids) are co-located into one split, so upstream re-publishing one sample under two ids cannot straddle the split. It is *not* disjoint in **images**: one screenshot legitimately backs many distinct samples, and only whole samples are co-located, so the same picture can appear on both sides. `test` is reserved for out-of-distribution benchmark datasets

## Stats

{stats_table}

## Local mirror & SFT export

For local workflows (SFT export, dedup, mixing across datasets), use
`lite.data.hf.download` to mirror this repo back to the canonical local
layout:

```
$CUA_LITE_DATASETS_ROOT/{org}/{name}/
  images/<hash[:2]>/<hash>.<ext>                          # content-addressed image store
  <platform>/<task_type>/<split>[/<variant>].parquet      # rows reference images by relative path
```

Rows in the local parquet have `images: list[str]`; bytes are extracted to
the image store. `lite.train.export.export_sft` consumes the local
form directly with `--image-root=$CUA_LITE_DATASETS_ROOT`.

- Total unique images: **{n_images:,}**
- Image store size: **{store_bytes_gb:.2f} GB**

## Notes

{extra_notes}

## License & citation

{license}

{citation}
"""


def render_card(*, name: str, repo: dict, stats: DatasetStats, org: str = ORG) -> str:
    """Render the canonical README for ``<org>/<name>``.

    ``org`` defaults to :data:`lite.data.staging.ORG` — the same constant that
    names the on-disk staging root segment and the ``images/`` path prefix baked
    into every published row — so the card's ``load_dataset(...)`` snippets and
    the repo actually created by ``upload.push_dataset`` cannot drift apart.

    ``repo`` may carry ``config_name_override`` (written by ``lite.data.hf.stage``
    when ``--config-names`` is given); when truthy the configs YAML names each
    config VERBATIM after its variant label instead of the derived cohort names.
    """
    config_name_override = bool(repo.get("config_name_override", False))
    return _TEMPLATE.format(
        name=name,
        org=org,
        description=repo["description"].strip(),
        origin_list="\n".join(f"- [{u}]({u})" for u in repo["original_urls"]),
        configs_yaml=configs_yaml(stats, config_name_override=config_name_override),
        config_examples=_config_examples(
            name, stats, org=org, config_name_override=config_name_override
        ),
        stats_table=stats_table(stats),
        n_images=stats.unique_images,
        store_bytes_gb=stats.image_store_bytes / 1e9,
        extra_notes=(repo.get("extra_notes") or "").strip() or "_(none)_",
        license=repo["license"].strip(),
        citation=repo["citation"].strip(),
    )


__all__ = [
    "load_repo_json",
    "configs_yaml",
    "render_card",
]
