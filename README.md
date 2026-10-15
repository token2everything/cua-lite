# WebArena-Verified v0.3 generation

Generate new single-site tasks against fixed WebArena-Verified 1.2.3 assets,
ground their answers with fresh browser evidence, collect independent Qwen
trajectories, and export cleaned JSON plus absolute-path JSONL.

All operator and authoring instructions are in the self-contained
[generation skill](/skills/webarena-verified-v03-generate/SKILL.md). Invoke it with
the repository path, credentials file, requested count/site coverage, and shared
export destination. It does not require Feishu or historical run documents.

```bash
export UV_PROJECT_ENVIRONMENT=/tmp/wav-v03-host-env
bash setup.sh
uv run --no-sync python -B pipeline.py --help
```

Runtime inputs are sandbox/model credentials and the mounted fixed site assets.
Executable source, author scripts, semantic seeds, evaluator build, desktop
dependencies, fixed46 profile, wheels, locks and the post-generation catalog are
bundled. `runs/`, secrets and generated trajectories are excluded from code
releases. The imported GUI source is Source19/episode-v6; provenance is recorded
in `SOURCE_PROVENANCE.json` and per-run source manifests. This repository supports
forward generation under that pinned policy; the historical final300 contains
multiple source revisions and is not silently rewritten.

Delivery remains pending human review. No generated data is promoted for
training automatically.
