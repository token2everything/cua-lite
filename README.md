# WebArena-Verified v0.4 generation

Generate entity-bound Shopping + GitLab tasks from fixed WebArena-Verified 1.2.3
assets. Each author, teacher and student attempt receives a fresh two-site
baseline. One GUI episode reads source facts, selects an intermediate entity,
uses its facts on the target site, and leaves replayable evidence.

The coordinator owns sandbox creation, original handles and cleanup. Author
scripts collect visible facts without model calls. Qwen3.8-max and GLM5.3-flash
execute public tasks independently; private evidence never enters model input.

```bash
export PYTHONDONTWRITEBYTECODE=1
export UV_PROJECT_ENVIRONMENT=/tmp/wav-v04-host-env
bash setup.sh
uv run --no-sync python -B pipeline.py init --run newbatch \
  --sites '["shopping","gitlab"]' --env /absolute/credentials.env
uv run --no-sync python -B pipeline.py probe --run newbatch --env /absolute/credentials.env
uv run --no-sync python -B pipeline.py prepare --run newbatch --env /absolute/credentials.env
uv run --no-sync python -B pipeline.py explore --run newbatch --env /absolute/credentials.env
```

Read [the generation skill](/skills/webarena-verified-v04-generate/SKILL.md) for
entity selection, author inputs, admission, independent sampling, fixed46
cleaning and absolute-path JSON/JSONL delivery. `pipeline.py` hands resource
creation to `infra/worker.py`, execution to `infra/run_case.sh`, and private
scoring to `devs/data/webarena_verified_generated`.

Each run freezes the complete source, dependencies, vendor inputs, cleaning
profile and coordinator in `release/`. Subsequent phases execute that release;
new code requires a new run. Published v0.3 provenance and its limited live
verification remain under `provenance/v03/`; they do not verify v0.4.

Pipeline version v0.4, site assets 1.2.3, and the V3 trajectory format are
separate contracts. Delivery is pending human review and is not promoted for
training. Credentials and generated runs are excluded from source releases.
