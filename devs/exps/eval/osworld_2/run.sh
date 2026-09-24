#!/usr/bin/env bash
# osworld_2 (OSWorld-V2 2026.06.24) single-phase eval — single-model invocation.
#
# Routes --log-root to <repo-root>/.exps/eval/osworld_2/<commit-ts>_<commit>/<run_id>/<slug>/.
# Repo root is derived from this script's location (worktree-safe — a stale
# CUA_LITE_ROOT inherited from another worktree's shell would otherwise
# silently redirect output).
# Resumes if the same (commit, run_id, model) was run before; completed tasks skipped.
#
# $EVAL_RUN_ID is OPTIONAL — see devs/exps/eval/AGENTS.md "Run id contract".
# If unset, run.sh auto-resolves to the highest-numbered run_<N>[_<label>] under the
# current commit dir (resume-to-latest), or `run_0` if this is the first campaign.
#
# Env-server prereq (workflow default): export CUA_LITE_ENV_SERVER_URL +
# CUA_LITE_ENV_SERVER_TOKEN before invoking. Missing vars now fail fast;
# set EVAL_ALLOW_DIRECT=1 to connect the Agent directly to its Worker.
# The env-server uses OSWORLD_WORKER_CONNECTION to reach a release-matched
# Worker; it does not create desktop VMs. Run the Agent and Worker in Sandbox.
# See /lite/gym/envs/osworld_2/WORKER.md. Multi-phase tasks use the task runner
# lite.gym.envs.osworld_2.rollout, which owns the full phase lifecycle.
#
# Usage:
#   # auto-resume to latest run_<N> at this commit (most common):
#   CUDA_VISIBLE_DEVICES=<gpus> ./devs/exps/eval/osworld_2/run.sh <model-id>
#   # or open a fresh campaign at this commit:
#   export EVAL_RUN_ID="run_1"        # bump past any existing run_0 / run_1 / ...
#   CUDA_VISIBLE_DEVICES=<gpus> ./devs/exps/eval/osworld_2/run.sh <model-id>
#
# Examples:
#   CUDA_VISIBLE_DEVICES=0       ./devs/exps/eval/osworld_2/run.sh Qwen/Qwen3-VL-8B-Instruct
#   CUDA_VISIBLE_DEVICES=0,1     ./devs/exps/eval/osworld_2/run.sh Qwen/Qwen3-VL-32B-Instruct
#   ./devs/exps/eval/osworld_2/run.sh gpt-5.5         # API model, no GPU
#
# tp_size is inferred from the GPU count in CUDA_VISIBLE_DEVICES (local HF models only).
# Pre-reqs:
#   - Provision a Worker for release osworld-v2-2026.06.24 as described in WORKER.md.
#     Its native runtime, gated task code, and VM image must match that release.
#   - Set OSWORLD_WORKER_CONNECTION to the private connection JSON on the
#     env-server host (or Agent host for EVAL_ALLOW_DIRECT=1).
#   - Configure task services and simulator/judge APIs on the Worker. Their
#     credentials are not read by the task registry or this campaign launcher.
#
# Env shape (see lite/gym/envs/osworld_2/README.md):
#   - eval split = the release's 108 tasks (ids 001-108). The filter uses only
#     the packaged protocol exclusions; missing deployment services fail during
#     execution, rather than silently shrinking the task catalog.
#   - The Worker owns one active task at a time. Default concurrency is 1;
#     higher concurrency requires independent Worker assignment per episode.
#   - Wrapper defaults come from the release's configs/default.yaml:
#     step_timeout=420, reset_timeout=2100, and loop_detect=0.
#   - max_steps: env default 200 (OSWorld-V2 official GPT run) — no override.
#     V2 trajectories are far longer than v1's 30-step runs; budget wall-clock
#     accordingly.
set -euo pipefail

MODEL="${1:?usage: CUDA_VISIBLE_DEVICES=<gpus> $0 <model-id>}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." &>/dev/null && pwd)"
[[ -n "$ROOT" && -d "$ROOT" ]] || { echo "$0: cannot resolve repo root from ${BASH_SOURCE[0]}" >&2; exit 1; }
cd "$ROOT"
EVAL_ENV_ID="osworld_2"
source "$ROOT/devs/exps/eval/utils/runtime_mode.sh"

SLUG="${MODEL//\//_}"
ENV_ROOT="$ROOT/.exps/eval/osworld_2"

# Pipeline-relevant paths: changes to these files are what advance the
# commit-id used for path keying. Pure docs (CHANGELOG, README, snapshots)
# and unrelated training/other-env code do NOT — campaigns reuse the
# existing commit dir if pipeline state hasn't moved since.
shopt -s nullglob
PIPELINE_PATHS=(
  devs/exps/eval/osworld_2/run.sh
  lite/core
  lite/agents
  lite/gym/envs/osworld_2
  lite/gym/utils
  lite/gym/__init__.py lite/gym/types.py lite/gym/registry.py
  lite/gym/factory.py lite/gym/services.py lite/gym/remote
  lite/gym/base.py lite/gym/wrappers.py
  scripts/serve_env.py devs/exps/eval/utils/runtime_mode.sh
  devs/exps/eval/utils/campaign_dir.sh
  lite/agents/factory.py lite/infer/serving.py lite/infer/rollout.py
  scripts/rollout.py
  scripts/configs/*/default/osworld_2.yaml
)
shopt -u nullglob

# Pre-flight: pipeline files must be committed (clean working tree + index).
DIRTY=$(git status --porcelain -- "${PIPELINE_PATHS[@]}" 2>/dev/null)
if [ -n "$DIRTY" ]; then
  echo "[run.sh] ERROR: pipeline files have uncommitted changes — commit first (path key would be ambiguous):" >&2
  echo "$DIRTY" | sed 's/^/  /' >&2
  exit 1
fi

# Resolve $COMMIT_DIR. Prefer reusing the latest existing campaign dir whose
# commit's pipeline state matches HEAD's (so doc-only commits between campaigns
# don't fragment paths). Fall through to a fresh HEAD-keyed dir otherwise.
source "$ROOT/devs/exps/eval/utils/campaign_dir.sh"
resolve_eval_commit_dir --allow-eval-commit-dir

# Resolve RUN_ID:
#   - If $EVAL_RUN_ID is set, use it verbatim (user-controlled).
#   - Else auto-resume to highest-numbered run_<N>[_<label>] under $COMMIT_DIR;
#     fall back to "run_0" when the commit dir is empty / missing.
if [ -n "${EVAL_RUN_ID:-}" ]; then
  RUN_ID="$EVAL_RUN_ID"
else
  if [ -d "$COMMIT_DIR" ]; then
    RUN_ID=$(ls -1 "$COMMIT_DIR" 2>/dev/null | grep -E '^run_[0-9]+(_|$)' | sort -t_ -k2,2n | tail -1)
  fi
  RUN_ID="${RUN_ID:-run_0}"
  echo "[run.sh] EVAL_RUN_ID unset → auto-resolved run_id=$RUN_ID (resume-to-latest at this commit)" >&2
fi
LOG_ROOT="$COMMIT_DIR/$RUN_ID/$SLUG"
export SESSION_ID="${SESSION_ID:-osworld_2-${RUN_ID}-${SLUG}}"

# Safety net: warn if starting a fresh run_id while other campaigns exist for this commit.
if [ ! -d "$COMMIT_DIR/$RUN_ID" ] && [ -d "$COMMIT_DIR" ]; then
  EXISTING=$(ls -1 "$COMMIT_DIR" 2>/dev/null | grep -vx "$RUN_ID" || true)
  if [ -n "$EXISTING" ]; then
    echo "[run.sh] WARNING: starting fresh at run_id=$RUN_ID, but other campaigns exist at this commit:" >&2
    echo "$EXISTING" | sed 's/^/  - /' >&2
    echo "  if you meant to RESUME, Ctrl-C now and re-export EVAL_RUN_ID to one of the above (or unset to auto-resume)." >&2
    sleep 5
  fi
fi

# model-family → rollout config (only these families have osworld_2 configs so far)
case "$MODEL" in
  Qwen/Qwen3-VL-*-Instruct|Qwen/Qwen3-VL-*-Thinking) CFG=scripts/configs/qwen3_vl/default/osworld_2.yaml ;;
  gpt-*)                           CFG=scripts/configs/gpt/default/osworld_2.yaml ;;
  claude-*)                        CFG=scripts/configs/claude/default/osworld_2.yaml ;;
  *) echo "unknown model: $MODEL — add a case (and a scripts/configs/<family>/default/osworld_2.yaml) in $0" >&2; exit 1 ;;
esac

CONCURRENCY="${EVAL_CONCURRENCY:-1}"

mkdir -p "$LOG_ROOT"
echo "[run.sh] $MODEL"
echo "         commit_dir=$(basename "$COMMIT_DIR")  run_id=$RUN_ID  GPUs=${CUDA_VISIBLE_DEVICES:-?}"
echo "         log_root=$LOG_ROOT"
echo "         config=$CFG"

HF_HUB_OFFLINE=1 exec uv run python scripts/rollout.py \
  --model-id "$MODEL" \
  --env-id osworld_2 --splits eval \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --concurrency "$CONCURRENCY" \
  --config-path "$CFG" \
  --log-root "$LOG_ROOT"
