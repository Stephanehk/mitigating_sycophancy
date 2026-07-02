#!/bin/bash
# Held-out eval of the UNTRAINED base Qwen3-8B on AITA and AITA-NTA-FLIP under
# the THREE remaining prompt-only baselines (base / be-critical / perspective-
# shift were already run; this fills in the rest, matching the OEQ/PAS table):
#   explicit    Cheng et al. (2026) EXPLICIT accommodation (PREPEND EXPLICIT_PREFIX)
#   sharma      Sharma et al. honesty prompt (APPEND SHARMA_HONESTY_SUFFIX)
#   dontbesyco  "Don't be sycophantic." (APPEND via --gen-suffix)
# No LLM judge -- the YTA/NTA verdict is parsed from the response
# (eval_aita_label_heldout.py, deterministic parse_yta_nta). The steer is added
# at generation time with AITA_LABEL_SUFFIX kept LAST so the verdict parses; the
# judged scenario is unchanged, so these are directly comparable to base.
#
# 18 array tasks, each on its own 2x a4000 (8B sharded via --device-map auto):
#   per condition -> AITA 2 shards of 1000 posts + AITA-NTA-FLIP 4 shards of 1000
#   pairs = 6 tasks; x3 conditions = 18. Decode temp1.0/seed0,
#   --sample-max-tokens 1536 (verbose explicit/sharma responses need room to reach
#   the trailing label instead of truncating into OTHER). Held-out ids reuse the
#   prebuilt runs/baseline_<env>/test_ids.json (same split base/be-critical used).
#
# Aggregate per (env, condition) after the array finishes, e.g.:
#   PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
#   for C in explicit sharma dontbesyco; do
#     $PY training_mitigations/eval/aggregate_aita_shards.py \
#         --run-dir training_mitigations/runs/baseline_AITA \
#         --env AITA --adapter-tag base-$C --num-shards 2
#     $PY training_mitigations/eval/aggregate_aita_shards.py \
#         --run-dir training_mitigations/runs/baseline_AITA-NTA-FLIP \
#         --env AITA-NTA-FLIP --adapter-tag base-$C --num-shards 4
#   done

#SBATCH --job-name=eval_aita_pbl
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next5,next6
#SBATCH --nodes=1
#SBATCH --array=0-17%10
#SBATCH --gres=gpu:a4000:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=training_mitigations/logs/eval_aita_pbl_%A_%a.out
#SBATCH --error=training_mitigations/logs/eval_aita_pbl_%A_%a.err

set -euo pipefail
cd /next/u/stephhk/sycophancy
export HF_HOME=/next/u/stephhk/.cache/huggingface
mkdir -p "$HF_HOME" training_mitigations/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
if [[ -n "${HF_TOKEN:-}" ]]; then export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"; fi

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
EV=training_mitigations/eval/eval_aita_label_heldout.py
AITA_RUN=training_mitigations/runs/baseline_AITA
FLIP_RUN=training_mitigations/runs/baseline_AITA-NTA-FLIP

# Job table: ENV  RUN_DIR  COND  NUM_SHARDS  SHARD_ID. Envs/conditions interleaved
# so each %10 wave mixes fast AITA and slower FLIP tasks.
JOBS=(
    "AITA          $AITA_RUN explicit   2 0"
    "AITA          $AITA_RUN explicit   2 1"
    "AITA-NTA-FLIP $FLIP_RUN explicit   4 0"
    "AITA-NTA-FLIP $FLIP_RUN explicit   4 1"
    "AITA-NTA-FLIP $FLIP_RUN explicit   4 2"
    "AITA-NTA-FLIP $FLIP_RUN explicit   4 3"
    "AITA          $AITA_RUN sharma     2 0"
    "AITA          $AITA_RUN sharma     2 1"
    "AITA-NTA-FLIP $FLIP_RUN sharma     4 0"
    "AITA-NTA-FLIP $FLIP_RUN sharma     4 1"
    "AITA-NTA-FLIP $FLIP_RUN sharma     4 2"
    "AITA-NTA-FLIP $FLIP_RUN sharma     4 3"
    "AITA          $AITA_RUN dontbesyco 2 0"
    "AITA          $AITA_RUN dontbesyco 2 1"
    "AITA-NTA-FLIP $FLIP_RUN dontbesyco 4 0"
    "AITA-NTA-FLIP $FLIP_RUN dontbesyco 4 1"
    "AITA-NTA-FLIP $FLIP_RUN dontbesyco 4 2"
    "AITA-NTA-FLIP $FLIP_RUN dontbesyco 4 3"
)
read -r ENV RUN COND N S <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"

# Map condition -> generation flags (kept as an array so the dontbesyco suffix
# with its space stays one argument).
case "$COND" in
    explicit)   FLAGS=(--explicit) ;;
    sharma)     FLAGS=(--sharma-honesty) ;;
    dontbesyco) FLAGS=(--gen-suffix "Don't be sycophantic." --gen-tag dontbesyco) ;;
    *) echo "ERROR: unknown COND=$COND" >&2; exit 1 ;;
esac

echo "[task $SLURM_ARRAY_TASK_ID] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[task $SLURM_ARRAY_TASK_ID] env=$ENV cond=$COND shard=$S/$N run=$RUN"

"$PY" "$EV" --run-dir "$RUN" --env "$ENV" --base \
    "${FLAGS[@]}" \
    --device-map auto --seed 0 \
    --sample-max-tokens 1536 \
    --num-shards "$N" --shard-id "$S"
echo "[task $SLURM_ARRAY_TASK_ID] done."
