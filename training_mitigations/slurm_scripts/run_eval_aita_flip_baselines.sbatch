#!/bin/bash
# Held-out eval of the UNTRAINED base Qwen3-8B on the two new label-parse envs,
# AITA and AITA-NTA-FLIP, in two prompt conditions each: bare and "Be critical."
# No LLM judge -- the YTA/NTA verdict is parsed from the response
# (eval_aita_label_heldout.py). Submit from repo root with HF_TOKEN exported:
#   sbatch training_mitigations/slurm_scripts/run_eval_aita_flip_baselines.sbatch
#
# 12 array tasks, each on its own 2x a4000 (the 8B is sharded across the pair
# via --device-map auto; no quantization). 10 run at once (%10) -> all 20 GPUs
# of next5+next6; the last 2 queue and start as slots free:
#   AITA          base         : 2 shards of 1000 posts  (~500 gens) tasks 0-1
#   AITA          be-critical   : 2 shards of 1000 posts  (~500 gens) tasks 2-3
#   AITA-NTA-FLIP base         : 4 shards of 1000 pairs  (~500 gens) tasks 4-7
#   AITA-NTA-FLIP be-critical   : 4 shards of 1000 pairs  (~500 gens) tasks 8-11
# Decode temp1.0/seed0, --sample-max-tokens 1536: the bare post emits its YTA/NTA
# verdict well within 512, but the verbose "Be critical." responses need the
# bigger budget to actually reach the trailing label instead of truncating into
# an OTHER (which would bias the parsed denominator). The per-shard n_dropped_other
# still records any response that never emits a verdict. Held-out ids come from the
# prebuilt runs/baseline_<env>/test_ids.json (seed-0 split, same the trained runs
# reuse; AITA order interleaved YTA/NTA so each contiguous shard is class-balanced).
#
# After the array finishes, recombine each (env, condition)'s shards (AITA: 2,
# FLIP: 4) with aggregate_aita_shards.py, e.g.:
#   PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
#   $PY training_mitigations/eval/aggregate_aita_shards.py \
#       --run-dir training_mitigations/runs/baseline_AITA \
#       --env AITA --adapter-tag base --num-shards 2
#   $PY training_mitigations/eval/aggregate_aita_shards.py \
#       --run-dir training_mitigations/runs/baseline_AITA \
#       --env AITA --adapter-tag base-becritical --num-shards 2
#   $PY training_mitigations/eval/aggregate_aita_shards.py \
#       --run-dir training_mitigations/runs/baseline_AITA-NTA-FLIP \
#       --env AITA-NTA-FLIP --adapter-tag base --num-shards 4
#   $PY training_mitigations/eval/aggregate_aita_shards.py \
#       --run-dir training_mitigations/runs/baseline_AITA-NTA-FLIP \
#       --env AITA-NTA-FLIP --adapter-tag base-becritical --num-shards 4

#SBATCH --job-name=eval_aita_flip
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next5,next6
#SBATCH --nodes=1
#SBATCH --array=0-11%10
#SBATCH --gres=gpu:a4000:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=training_mitigations/logs/eval_aita_flip_%A_%a.out
#SBATCH --error=training_mitigations/logs/eval_aita_flip_%A_%a.err

set -euo pipefail

export HF_HOME=/next/u/stephhk/.cache/huggingface
mkdir -p "$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Model is fully pre-cached; go offline to dodge flaky node DNS to the HF hub.
export HF_HUB_OFFLINE=1
if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python

# Job table: ENV  RUN_DIR  COND  NUM_SHARDS  SHARD_ID
JOBS=(
    "AITA          training_mitigations/runs/baseline_AITA          base 2 0"
    "AITA          training_mitigations/runs/baseline_AITA          base 2 1"
    "AITA          training_mitigations/runs/baseline_AITA          crit 2 0"
    "AITA          training_mitigations/runs/baseline_AITA          crit 2 1"
    "AITA-NTA-FLIP training_mitigations/runs/baseline_AITA-NTA-FLIP base 4 0"
    "AITA-NTA-FLIP training_mitigations/runs/baseline_AITA-NTA-FLIP base 4 1"
    "AITA-NTA-FLIP training_mitigations/runs/baseline_AITA-NTA-FLIP base 4 2"
    "AITA-NTA-FLIP training_mitigations/runs/baseline_AITA-NTA-FLIP base 4 3"
    "AITA-NTA-FLIP training_mitigations/runs/baseline_AITA-NTA-FLIP crit 4 0"
    "AITA-NTA-FLIP training_mitigations/runs/baseline_AITA-NTA-FLIP crit 4 1"
    "AITA-NTA-FLIP training_mitigations/runs/baseline_AITA-NTA-FLIP crit 4 2"
    "AITA-NTA-FLIP training_mitigations/runs/baseline_AITA-NTA-FLIP crit 4 3"
)
read -r ENV RUN COND NUM_SHARDS SHARD <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"

echo "[task $SLURM_ARRAY_TASK_ID] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[task $SLURM_ARRAY_TASK_ID] env=$ENV cond=$COND shard=$SHARD/$NUM_SHARDS run=$RUN"

COMMON=(--run-dir "$RUN" --env "$ENV" --base --device-map auto --seed 0 \
        --sample-max-tokens 1536 \
        --num-shards "$NUM_SHARDS" --shard-id "$SHARD")
if [[ "$COND" == "crit" ]]; then
    "$PY" training_mitigations/eval/eval_aita_label_heldout.py "${COMMON[@]}" --be-critical
else
    "$PY" training_mitigations/eval/eval_aita_label_heldout.py "${COMMON[@]}"
fi
echo "[task $SLURM_ARRAY_TASK_ID] done."
