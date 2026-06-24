#!/bin/bash
# Generation-only (--no-judge) over a 1000-prompt held-out set for ONE trained
# run, using its FINAL saved adapter. Like run_gen_heldout1000.sbatch but the
# shard COUNT is parametric, so a slow card (e.g. the 8B split across a 2x a4000
# pair via device_map=auto, ~1.6 resp/min) can be split into enough shards that
# every shard finishes inside the 4h wall. Each shard samples 1000/N prompts
# (temp1.0/seed0) and uses 2 a4000.
#
# Parametrized via --export (set at submit time):
#   GEN_RUNDIR     : run dir holding the final adapter (and test_ids.json)
#   GEN_IDS        : explicit 1000-id held-out json (--test-ids-file)
#   GEN_ENV        : OEQ | PAS
#   GEN_NUM_SHARDS : shard count (must match the sbatch --array=0-(N-1)%N)
#
# Outputs per shard under <run>/eval/:
#   responses_final_final1000_temp1.0_seed0_shard{i}of{N}.csv  (id,sentence,response)
#   responses_final_final1000_temp1.0_seed0_shard{i}of{N}.jsonl
#   summary_final_final1000_temp1.0_seed0_shard{i}of{N}.json   (judged=false)

#SBATCH --job-name=gen_hd1000n
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --gres=gpu:a4000:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=training_mitigations/logs/gen_hd1000n_%A_%a.out
#SBATCH --error=training_mitigations/logs/gen_hd1000n_%A_%a.err

set -euo pipefail

export HF_HOME=/next/u/stephhk/.cache/huggingface
mkdir -p "$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set." >&2; exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
# No Azure creds needed: generation runs with --no-judge.

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
NUM_SHARDS="${GEN_NUM_SHARDS:?set GEN_NUM_SHARDS}"
si=$SLURM_ARRAY_TASK_ID

echo "[task $si] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[task $si] run=$GEN_RUNDIR env=${GEN_ENV:-auto} ids=$GEN_IDS shard=$si/$NUM_SHARDS (FINAL adapter)"

# No --adapter / --base => evaluates the FINAL saved adapter (= run dir).
"$PY" training_mitigations/eval/eval_endorsement_heldout.py \
    --run-dir "$GEN_RUNDIR" --env "${GEN_ENV}" \
    --test-ids-file "$GEN_IDS" --tag-extra final1000 \
    --device-map auto --seed 0 \
    --num-shards "$NUM_SHARDS" --shard-id "$si" --no-judge
echo "[task $si] done."
