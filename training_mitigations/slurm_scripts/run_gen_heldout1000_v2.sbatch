#!/bin/bash
# Generation-only (--no-judge) over a 1000-prompt held-out set for ONE trained
# run. Generalizes run_gen_heldout1000.sbatch:
#   - evaluates an ARBITRARY adapter dir (GEN_ADAPTER, e.g. a per-iteration
#     adapter_iter_2) instead of only the final merged adapter, OR the final
#     adapter when GEN_ADAPTER is empty;
#   - shard count is parametrized (NUM_SHARDS) and the array index is the shard
#     id, so the caller picks how finely to fan out across GPUs;
#   - --gres / --array / --nodelist / --time are meant to be overridden on the
#     sbatch command line per node (a4000 pair vs single a6000).
#
# Each shard samples NUM_SHARDS-way-split block of the 1000 held-out prompts
# (temp1.0/seed0) and writes, on completion, responses_<tag>_shard<i>of<N>.csv
# under <run>/eval/. The 8B is placed with --device-map auto (no quant): a
# single a6000 holds it; a 2x a4000 pair splits it.
#
# Parametrized via --export:
#   GEN_RUNDIR : run dir holding the adapter and (default) test_ids.json
#   GEN_ENV    : OEQ | PAS
#   GEN_IDS    : explicit 1000-id held-out json (--test-ids-file)
#   GEN_ADAPTER: adapter dir to evaluate (empty => final adapter = GEN_RUNDIR)
#   NUM_SHARDS : total shards (this task does shard $SLURM_ARRAY_TASK_ID)

#SBATCH --job-name=gen_hd1000_v2
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --gres=gpu:a4000:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=training_mitigations/logs/gen2/%x_%A_%a.out
#SBATCH --error=training_mitigations/logs/gen2/%x_%A_%a.err

set -euo pipefail

cd /next/u/stephhk/sycophancy
export HF_HOME=/next/u/stephhk/.cache/huggingface
mkdir -p "$HF_HOME" training_mitigations/logs/gen2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1   # model is pre-cached; never hit the network mid-run

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set." >&2; exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

: "${GEN_RUNDIR:?set GEN_RUNDIR}"
: "${GEN_ENV:?set GEN_ENV}"
: "${GEN_IDS:?set GEN_IDS}"
: "${NUM_SHARDS:?set NUM_SHARDS}"
GEN_ADAPTER="${GEN_ADAPTER:-}"

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
si=$SLURM_ARRAY_TASK_ID

adapter_flag=()
if [[ -n "$GEN_ADAPTER" ]]; then
    adapter_flag=(--adapter "$GEN_ADAPTER")
fi

echo "[task $si] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[task $si] run=$GEN_RUNDIR env=$GEN_ENV ids=$GEN_IDS adapter='${GEN_ADAPTER:-FINAL}' shard=$si/$NUM_SHARDS"

"$PY" training_mitigations/eval/eval_endorsement_heldout.py \
    --run-dir "$GEN_RUNDIR" --env "$GEN_ENV" \
    "${adapter_flag[@]}" \
    --test-ids-file "$GEN_IDS" --tag-extra final1000 \
    --device-map auto --seed 0 \
    --num-shards "$NUM_SHARDS" --shard-id "$si" --no-judge
echo "[task $si] done."
