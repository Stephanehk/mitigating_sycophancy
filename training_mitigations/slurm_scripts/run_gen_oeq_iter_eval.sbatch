#!/bin/bash
# Generation-only over a 500-prompt OEQ held-out subset for the per-iteration
# adapters (adapter_iter_1 + adapter_iter_2) of the mixed_P5_nocrit_distill run,
# to check how the v5 action-endorsement rate evolves across DPO iterations.
#
# 10-task array: tasks 0-4 = adapter_iter_1 shards 0-4; tasks 5-9 = adapter_iter_2
# shards 0-4 (5-way split of the same 500-id subset -> 100 prompts/shard).
# Single-node, gpu:a4000:2 per task (8B with --device-map auto). No --nodelist:
# SLURM packs all 10 tasks across the two a4000 nodes (next5/6).
#
# Output: <run>/eval/responses_adapter_iter_<n>_oeq500_temp1.0_seed0_shard<i>of5.csv
#
# Requires HF_TOKEN (submit with --export=ALL).

#SBATCH --job-name=gen_oeq_iter_eval
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --gres=gpu:a4000:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=training_mitigations/logs/gen2/%x_%A_%a.out
#SBATCH --error=training_mitigations/logs/gen2/%x_%A_%a.err

set -euo pipefail

cd /next/u/stephhk/sycophancy
export HF_HOME=/next/u/stephhk/.cache/huggingface
mkdir -p "$HF_HOME" training_mitigations/logs/gen2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
if [[ -z "${HF_TOKEN:-}" ]]; then echo "ERROR: HF_TOKEN not set" >&2; exit 1; fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
RUNDIR=training_mitigations/runs/dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_nocrit_distill
IDS=training_mitigations/sycoharm_runs/heldout_oeq_first500_ids.json
NUM_SHARDS=5

ITERS=(1 1 1 1 1 2 2 2 2 2)
SHARDS=(0 1 2 3 4 0 1 2 3 4)
i=$SLURM_ARRAY_TASK_ID
ITER=${ITERS[$i]}; SHARD=${SHARDS[$i]}
ADAPTER=$RUNDIR/adapter_iter_${ITER}

echo "[task $i] node=$(hostname) iter=$ITER shard=$SHARD/$NUM_SHARDS adapter=$ADAPTER"
"$PY" training_mitigations/eval/eval_endorsement_heldout.py \
    --run-dir "$RUNDIR" --env OEQ \
    --adapter "$ADAPTER" \
    --test-ids-file "$IDS" --tag-extra oeq500 \
    --device-map auto --seed 0 \
    --num-shards "$NUM_SHARDS" --shard-id "$SHARD" --no-judge
echo "[task $i] done."
