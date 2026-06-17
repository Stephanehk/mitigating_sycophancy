#!/bin/bash
# Sharded held-out eval of the UNTRAINED base Qwen3-8B with the prompt-only
# "Be critical." baseline appended at generation time, on the PAS held-out 500.
# Submit (from repo root, with AZURE_OPENAI_* and HF_TOKEN exported):
#   sbatch training_mitigations/slurm_scripts/run_eval_pas_becritical.sbatch
#
# 2 array tasks = base model x 2 shards, each on its own 2x a4000 (the 8B is
# sharded across the pair via --device-map auto; no quantization). Each shard
# evaluates 250 of the 500 held-out PAS prompts. Decode temp1.0/seed0 matches
# the existing bare-post base eval so the only difference is the appended
# "Be critical." instruction. Judge = paper's Azure gpt-4o endorse judge.
#
# Outputs land under runs/dpo_lora_pas_qwen3-8b_n1000/eval/ tagged
# base-becritical_temp1.0_seed0_shard{i}of2. After the array finishes:
#   PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
#   RUN=training_mitigations/runs/dpo_lora_pas_qwen3-8b_n1000
#   $PY training_mitigations/eval/aggregate_endorsement_shards.py --run-dir $RUN \
#       --adapter-tag base-becritical --num-shards 2

#SBATCH --job-name=eval_pas_becritical
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:a4000:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=training_mitigations/logs/eval_pas_becritical_%A_%a.out
#SBATCH --error=training_mitigations/logs/eval_pas_becritical_%A_%a.err

set -euo pipefail

export HF_HOME=/next/u/stephhk/.cache/huggingface
mkdir -p "$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set." >&2; exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
if [[ -z "${AZURE_OPENAI_API_KEY:-}" || -z "${AZURE_OPENAI_ENDPOINT:-}" ]]; then
    echo "ERROR: AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT must be set." >&2; exit 1
fi

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
RUN=training_mitigations/runs/dpo_lora_pas_qwen3-8b_n1000
NUM_SHARDS=2
si=$SLURM_ARRAY_TASK_ID   # shard id (0 or 1)

echo "[task $SLURM_ARRAY_TASK_ID] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[task $SLURM_ARRAY_TASK_ID] model=base+becritical shard=$si/$NUM_SHARDS"

"$PY" training_mitigations/eval/eval_endorsement_heldout.py \
    --run-dir "$RUN" --base --be-critical \
    --device-map auto --seed 0 \
    --num-shards "$NUM_SHARDS" --shard-id "$si"
echo "[task $SLURM_ARRAY_TASK_ID] done."
