#!/bin/bash
# Submit with (from the repo root, with AZURE_OPENAI_* and HF_TOKEN exported):
#   sbatch training_mitigations/slurm_scripts/run_eval_pas_base_final_and_qwen.sbatch
#
# Runs eval_endorsement_heldout.py on the 500 held-out PAS prompts for:
#   (1) the FINAL adapter of the completed PAS base-method run
#       (runs/dpo_lora_pas_qwen3-8b_n1000) — only the final iteration is
#       available (per-iteration adapters were not saved for that run).
#   (2) the untrained base Qwen3-8B (--base) — the iteration-0 baseline.
# Both sample one response per prompt, judge with the Azure gpt-4o endorse
# judge, and report the BROAD endorsement rate (#1+#2)/(#0+#1+#2+#3).
#
# GPU: the 8B does not fit one 16 GB a4000 in bf16, so we request TWO and
# shard the weights with --device-map auto (no quantization; faithful bf16).
# The only 48 GB a6000 node (next7) is occupied by the training jobs.

#SBATCH --job-name=eval_pas_base_qwen
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --gres=gpu:a4000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=training_mitigations/logs/eval_pas_base_qwen_%j.out
#SBATCH --error=training_mitigations/logs/eval_pas_base_qwen_%j.err

set -euo pipefail

export HF_HOME=/next/u/stephhk/.cache/huggingface
mkdir -p "$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. Export it before sbatch." >&2; exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
if [[ -z "${AZURE_OPENAI_API_KEY:-}" || -z "${AZURE_OPENAI_ENDPOINT:-}" ]]; then
    echo "ERROR: AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT must be set for the judge." >&2; exit 1
fi

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
RUN=training_mitigations/runs/dpo_lora_pas_qwen3-8b_n1000

echo "[sbatch] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[sbatch] run_dir=$RUN"

echo "=== (1) PAS base method — FINAL adapter ==="
"$PY" training_mitigations/eval/eval_endorsement_heldout.py \
    --run-dir "$RUN" \
    --device-map auto \
    --seed 0

echo "=== (2) Untrained base Qwen3-8B (iteration-0 baseline) ==="
"$PY" training_mitigations/eval/eval_endorsement_heldout.py \
    --run-dir "$RUN" \
    --base \
    --device-map auto \
    --seed 0

echo "[sbatch] done. Summaries under $RUN/eval/"
