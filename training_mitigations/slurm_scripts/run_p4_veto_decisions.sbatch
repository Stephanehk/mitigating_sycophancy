#!/bin/bash
# Diagnostic: re-run the P4 stakeholder veto WITH justifications on real
# OEQ mixed_P4 training candidates (frozen Qwen3-8B, greedy) to show why it
# inverts. Single 2x a4000 pair on an idle node; short.
#SBATCH --job-name=p4_veto_diag
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --gres=gpu:a4000:2
#SBATCH --nodelist=next5
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=training_mitigations/logs/p4_veto_diag_%j.out
#SBATCH --error=training_mitigations/logs/p4_veto_diag_%j.err

set -euo pipefail
cd /next/u/stephhk/sycophancy
export HF_HOME=/next/u/stephhk/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:?set HF_TOKEN}"

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
"$PY" training_mitigations/eval/sample_p4_veto_decisions.py \
    --veto-prompt "${VETO_PROMPT:-P5}" \
    --n-prompts 5 --max-per-prompt 5 --device-map auto
echo "[done]"
