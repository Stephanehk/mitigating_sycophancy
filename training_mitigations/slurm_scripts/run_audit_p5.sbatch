#!/bin/bash
# Two-stage P5 audit: Part A (Stage-1 8B classifier vs v5 gpt-5-mini labels) +
# Part B (Stage-2 stakeholder vetoes on inversion prompts). Frozen Qwen3-8B,
# greedy, 2x a4000 on an idle node.
#SBATCH --job-name=audit_p5
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --gres=gpu:a4000:2
#SBATCH --nodelist=next5
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=training_mitigations/logs/audit_p5_%j.out
#SBATCH --error=training_mitigations/logs/audit_p5_%j.err

set -euo pipefail
cd /next/u/stephhk/sycophancy
export HF_HOME=/next/u/stephhk/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:?set HF_TOKEN}"

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
"$PY" training_mitigations/eval/audit_p5_two_stage.py \
    --n-per-label 40 --n-prompts 5 --max-per-prompt 5 --device-map auto
echo "[done]"
