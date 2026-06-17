#!/bin/bash
# Stage-1 parity check: vLLM base vs HF base, and vLLM base+adapter vs HF policy.
#   sbatch training_mitigations/slurm_scripts/run_parity_check.sbatch
#SBATCH --job-name=vllm_parity
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=training_mitigations/logs/vllm_parity_%j.out
#SBATCH --error=training_mitigations/logs/vllm_parity_%j.err

set -euo pipefail
export HF_HOME=/next/u/stephhk/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
echo "[sbatch] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"

"$PY" training_mitigations/legacy/build_vllm_parity_check.py \
    --adapter training_mitigations/runs/dpo_lora_oeq_qwen3-8b_n1000_mixed_selfpref/adapter_iter_1 \
    --device cuda:0 --sim-device cuda:1
