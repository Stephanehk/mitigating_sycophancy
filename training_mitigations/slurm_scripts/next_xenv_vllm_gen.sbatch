#!/bin/bash
# Rebuttal cross-dataset GENERATION on batched offline vLLM (2026-11-06): one
# array task per (train, eval) condition, index -> eval/xenv_vllm_gen.py
# conditions() (0..11). Replaces the 138-shard HF path for these 12 conditions
# only; see the parity notes in xenv_vllm_gen.py. Generation only, no API.
#
# Submit (login node):
#   sbatch --array=0-11 training_mitigations/slurm_scripts/next_xenv_vllm_gen.sbatch
# Paths: XENV_RUNS (containers, default runs/), XENV_ADAPTER_ROOT (the four
# source adapters, default runs_rebuttal/). A finished condition exits at once.
#SBATCH --job-name=xvg
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --open-mode=append
#SBATCH --output=/next/u/stephhk/sycophancy/training_mitigations/logs/xvg_%A_%a.out
#SBATCH --error=/next/u/stephhk/sycophancy/training_mitigations/logs/xvg_%A_%a.err

source /next/u/stephhk/sycophancy/training_mitigations/slurm_scripts/_next_eval_env.sh
set -e
# _next_eval_env.sh pins HF_HOME but not HF_HUB_CACHE; an inherited login-shell
# HF_HUB_CACHE would silently redirect vLLM's hub lookup. Pin it to the
# canonical $HF_HOME/hub layout (Qwen3-8B snapshot b968826 is pre-cached there).
export HF_HUB_CACHE="$HF_HOME/hub"

I=${SLURM_ARRAY_TASK_ID:?run as an array job}
echo "[xvg] job=${SLURM_JOB_ID} task=$I node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd,)"
"$PY" training_mitigations/eval/xenv_vllm_gen.py --index "$I" \
  --runs "${XENV_RUNS:-$REPO/training_mitigations/runs}" \
  --adapter-root "${XENV_ADAPTER_ROOT:-$REPO/training_mitigations/runs_rebuttal}" \
  ${XENV_LIMIT:+--limit "$XENV_LIMIT"}
echo "[xvg] task $I done"
