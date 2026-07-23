#!/bin/bash
#
# Validate ASSISTANT_ROLE_VETO_PROMPT on the LOCAL frozen Qwen3-8B judge — the
# sim the marlowe_*_arole training arms use. Scores eval/arole_validation_set.csv
# (30 known role-abandoning responses + 15 known-good controls) through the exact
# production classifier path (SimulatedStakeholders._classify_assistant_role), on
# a vLLM server started the same way the training runs start theirs.
#
# The Gemini half of this comparison needs no GPU and runs from the login node:
#   python eval/score_arole_validation.py --backend gemini
#
# One GPU, a few minutes: the model load dominates, the 45 classify calls do not.
#SBATCH --job-name=arole_validation_qwen
#SBATCH --account=marlowe-m000254-free
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:45:00
#SBATCH --open-mode=append
#SBATCH --output=/projects/m000254/logs/arole_validation_qwen_%j.out
#SBATCH --error=/projects/m000254/logs/arole_validation_qwen_%j.err

set -euo pipefail

PROJECT=m000254
REPO=/projects/$PROJECT/mitigating_sycophancy/training_mitigations
ENV_PREFIX=/projects/$PROJECT/envs/llm126     # cu126 env (matches Marlowe's r565 driver)
export HF_HOME=/projects/$PROJECT/hf_cache

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN not set; export it before sbatch (Qwen3-8B is gated)." >&2
  exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1

cd "$REPO"
echo "[arole-val] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"

"$ENV_PREFIX/bin/python" eval/score_arole_validation.py \
    --backend vllm \
    --model "${SIM_MODEL:-Qwen/Qwen3-8B}" \
    --concurrency 8
echo "[arole-val] rc=$?"
