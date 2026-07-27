#!/bin/bash
#
# Gemma-4-12B-it counterpart of run_arole_validation.sbatch: score the
# assistant-role veto classifier on BOTH labelled sets (arole_validation_set.csv
# 30+15, arole_heldout_set.csv 20+40) with BOTH prompt variants (base + the
# Qwen-specific variant), through the exact production classifier path
# (SimulatedStakeholders._classify_assistant_role), on a vLLM server serving
# google/gemma-4-12B-it. Lets us compare Gemma-4-12B as the stakeholder/veto
# judge against the existing Gemini-Flash-Lite and Qwen3-8B numbers.
#
# Runs on the CUDA-13 forward-compat stack (gemma4vllm env + staged compat
# libcuda + VLLM_USE_FLASHINFER_SAMPLER=0), same as the Gemma-4 training body.
# Outputs are suffixed _gemma4_<prompt> so they never overwrite the existing
# scored_*_vllm_<prompt>.csv (Qwen) or scored_*_gemini_base.csv files.
#
# One GPU, ~15-20 min: four model loads dominate, the classify calls do not.
#SBATCH --job-name=arole_validation_gemma4
#SBATCH --account=marlowe-m000254-free
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
#SBATCH --output=/projects/m000254/logs/arole_validation_gemma4_%j.out
#SBATCH --error=/projects/m000254/logs/arole_validation_gemma4_%j.err

set -uo pipefail

PROJECT=m000254
REPO=/projects/$PROJECT/mitigating_sycophancy/training_mitigations
ENV_PREFIX=/projects/$PROJECT/envs/gemma4vllm      # cu13 env for gemma4_unified
COMPAT_LIB=/projects/$PROJECT/cuda-compat-13/lib   # CUDA-13 forward-compat libcuda
MODEL=google/gemma-4-12B-it

export HF_HOME=/projects/$PROJECT/hf_cache
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Forward-compat + flashinfer + offline (see _marlowe_dpo_body_gemma4.sh).
export LD_LIBRARY_PATH="$COMPAT_LIB:${LD_LIBRARY_PATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[[ -n "${HF_TOKEN:-}" ]] && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"  # optional; model is public

cd "$REPO"
PY="$ENV_PREFIX/bin/python"
echo "[arole-val-gemma4] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) model=$MODEL"

rc=0
for SET in arole_validation_set arole_heldout_set; do
  for PROMPT in base qwen; do
    echo "======================================================================"
    echo "[arole-val-gemma4] set=$SET prompt=$PROMPT"
    echo "======================================================================"
    "$PY" eval/score_arole_validation.py \
        --backend vllm \
        --model "$MODEL" \
        --prompt "$PROMPT" \
        --concurrency 8 \
        --input "$REPO/eval/${SET}.csv" \
        --output "$REPO/eval/scored_${SET}_gemma4_${PROMPT}.csv" || rc=$?
  done
done
echo "[arole-val-gemma4] done rc=$rc"
exit "$rc"
