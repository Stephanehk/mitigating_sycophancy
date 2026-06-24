#!/bin/bash
# Submit with:
#   sbatch training_mitigations/slurm_scripts/run_train_oeq_neutral_grpo.sbatch
# (from the repo root). NEUTRALITY GRPO BASELINE on OEQ: on-policy RL whose
# reward is the v5 gpt-5-mini endorse judge — +1 if the policy's response is
# labeled 3 (NEUTRAL toward the user's action), 0 otherwise. G=8 completions per
# prompt, group-normalized advantages, KL-to-reference (--beta). The on-policy
# counterpart to run_train_oeq_neutral_dpo.sbatch (same v5 reward criterion,
# same env/seed/split). See train_grpo_lora.py and core/neutrality_judge.py.
#
# REQUIRES OUTBOUND NETWORK: the reward judge calls OpenRouter (gpt-5-mini) for
# every completion every step (G x prompts-per-step). Export OPENROUTER_API_KEY
# before sbatch; the client retries (max_retries=12). Requires $HF_TOKEN for the
# (offline-cached) policy weights. Judge spend is capped at
# --judge-max-cost (1.5x the ~$4.3 estimate); the run aborts if it is hit.
#
# Single GPU: default HF generation (slow but simple). Pass --use-vllm to
# colocate a vLLM engine on the same GPU for faster generation (more memory).

#SBATCH --job-name=grpo_oeq_neutral
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8-08:00:00
#SBATCH --output=training_mitigations/logs/grpo_oeq_neutral_%j.out
#SBATCH --error=training_mitigations/logs/grpo_oeq_neutral_%j.err

set -euo pipefail

export HF_HOME=/next/u/stephhk/.cache/huggingface
mkdir -p "$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. Export it in your shell before sbatch." >&2
    exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY is not set (the v5 neutrality reward needs it)." >&2
    exit 1
fi

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
OUT_DIR=training_mitigations/runs/grpo_lora_oeq_qwen3-8b_n1000_neutral

echo "[sbatch] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[sbatch] HF_HOME=$HF_HOME"
echo "[sbatch] python=$PY"
echo "[sbatch] output_dir=$OUT_DIR"

"$PY" training_mitigations/train_grpo_lora.py \
    --env OEQ \
    --output-dir "$OUT_DIR" \
    --judge-model openai/gpt-5-mini \
    --judge-reasoning minimal \
    --judge-concurrency 16 \
    --judge-max-cost 6.5 \
    --seed 0 \
    --device cuda:0 \
    --train-n 1000 \
    --test-n 500 \
    --num-generations 8 \
    --num-train-epochs 1 \
    --per-device-batch-size 2 \
    --grad-accum 4 \
    --learning-rate 1e-6 \
    --beta 0.04 \
    --temperature 1.0 \
    --top-p 1.0 \
    --max-completion-length 1024 \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05
