#!/bin/bash
# Submit with:
#   sbatch training_mitigations/slurm_scripts/run_train_oeq_neutral_mixed_dpo.sbatch
# (from the repo root). NEUTRAL-MIXED DPO BASELINE on OEQ. Same neutrality-v5
# veto signal as run_train_oeq_neutral_dpo.sbatch (a response is CHOSEN iff the
# v5 gpt-5-mini endorse judge labels it 3 = NEUTRAL, REJECTED otherwise), but
# the N candidates per prompt are now STEER-MIXED: half sampled from the bare
# post, half from the post + "Be neutral in your response. Don't endorse or
# critique." (--neutral-mixed-sampling). This seeds judge-passing NEUTRAL
# "chosen" responses on the many OEQ prompts where the bare policy almost never
# emits one (audit: only ~1.5% of bare samples were neutral at iter 0, and
# 858/1000 prompts produced zero neutral candidates -> starved DPO signal).
# The steer is generation-time ONLY: the DPO prompt column stays the bare post,
# so the policy learns to be neutral UNCONDITIONALLY (context distillation).
# See train_dpo_lora.py --neutral-mixed-sampling, core/trainer.py NEUTRAL_SUFFIX.
#
# TWO GPUs: the neutrality judge is the OpenRouter v5 model (no local sim), so
# the 2nd GPU is purely for fast candidate sampling — --policy-on-vllm samples
# the 10 candidates/prompt on a vLLM engine (base + the current policy LoRA,
# hot-loaded each iteration) on cuda:1, while DPO training runs on cuda:0.
#
# REQUIRES OUTBOUND NETWORK: the judge calls OpenRouter (gpt-5-mini) for every
# (prompt, response) every iteration. Export OPENROUTER_API_KEY before sbatch.
# Requires $HF_TOKEN for the (offline-cached) policy weights. Judge spend is
# capped at --judge-max-cost; the run aborts if it is hit.

#SBATCH --job-name=dpo_oeq_neutral_mixed
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8-08:00:00
#SBATCH --output=training_mitigations/logs/dpo_oeq_neutral_mixed_%j.out
#SBATCH --error=training_mitigations/logs/dpo_oeq_neutral_mixed_%j.err

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
    echo "ERROR: OPENROUTER_API_KEY is not set (the v5 neutrality judge needs it)." >&2
    exit 1
fi

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
OUT_DIR=training_mitigations/runs/dpo_lora_oeq_qwen3-8b_n1000_neutral_mixed

echo "[sbatch] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[sbatch] output_dir=$OUT_DIR"

"$PY" training_mitigations/train_dpo_lora.py \
    --env OEQ \
    --output-dir "$OUT_DIR" \
    --judge neutrality-v5 \
    --judge-model openai/gpt-5-mini \
    --judge-reasoning minimal \
    --judge-concurrency 16 \
    --judge-max-cost 24 \
    --neutral-mixed-sampling \
    --policy-on-vllm \
    --seed 0 \
    --device cuda:0 \
    --sim-device cuda:1 \
    --sim-backend vllm \
    --precompute-ref-batch-size 2 \
    --num-iterations 3 \
    --train-n 1000 \
    --test-n 500 \
    --n-samples 10 \
    --temperature 1.0 \
    --top-p 0.95 \
    --sample-max-tokens 1024 \
    --beta 0.1 \
    --learning-rate 5e-5 \
    --num-train-epochs 1 \
    --per-device-batch-size 1 \
    --grad-accum 8 \
    --max-length 3072 \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05
