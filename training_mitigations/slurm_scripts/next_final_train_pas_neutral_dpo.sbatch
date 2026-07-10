#!/bin/bash
# Submit with:
#   sbatch training_mitigations/slurm_scripts/run_train_pas_neutral_dpo.sbatch
# (from the repo root). NEUTRALITY DPO BASELINE on PAS: instead of the
# SimulatedStakeholders veto, the (chosen, rejected) split comes from the v5
# gpt-5-mini endorse judge — a response is CHOSEN iff the judge labels it 3
# (NEUTRAL toward the user's action) and REJECTED otherwise (challenge/affirm).
# All 10 candidate responses per prompt are sampled from the policy itself
# (source=model, the deployment-relevant distribution). Same iterative DPO+LoRA
# loop as the stakeholder runs; only the veto signal differs. See
# train_dpo_lora.py --judge neutrality-v5 and core/neutrality_judge.py.
#
# TWO GPUs: the neutrality judge is the OpenRouter v5 model (no local sim), so
# the 2nd GPU is purely for fast candidate sampling — --policy-on-vllm samples
# the 10 candidates/prompt on a vLLM engine (base + the current policy LoRA,
# hot-loaded each iteration) on cuda:1, while DPO training runs on cuda:0. vLLM
# needs the whole GPU, so the two devices must differ. (To run on ONE GPU
# instead, drop --policy-on-vllm + set --sim-backend hf + --gres=gpu:1; that
# samples via HF model.generate on cuda:0 — correct but slower.)
#
# REQUIRES OUTBOUND NETWORK: the judge calls OpenRouter (gpt-5-mini) for every
# (prompt, response) every iteration. Export OPENROUTER_API_KEY before sbatch.
# The OpenAI client retries (max_retries=12); if the GPU node's outbound DNS is
# too flaky for the judge, move to a node with reliable egress. Requires
# $HF_TOKEN for the (offline-cached) policy weights. Judge spend is capped at
# --judge-max-cost (1.5x the ~$14 estimate); the run aborts if it is hit.

#SBATCH --job-name=dpo_pas_neutral_final
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8-08:00:00
#SBATCH --output=training_mitigations/logs/dpo_pas_neutral_%j.out
#SBATCH --error=training_mitigations/logs/dpo_pas_neutral_%j.err

set -euo pipefail

export HF_HOME=/next/u/stephhk/.cache/huggingface
mkdir -p "$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Policy weights are pre-cached; only HF hub access is offline (the judge's
# OpenRouter egress is unaffected by this).
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
OUT_DIR=training_mitigations/runs/dpo_lora_pas_qwen3-8b_n1000_neutral_final

echo "[sbatch] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[sbatch] HF_HOME=$HF_HOME"
echo "[sbatch] python=$PY"
echo "[sbatch] output_dir=$OUT_DIR"

"$PY" training_mitigations/train_dpo_lora.py \
    --env PAS \
    --output-dir "$OUT_DIR" \
    --use-final-data \
    --judge neutrality-v5 \
    --judge-model openai/gpt-5-mini \
    --judge-reasoning minimal \
    --judge-concurrency 16 \
    --judge-max-cost 21 \
    --policy-on-vllm \
    --seed 0 \
    --device cuda:0 \
    --sim-device cuda:1 \
    --sim-backend vllm \
    --precompute-ref-batch-size 8 \
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
