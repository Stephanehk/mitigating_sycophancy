#!/bin/bash
# Submit with:
#   sbatch training_mitigations/slurm_scripts/run_train_pas_neutral_mixed_dpo.sbatch
# (from the repo root). NEUTRAL-MIXED DPO BASELINE on PAS — the PAS counterpart
# of run_train_oeq_neutral_mixed_dpo.sbatch, for a matched OEQ/PAS pair in the
# table. Same neutrality-v5 veto (CHOSEN iff the v5 gpt-5-mini endorse judge
# labels a response 3 = NEUTRAL), with STEER-MIXED candidate sampling: half the
# N per prompt from the bare post, half from the post + "Be neutral in your
# response. Don't endorse or critique." (--neutral-mixed-sampling). On PAS the
# plain neutral DPO baseline already reaches a high neutral rate (chosen-signal
# is not starved the way it is on OEQ), so this run mainly establishes the
# matched comparison; the steer is generation-time ONLY (DPO prompt column stays
# the bare post -> the policy learns neutrality unconditionally).
# See train_dpo_lora.py --neutral-mixed-sampling, core/trainer.py NEUTRAL_SUFFIX.
#
# TWO GPUs: --policy-on-vllm samples the 10 candidates/prompt on a vLLM engine
# on cuda:1 while DPO training runs on cuda:0. The neutrality judge is the
# OpenRouter v5 model (no local sim).
#
# REQUIRES OUTBOUND NETWORK: the judge calls OpenRouter (gpt-5-mini) for every
# (prompt, response) every iteration. Export OPENROUTER_API_KEY before sbatch.
# Requires $HF_TOKEN for the (offline-cached) policy weights. Judge spend is
# capped at --judge-max-cost; the run aborts if it is hit.

#SBATCH --job-name=dpo_pas_neutral_mixed_final
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8-08:00:00
#SBATCH --output=training_mitigations/logs/dpo_pas_neutral_mixed_%j.out
#SBATCH --error=training_mitigations/logs/dpo_pas_neutral_mixed_%j.err

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
OUT_DIR=training_mitigations/runs/dpo_lora_pas_qwen3-8b_n1000_neutral_mixed_final

echo "[sbatch] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[sbatch] output_dir=$OUT_DIR"

"$PY" training_mitigations/train_dpo_lora.py \
    --env PAS \
    --output-dir "$OUT_DIR" \
    --use-final-data \
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
