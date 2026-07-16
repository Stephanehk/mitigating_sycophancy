#!/bin/bash
# RESUME of the FINAL-DATA PAS neutral DPO baseline (training job 16127527, which
# died at iteration 2 with a CUDA OOM inside TRL's DPO reference-logprob
# precompute: 16.39 GiB tried with --precompute-ref-batch-size 8, only ~14.7 GiB
# free on the a6000). Iteration 1 completed and adapter_iter_1 was saved, so we
# resume from it (--start-iteration 1) and train iterations 2..3.
#
# The ONLY change vs next_final_train_pas_neutral_dpo.sbatch is
# --precompute-ref-batch-size 8 -> 2: a pure precompute micro-batching knob that
# does NOT change the trained model. Same fix that unblocked the P5 runs.
#
# --resume-adapter makes the run reuse the EXACT train_ids.json the original
# final-data run wrote, so the continued iterations train on the same frozen
# final split (--use-final-data is kept for provenance; the split is read back
# from disk, not resampled).
#
# Requires $HF_TOKEN and $OPENROUTER_API_KEY (the v5 neutrality judge is the veto).

#SBATCH --job-name=resume_pas_neutral_final
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8-08:00:00
#SBATCH --output=training_mitigations/logs/resume_pas_neutral_final_%j.out
#SBATCH --error=training_mitigations/logs/resume_pas_neutral_final_%j.err

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
OUT_DIR=training_mitigations/runs/dpo_lora_pas_qwen3-8b_n1000_neutral_final

test -d "$OUT_DIR/adapter_iter_1" || { echo "ERROR: $OUT_DIR/adapter_iter_1 missing" >&2; exit 1; }

echo "[sbatch] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[sbatch] resume from $OUT_DIR/adapter_iter_1 (start-iteration 1), precompute-ref-batch-size 2"

"$PY" training_mitigations/train_dpo_lora.py \
    --env PAS \
    --output-dir "$OUT_DIR" \
    --use-final-data \
    --resume-adapter "$OUT_DIR/adapter_iter_1" \
    --start-iteration 1 \
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
