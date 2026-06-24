#!/bin/bash
# RESUME of the MIXED OEQ n1000 P5 run (training job 15945440). PREPARED IN
# ADVANCE in case it FAILS at iteration 2 with the same CUDA OOM that killed the
# base P5 run (15945439) in the DPO reference-logprob precompute (16.25 GiB tried
# with --precompute-ref-batch-size 8; only ~14.8 GiB free on the a6000). Iteration
# 1 completed and adapter_iter_1 was saved, so resume from it (--start-iteration 1)
# and rebuild/train iterations 2..3. The ONLY change vs the original is
# --precompute-ref-batch-size 2 (8 -> 2): a pure precompute micro-batching knob
# that does NOT change the trained model. Mirrors the mixed-sampling source mix
# (2 policy / 2 "Be critical." / 6 stakeholder). Requires $HF_TOKEN.
#
# DO NOT SUBMIT while 15945440 is still RUNNING -- only after it exits.

#SBATCH --job-name=resume_oeq_n1000_mixed_P5
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8-08:00:00
#SBATCH --output=training_mitigations/logs/resume_oeq_n1000_mixed_P5_%j.out
#SBATCH --error=training_mitigations/logs/resume_oeq_n1000_mixed_P5_%j.err

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

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
OUT_DIR=training_mitigations/runs/dpo_lora_oeq_qwen3-8b_n1000_mixed_P5

echo "[sbatch] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[sbatch] resume from $OUT_DIR/adapter_iter_1 (start-iteration 1), precompute-ref-batch-size 2"

"$PY" training_mitigations/train_dpo_lora.py \
    --env OEQ \
    --output-dir "$OUT_DIR" \
    --resume-adapter "$OUT_DIR/adapter_iter_1" \
    --start-iteration 1 \
    --veto-prompt P5 \
    --policy-on-vllm \
    --seed 0 \
    --device cuda:0 \
    --sim-device cuda:1 \
    --sim-backend vllm \
    --veto-concurrency 16 \
    --precompute-ref-batch-size 2 \
    --num-iterations 3 \
    --train-n 1000 \
    --test-n 500 \
    --n-samples 10 \
    --mixed-sampling \
    --n-source-model 2 \
    --n-source-critical 2 \
    --temperature 1.0 \
    --top-p 0.95 \
    --max-tokens 1024 \
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
