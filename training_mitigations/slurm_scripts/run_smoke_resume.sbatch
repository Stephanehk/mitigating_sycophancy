#!/bin/bash
#SBATCH --job-name=smoke_resume
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=training_mitigations/logs/smoke_resume_%j.out
#SBATCH --error=training_mitigations/logs/smoke_resume_%j.err
set -euo pipefail
export HF_HOME=/next/u/stephhk/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
OUT=training_mitigations/runs/smoke_p4vllm2
echo "[sbatch] resume smoke from $OUT/adapter_iter_1"
"$PY" training_mitigations/train_dpo_lora.py \
    --env OEQ --output-dir "$OUT" \
    --resume-adapter "$OUT/adapter_iter_1" --start-iteration 1 \
    --policy-on-vllm --veto-prompt baseline \
    --seed 0 --device cuda:0 --sim-device cuda:1 \
    --sim-backend vllm --veto-concurrency 8 --precompute-ref-batch-size 4 \
    --num-iterations 2 --train-n 6 --test-n 3 --n-samples 8 \
    --mixed-sampling --n-source-model 2 --n-source-critical 2 \
    --self-preference --preference-max-tokens 64 --preference-max-pairs 4 \
    --temperature 1.0 --top-p 0.95 --max-tokens 512 --sample-max-tokens 256 \
    --beta 0.1 --learning-rate 5e-5 --num-train-epochs 1 \
    --per-device-batch-size 1 --grad-accum 4 --max-length 2048 \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05
echo "RESUME SMOKE EXIT=$?"
