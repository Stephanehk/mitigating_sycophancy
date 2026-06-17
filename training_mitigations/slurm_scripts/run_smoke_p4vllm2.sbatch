#!/bin/bash
# Stage-2 smoke: tiny end-to-end run exercising --policy-on-vllm (policy sampled
# on the shared vLLM engine + per-iteration adapter hot-load) with mixed +
# self-preference + P4 veto, over 2 iterations so the adapter sync between
# iterations is exercised. NOT a real run (train-n 3).
#SBATCH --job-name=smoke_p4vllm2
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:50:00
#SBATCH --output=training_mitigations/logs/smoke_p4vllm2_%j.out
#SBATCH --error=training_mitigations/logs/smoke_p4vllm2_%j.err

set -euo pipefail
export HF_HOME=/next/u/stephhk/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
OUT=training_mitigations/runs/smoke_p4vllm2
rm -rf "$OUT"
echo "[sbatch] node=$(hostname)"

"$PY" training_mitigations/train_dpo_lora.py \
    --env OEQ --output-dir "$OUT" \
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
echo "SMOKE EXIT=$?"
echo "=== adapters produced ==="
ls -ld "$OUT"/adapter_iter_* "$OUT"/adapter_model.safetensors 2>/dev/null
