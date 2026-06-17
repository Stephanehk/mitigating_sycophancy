#!/bin/bash
# Smoke for the per-stage (breadth-first) BATCHED build loop. Exercises:
#   - multiple chunks per iteration (train-n 24, --chunk-size 8 => 3 chunks)
#   - concurrent fan-out within each stage (--stage-concurrency 64)
#   - all stages: prime stakeholders / sample / veto / self-preference
#   - --policy-on-vllm adapter hot-load across 2 iterations (sync still works
#     with the rewritten _build_iteration)
# NOT a real run. Validates correctness + that throughput >> the ~90s/prompt
# per-prompt-loop baseline.
#SBATCH --job-name=smoke_batched
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=training_mitigations/logs/smoke_batched_%j.out
#SBATCH --error=training_mitigations/logs/smoke_batched_%j.err

set -euo pipefail
export HF_HOME=/next/u/stephhk/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
OUT=training_mitigations/runs/smoke_batched
rm -rf "$OUT"
echo "[sbatch] node=$(hostname) start=$(date +%s)"

"$PY" training_mitigations/train_dpo_lora.py \
    --env OEQ --output-dir "$OUT" \
    --policy-on-vllm --veto-prompt P4 \
    --seed 0 --device cuda:0 --sim-device cuda:1 \
    --sim-backend vllm --precompute-ref-batch-size 4 \
    --chunk-size 8 --stage-concurrency 64 \
    --num-iterations 2 --train-n 24 --test-n 8 --n-samples 4 \
    --mixed-sampling --n-source-model 1 --n-source-critical 1 \
    --self-preference --preference-max-tokens 64 --preference-max-pairs 4 \
    --temperature 1.0 --top-p 0.95 --max-tokens 512 --sample-max-tokens 256 \
    --beta 0.1 --learning-rate 5e-5 --num-train-epochs 1 \
    --per-device-batch-size 1 --grad-accum 4 --max-length 2048 \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05
echo "SMOKE EXIT=$?  end=$(date +%s)"
echo "=== adapters produced ==="
ls -ld "$OUT"/adapter_iter_* "$OUT"/adapter_model.safetensors 2>/dev/null
echo "=== responses jsonl line counts (expect 24/iter) ==="
wc -l "$OUT"/responses_iter_*.jsonl 2>/dev/null
echo "=== iter_stats ==="
cat "$OUT"/iter_stats.json 2>/dev/null
