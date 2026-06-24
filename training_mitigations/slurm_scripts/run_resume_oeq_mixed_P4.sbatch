#!/bin/bash
# RESUME the OEQ n1000 MIXED (P4 veto) run from its saved adapter_iter_1
# (iteration 1 already done; iteration 2 OOM'd in the ref-logp precompute).
# Continues iterations 2-3 in the SAME output dir, reusing the original train
# split via train_ids.json. Identical args to run_train_oeq_n1000_mixed_P4.sbatch
# plus --resume-adapter / --start-iteration. Picks up the new between-iteration
# torch.cuda.empty_cache() fragmentation fix in core/trainer.py. Requires $HF_TOKEN.

#SBATCH --job-name=resume_oeq_mixed_P4
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8-08:00:00
#SBATCH --output=training_mitigations/logs/resume_oeq_mixed_P4_%j.out
#SBATCH --error=training_mitigations/logs/resume_oeq_mixed_P4_%j.err

set -euo pipefail

export HF_HOME=/next/u/stephhk/.cache/huggingface
mkdir -p "$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# next7 has flaky outbound DNS and Qwen3-8B is fully pre-cached under $HF_HOME.
export HF_HUB_OFFLINE=1

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. Export it in your shell before sbatch." >&2
    exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
OUT_DIR=training_mitigations/runs/dpo_lora_oeq_qwen3-8b_n1000_mixed_P4

echo "[sbatch] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[sbatch] output_dir=$OUT_DIR (RESUME from adapter_iter_1, start-iteration 1)"

"$PY" training_mitigations/train_dpo_lora.py \
    --env OEQ \
    --output-dir "$OUT_DIR" \
    --resume-adapter "$OUT_DIR/adapter_iter_1" \
    --start-iteration 1 \
    --veto-prompt P4 \
    --policy-on-vllm \
    --seed 0 \
    --device cuda:0 \
    --sim-device cuda:1 \
    --sim-backend vllm \
    --veto-concurrency 16 \
    --precompute-ref-batch-size 8 \
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
