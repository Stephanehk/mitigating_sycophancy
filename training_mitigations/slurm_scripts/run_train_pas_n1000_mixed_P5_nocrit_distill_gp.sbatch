#!/bin/bash
# mixed_P5_nocrit_distill_gp (PAS): EXACT clone of
# run_train_pas_n1000_mixed_P5_nocrit_distill.sbatch (single-GPU recipe: policy
# on cuda:0, vLLM sim on cuda:1; 2xA6000; P5 veto, mixed 4-model/0-critical/
# 6-stakeholder, --rpo-alpha 1.0 distill SFT term, --max-length 3072, seed 0,
# identical DPO hyperparameters) with ONE change: --stakeholder-voice, so the
# stakeholder candidates are generated in second-person advice voice instead of
# first-person role-play (dropping the role-play surface from the chosen DPO
# exemplars). Separate _gp OUT_DIR so the original distill run is untouched.
#
# Requires 2 free A6000s on next7 and $HF_TOKEN (next7 runs HF offline).

#SBATCH --job-name=dpo_pas_n1000_mixed_P5_nocrit_distill_gp
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8-08:00:00
#SBATCH --output=training_mitigations/logs/dpo_pas_n1000_mixed_P5_nocrit_distill_gp_%j.out
#SBATCH --error=training_mitigations/logs/dpo_pas_n1000_mixed_P5_nocrit_distill_gp_%j.err

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
OUT_DIR=training_mitigations/runs/dpo_lora_pas_qwen3-8b_n1000_mixed_P5_nocrit_distill_gp

echo "[sbatch] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[sbatch] output_dir=$OUT_DIR"

"$PY" training_mitigations/train_dpo_lora.py \
    --env PAS \
    --output-dir "$OUT_DIR" \
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
    --n-source-model 4 \
    --n-source-critical 0 \
    --stakeholder-voice \
    --rpo-alpha 1.0 \
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
