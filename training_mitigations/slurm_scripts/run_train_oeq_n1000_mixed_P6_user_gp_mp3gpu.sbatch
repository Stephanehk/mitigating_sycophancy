#!/bin/bash
# mixed_P6_user_gp (OEQ): EXACT clone of run_train_oeq_n1000_mixed_P5_nocrit_
# distill_gp_mp3gpu.sbatch (same 3xA6000 pipeline-MP recipe, --stakeholder-voice
# generation, mixed 4-model/0-critical/6-stakeholder, --rpo-alpha 1.0 distill SFT
# term, --max-length 3072, seed 0, identical DPO hyperparameters) with the ONE
# method change under test:
#   --veto-prompt P6  : BIDIRECTIONAL (stance-mismatch) veto — each voter states
#                       its own preferred stance and vetoes the response on
#                       disagreement (endorse-when-they'd-challenge OR challenge-
#                       when-they'd-endorse), so over-critical responses can be
#                       demoted too, not only affirming ones.
#   --user-stakeholder: add a 4th narrator/user voice to the veto panel (veto-
#                       only; NOT sampled for candidates) to lift endorsement
#                       toward target. See simulated_stakeholders.py VETO_PROMPT_P6
#                       / VETO_PROMPT_USER. Generation is byte-identical to _gp,
#                       so this isolates the veto change. Separate OUT_DIR.
#
# Requires 3 free A6000s on next7 and $HF_TOKEN (next7 runs HF offline).

#SBATCH --job-name=dpo_oeq_n1000_mixed_P6_user_gp_mp3
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --gres=gpu:a6000:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=8-08:00:00
#SBATCH --output=training_mitigations/logs/dpo_oeq_n1000_mixed_P6_user_gp_mp3_%j.out
#SBATCH --error=training_mitigations/logs/dpo_oeq_n1000_mixed_P6_user_gp_mp3_%j.err

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
OUT_DIR=training_mitigations/runs/dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_user_gp

echo "[sbatch] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[sbatch] output_dir=$OUT_DIR"

"$PY" training_mitigations/train_dpo_lora.py \
    --env OEQ \
    --output-dir "$OUT_DIR" \
    --veto-prompt P6 \
    --user-stakeholder \
    --policy-on-vllm \
    --seed 0 \
    --device cuda:0 \
    --policy-device-map 0,2 \
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
