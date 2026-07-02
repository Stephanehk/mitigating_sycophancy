#!/bin/bash
# Submit with:
#   sbatch training_mitigations/slurm_scripts/run_train_oeq_n1000_mixed_P5_perspective_shift.sbatch
# (from the repo root). EXACT clone of run_train_oeq_n1000_mixed_P5.sbatch (same
# 2xA6000 next7 recipe, P5 two-stage veto, 2 bare / 2 steer / 6 stakeholder mix,
# identical DPO hyperparameters, seed 0) with the ONE method change under test:
#
#   --perspective-shift-sampling : the 2 "steer" candidates per prompt are drawn
#       NOT from post + "\n\nBe critical." but from the ELEPHANT (Cheng et al.
#       2025, App G.2) THIRD-PERSON REWRITE of the post. Each prompt is rewritten
#       first->third-person once (greedy, via the policy, --rewrite-max-tokens
#       1536), then the --n-source-critical (=2) samples are generated on that
#       rewrite. The DPO prompt column stays the bare first-person post.
#   --perspshift-assess : append the paper's OEQ addendum "Assess this
#       narrative." after the rewrite. This matches the OEQ perspshift eval
#       baseline (run_gen_perspshift_final1000.sbatch adds --assess-narrative for
#       OEQ, since long narratives keep re-addressing "you" after the rewrite);
#       PAS does not use it. See core/trainer.py / core/perspective_shift.py.
#
# --precompute-ref-batch-size 2 (not 8): matches mixed_P5 (OEQ P5 OOM'd at 8).
# Pinned to next7 (2x A6000, 48GB); HF offline (Qwen3-8B pre-cached).

#SBATCH --job-name=dpo_oeq_n1000_mixed_P5_perspshift
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8-08:00:00
#SBATCH --output=training_mitigations/logs/dpo_oeq_n1000_mixed_P5_perspshift_%j.out
#SBATCH --error=training_mitigations/logs/dpo_oeq_n1000_mixed_P5_perspshift_%j.err

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
OUT_DIR=training_mitigations/runs/dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_perspective_shift

echo "[sbatch] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[sbatch] HF_HOME=$HF_HOME"
echo "[sbatch] python=$PY"
echo "[sbatch] output_dir=$OUT_DIR"

"$PY" training_mitigations/train_dpo_lora.py \
    --env OEQ \
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
    --n-source-model 2 \
    --n-source-critical 2 \
    --perspective-shift-sampling \
    --perspshift-assess \
    --rewrite-max-tokens 1536 \
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
