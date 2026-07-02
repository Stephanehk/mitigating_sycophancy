#!/bin/bash
# Submit with:
#   sbatch training_mitigations/slurm_scripts/run_train_oeq_n1000_mixed_P5_nocrit_distill_mp3gpu.sbatch
# (from the repo root).
#
# PIPELINE-MODEL-PARALLEL variant of run_train_oeq_n1000_mixed_P5_nocrit_distill.sbatch.
# Identical training recipe (P5 veto, mixed sampling 4 model / 0 critical / 6
# stakeholder, --rpo-alpha 1.0 distill term, seed, decode, DPO hyperparameters,
# --max-length 3072) — the ONLY difference is GPU placement:
#
#   single-GPU recipe : policy training on cuda:0, vLLM (sim + policy sampling)
#                       on cuda:1. At --max-length 3072 the ~7GB full-vocab
#                       logits/CE spike (DPO concat + distill SFT term) OOM'd the
#                       48GB A6000 at iter1/step34 on long OEQ sequences.
#   this recipe       : policy training pipeline-SHARDED across cuda:0 + cuda:2
#                       (--policy-device-map 0,2); vLLM stays on cuda:1. Sharding
#                       halves each card's weight+activation load so the lm_head
#                       card has ample room for the logits spike — keeping
#                       --max-length 3072 (for apples-to-apples comparison with
#                       the other methods) instead of having to truncate to 2048.
#
# Pipeline MP is NOT faster (one shard's GPU computes at a time + cross-device
# copies); the 2nd training card buys memory headroom, not throughput. The DPO
# math is unchanged vs single-GPU (no gradient all-reduce, no effective-batch-
# size change); only LoRA-dropout RNG draws + bf16 copy-boundary rounding differ,
# at run-to-run-noise (reseed) magnitude. Same OUT_DIR + seed as the single-GPU
# recipe so eval/comparison is unaffected.
#
# Requires 3 free A6000s on next7 and $HF_TOKEN. next7 has flaky outbound DNS and
# Qwen3-8B is fully pre-cached under $HF_HOME, so we run HF offline.

#SBATCH --job-name=dpo_oeq_n1000_mixed_P5_nocrit_distill_mp3
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --gres=gpu:a6000:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=8-08:00:00
#SBATCH --output=training_mitigations/logs/dpo_oeq_n1000_mixed_P5_nocrit_distill_mp3_%j.out
#SBATCH --error=training_mitigations/logs/dpo_oeq_n1000_mixed_P5_nocrit_distill_mp3_%j.err

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
OUT_DIR=training_mitigations/runs/dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_nocrit_distill

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
