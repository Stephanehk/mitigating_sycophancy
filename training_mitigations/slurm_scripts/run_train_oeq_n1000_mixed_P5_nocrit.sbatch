#!/bin/bash
# Submit with:
#   sbatch training_mitigations/slurm_scripts/run_train_oeq_n1000_mixed_P5_nocrit.sbatch
# (from the repo root).
#
# ABLATION of run_train_oeq_n1000_mixed_P5.sbatch: drop the "Be critical."
# candidate source. The 2 critical samples are replaced by 2 MORE bare-policy
# samples, so the mixed-sampling split becomes 4 bare policy / 0 critical / 6
# round-robin over identified stakeholders (was 2 / 2 / 6). Everything else —
# the P5 TWO-STAGE veto (Stage 1 ENDORSE_CLASSIFY_PROMPT_P5 endorsement
# classifier, Stage 2 per-stakeholder VETO_PROMPT_P5), seed, decode, and DPO
# hyperparameters — is identical to the original mixed_P5 run.
#
# Purpose: test whether the "be critical" source is what drives mixed_P5's
# empathy drop / message-as-artifact register. Writes to a distinct output dir
# so the original mixed_P5 run is not clobbered. Requires $HF_TOKEN.
#
# --precompute-ref-batch-size 2 (not 8): the OEQ P5 runs OOM'd at the iter-2
# reference-logp precompute with batch-size 8 and had to be resumed at 2. Batch
# size only affects ref-logp throughput (semantics-preserving), so we use 2 from
# the start to avoid the crash + manual resume.
#
# Pinned to next7 (2x A6000, 48GB): the policy+training on cuda:0 needs ~17GB
# which does not fit the idle a4000 (16GB) nodes; this matches the node the
# successful P5 runs used. next7 has flaky outbound DNS and Qwen3-8B is fully
# pre-cached under $HF_HOME, so we run HF offline.

#SBATCH --job-name=dpo_oeq_n1000_mixed_P5_nocrit
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8-08:00:00
#SBATCH --output=training_mitigations/logs/dpo_oeq_n1000_mixed_P5_nocrit_%j.out
#SBATCH --error=training_mitigations/logs/dpo_oeq_n1000_mixed_P5_nocrit_%j.err

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
OUT_DIR=training_mitigations/runs/dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_nocrit

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
    --n-source-model 4 \
    --n-source-critical 0 \
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
