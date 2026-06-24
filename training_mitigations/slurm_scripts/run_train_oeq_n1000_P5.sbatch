#!/bin/bash
# Submit with:
#   sbatch training_mitigations/slurm_scripts/run_train_oeq_n1000_P5.sbatch
# (from the repo root). BASE (non-mixed) OEQ n1000 run: all 10 candidate
# responses per prompt are sampled from the policy itself (source=model, the
# deployment-relevant distribution -- no "Be critical." or stakeholder-voice
# sources). Uses the P5 TWO-STAGE veto: Stage 1 is the stakeholder-INDEPENDENT
# endorsement classifier (ENDORSE_CLASSIFY_PROMPT_P5 = the "V3_minimal" prompt
# from the eval/iterate_stage1.py sweep), run once per response; Stage 2 asks
# each cached stakeholder to VETO/ACCEPT only the responses Stage 1 flagged as
# endorsing (VETO_PROMPT_P5). See simulated_stakeholders.py. No self-preference.
# Writes to a distinct output dir so no other run is clobbered. Requires $HF_TOKEN.

#SBATCH --job-name=dpo_oeq_n1000_P5
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8-08:00:00
#SBATCH --output=training_mitigations/logs/dpo_oeq_n1000_P5_%j.out
#SBATCH --error=training_mitigations/logs/dpo_oeq_n1000_P5_%j.err

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
OUT_DIR=training_mitigations/runs/dpo_lora_oeq_qwen3-8b_n1000_P5

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
    --precompute-ref-batch-size 8 \
    --num-iterations 3 \
    --train-n 1000 \
    --test-n 500 \
    --n-samples 10 \
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
