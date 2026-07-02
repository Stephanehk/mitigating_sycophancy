#!/bin/bash
# Bidirectional (stance-mismatch) veto probe. Generates _gp-style candidates
# (4 bare-post + 6 stakeholder-voiced) for 100 OEQ train prompts, then runs the
# proposed symmetric veto and measures how many NON-endorsing responses get
# vetoed. Qwen3-8B served tensor-parallel across 4 idle a4000s on next5.

#SBATCH --job-name=probe_bidir_veto
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodes=1
#SBATCH --nodelist=next5
#SBATCH --gres=gpu:a4000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=training_mitigations/logs/probe_bidir_veto_%j.out
#SBATCH --error=training_mitigations/logs/probe_bidir_veto_%j.err

set -euo pipefail

export HF_HOME=/next/u/stephhk/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ -n "${HF_TOKEN:-}" ]]; then export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"; fi

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
OUT_DIR=training_mitigations/sycoharm_runs/probe_bidir_veto

echo "[sbatch] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[sbatch] out_dir=$OUT_DIR"

"$PY" training_mitigations/eval/probe_bidirectional_veto.py \
    --env OEQ \
    --n-prompts 100 \
    --n-gpus 4 \
    --gpu-mem-util 0.90 \
    --out-dir "$OUT_DIR"

echo "[sbatch] done rc=$?"
