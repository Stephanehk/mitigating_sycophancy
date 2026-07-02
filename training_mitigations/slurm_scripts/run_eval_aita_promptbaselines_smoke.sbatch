#!/bin/bash
# One-task GPU smoke of the THREE new AITA prompt-only baselines (explicit /
# sharma / dontbesyco) at tiny --n: validates the PREPEND (explicit) and APPEND
# (sharma / gen-suffix) generation paths and that the YTA/NTA label still parses
# with the steer in place. AITA only (fast); FLIP shares the same _respond path.
#SBATCH --job-name=eval_aita_pbl_smoke
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next5,next6
#SBATCH --nodes=1
#SBATCH --gres=gpu:a4000:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --output=training_mitigations/logs/eval_aita_pbl_smoke_%j.out
#SBATCH --error=training_mitigations/logs/eval_aita_pbl_smoke_%j.err

set -euo pipefail
cd /next/u/stephhk/sycophancy
export HF_HOME=/next/u/stephhk/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
if [[ -n "${HF_TOKEN:-}" ]]; then export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"; fi
PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
EV=training_mitigations/eval/eval_aita_label_heldout.py
AITA_RUN=training_mitigations/runs/baseline_AITA

echo "[smoke] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"

echo "[smoke] AITA explicit n=4"
"$PY" "$EV" --run-dir "$AITA_RUN" --env AITA --base --explicit \
    --device-map auto --seed 0 --sample-max-tokens 1536 --n 4 --tag-extra pblsmoke

echo "[smoke] AITA sharma n=4"
"$PY" "$EV" --run-dir "$AITA_RUN" --env AITA --base --sharma-honesty \
    --device-map auto --seed 0 --sample-max-tokens 1536 --n 4 --tag-extra pblsmoke

echo "[smoke] AITA dontbesyco n=4"
"$PY" "$EV" --run-dir "$AITA_RUN" --env AITA --base \
    --gen-suffix "Don't be sycophantic." --gen-tag dontbesyco \
    --device-map auto --seed 0 --sample-max-tokens 1536 --n 4 --tag-extra pblsmoke

echo "[smoke] done."
