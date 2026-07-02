#!/bin/bash
# One-task smoke test of eval_aita_label_heldout.py on real GPUs before the full
# 20-GPU array: tiny --n over AITA (base + be-critical) and AITA-NTA-FLIP, to
# confirm device-map load, generation, label parsing, and both metrics work.
#SBATCH --job-name=eval_aita_smoke
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next5,next6
#SBATCH --nodes=1
#SBATCH --gres=gpu:a4000:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=training_mitigations/logs/eval_aita_smoke_%j.out
#SBATCH --error=training_mitigations/logs/eval_aita_smoke_%j.err

set -euo pipefail
export HF_HOME=/next/u/stephhk/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
if [[ -n "${HF_TOKEN:-}" ]]; then export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"; fi
PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
EV=training_mitigations/eval/eval_aita_label_heldout.py

echo "[smoke] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"

echo "[smoke] AITA base n=6"
"$PY" "$EV" --run-dir training_mitigations/runs/baseline_AITA --env AITA --base \
    --device-map auto --seed 0 --n 6 --tag-extra smoke

echo "[smoke] AITA be-critical n=6"
"$PY" "$EV" --run-dir training_mitigations/runs/baseline_AITA --env AITA --base --be-critical \
    --device-map auto --seed 0 --n 6 --tag-extra smoke

echo "[smoke] AITA-NTA-FLIP base n=6 (two gens/pair)"
"$PY" "$EV" --run-dir training_mitigations/runs/baseline_AITA-NTA-FLIP --env AITA-NTA-FLIP --base \
    --device-map auto --seed 0 --n 6 --tag-extra smoke

echo "[smoke] done."
