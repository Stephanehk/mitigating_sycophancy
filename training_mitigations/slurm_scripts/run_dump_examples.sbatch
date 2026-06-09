#!/bin/bash
# Submit from the repo root with:
#   sbatch training_mitigations/slurm_scripts/run_dump_examples.sbatch
# Regenerates representative stakeholder-simulation examples (identified
# stakeholders, concerns, veto reasoning, vetoed responses) for the OEQ
# and PAS runs by replaying the frozen-base greedy pipeline. One GPU,
# inference only, short job.
#SBATCH --job-name=dump_examples
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next7
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=training_mitigations/logs/dump_examples_%j.out
#SBATCH --error=training_mitigations/logs/dump_examples_%j.err

set -euo pipefail
export HF_HOME=/next/u/stephhk/.cache/huggingface
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. Export it before sbatch." >&2
    exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
echo "[sbatch] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
"$PY" training_mitigations/dump_stakeholder_examples.py
