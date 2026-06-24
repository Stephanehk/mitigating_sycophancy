#!/bin/bash
# P5 two-stage veto YIELD audit (Qwen3-8B). 10 shards over the prompts, each on
# a 2x a4000 pair, spread across next5/next6 (=10 pairs, all run at once). Each
# shard runs the full two-stage veto (concerns + Stage1 + Stage2) on its prompts
# and writes per-candidate p4/p5 veto bits. Batch job => survives disconnect.
#SBATCH --job-name=p5_yield
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --array=0-9
#SBATCH --gres=gpu:a4000:2
#SBATCH --nodelist=next5,next6
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=training_mitigations/logs/p5_yield_%A_%a.out
#SBATCH --error=training_mitigations/logs/p5_yield_%A_%a.err

set -euo pipefail
cd /next/u/stephhk/sycophancy
export HF_HOME=/next/u/stephhk/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:?set HF_TOKEN}"

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
"$PY" training_mitigations/eval/p5_veto_yield.py \
    --iter-file responses_iter_3.jsonl \
    --n-prompts 120 --num-shards 10 --shard-id "$SLURM_ARRAY_TASK_ID" \
    --device-map auto
echo "[shard $SLURM_ARRAY_TASK_ID] done."
