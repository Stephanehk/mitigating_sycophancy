#!/bin/bash
# Held-out GENERATION for the TRAINED P6 AITA-NTA-FLIP adapter
# (dpo_lora_flip_qwen3-8b_n500_mixed_P6_perspshift_distill_gp_rpo0p5_final) on the
# frozen final TEST split (1000 pairs -> 2000 free-form generations). GPU array,
# GENERATION ONLY (no verdict): the YTA/NTA/UNCLEAR verdict + both-NTA sycophancy
# metrics come afterward from the network-bound classifier stage
# (classify_aita_verdicts.py, gpt-5-mini). Submit from repo root with HF_TOKEN set:
#   sbatch training_mitigations/slurm_scripts/next_final_gen_flip_P6_perspshift.sbatch
#
# 5 array tasks, ALL on next6 (10x a4000, idle). Each task takes 2 a4000 and shards
# the 8B across the pair via --device-map auto (bf16 weights ~15GB don't fit one
# 16GB a4000), so 5 tasks x 2 GPUs = all 10 next6 GPUs run at once:
#   shard i of 5 -> 200 pairs (~400 generations) each.
# Decode temp1.0 / seed0 / --sample-max-tokens 1536. Each shard writes
# aita_responses_final_finaldata_temp1.0_seed0_shard{i}of5.csv under <run>/eval/.
#
# After the array finishes, run the classifier stage (needs OPENROUTER_API_KEY;
# CPU/login, no GPU) to produce verdicts + both-NTA sycophancy metrics:
#   PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
#   $PY training_mitigations/eval/classify_aita_verdicts.py \
#       --run-dir training_mitigations/runs/dpo_lora_flip_qwen3-8b_n500_mixed_P6_perspshift_distill_gp_rpo0p5_final \
#       --env AITA-NTA-FLIP --tag final_finaldata_temp1.0_seed0 --num-shards 5

#SBATCH --job-name=gen_flip_p6
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next6
#SBATCH --nodes=1
#SBATCH --array=0-4%5
#SBATCH --gres=gpu:a4000:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=training_mitigations/logs/gen_flip_p6_%A_%a.out
#SBATCH --error=training_mitigations/logs/gen_flip_p6_%A_%a.err

set -euo pipefail

export HF_HOME=/next/u/stephhk/.cache/huggingface
mkdir -p "$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Model is fully pre-cached; go offline to dodge flaky node DNS to the HF hub.
export HF_HUB_OFFLINE=1
if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
RUN=training_mitigations/runs/dpo_lora_flip_qwen3-8b_n500_mixed_P6_perspshift_distill_gp_rpo0p5_final
NUM_SHARDS=5
SHARD=$SLURM_ARRAY_TASK_ID

echo "[task $SLURM_ARRAY_TASK_ID] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[task $SLURM_ARRAY_TASK_ID] shard=$SHARD/$NUM_SHARDS run=$RUN"

"$PY" training_mitigations/eval/eval_aita_label_heldout.py \
    --run-dir "$RUN" \
    --env AITA-NTA-FLIP --use-final-data --final-split test --tag-extra finaldata \
    --sample-max-tokens 1536 --num-shards "$NUM_SHARDS" --shard-id "$SHARD" \
    --temperature 1.0 --seed 0 --device-map auto
echo "[task $SLURM_ARRAY_TASK_ID] done."
