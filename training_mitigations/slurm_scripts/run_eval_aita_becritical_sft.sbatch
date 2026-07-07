#!/bin/bash
# Held-out eval of the AITA / AITA-NTA-FLIP "Be critical." SFT adapters
# (run_train_aita_becritical_sft) on the CANONICAL test set (each SFT run dir's
# test_ids.json == baseline_<env>/test_ids.json), so the SFT numbers sit on the
# same 1000-prompt held-out as base / be-critical / explicit / sharma / dontbesyco
# / perspective-shift. The student sees the BARE post (no steer, no verdict nudge);
# this GPU array only samples responses. 2x a4000 via --device-map auto.
# Gate afterok on run_train_aita_becritical_sft.
#
# 6 array tasks = AITA 2 shards + FLIP 4 shards. adapter==run_dir -> tag 'final'.
# After the array, run the verdict + metrics stage per env (needs OPENROUTER_API_KEY):
#   classify_aita_verdicts.py --run-dir <sft_run> --env <ENV> \
#       --tag final_temp1.0_seed0 --num-shards {2|4}

#SBATCH --job-name=eval_aita_sft
#SBATCH --account=next
#SBATCH --partition=next
#SBATCH --nodelist=next5,next6
#SBATCH --nodes=1
#SBATCH --array=0-5%6
#SBATCH --gres=gpu:a4000:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=training_mitigations/logs/eval_aita_sft_%A_%a.out
#SBATCH --error=training_mitigations/logs/eval_aita_sft_%A_%a.err

set -euo pipefail
cd /next/u/stephhk/sycophancy
export HF_HOME=/next/u/stephhk/.cache/huggingface
mkdir -p "$HF_HOME" training_mitigations/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
if [[ -n "${HF_TOKEN:-}" ]]; then export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"; fi

PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
EV=training_mitigations/eval/eval_aita_label_heldout.py
AITA_DIR=training_mitigations/runs/sft_lora_aita_qwen3-8b_n1000_becritical
FLIP_DIR=training_mitigations/runs/sft_lora_flip_qwen3-8b_n500_becritical

# Job table: ENV  SFT_DIR  NUM_SHARDS  SHARD_ID.
JOBS=(
    "AITA          $AITA_DIR 2 0"
    "AITA          $AITA_DIR 2 1"
    "AITA-NTA-FLIP $FLIP_DIR 4 0"
    "AITA-NTA-FLIP $FLIP_DIR 4 1"
    "AITA-NTA-FLIP $FLIP_DIR 4 2"
    "AITA-NTA-FLIP $FLIP_DIR 4 3"
)
read -r ENV DIR N S <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"

echo "[sft-eval $SLURM_ARRAY_TASK_ID] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "[sft-eval $SLURM_ARRAY_TASK_ID] env=$ENV shard=$S/$N dir=$DIR"

# --adapter == --run-dir -> the saved SFT adapter; tag 'final'. Evaluates the
# canonical test_ids.json in the run dir (no --test-ids-file override).
"$PY" "$EV" --run-dir "$DIR" --adapter "$DIR" --env "$ENV" \
    --device-map auto --seed 0 \
    --sample-max-tokens 1536 \
    --num-shards "$N" --shard-id "$S"
echo "[sft-eval $SLURM_ARRAY_TASK_ID] done."
