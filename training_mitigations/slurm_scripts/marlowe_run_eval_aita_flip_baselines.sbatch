#!/bin/bash
# MARLOWE port of run_eval_aita_flip_baselines: held-out GENERATION for the
# UNTRAINED base Qwen3-8B on AITA and AITA-NTA-FLIP in two prompt conditions,
# BARE and "Be critical.". GPU generation only (no verdict); the gpt-5-mini
# classifier stage runs afterward on a login node (marlowe_classify_all.sh).
# Submit from a Marlowe login node:
#   sbatch training_mitigations/slurm_scripts/marlowe_run_eval_aita_flip_baselines.sbatch
#
# 4 array tasks, each on ONE H100 80GB (the 8B fits on a single card). ONE
# unsharded task per (env, condition) -- the full held-out set in a single job
# (--num-shards 1):
#   AITA          base         : 1000 posts   task 0
#   AITA          be-critical   : 1000 posts   task 1
#   AITA-NTA-FLIP base         : 1000 pairs   task 2   (~2000 gens: original+flipped)
#   AITA-NTA-FLIP be-critical   : 1000 pairs   task 3   (~2000 gens)
# Decode temp1.0/seed0, --sample-max-tokens 1536. Held-out ids come from the
# in-repo runs/baseline_<env>/test_ids.json (seed-0 split). Each task writes
# aita_responses_<tag>_shard0of1.csv into the baseline run dir.
#
# Tags produced: base / base-becritical. Classify afterward on a login node:
#   see marlowe_classify_all.sh (or the per-tag commands in that script's header).
#SBATCH --job-name=meval_aita_flip
#SBATCH --account=marlowe-m000254-free
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --array=0-3
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/projects/m000254/logs/meval_aita_flip_%A_%a.out
#SBATCH --error=/projects/m000254/logs/meval_aita_flip_%A_%a.err

source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_eval_env.sh

AITA_RUN=training_mitigations/runs/baseline_AITA
FLIP_RUN=training_mitigations/runs/baseline_AITA-NTA-FLIP

# Job table: ENV  RUN_DIR  COND  NUM_SHARDS  SHARD_ID.  One unsharded task each.
JOBS=(
    "AITA          $AITA_RUN base        1 0"
    "AITA          $AITA_RUN becritical  1 0"
    "AITA-NTA-FLIP $FLIP_RUN base        1 0"
    "AITA-NTA-FLIP $FLIP_RUN becritical  1 0"
)
read -r ENV RUN COND N S <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"

case "$COND" in
    base)       FLAGS=() ;;
    becritical) FLAGS=(--be-critical) ;;
    *) echo "ERROR: unknown COND=$COND" >&2; exit 1 ;;
esac

echo "[meval $SLURM_ARRAY_TASK_ID] env=$ENV cond=$COND shard=$S/$N run=$RUN"
"$PY" "$EV" --run-dir "$RUN" --env "$ENV" --base \
    "${FLAGS[@]}" \
    --seed 0 --sample-max-tokens 1536 \
    --num-shards "$N" --shard-id "$S"
echo "[meval $SLURM_ARRAY_TASK_ID] done."
