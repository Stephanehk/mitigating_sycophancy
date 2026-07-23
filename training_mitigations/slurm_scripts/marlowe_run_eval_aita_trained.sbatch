#!/bin/bash
# MARLOWE held-out GENERATION for the two remaining TRAINED mitigations on AITA
# and AITA-NTA-FLIP: the DPO adapters
#   frozenstake : mixed_P5_nocrit_distill_gp_frozenstake
#   P6          : mixed_P6_perspshift_distill_gp
# The student sees the BARE post (no steer, no verdict nudge); GPU generation
# only. Verdicts+metrics come from the login-node gpt-5-mini classifier stage
# (marlowe_classify_all.sh). Submit from a Marlowe login node, AFTER the adapters
# have (re)trained on the new bare-post code:
#   sbatch training_mitigations/slurm_scripts/marlowe_run_eval_aita_trained.sbatch
#
# 4 array tasks, each on ONE H100 80GB (--adapter == --run-dir -> tag 'final').
# One unsharded task per (mitigation, env) -- the full held-out in a single job:
#   frozenstake AITA (1000 posts) + FLIP (1000 pairs)   tasks 0-1
#   P6          AITA (1000 posts) + FLIP (1000 pairs)   tasks 2-3
# Decode temp1.0/seed0, --sample-max-tokens 1536.
#
# COMPARABILITY: each adapter is evaluated on its OWN run-dir test_ids.json. For
# the FNR/FPR numbers to sit on the SAME 1000-prompt held-out as the prompt
# baselines, the (re)training MUST pin the baseline split so the adapter's
# test_ids == runs/baseline_<env>/test_ids.json (today they differ). If that is
# not yet guaranteed, add --test-ids-file training_mitigations/runs/baseline_<ENV>/
# test_ids.json below — but ONLY once the adapter's TRAIN ids are confirmed
# disjoint from the baseline test ids (else leakage).
#SBATCH --job-name=meval_aita_trained
#SBATCH --account=marlowe-m000254-free
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --array=0-3
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/projects/m000254/logs/meval_aita_trained_%A_%a.out
#SBATCH --error=/projects/m000254/logs/meval_aita_trained_%A_%a.err

source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_eval_env.sh

RUNS_ROOT=/projects/m000254/runs
FROZEN_AITA=$RUNS_ROOT/dpo_lora_aita_qwen3-8b_n1000_mixed_P5_nocrit_distill_gp_frozenstake
FROZEN_FLIP=$RUNS_ROOT/dpo_lora_flip_qwen3-8b_n500_mixed_P5_nocrit_distill_gp_frozenstake
P6_AITA=$RUNS_ROOT/dpo_lora_aita_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp
P6_FLIP=$RUNS_ROOT/dpo_lora_flip_qwen3-8b_n500_mixed_P6_perspshift_distill_gp

# Job table: ENV  ADAPTER_DIR  NUM_SHARDS  SHARD_ID.  One unsharded task each.
JOBS=(
    "AITA          $FROZEN_AITA 1 0"
    "AITA-NTA-FLIP $FROZEN_FLIP 1 0"
    "AITA          $P6_AITA     1 0"
    "AITA-NTA-FLIP $P6_FLIP     1 0"
)
read -r ENV DIR N S <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"

echo "[meval-trained $SLURM_ARRAY_TASK_ID] env=$ENV shard=$S/$N adapter=$DIR"
"$PY" "$EV" --run-dir "$DIR" --adapter "$DIR" --env "$ENV" \
    --seed 0 --sample-max-tokens 1536 \
    --num-shards "$N" --shard-id "$S"
echo "[meval-trained $SLURM_ARRAY_TASK_ID] done."
