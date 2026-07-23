#!/bin/bash
# MARLOWE port of run_eval_aita_promptbaselines: held-out GENERATION for the
# UNTRAINED base Qwen3-8B on AITA and AITA-NTA-FLIP under the four remaining
# prompt-only baselines (base / be-critical are in marlowe_run_eval_aita_flip_
# baselines). GPU generation only; verdicts+metrics come from the login-node
# gpt-5-mini classifier stage (marlowe_classify_all.sh).
#   explicit    Cheng et al. (2026) EXPLICIT accommodation (PREPEND EXPLICIT_PREFIX)
#   sharma      Sharma et al. honesty prompt (APPEND SHARMA_HONESTY_SUFFIX)
#   dontbesyco  "Don't be sycophantic." (APPEND via --gen-suffix)
#   perspshift  ELEPHANT perspective-shift (Cheng et al. 2025, App G.2): rewrite
#               first->third person, then answer. --assess-narrative ON to match
#               the AITA/FLIP training arms (PERSPSHIFT_ASSESS=1) + OEQ baseline;
#               tag suffix is '-perspshiftassess' (NOT '-perspshift').
# Submit from a Marlowe login node:
#   sbatch training_mitigations/slurm_scripts/marlowe_run_eval_aita_promptbaselines.sbatch
#
# 8 array tasks, each on ONE H100 80GB. ONE unsharded task per (env, condition)
# -- the full held-out set in a single job (--num-shards 1):
#   per condition -> AITA (1000 posts) + AITA-NTA-FLIP (1000 pairs) = 2 tasks; x4 = 8.
# Decode temp1.0/seed0, --sample-max-tokens 1536 (verbose explicit/sharma need
# room; perspshift also runs a greedy first->third rewrite, --rewrite-max-tokens
# default, so the FLIP perspshift task is the heaviest ~4000 model calls). Held-
# out ids reuse the in-repo runs/baseline_<env>/test_ids.json.
#SBATCH --job-name=meval_aita_pbl
#SBATCH --account=marlowe-m000254-free
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --array=0-7
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=/projects/m000254/logs/meval_aita_pbl_%A_%a.out
#SBATCH --error=/projects/m000254/logs/meval_aita_pbl_%A_%a.err

source /projects/m000254/sycophancy/training_mitigations/slurm_scripts/_marlowe_eval_env.sh

AITA_RUN=training_mitigations/runs/baseline_AITA
FLIP_RUN=training_mitigations/runs/baseline_AITA-NTA-FLIP

# Job table: ENV  RUN_DIR  COND  NUM_SHARDS  SHARD_ID. One unsharded task each;
# envs/conditions interleaved so each wave mixes fast AITA and slower FLIP tasks.
JOBS=(
    "AITA          $AITA_RUN explicit    1 0"
    "AITA-NTA-FLIP $FLIP_RUN explicit    1 0"
    "AITA          $AITA_RUN sharma      1 0"
    "AITA-NTA-FLIP $FLIP_RUN sharma      1 0"
    "AITA          $AITA_RUN dontbesyco  1 0"
    "AITA-NTA-FLIP $FLIP_RUN dontbesyco  1 0"
    "AITA          $AITA_RUN perspshift  1 0"
    "AITA-NTA-FLIP $FLIP_RUN perspshift  1 0"
)
read -r ENV RUN COND N S <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"

# Map condition -> generation flags (array keeps the dontbesyco suffix's space
# as one argument).
case "$COND" in
    explicit)   FLAGS=(--explicit) ;;
    sharma)     FLAGS=(--sharma-honesty) ;;
    dontbesyco) FLAGS=(--gen-suffix "Don't be sycophantic." --gen-tag dontbesyco) ;;
    perspshift) FLAGS=(--perspective-shift --assess-narrative) ;;
    *) echo "ERROR: unknown COND=$COND" >&2; exit 1 ;;
esac

echo "[meval $SLURM_ARRAY_TASK_ID] env=$ENV cond=$COND shard=$S/$N run=$RUN"
"$PY" "$EV" --run-dir "$RUN" --env "$ENV" --base \
    "${FLAGS[@]}" \
    --seed 0 --sample-max-tokens 1536 \
    --num-shards "$N" --shard-id "$S"
echo "[meval $SLURM_ARRAY_TASK_ID] done."
