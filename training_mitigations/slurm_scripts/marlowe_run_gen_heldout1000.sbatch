#!/bin/bash
# Marlowe `preempt` held-out GENERATION (no judge) for ONE trained adapter over
# the 1000-prompt held-out set. Parametrized via --export (set by
# run_gen_heldout1000_marlowe_submit.sh):
#   GEN_RUNDIR : abs run dir holding the FINAL adapter (adapter_config.json)
#   GEN_ENV    : PAS | OEQ | AITA | AITA-NTA-FLIP
#   GEN_IDS    : abs path to the held-out id json (passed as --test-ids-file)
#
# Array x2: each task samples 500 of the 1000 held-out prompts (temp1.0/seed0,
# the driver defaults) and writes, under <run>/eval/:
#   responses_final_final1000_temp1.0_seed0_shard{0,1}of2.{jsonl,csv}
# The FINAL saved adapter is evaluated (no --adapter/--base => adapter = run dir).
#
# Generation is NOT checkpoint-resumable, so this is a PLAIN batch job (no
# self-resubmit chain): a single 500-prompt shard of batch-1 H100 inference
# finishes well within the 4h window. If a shard dies, just re-run the
# submit helper — this job skips any shard whose responses_*.csv already exists,
# and the helper re-submits the rest. 1 GPU: an 8B in bf16 fits one H100 (80G).
#SBATCH --job-name=genhd1000
#SBATCH --account=marlowe-m000254-free
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
# 12h (batch MaxTime is 2 days). The old 4h was sized for the fast PAS path (512
# tokens); the AITA/FLIP label eval at 1536 tokens — 2 gens/row for FLIP — and
# verbose non-distill (plain-P5) policies can exceed 4h for a 500-row shard.
# Reshard a slow run via GEN_NUM_SHARDS + a matching --array to finish sooner.
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --open-mode=append
#SBATCH --output=/projects/m000254/logs/genhd1000_%A_%a.out
#SBATCH --error=/projects/m000254/logs/genhd1000_%A_%a.err

# -e so a failed driver aborts the job (FAILED) instead of falling through to the
# success echo and reporting COMPLETED with no output (silent-failure masking).
set -euo pipefail

# ===================== Marlowe site paths (mirror _marlowe_dpo_body.sh) =======
PROJECT=m000254
REPO=/projects/$PROJECT/sycophancy
ENV_PREFIX=/projects/$PROJECT/envs/llm126
export HF_HOME=/projects/$PROJECT/hf_cache
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# =============================================================================

: "${GEN_RUNDIR:?set GEN_RUNDIR via --export}"
: "${GEN_ENV:?set GEN_ENV via --export}"
: "${GEN_IDS:?set GEN_IDS via --export}"
[[ -n "${HF_TOKEN:-}" ]] || { echo "ERROR: HF_TOKEN not set; export before sbatch." >&2; exit 1; }
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

cd "$REPO"
PY="$ENV_PREFIX/bin/python"
SHARD="$SLURM_ARRAY_TASK_ID"
# Shards default to 2 (--array=0-1). Override with GEN_NUM_SHARDS + a matching
# --array (e.g. GEN_NUM_SHARDS=4 --array=0-3) to split a slow run into smaller,
# disjoint slices of the same fixed 1000-id set so it finishes within the window.
NUM_SHARDS="${GEN_NUM_SHARDS:-2}"

# Dispatch driver + per-env invocation by env. The two eval paths DIFFER:
#   PAS / OEQ -> eval_endorsement_heldout.py: an LLM judge scores later, so here
#           --no-judge means "generation only"; it writes responses_<tag>.csv and
#           its default --sample-max-tokens (512) is the established PAS/OEQ config.
#           Score afterward with the gpt-5-mini action-endorsement judge:
#           stage_heldout_inputs.py -> run_endorse_gpt5mini.py --prompt-version v5
#           (see marlowe_endorse_judge_all.sh). NOTE the judge's argparse default
#           is v1; v5 is the established prompt and must be passed explicitly.
#   AITA / AITA-NTA-FLIP -> eval_aita_label_heldout.py: SAMPLES the response and
#           parses the trailing YTA/NTA verdict itself (NO separate judge). It has
#           no --no-judge flag, needs --sample-max-tokens 1536 so verbose answers
#           reach the verdict instead of truncating into an OTHER (which would bias
#           the parsed denominator; see run_eval_aita_flip_baselines.sbatch), and
#           writes aita_scored_<tag>.csv (the shard's done-marker).
case "$GEN_ENV" in
  PAS|OEQ)
    DRIVER=training_mitigations/eval/eval_endorsement_heldout.py
    EXTRA_ARGS=(--no-judge)
    DONE_CSV="$GEN_RUNDIR/eval/responses_final_final1000_temp1.0_seed0_shard${SHARD}of${NUM_SHARDS}.csv"
    ;;
  AITA|AITA-NTA-FLIP)
    DRIVER=training_mitigations/eval/eval_aita_label_heldout.py
    EXTRA_ARGS=(--sample-max-tokens 1536)
    DONE_CSV="$GEN_RUNDIR/eval/aita_scored_final_final1000_temp1.0_seed0_shard${SHARD}of${NUM_SHARDS}.csv"
    ;;
  *) echo "ERROR: unknown GEN_ENV=$GEN_ENV" >&2; exit 1 ;;
esac

[[ -f "$GEN_RUNDIR/adapter_config.json" ]] || { echo "ERROR: no final adapter in $GEN_RUNDIR" >&2; exit 1; }
[[ -f "$GEN_IDS" ]] || { echo "ERROR: held-out ids file missing: $GEN_IDS" >&2; exit 1; }

# Idempotent: skip this shard if its done-marker csv already exists (written only
# after the shard's last prompt), so a re-run doesn't redo finished shards.
if [[ -f "$DONE_CSV" ]]; then
  echo "[gen] shard $SHARD already done ($DONE_CSV) — skipping."
  exit 0
fi

echo "[gen] env=$GEN_ENV run=$GEN_RUNDIR shard=$SHARD/$NUM_SHARDS ids=$GEN_IDS"
echo "[gen] node=$(hostname) driver=$DRIVER python=$PY"
"$PY" "$DRIVER" \
    --run-dir "$GEN_RUNDIR" --env "$GEN_ENV" \
    --test-ids-file "$GEN_IDS" --tag-extra final1000 \
    --device cuda:0 --seed 0 \
    --num-shards "$NUM_SHARDS" --shard-id "$SHARD" "${EXTRA_ARGS[@]}"
echo "[gen] shard $SHARD done."
