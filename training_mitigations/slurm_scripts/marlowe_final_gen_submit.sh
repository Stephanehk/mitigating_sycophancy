#!/bin/bash
# Arm/launch FINAL-RUN held-out GENERATION on Marlowe for every method in the
# paper table, over the frozen final_training_eval_data/<env>_test split.
# Submits marlowe_final_gen.sbatch once per (env, condition) with the right
# --export + --array + a UNIQUE --job-name.
#
# THROTTLED + IDEMPOTENT (the account's `medium` QOS caps submitted jobs at 32
# ACROSS the whole account, shared with other experiments), so it is safe to run
# repeatedly / in a loop:
#   - a TRAINED condition whose run dir has no .complete is skipped (still training
#     — or, for the neutral arms, not yet relayed from `next`)
#   - a condition whose gen shard csv(s) already exist is skipped (done)
#   - a condition whose unique job-name is already in the queue is skipped (in flight)
#   - if submitting would push the ACCOUNT task count over --target, it HOLDS
#     (leaves it for a later pass) so we never trip MaxSubmitJobsPerAccount or
#     starve co-running jobs
#
# Usage (Marlowe login node):
#   bash marlowe_final_gen_submit.sh [--dry-run] [--target N] [--loop SECONDS]
#   --loop keeps re-scanning every SECONDS until every condition is done or
#   queued (picks up trained arms as their .complete lands); launch detached:
#     nohup bash marlowe_final_gen_submit.sh --loop 300 \
#       > /projects/m000254/logs/final_gen_drain.log 2>&1 &
set -uo pipefail

PROJECT=m000254
REPO=/projects/$PROJECT/sycophancy
RUNS_ROOT=/projects/$PROJECT/runs
ACCT=marlowe-$PROJECT-free
SS="$REPO/training_mitigations/slurm_scripts"
GENJOB="$SS/marlowe_final_gen.sbatch"

DRY=0; LOOP=0; TARGET=28
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --loop) LOOP="$2"; shift ;;
    --target) TARGET="$2"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac; shift
done

adapter_tag() { # kind steer -> the eval's adapter_tag (mirrors marlowe_final_gen.sbatch)
  if [[ "$1" == "trained" ]]; then echo final; return; fi
  case "$2" in
    none) echo base ;; becritical) echo base-becritical ;;
    explicit) echo base-explicit ;; sharma) echo base-sharma ;;
    dontbesyco) echo base-dontbesyco ;; perspshift) echo base-perspshiftassess ;;
    *) echo UNKNOWN ;;
  esac
}
acct_count() { squeue -A "$ACCT" -h -r -o "%i" 2>/dev/null | wc -l; }

OEQ_BASE=baseline_final_OEQ ; PAS_BASE=baseline_final_PAS
AITA_BASE=baseline_final_AITA ; FLIP_BASE=baseline_final_AITA-NTA-FLIP
# ENV  KIND  STEER  RUNDIR(rel $RUNS_ROOT)
JOBS=(
  "OEQ base none $OEQ_BASE"  "OEQ base becritical $OEQ_BASE"  "OEQ base explicit $OEQ_BASE"
  "OEQ base sharma $OEQ_BASE" "OEQ base dontbesyco $OEQ_BASE" "OEQ base perspshift $OEQ_BASE"
  "OEQ trained none dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final"
  "OEQ trained none dpo_lora_oeq_qwen3-8b_n1000_neutral_final"
  "OEQ trained none dpo_lora_oeq_qwen3-8b_n1000_neutral_mixed_final"
  "OEQ trained none sft_lora_oeq_qwen3-8b_becritical_final"
  "PAS base none $PAS_BASE"  "PAS base becritical $PAS_BASE"  "PAS base explicit $PAS_BASE"
  "PAS base sharma $PAS_BASE" "PAS base dontbesyco $PAS_BASE" "PAS base perspshift $PAS_BASE"
  "PAS trained none dpo_lora_pas_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final"
  "PAS trained none dpo_lora_pas_qwen3-8b_n1000_neutral_final"
  "PAS trained none dpo_lora_pas_qwen3-8b_n1000_neutral_mixed_final"
  "PAS trained none sft_lora_pas_qwen3-8b_becritical_final"
  "AITA base none $AITA_BASE" "AITA base becritical $AITA_BASE" "AITA base explicit $AITA_BASE"
  "AITA base sharma $AITA_BASE" "AITA base dontbesyco $AITA_BASE" "AITA base perspshift $AITA_BASE"
  "AITA trained none dpo_lora_aita_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final"
  "AITA-NTA-FLIP base none $FLIP_BASE" "AITA-NTA-FLIP base becritical $FLIP_BASE" "AITA-NTA-FLIP base explicit $FLIP_BASE"
  "AITA-NTA-FLIP base sharma $FLIP_BASE" "AITA-NTA-FLIP base dontbesyco $FLIP_BASE" "AITA-NTA-FLIP base perspshift $FLIP_BASE"
  "AITA-NTA-FLIP trained none dpo_lora_flip_qwen3-8b_n500_mixed_P6_perspshift_distill_gp_rpo0p5_final"
)

pass() {
  # returns via globals: N_DONE N_QUEUED N_SUB N_TRAIN N_HOLD
  N_DONE=0; N_QUEUED=0; N_SUB=0; N_TRAIN=0; N_HOLD=0
  mapfile -t QN < <(squeue -u "$USER" -h -O "Name:512" 2>/dev/null | sed 's/[[:space:]]*$//')
  local ac; ac=$(acct_count)
  printf "%-14s %-8s %-11s %-9s %s\n" ENV KIND STEER STATE ACTION
  for row in "${JOBS[@]}"; do
    read -r ENV KIND STEER RUN <<< "$row"
    rundir="$RUNS_ROOT/$RUN"
    case "$ENV" in
      OEQ|PAS)       ARRAY=0-1; N=2; PREFIX=responses ;;
      AITA)          ARRAY=0;   N=1; PREFIX=aita_responses ;;
      AITA-NTA-FLIP) ARRAY=0-1; N=2; PREFIX=aita_responses ;;
    esac
    TAG="$(adapter_tag "$KIND" "$STEER")_finaldata_temp1.0_seed0"
    if [[ "$KIND" == "trained" ]]; then JOBNAME="fg_${RUN}"; else JOBNAME="fg_${ENV}_${STEER}"; fi

    if [[ "$KIND" == "trained" && ! -f "$rundir/.complete" ]]; then
      N_TRAIN=$((N_TRAIN+1)); printf "%-14s %-8s %-11s %-9s %s\n" "$ENV" "$KIND" "$STEER" "training" "wait .complete"; continue
    fi
    local done=1; for ((s=0;s<N;s++)); do [[ -f "$rundir/eval/${PREFIX}_${TAG}_shard${s}of${N}.csv" ]] || done=0; done
    if (( done )); then N_DONE=$((N_DONE+1)); printf "%-14s %-8s %-11s %-9s %s\n" "$ENV" "$KIND" "$STEER" "done" "skip"; continue; fi
    if printf '%s\n' "${QN[@]}" | grep -qxF "$JOBNAME"; then
      N_QUEUED=$((N_QUEUED+1)); printf "%-14s %-8s %-11s %-9s %s\n" "$ENV" "$KIND" "$STEER" "queued" "in flight ($JOBNAME)"; continue
    fi
    if (( ac + N > TARGET )); then
      N_HOLD=$((N_HOLD+1)); printf "%-14s %-8s %-11s %-9s %s\n" "$ENV" "$KIND" "$STEER" "hold" "acct=$ac+$N>$TARGET"; continue
    fi
    if (( DRY )); then
      N_SUB=$((N_SUB+1)); printf "%-14s %-8s %-11s %-9s %s\n" "$ENV" "$KIND" "$STEER" "ready" "would submit ($JOBNAME array $ARRAY)"; continue
    fi
    if jid=$(sbatch --parsable --job-name="$JOBNAME" --array="$ARRAY" \
        --export=ALL,GEN_ENV="$ENV",GEN_RUNDIR="$rundir",GEN_KIND="$KIND",GEN_STEER="$STEER",GEN_NUM_SHARDS="$N" \
        "$GENJOB" 2>/dev/null); then
      N_SUB=$((N_SUB+1)); ac=$((ac+N)); printf "%-14s %-8s %-11s %-9s %s\n" "$ENV" "$KIND" "$STEER" "SUBMIT" "$jid ($JOBNAME array $ARRAY)"
    else
      N_HOLD=$((N_HOLD+1)); printf "%-14s %-8s %-11s %-9s %s\n" "$ENV" "$KIND" "$STEER" "hold" "sbatch rejected (cap?) — retry"
    fi
  done
  echo "[pass] done=$N_DONE queued=$N_QUEUED submitted=$N_SUB training=$N_TRAIN hold=$N_HOLD  acct_tasks=$(acct_count)/$TARGET(cap32)"
}

while :; do
  echo "==================== $(date -u +%H:%M:%S)UTC  target=$TARGET ===================="
  pass
  remaining=$((N_TRAIN + N_HOLD))
  (( LOOP > 0 )) || break
  (( remaining > 0 )) || { echo "[drain] all conditions done or queued — exiting loop."; break; }
  echo "[drain] $remaining not yet submittable (training/hold); sleeping ${LOOP}s..."
  sleep "$LOOP"
done
