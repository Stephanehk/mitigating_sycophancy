#!/bin/bash
# Targeted held-out GEN auto-submitter for the two finishing retrains ONLY
# (OEQ + AITA P6_perspshift rpo0.5 finals). Mirrors marlowe_final_gen_submit.sh's
# exact per-condition invocation, but touches nothing else (no FLIP re-queue, no
# neutral arms). Waits for each run's .complete, then submits that one gen.
# Idempotent: skips an arm already done (gen csvs present) or already in queue.
set -uo pipefail
SS=/projects/m000254/mitigating_sycophancy/training_mitigations/slurm_scripts
RUNS=/projects/m000254/runs
GENJOB="$SS/marlowe_final_gen.sbatch"
TAG=final_finaldata_temp1.0_seed0
POLL=120
MAX_ITERS=180   # ~6h safety cap

# arm: ENV|RUN|ARRAY|N|PREFIX
ARMS=(
  "OEQ|dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final|0-1|2|responses"
  "AITA|dpo_lora_aita_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final|0|1|aita_responses"
)
declare -A DONE_ARM

gen_done() { local run="$1" n="$2" pref="$3"; local ok=1
  for ((s=0;s<n;s++)); do [[ -f "$RUNS/$run/eval/${pref}_${TAG}_shard${s}of${n}.csv" ]] || ok=0; done
  (( ok )); }

echo "[watch] start $(date -u +%FT%TZ)  polling every ${POLL}s"
for ((i=0;i<MAX_ITERS;i++)); do
  all_settled=1
  for row in "${ARMS[@]}"; do
    IFS='|' read -r ENV RUN ARRAY N PREFIX <<< "$row"
    [[ -n "${DONE_ARM[$RUN]:-}" ]] && continue
    JOBNAME="fg_${RUN}"
    if gen_done "$RUN" "$N" "$PREFIX"; then
      echo "[watch] $ENV already has gen csvs — nothing to do."; DONE_ARM[$RUN]=1; continue; fi
    if squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qxF "$JOBNAME"; then
      echo "[watch] $ENV gen already in queue ($JOBNAME) — marking handled."; DONE_ARM[$RUN]=1; continue; fi
    if [[ ! -f "$RUNS/$RUN/.complete" ]]; then
      all_settled=0; continue; fi   # still training
    # .complete present, not done, not queued -> submit
    if jid=$(sbatch --parsable --job-name="$JOBNAME" --array="$ARRAY" \
        --export=ALL,GEN_ENV="$ENV",GEN_RUNDIR="$RUNS/$RUN",GEN_KIND=trained,GEN_STEER=none,GEN_NUM_SHARDS="$N" \
        "$GENJOB" 2>&1); then
      echo "[watch] SUBMITTED $ENV gen: job $jid ($JOBNAME array $ARRAY)"; DONE_ARM[$RUN]=1
    else
      echo "[watch] $ENV sbatch failed (cap?/retry next pass): $jid"; all_settled=0
    fi
  done
  (( all_settled )) && { echo "[watch] both arms submitted/done — exiting."; exit 0; }
  sleep "$POLL"
done
echo "[watch] MAX_ITERS reached — exiting (check retrains manually)."; exit 1
