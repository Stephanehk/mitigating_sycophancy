#!/bin/bash
# Follow-on GEN for OEQ P5_nocrit frozenstake (job 380019), on the PREVIOUS
# tuning-split held-out 1000 set (heldout_oeq_final1000_ids.json, final1000 tag),
# NOT final_data. Verified clean: train ∩ held1000 = 0.
#
# Ordering: the two OEQ/AITA P6_perspshift gens are HIGHER PRIORITY. This waits
# until BOTH of those gens have STARTED (appear in the queue, or their csvs exist)
# before submitting the P5_nocrit gen, so perspshift gets its node request in first.
# Idempotent: skips if the P5_nocrit gen csvs already exist or its job is queued.
set -uo pipefail
SS=/projects/m000254/mitigating_sycophancy/training_mitigations/slurm_scripts
RUNS=/projects/m000254/runs
GENJOB="$SS/marlowe_run_gen_heldout1000.sbatch"
OEQ_IDS=/projects/m000254/mitigating_sycophancy/training_mitigations/sycoharm_runs/heldout_oeq_final1000_ids.json
P5DIR="$RUNS/dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_nocrit_distill_gp_frozenstake"
P5JOB=genhd1000_oeq_P5nocrit_frozenstake
POLL=120
MAX_ITERS=240   # ~8h safety cap (perspshift training + gens can take a while)

# the two higher-priority perspshift gen job-names + a "done csv" for each
PSHIFT_OEQ_JOB=fg_dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final
PSHIFT_AITA_JOB=fg_dpo_lora_aita_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final
PSHIFT_OEQ_CSV="$RUNS/dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final/eval/responses_final_finaldata_temp1.0_seed0_shard0of2.csv"
PSHIFT_AITA_CSV="$RUNS/dpo_lora_aita_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final/eval/aita_responses_final_finaldata_temp1.0_seed0_shard0of1.csv"

inq(){ squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qxF "$1"; }
started(){ # $1 job-name  $2 done-csv : started == in queue OR its csv exists
  inq "$1" || [[ -f "$2" ]]; }

echo "[p5watch] start $(date -u +%FT%TZ)  waiting for BOTH perspshift gens to start; poll ${POLL}s"
for ((i=0;i<MAX_ITERS;i++)); do
  # already handled?
  if [[ -f "$P5DIR/eval/responses_final_final1000_temp1.0_seed0_shard0of2.csv" \
     && -f "$P5DIR/eval/responses_final_final1000_temp1.0_seed0_shard1of2.csv" ]]; then
    echo "[p5watch] P5_nocrit gen already complete — nothing to do."; exit 0; fi
  if inq "$P5JOB"; then echo "[p5watch] P5_nocrit gen already in queue ($P5JOB) — done."; exit 0; fi

  if started "$PSHIFT_OEQ_JOB" "$PSHIFT_OEQ_CSV" && started "$PSHIFT_AITA_JOB" "$PSHIFT_AITA_CSV"; then
    echo "[p5watch] both perspshift gens have started — submitting P5_nocrit gen."
    if jid=$(sbatch --parsable --job-name="$P5JOB" \
        --export=ALL,GEN_RUNDIR="$P5DIR",GEN_ENV=OEQ,GEN_IDS="$OEQ_IDS" \
        "$GENJOB" 2>&1); then
      echo "[p5watch] SUBMITTED P5_nocrit gen: job $jid ($P5JOB, array 0-1)"; exit 0
    else
      echo "[p5watch] sbatch failed (cap?/retry next pass): $jid"
    fi
  else
    echo "[p5watch] waiting: OEQ_pshift started=$(started "$PSHIFT_OEQ_JOB" "$PSHIFT_OEQ_CSV" && echo y || echo n)  AITA_pshift started=$(started "$PSHIFT_AITA_JOB" "$PSHIFT_AITA_CSV" && echo y || echo n)"
  fi
  sleep "$POLL"
done
echo "[p5watch] MAX_ITERS reached — exiting without submit (check manually)."; exit 1
