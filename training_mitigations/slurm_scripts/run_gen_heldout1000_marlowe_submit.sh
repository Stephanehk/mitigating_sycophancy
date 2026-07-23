#!/bin/bash
# Arm/launch held-out GENERATION on Marlowe for the trained adapters.
#
# For each training wrapper below, this derives (RUN_ENV, OUT_NAME) straight from
# the wrapper file (no hardcoded run names to drift), maps env -> held-out id
# file, and submits marlowe_run_gen_heldout1000.sbatch (2 shards x 500 prompts)
# over the 1000-prompt held-out set for that model's FINAL adapter.
#
# It is GATED and IDEMPOTENT, so it is safe to run repeatedly as training lands:
#   - skips a run that has NOT yet written <run>/.complete (still training)
#   - skips a run whose both shard responses csvs already exist (already generated)
#   - otherwise submits the gen job on the award account (batch partition)
#
# Usage (from anywhere; award account + batch partition are baked into the sbatch):
#   export HF_TOKEN=...
#   bash /projects/m000254/sycophancy/training_mitigations/slurm_scripts/run_gen_heldout1000_marlowe_submit.sh [--dry-run]
set -uo pipefail

PROJECT=m000254
REPO=/projects/$PROJECT/sycophancy
RUNS_ROOT=/projects/$PROJECT/runs
SS="$REPO/training_mitigations/slurm_scripts"
GENJOB="$SS/marlowe_run_gen_heldout1000.sbatch"

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1
[[ -n "${HF_TOKEN:-}" ]] || { echo "ERROR: export HF_TOKEN before running (Qwen3-8B is gated)." >&2; exit 1; }

# env -> absolute held-out id json (the verified 1000-prompt sets).
ids_for_env() {
  case "$1" in
    PAS)           echo "$REPO/training_mitigations/sycoharm_runs/heldout_pas_final1000_ids.json" ;;
    AITA)          echo "$REPO/training_mitigations/runs/baseline_AITA/test_ids.json" ;;
    AITA-NTA-FLIP) echo "$REPO/training_mitigations/runs/baseline_AITA-NTA-FLIP/test_ids.json" ;;
    *) return 1 ;;
  esac
}

# The 8 trained arms (their training wrappers). RUN_ENV/OUT_NAME are read from each.
WRAPPERS=(
  marlowe_run_train_pas_n1000_mixed_P5_nocrit_distill_gp_frozenstake.sbatch
  marlowe_run_train_pas_n1000_mixed_P6_perspshift_distill_gp.sbatch
  marlowe_run_train_aita_n1000_mixed_P5_nocrit_distill_gp_frozenstake.sbatch
  marlowe_run_train_aita_n1000_mixed_P6_perspshift_distill_gp.sbatch
  marlowe_run_train_aita_n1000_mixed_P5.sbatch
  marlowe_run_train_flip_n500_mixed_P5_nocrit_distill_gp_frozenstake.sbatch
  marlowe_run_train_flip_n500_mixed_P6_perspshift_distill_gp.sbatch
  marlowe_run_train_flip_n500_mixed_P5.sbatch
)

printf "%-56s %-14s %-9s %s\n" "RUN" "ENV" "STATE" "ACTION"
for w in "${WRAPPERS[@]}"; do
  env=$(grep -oP '^RUN_ENV=\K.*'  "$SS/$w")
  out=$(grep -oP '^OUT_NAME=\K.*' "$SS/$w")
  rundir="$RUNS_ROOT/$out"
  ids=$(ids_for_env "$env") || { echo "no ids mapping for env=$env ($w)"; exit 1; }

  if [[ ! -f "$rundir/.complete" ]]; then
    printf "%-56s %-14s %-9s %s\n" "$out" "$env" "training" "wait (.complete absent)"
    continue
  fi
  base="$rundir/eval/responses_final_final1000_temp1.0_seed0"
  if [[ -f "${base}_shard0of2.csv" && -f "${base}_shard1of2.csv" ]]; then
    printf "%-56s %-14s %-9s %s\n" "$out" "$env" "done" "skip (already generated)"
    continue
  fi
  if (( DRY )); then
    printf "%-56s %-14s %-9s %s\n" "$out" "$env" "ready" "would submit gen array 0-1"
  else
    jid=$(sbatch --parsable --export=ALL,GEN_RUNDIR="$rundir",GEN_ENV="$env",GEN_IDS="$ids" "$GENJOB")
    printf "%-56s %-14s %-9s %s\n" "$out" "$env" "ready" "submitted job $jid (array 0-1)"
  fi
done
