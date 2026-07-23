#!/bin/bash
# Relay held-out generation outputs from Marlowe to the `next`/SC cluster, via
# THIS laptop, so the endorse-judge (run_endorse_gpt5mini.py) can run on next
# where /next/u/stephhk/sycophancy lives.
#
# Why two hops: Marlowe's login node has no key auth to SC (SC needs your
# password/Duo), so a direct Marlowe->SC scp fails. Your laptop can auth to both,
# so we go Marlowe -> laptop (stage) -> next. RUN THIS ON YOUR LAPTOP.
#
# It only moves the small eval artifacts (default glob *final1000* = the per-shard
# responses csv/jsonl + summary json), NOT adapters/checkpoints. Idempotent
# (rsync), so re-running only ships what changed.
#
# Usage (from your laptop):
#   # defaults to the 2 PAS runs; override hosts to your ssh aliases:
#   MARLOWE_HOST=marlowe NEXT_HOST=sc bash sync_heldout_to_next.sh
#   # or name specific runs (any env), and/or dry-run first:
#   bash sync_heldout_to_next.sh --dry-run dpo_lora_pas_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp
#
# Then, on next, judge each run, e.g.:
#   python training_mitigations/eval/stage_heldout_inputs.py \
#       --run-dir training_mitigations/runs/<run> --name <run>
#   # -> feeds training_mitigations/eval/run_endorse_gpt5mini.py (GPT-5-mini judge)
set -euo pipefail

# --- parameters (override via env) --------------------------------------------
MARLOWE_HOST=${MARLOWE_HOST:-marlowe}          # your ssh alias/host for Marlowe login
NEXT_HOST=${NEXT_HOST:-sc}                      # your ssh alias/host for next / SC
MARLOWE_RUNS=${MARLOWE_RUNS:-/projects/m000254/runs}
NEXT_RUNS=${NEXT_RUNS:-/next/u/stephhk/sycophancy/training_mitigations/runs}
STAGE=${STAGE_DIR:-$HOME/marlowe_heldout_stage}   # local staging dir on the laptop
GLOB=${FILE_GLOB:-'*final1000*'}               # which eval files to relay

DRYRUN=()
if [[ "${1:-}" == "--dry-run" ]]; then DRYRUN=(-n); shift; echo "[dry-run] no files written"; fi

RUNS=("$@")
if [[ ${#RUNS[@]} -eq 0 ]]; then
  # default: the two PAS runs (AITA/FLIP are judged locally on Marlowe, no relay)
  RUNS=(dpo_lora_pas_qwen3-8b_n1000_mixed_P5_nocrit_distill_gp_frozenstake
        dpo_lora_pas_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp)
fi

echo "marlowe=$MARLOWE_HOST:$MARLOWE_RUNS  ->  laptop:$STAGE  ->  next=$NEXT_HOST:$NEXT_RUNS"
echo "runs: ${RUNS[*]}"
echo "glob: $GLOB"
echo

for run in "${RUNS[@]}"; do
  echo "==================== $run ===================="
  mkdir -p "$STAGE/$run/eval"
  # hop 1: Marlowe -> laptop (remote shell expands $GLOB)
  rsync "${DRYRUN[@]}" -avz --progress \
    "$MARLOWE_HOST:$MARLOWE_RUNS/$run/eval/$GLOB" \
    "$STAGE/$run/eval/"
  # hop 2: laptop -> next
  ssh "$NEXT_HOST" "mkdir -p '$NEXT_RUNS/$run/eval'"
  rsync "${DRYRUN[@]}" -avz --progress \
    "$STAGE/$run/eval/" \
    "$NEXT_HOST:$NEXT_RUNS/$run/eval/"
  echo "[ok] $run relayed. On next: $NEXT_RUNS/$run/eval/"
done

echo
echo "done. Verify on next:  ssh $NEXT_HOST 'ls $NEXT_RUNS/${RUNS[0]}/eval/'"
