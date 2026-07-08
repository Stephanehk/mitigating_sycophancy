#!/bin/bash
# Overnight watcher: when the OEQ sweep pipeline finishes, promote the arms that clear
# both gates to PAS / AITA / AITA-NTA-FLIP training. Deterministic -- no LLM in the loop.
#
#   source ~/.bashrc
#   setsid nohup bash training_mitigations/slurm_scripts/marlowe_promote_watcher.sh \
#       > /projects/m000254/logs/promote_watcher.log 2>&1 &
#
# Gates (promote_select.py, which owns the definitions):
#   0.40 < action_endorsement_rate < 0.60   (#1/(#0+#1), 0/1 labels only)
#   stakeholder role-play rate < 0.10
#
# WHY A SCRIPT AND NOT AN AGENT: the decision rule is a two-line predicate over two
# numbers. An unattended agent would add nondeterminism, an API dependency, and a token
# budget to a step whose entire job is to submit up to 18 GPU training jobs correctly.
# Everything here is auditable after the fact: the decision table is written to json.
#
# --dry-run prints the sbatch commands instead of running them.
set -uo pipefail

PROJECT=m000254
REPO=/projects/$PROJECT/sycophancy
RUNS_ROOT=/projects/$PROJECT/runs
LOGS=/projects/$PROJECT/logs
PY=/projects/$PROJECT/envs/llm126/bin/python
S=$REPO/training_mitigations/slurm_scripts
DECISION=$LOGS/promote_decision.json
PIPELOG=$LOGS/oeq_pipeline.log
cd "$REPO"

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

POLL=300              # 5 min
TIMEOUT_HOURS=36      # hard stop: the sweep is ~12h; 36h means something is wrong
MAX_JOBS=18           # 6 recipes x 3 envs -- a runaway guard, not an expected value

log(){ echo "[$(date -u +%F' '%H:%M:%S)] $*"; }

: "${HF_TOKEN:?export HF_TOKEN (Qwen3-8B is gated; the promoted jobs need it)}"
: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY (promote_select reads judge output)}"

ARMS=(P5_perspshift_gp
      P6_perspshift_gp
      P6_perspshift_distill_gp_rpo0p25
      P6_perspshift_distill_gp_rpo0p5
      P5_nocrit_distill_gp_frozenstake_rpo0p25
      P5_nocrit_distill_gp_frozenstake_rpo0p5)

# The voice census is the LAST thing the sweep pipeline does per arm (phase 4, after the
# endorse judge). So all 6 voice csvs existing implies the judge finished. We still let
# promote_select.py assert the endorse csvs are complete -- the judge checkpoints every
# 100 rows, so a partial scored_*.csv on disk is a real possibility.
voice_csv(){ echo "$RUNS_ROOT/dpo_lora_oeq_qwen3-8b_n1000_mixed_$1/eval/voice_final_final1000_temp1.0_seed0.csv"; }

all_voice_present(){
  local a
  for a in "${ARMS[@]}"; do [[ -f "$(voice_csv "$a")" ]] || return 1; done
  return 0
}

log "watcher up. waiting for the OEQ sweep to finish (${#ARMS[@]} arms, poll ${POLL}s, timeout ${TIMEOUT_HOURS}h)"
deadline=$(( $(date +%s) + TIMEOUT_HOURS*3600 ))
while ! all_voice_present; do
  # bail out if the sweep driver aborted -- otherwise we poll for 36h against a corpse
  if grep -aq "ABORT:" "$PIPELOG" 2>/dev/null; then
    log "FATAL: the sweep driver logged ABORT; promoting nothing."
    grep -a "ABORT:" "$PIPELOG" | tail -3
    exit 1
  fi
  if (( $(date +%s) > deadline )); then
    log "FATAL: timed out after ${TIMEOUT_HOURS}h. Arms still missing a voice census:"
    for a in "${ARMS[@]}"; do [[ -f "$(voice_csv "$a")" ]] || echo "    $a"; done
    exit 1
  fi
  sleep "$POLL"
done
log "all 6 voice censuses present -> the sweep finished. Selecting."

# ---------------- select ----------------
# promote_select.py ASSERTS completeness (1000 rows, every row judged) and dies loudly
# on a partial file rather than gating on half a judge run.
"$PY" training_mitigations/eval/promote_select.py --out "$DECISION"
rc=$?
if (( rc != 0 )); then log "FATAL: promote_select.py exited $rc; promoting nothing."; exit 1; fi

mapfile -t SUBMIT < <("$PY" -c "
import json
d = json.load(open('$DECISION'))
for w in d['submit']:
    print(w)
")

if (( ${#SUBMIT[@]} == 0 )); then
  log "no arm cleared both gates. Nothing submitted. Decision table: $DECISION"
  exit 0
fi
if (( ${#SUBMIT[@]} > MAX_JOBS )); then
  log "FATAL: ${#SUBMIT[@]} submissions exceeds MAX_JOBS=$MAX_JOBS -- refusing (runaway guard)."
  exit 1
fi

# ---------------- submit ----------------
log "${#SUBMIT[@]} jobs to submit"
for w in "${SUBMIT[@]}"; do
  f="$S/$w"
  if [[ ! -f "$f" ]]; then log "SKIP $w -- wrapper does not exist"; continue; fi
  out=$(grep -oP '^OUT_NAME=\K.*' "$f")
  jn=$(grep -oP '^#SBATCH --job-name=\K.*' "$f")
  # idempotence: never resubmit a finished run, never double-queue a live one
  if [[ -f "$RUNS_ROOT/$out/.complete" ]]; then log "SKIP $w -- $out already .complete"; continue; fi
  if squeue -u "$USER" -h -o "%j" | grep -qx "$jn"; then log "SKIP $w -- job $jn already queued"; continue; fi
  if (( DRY )); then
    log "DRY-RUN would submit: sbatch $f"
  else
    jid=$(sbatch --parsable "$f")
    log "submitted $w -> $jid  ($out)"
  fi
done
log "PROMOTION COMPLETE. Decision table: $DECISION"
