#!/bin/bash
# LOGIN-NODE watcher that AUTO-LAUNCHES the paid scoring for the `stakeholder`
# prompt baseline (added 2026-09-10) as soon as its held-out generations land.
# NO GPU, NO sbatch — pure OpenRouter gpt-5-mini, because Marlowe compute nodes
# are air-gapped and cannot reach the API.
#
# The arm is the Qwen3-8B prompt-only baseline whose generation appends
# "consider stakeholders that may be affected by user behavior, and provide an
# objective response" to each held-out post (GEN_STEER=stakeholder ->
# tag base-stakeholder; see marlowe_final_gen.sbatch).
#
# It does NOT reimplement any scoring: it waits for gen and then calls the two
# canonical drivers, which are already idempotent and per-row ready-checked:
#   OEQ/PAS  -> marlowe_final_endorse_all.sh all   (stage + endorse judge
#               + OEQ voice census)
#   AITA/FLIP-> marlowe_final_classify_all.sh all  (verdict classifier)
# Both skip any row already scored and any row whose gen shards are absent, so
# calling the `all` group is safe: as of 2026-09-10 every other row in those
# groups is already scored, so the only API spend is this arm's 4 envs. That is
# ASSERTED below before each driver call — if some other arm goes unscored later,
# this watcher refuses to run rather than quietly widening the spend.
#
# Launch detached (login node):
#   source ~/.bashrc                       # OPENROUTER_API_KEY
#   setsid nohup bash training_mitigations/slurm_scripts/marlowe_stakeholder_score_watcher.sh \
#       > /projects/m000254/logs/stakeholder_score_watcher.log 2>&1 &
#
# Stop it (kills the whole process group, i.e. the poll sleep too -- `pkill -f`
# alone leaves the sleep child behind):
#   kill -- -"$(cat /projects/m000254/logs/.stakeholder_score_watcher.pid)"
#
# Inspect what it would do, spending nothing and without taking the lock:
#   bash training_mitigations/slurm_scripts/marlowe_stakeholder_score_watcher.sh --check
#
# Options (env):
#   POLL_SECONDS  gen-readiness poll interval          (default 300)
#   MAX_HOURS     give up and exit rather than poll    (default 72)
#                 forever if the batch partition never
#                 comes back
set -uo pipefail

PROJECT=m000254
# Site paths are overridable, same convention as marlowe_final_endorse_all.sh
# (which is scored in place on `next` for the arms that never leave it). Also
# what lets tests/test_stakeholder_prompt_baseline.py exercise the spend guard
# against a synthetic tree instead of the live one.
REPO="${REPO:-/projects/$PROJECT/sycophancy}"
RUNS_ROOT="${RUNS_ROOT:-/projects/$PROJECT/runs}"
SS="${SS:-$REPO/training_mitigations/slurm_scripts}"
SCORED_DIR="${SCORED_DIR:-$REPO/training_mitigations/sycoharm_runs/endorse_gpt5mini_v5_temp0}"
PY="${PY:-/projects/$PROJECT/envs/llm126/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MAX_HOURS="${MAX_HOURS:-72}"
LOCK="${LOCK:-/projects/$PROJECT/logs/.stakeholder_score_watcher.lock}"
PIDFILE="${PIDFILE:-/projects/$PROJECT/logs/.stakeholder_score_watcher.pid}"
cd "$REPO"

log(){ echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
[[ "${1:-}" == "--check" ]] || \
  : "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY (it is in ~/.bashrc on Marlowe)}"

# Single-instance: two watchers would double-submit the same judge calls in the
# window between one starting a judge and its scored_*.csv appearing.
if [[ "${1:-}" != "--check" ]]; then
  exec 9>"$LOCK" || { echo "cannot open lock $LOCK" >&2; exit 1; }
  flock -n 9 || { echo "another stakeholder_score_watcher holds $LOCK — exiting." >&2; exit 0; }
  echo $$ > "$PIDFILE"
  trap 'rm -f "$PIDFILE"' EXIT
fi

TAG=base-stakeholder_finaldata_temp1.0_seed0

# --- the arm's four envs -----------------------------------------------------
# env | run dir | gen shards | scoring path | completion probe
#   endorse path -> $SCORED_DIR/scored_<name>.csv
#   verdict path -> <run>/eval/aita_scored_<tag>_shard{i}of{N}.csv
ENDORSE_ARMS=(   # env|rundir|name
  "OEQ|baseline_final_OEQ|oeq_stakeholder_finaldata"
  "PAS|baseline_final_PAS|pas_stakeholder_finaldata"
)
VERDICT_ARMS=(   # env|rundir|nshards
  "AITA|baseline_final_AITA|1"
  "AITA-NTA-FLIP|baseline_final_AITA-NTA-FLIP|2"
)

gen_ready(){ # rundir prefix nshards -> all gen shards present
  local d=$RUNS_ROOT/$1/eval pre=$2 n=$3 s
  for ((s=0; s<n; s++)); do [[ -f "$d/${pre}_${TAG}_shard${s}of${n}.csv" ]] || return 1; done
}
endorse_done(){ [[ -f "$SCORED_DIR/scored_$1.csv" ]]; }
verdict_done(){ # rundir nshards
  local d=$RUNS_ROOT/$1/eval n=$2 s
  for ((s=0; s<n; s++)); do [[ -f "$d/aita_scored_${TAG}_shard${s}of${n}.csv" ]] || return 1; done
}

# --- spend guard -------------------------------------------------------------
# Re-derive, from the driver itself, which of its rows would actually incur API
# calls right now. Anything other than this arm means the driver's scope has
# drifted since 2026-09-10, so stop instead of spending on someone else's arm.
endorse_extra_rows(){
  awk '/^OEQ_JOBS=\(/,/^\)/' "$SS/marlowe_final_endorse_all.sh" \
    | cat - <(awk '/^PAS_JOBS=\(/,/^\)/' "$SS/marlowe_final_endorse_all.sh") \
    | grep '^  "' | tr -d '"' | while read -r env run tag name dov; do
        [[ "$name" == *_stakeholder_finaldata ]] && continue
        [[ -f "$SCORED_DIR/scored_$name.csv" ]] && continue
        [[ -f "$RUNS_ROOT/$run/eval/responses_${tag}_shard0of2.csv" ]] || continue
        [[ -f "$RUNS_ROOT/$run/eval/responses_${tag}_shard1of2.csv" ]] || continue
        echo "$name"
      done
}
verdict_extra_rows(){
  awk '/^AITA_JOBS=\(/,/^\)/' "$SS/marlowe_final_classify_all.sh" \
    | cat - <(awk '/^FLIP_JOBS=\(/,/^\)/' "$SS/marlowe_final_classify_all.sh") \
    | grep '^  "' | tr -d '"' | while read -r env run tag n; do
        [[ "$tag" == "$TAG" ]] && continue
        local_ready=1; local_done=1
        for ((s=0; s<n; s++)); do
          [[ -f "$RUNS_ROOT/$run/eval/aita_responses_${tag}_shard${s}of${n}.csv" ]] || local_ready=0
          [[ -f "$RUNS_ROOT/$run/eval/aita_scored_${tag}_shard${s}of${n}.csv" ]] || local_done=0
        done
        (( local_ready && ! local_done )) && echo "$env:$tag"
      done
}

# --check: print the state of all four envs + the spend guards, then exit. Lets
# you confirm what the watcher would do WITHOUT it taking the lock long-term or
# spending anything.
if [[ "${1:-}" == "--check" ]]; then
  printf "%-14s %-10s %-10s %s\n" ENV GEN SCORED RUNDIR
  for row in "${ENDORSE_ARMS[@]}"; do
    IFS='|' read -r env run name <<< "$row"
    g=no; d=no; gen_ready "$run" responses 2 && g=ready; endorse_done "$name" && d=done
    printf "%-14s %-10s %-10s %s\n" "$env" "$g" "$d" "$run"
  done
  for row in "${VERDICT_ARMS[@]}"; do
    IFS='|' read -r env run n <<< "$row"
    g=no; d=no; gen_ready "$run" aita_responses "$n" && g=ready; verdict_done "$run" "$n" && d=done
    printf "%-14s %-10s %-10s %s\n" "$env" "$g" "$d" "$run"
  done
  # Report the guards as "<count>" + indented rows, never as a bare multi-line
  # blob: the count is what tells you at a glance whether the `all` group is
  # still scoped to this arm alone.
  for kind in endorse verdict; do
    rows="$(${kind}_extra_rows)"
    n=$([[ -z "$rows" ]] && echo 0 || echo "$rows" | wc -l)
    echo "$kind rows outside this arm that would ALSO be scored: $n"
    [[ -n "$rows" ]] && echo "$rows" | sed 's/^/    /'
  done
  exit 0
fi

log "watcher up: poll=${POLL_SECONDS}s max=${MAX_HOURS}h tag=$TAG"
log "waiting on gen: $(squeue -u "$USER" -h -o '%j' | grep -c '^fg_.*_stakeholder$' || true) stakeholder gen job(s) in queue"
DEADLINE=$(( $(date +%s) + MAX_HOURS * 3600 ))

while :; do
  # --- OEQ/PAS endorse path -------------------------------------------------
  pending_endorse=0
  for row in "${ENDORSE_ARMS[@]}"; do
    IFS='|' read -r env run name <<< "$row"
    endorse_done "$name" && continue
    pending_endorse=1
    gen_ready "$run" responses 2 || { log "$env: gen not ready — waiting."; continue; }
    extra="$(endorse_extra_rows)"
    if [[ -n "$extra" ]]; then
      log "REFUSING endorse_all: it would also score rows outside this arm: $(echo $extra)"
      log "Scope the spend by hand (or add a driver group) and re-launch."
      exit 3
    fi
    log "$env: gen landed -> launching endorse_all (scores $name only)"
    bash "$SS/marlowe_final_endorse_all.sh" all
    break   # the driver covers both OEQ and PAS in one pass
  done

  # --- AITA/FLIP verdict path ----------------------------------------------
  pending_verdict=0
  for row in "${VERDICT_ARMS[@]}"; do
    IFS='|' read -r env run n <<< "$row"
    verdict_done "$run" "$n" && continue
    pending_verdict=1
    gen_ready "$run" aita_responses "$n" || { log "$env: gen not ready — waiting."; continue; }
    extra="$(verdict_extra_rows)"
    if [[ -n "$extra" ]]; then
      log "REFUSING classify_all: it would also classify rows outside this arm: $(echo $extra)"
      log "Scope the spend by hand (or add a driver group) and re-launch."
      exit 3
    fi
    log "$env: gen landed -> launching classify_all (classifies this arm only)"
    bash "$SS/marlowe_final_classify_all.sh" all
    break   # the driver covers both AITA and FLIP in one pass
  done

  if (( ! pending_endorse && ! pending_verdict )); then
    log "all four envs scored:"
    for row in "${ENDORSE_ARMS[@]}"; do IFS='|' read -r env run name <<< "$row"
      log "  $env -> $SCORED_DIR/scored_$name.csv"; done
    for row in "${VERDICT_ARMS[@]}"; do IFS='|' read -r env run n <<< "$row"
      log "  $env -> $RUNS_ROOT/$run/eval/aita_scored_${TAG}_shard*of${n}.csv"; done
    # Final step, and free (no API): rebuild the four paper tables so the new
    # row is actually in them. Non-fatal -- the scoring above is the expensive
    # part and is already durable on disk, so a table-build failure must not
    # read as a scoring failure.
    log "rebuilding paper tables (build_finaldata_tables.py)"
    if "$PY" training_mitigations/eval/build_finaldata_tables.py; then
      log "tables rebuilt -> $REPO/training_mitigations/sycoharm_runs/"
    else
      log "WARN table build failed (scoring above is complete and on disk);"
      log "     re-run by hand: $PY training_mitigations/eval/build_finaldata_tables.py"
    fi
    exit 0
  fi
  if (( $(date +%s) >= DEADLINE )); then
    log "GIVING UP after ${MAX_HOURS}h — gen never landed for some env."
    log "Check: squeue -u $USER | grep stakeholder ; sinfo -p batch"
    exit 4
  fi
  # 9>&- : do NOT leak the flock fd into the sleep child. Otherwise killing the
  # watcher leaves the orphaned sleep holding the lock for up to POLL_SECONDS,
  # and an immediate relaunch bounces off its own stale lock.
  sleep "$POLL_SECONDS" 9>&-
done
