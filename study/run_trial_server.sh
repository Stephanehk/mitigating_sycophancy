#!/bin/bash
# Start / stop / restart the detached local-trial study server on next1:8750.
#
# Exists because the inline launch command contains the string
# "uvicorn server:app", so a `pkill -f` issued in the same shell command matches
# its OWN command line and kills the calling shell. Writing a pidfile and
# killing that pid instead removes the whole class of mistake.
#
# The raised timeouts are the ones §4b of README.md requires: the defaults
# (lease 180s, deadline 420s) reap jobs a contended a4000 node is still
# generating.
#
# THE DATABASE LIVES ON /scr, NOT /next. It used to be under study/data/, which
# is NFS, and that cost ~213ms on every request with 10-14 SECOND stalls when
# the GPU worker's long poll contended for the write lock. /scr is next1's local
# NVMe. Two consequences: the server must always run on next1, and /scr is
# purge-eligible, so `backup` the submissions file to /next before you care
# about it.
#
# Supabase credentials are sourced from .supabase_env if it exists (see
# .supabase_env.example). Without it the server still runs and writes every
# submission to the local JSONL, which is the right setup for a trial.
#
#   ./run_trial_server.sh start | stop | restart | status
#   ./run_trial_server.sh restart --wipe     # also delete the trial database
#   ./run_trial_server.sh backup             # copy submissions.jsonl to /next
#   ./run_trial_server.sh export             # submissions -> tidy CSVs

set -euo pipefail

STUDY_DIR=/next/u/stephhk/sycophancy/study
PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
PIDFILE="$STUDY_DIR/logs/trial_server.pid"

# Local NVMe on next1. See the note above before moving this back under /next.
SCRATCH="/scr/$USER/syco_study"
DB="$SCRATCH/local_trial.db"
SUBMISSIONS="$SCRATCH/submissions.jsonl"
BACKUP_DIR="$STUDY_DIR/data"

PORT=8750
HOST_IP=10.79.12.73

cd "$STUDY_DIR"
mkdir -p "$SCRATCH" logs

running_pid() {
    [ -f "$PIDFILE" ] || return 1
    local pid
    pid=$(cat "$PIDFILE")
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && echo "$pid"
}

stop_server() {
    local pid
    if pid=$(running_pid); then
        echo "stopping server pid=$pid"
        kill "$pid"
        for _ in $(seq 1 20); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.5
        done
        kill -0 "$pid" 2>/dev/null && { echo "force killing $pid"; kill -9 "$pid"; }
    else
        echo "no server recorded as running"
    fi
    rm -f "$PIDFILE"
}

start_server() {
    if pid=$(running_pid); then
        echo "already running (pid=$pid); use restart" >&2
        exit 1
    fi
    : "${STUDY_WORKER_TOKEN:=$(cat "$STUDY_DIR/.trial_token")}"

    # Optional. Absent -> submissions land only in $SUBMISSIONS, which is what
    # you want for a trial; the server logs a warning saying so.
    #
    # An UNFILLED .supabase_env is worse than no file at all: the values are
    # non-empty, so the server would report supabase_configured=true and then
    # 401 on every submit — discovered at the end of a participant's session.
    # Copying the example and forgetting to edit it is the obvious way to get
    # there, so detect the placeholders and fall back to local-only.
    if [ -f "$STUDY_DIR/.supabase_env" ]; then
        # shellcheck disable=SC1091
        . "$STUDY_DIR/.supabase_env"
        case "${STUDY_SUPABASE_URL:-}${STUDY_SUPABASE_SERVICE_KEY:-}" in
            *"<"*|*">"*|"")
                echo "supabase: .supabase_env still has PLACEHOLDER values -> local JSONL only"
                echo "          edit $STUDY_DIR/.supabase_env with the real Project URL"
                echo "          and service_role key, then restart."
                unset STUDY_SUPABASE_URL STUDY_SUPABASE_SERVICE_KEY
                ;;
            *)
                echo "supabase: ${STUDY_SUPABASE_URL} table=${STUDY_SUPABASE_TABLE:-study_submissions}"
                ;;
        esac
    else
        echo "supabase: not configured (.supabase_env absent) -> local JSONL only"
    fi

    setsid env \
        STUDY_DATABASE_URL="sqlite:///$DB" \
        STUDY_SUBMISSION_FALLBACK_PATH="$SUBMISSIONS" \
        STUDY_SUPABASE_URL="${STUDY_SUPABASE_URL:-}" \
        STUDY_SUPABASE_SERVICE_KEY="${STUDY_SUPABASE_SERVICE_KEY:-}" \
        STUDY_SUPABASE_TABLE="${STUDY_SUPABASE_TABLE:-study_submissions}" \
        STUDY_WORKER_TOKEN="$STUDY_WORKER_TOKEN" \
        STUDY_JOB_LEASE_SECONDS=1500 \
        STUDY_JOB_DEADLINE_SECONDS=3600 \
        STUDY_CLIENT_SLOW_NOTICE_MS=15000 \
        "$PY" -m uvicorn server:app --host 0.0.0.0 --port "$PORT" \
        > logs/trial_server.log 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    disown || true
    local pid
    pid=$(cat "$PIDFILE")
    echo "started pid=$pid"
    echo "db:          $DB"
    echo "submissions: $SUBMISSIONS"

    # Poll for readiness rather than sleeping a fixed interval. A flat `sleep 5`
    # was racy — startup has been observed taking ~10s — and printed an empty
    # health line, which reads exactly like a server that failed to boot.
    local health=""
    for _ in $(seq 1 40); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "SERVER DIED during startup; last log lines:" >&2
            tail -20 logs/trial_server.log >&2
            exit 1
        fi
        health=$(curl -s -m 2 "http://$HOST_IP:$PORT/api/health" || true)
        [ -n "$health" ] && break
        sleep 1
    done
    if [ -z "$health" ]; then
        echo "server did not answer /api/health within 40s; last log lines:" >&2
        tail -20 logs/trial_server.log >&2
        exit 1
    fi
    echo "$health"
}

case "${1:-status}" in
    start)   start_server ;;
    stop)    stop_server ;;
    status)
        if pid=$(running_pid); then
            echo "running pid=$pid"
            curl -s -m 3 "http://$HOST_IP:$PORT/api/health"; echo
        else
            echo "not running"
        fi
        ;;
    restart)
        stop_server
        if [ "${2:-}" = "--wipe" ]; then
            rm -f "$DB" "$DB-wal" "$DB-shm" "$SUBMISSIONS"
            echo "wiped $DB and $SUBMISSIONS"
        fi
        start_server
        ;;
    backup)
        # /scr is node-local and purge-eligible. The submissions file is the
        # only copy of a participant's answers until Supabase has them, so it
        # is the one thing worth copying back to /next.
        [ -f "$SUBMISSIONS" ] || { echo "nothing to back up at $SUBMISSIONS"; exit 0; }
        mkdir -p "$BACKUP_DIR"
        cp "$SUBMISSIONS" "$BACKUP_DIR/submissions.jsonl"
        echo "copied $(wc -l < "$SUBMISSIONS") rows -> $BACKUP_DIR/submissions.jsonl"
        ;;
    export)
        "$PY" analysis/export_responses.py --source local \
            --local-path "$SUBMISSIONS" --out-dir "${2:-/tmp/$USER/trial_export}"
        ;;
    *) echo "usage: $0 {start|stop|restart [--wipe]|status|backup|export [dir]}" >&2
       exit 2 ;;
esac
