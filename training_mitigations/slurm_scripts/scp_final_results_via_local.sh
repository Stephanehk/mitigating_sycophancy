#!/bin/bash
# Relay the FINAL-DATA results produced on `next` to Marlowe, via your laptop:
#   next -> laptop -> Marlowe.
#
# RUN THIS ON YOUR LOCAL MACHINE (it must have ssh to BOTH next and Marlowe).
# Everything it moves is gitignored (training_mitigations/runs/ and
# .../sycoharm_runs/), so `git pull` on Marlowe will NOT bring any of it.
#
# WHAT IT MOVES (two experiments, both already fully generated AND scored on next):
#
#   1. P6 method on AITA-NTA-FLIP  — dpo_lora_flip_..._rpo0p5_final
#      Adapter was TRAINED on Marlowe and copied here; generation (5 shards, on
#      next6) + gpt-5-mini verdict classify ran on next. So only eval/ goes back;
#      the adapter is already on Marlowe and is NOT re-sent.
#
#   2. Neutral / neutral-mixed DPO on OEQ + PAS (4 arms)
#      Trained on next (their veto is the OpenRouter gpt-5-mini judge, which an
#      air-gapped Marlowe compute node cannot call), then generated AND endorse-
#      scored on next. Adapters are included for archival: FINAL_RUNS.md Phase 1b
#      relays them so Marlowe can generate, but that is already done, so nothing
#      on Marlowe actually consumes them.
#
# TWO DESTINATION ROOTS (this is why the script stages by absolute path rather
# than reusing scp_env_csvs_via_local.sh's repo-relative layout):
#   scored/staged CSVs -> $MARLOWE_REPO/training_mitigations/sycoharm_runs/...
#   run dirs           -> $MARLOWE_RUNS/<run>/...        (OUTSIDE the repo)
# marlowe_final_endorse_all.sh reads RUNS_ROOT=/projects/m000254/runs but writes
# SCORED_DIR=$REPO/training_mitigations/sycoharm_runs/..., so both must be right.
#
# Landing scored_<name>.csv is self-enforcing: marlowe_final_endorse_all.sh skips
# any condition whose scored CSV already exists, so the 4 neutral arms will not be
# re-scored (and not re-billed) on Marlowe.
#
# The push hop uses ONE `sftp` batch instead of `ssh host mkdir` + `scp`, because
# Marlowe's login node authenticates but then closes non-interactive remote COMMAND
# execution (`ssh host "mkdir ..."` -> "Connection closed"). The SFTP subsystem is
# still permitted, so we mkdir and upload over sftp — one Duo prompt for the lot.
# `sftp -b` implies BatchMode=yes, which DISABLES interactive/Duo auth -> instant
# "Permission denied"; `-o BatchMode=no` keeps the single prompt.
#
# Usage (on your laptop):
#   NEXT_HOST=stephhk@scdt.stanford.edu \
#   [MARLOWE_HOST=stephhk@login-01.marlowe.stanford.edu] \
#   [MARLOWE_REPO=/projects/m000254/sycophancy] [MARLOWE_RUNS=/projects/m000254/runs] \
#   [STAGE=./marlowe_results_stage] [KEEP_STAGE=1] [SKIP_ADAPTERS=1] [DRY_RUN=1] \
#   bash scp_final_results_via_local.sh
set -euo pipefail

: "${NEXT_HOST:?set NEXT_HOST=user@next-login-hostname}"
MARLOWE_HOST="${MARLOWE_HOST:-stephhk@login-01.marlowe.stanford.edu}"
NEXT_REPO="${NEXT_REPO:-/next/u/stephhk/sycophancy}"
MARLOWE_REPO="${MARLOWE_REPO:-/projects/m000254/sycophancy}"
MARLOWE_RUNS="${MARLOWE_RUNS:-/projects/m000254/runs}"
STAGE="${STAGE:-./marlowe_results_stage}"

FLIP_RUN=dpo_lora_flip_qwen3-8b_n500_mixed_P6_perspshift_distill_gp_rpo0p5_final
NEUTRAL_RUNS=(
    dpo_lora_oeq_qwen3-8b_n1000_neutral_final
    dpo_lora_oeq_qwen3-8b_n1000_neutral_mixed_final
    dpo_lora_pas_qwen3-8b_n1000_neutral_final
    dpo_lora_pas_qwen3-8b_n1000_neutral_mixed_final
)
SCORED_NAMES=(oeq_neutral oeq_neutralmixed pas_neutral pas_neutralmixed)

echo "[relay] next    : $NEXT_HOST:$NEXT_REPO"
echo "[relay] stage   : $STAGE (local)"
echo "[relay] marlowe : $MARLOWE_HOST  repo=$MARLOWE_REPO  runs=$MARLOWE_RUNS"
echo

# ---------------------------------------------------------------------------
# Stage layout mirrors the ABSOLUTE destination path, minus the leading '/':
#   $STAGE/projects/m000254/runs/<run>/eval/...
# so hop 2 can derive every remote path mechanically as "/${relpath}" and stays
# agnostic to the two different roots.
# ---------------------------------------------------------------------------
pull() {  # pull <src-abs-on-next> <dst-abs-on-marlowe> [rsync-args...]
    local src="$1" dst="$2"; shift 2
    local local_path="$STAGE/${dst#/}"
    mkdir -p "$(dirname "$local_path")"
    # -t preserves mtimes so a re-run after a failed push skips already-pulled
    # bytes; --partial keeps interrupted 167M adapters resumable.
    rsync -rt --partial --info=name0 "$@" "$NEXT_HOST:$src" "$local_path"
}

echo "[relay] --- hop 1: next -> laptop ---"

# 1a. Endorse-scored CSVs + the staged judge inputs that produced them (repo-relative).
for n in "${SCORED_NAMES[@]}"; do
    pull "$NEXT_REPO/training_mitigations/sycoharm_runs/endorse_gpt5mini_v5_temp0/scored_${n}_finaldata.csv" \
         "$MARLOWE_REPO/training_mitigations/sycoharm_runs/endorse_gpt5mini_v5_temp0/scored_${n}_finaldata.csv"
    pull "$NEXT_REPO/training_mitigations/sycoharm_runs/inputs/${n}_finaldata.csv" \
         "$MARLOWE_REPO/training_mitigations/sycoharm_runs/inputs/${n}_finaldata.csv"
done

# 1b. FLIP P6: eval/ only. The adapter originated on Marlowe — do not ship it back.
pull "$NEXT_REPO/training_mitigations/runs/$FLIP_RUN/eval/" \
     "$MARLOWE_RUNS/$FLIP_RUN/eval/"

# 1c. Neutral runs: eval/ + final adapter + split provenance. Explicitly EXCLUDE
# the bulk that Marlowe has no use for (~1.7G/run): per-iteration adapters and
# checkpoints, the frozen ref copy, the live vLLM policy dir, training-time
# sample dumps, and server/memory logs.
for r in "${NEUTRAL_RUNS[@]}"; do
    pull "$NEXT_REPO/training_mitigations/runs/$r/eval/" "$MARLOWE_RUNS/$r/eval/"
    for f in adapter_config.json .complete train_ids.json test_ids.json iter_stats.json README.md; do
        pull "$NEXT_REPO/training_mitigations/runs/$r/$f" "$MARLOWE_RUNS/$r/$f" || \
            echo "[relay] note: $r/$f absent on next — skipping"
    done
    if [[ "${SKIP_ADAPTERS:-0}" == "1" ]]; then
        echo "[relay] SKIP_ADAPTERS=1 — omitting $r/adapter_model.safetensors (167M)"
    else
        pull "$NEXT_REPO/training_mitigations/runs/$r/adapter_model.safetensors" \
             "$MARLOWE_RUNS/$r/adapter_model.safetensors"
    fi
done

echo
echo "[relay] staged $(find "$STAGE" -type f | wc -l | tr -d ' ') files, $(du -sh "$STAGE" | cut -f1) total"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[relay] DRY_RUN=1 — stopping before the push. Inspect $STAGE, then re-run without DRY_RUN."
    exit 0
fi

# ---------------------------------------------------------------------------
# Hop 2: laptop -> Marlowe, ONE sftp batch.
# ---------------------------------------------------------------------------
batch=$(mktemp)

# PREFLIGHT (listing only, never deletes): if Marlowe ALSO generated/classified
# FLIP P6 under its own 2-shard regime (marlowe_final_gen_submit.sh uses N=2 for
# FLIP), its eval/ already holds *of2* files. Ours are *of5*. A mixed scored set
# makes build_aita_flip_tables.py's load_concat abort with "inconsistent shard
# counts {2, 5}". The listing below prints BEFORE the uploads — if you see
# aita_scored_*of2* for this run, remove those stale files on Marlowe by hand.
echo "-ls -l $MARLOWE_RUNS/$FLIP_RUN/eval/" >> "$batch"

# mkdir every ancestor dir (leading '-' => ignore "already exists"), then put.
# Sorting the dir list guarantees parents precede children.
find "$STAGE" -type f | sed "s|^$STAGE||" | while read -r rel; do dirname "$rel"; done \
    | sort -u | while read -r d; do
        path=""
        IFS='/' read -ra parts <<< "${d#/}"
        for p in "${parts[@]}"; do
            path="$path/$p"
            echo "-mkdir $path"
        done
      done | awk '!seen[$0]++' >> "$batch"
find "$STAGE" -type f | sed "s|^$STAGE||" | while read -r rel; do
    echo "put $STAGE$rel $rel"
done >> "$batch"

echo "[relay] --- hop 2: laptop -> Marlowe (single sftp session) ---"
echo "[relay] NOTE: check the FLIP eval listing below for stale aita_scored_*of2* files."
sftp -o BatchMode=no -b "$batch" "$MARLOWE_HOST"
rm -f "$batch"

if [[ "${KEEP_STAGE:-0}" == "1" ]]; then
    echo "[relay] kept local stage at $STAGE"
else
    rm -rf "$STAGE"
    echo "[relay] removed local stage $STAGE (set KEEP_STAGE=1 to keep)"
fi

cat <<EOF

[relay] done. Verify on Marlowe:
    ls -la $MARLOWE_RUNS/$FLIP_RUN/eval/
    ls -la $MARLOWE_REPO/training_mitigations/sycoharm_runs/endorse_gpt5mini_v5_temp0/scored_*_finaldata.csv
Then the two read-out stages should report these arms as already done:
    bash training_mitigations/slurm_scripts/marlowe_final_endorse_all.sh all    # 4 neutral -> "already scored — skipping"
    bash training_mitigations/slurm_scripts/marlowe_final_classify_all.sh flip  # FLIP P6 reads the of5 shards
EOF
