#!/bin/bash
# Copy the three env CSVs the AITA/FLIP/PAS Marlowe DPO runs need, RELAYING
# through your local machine: Stanford `next` -> laptop -> Marlowe. Use this when
# `next` cannot reach Marlowe directly (so scp_env_csvs_to_marlowe.sh won't work).
#
# RUN THIS ON YOUR LOCAL MACHINE (it must have ssh to BOTH next and Marlowe).
# The files are gitignored, so `git pull` on Marlowe will NOT bring them.
# resolve_csv() anchors each path to the repo root, so they must land at
# $MARLOWE_REPO/<same repo-relative path>; this script recreates that layout.
#
#   AITA  data/aita_clean.csv                                                (~175M)
#   FLIP  ELEPHANT_datasets/AITA-NTA-FLIP.csv                                (~4.9M)
#   PAS   osfstorage-archive/Data/PAS_results/endorse4_elephant_oeq_pas_claude.csv (~25M,
#         usually already on Marlowe from prior PAS runs — set SKIP_PAS=1 to omit)
#
# The push hop uses ONE `sftp` batch instead of `ssh host mkdir` + `scp`, because
# Marlowe's login node authenticates but then closes non-interactive remote
# COMMAND execution (`ssh host "mkdir ..."` -> "Connection closed"). The SFTP
# subsystem is still permitted, so we make dirs and upload over sftp — which also
# means Marlowe prompts for password/Duo only ONCE for the whole push.
#
# Usage (on your laptop). MARLOWE_HOST defaults to login-01.marlowe.stanford.edu:
#   NEXT_HOST=stephhk@scdt.stanford.edu \
#   [MARLOWE_HOST=stephhk@login-01.marlowe.stanford.edu] \
#   [NEXT_REPO=/next/u/stephhk/sycophancy] \
#   [MARLOWE_REPO=/projects/m000254/sycophancy] \
#   [STAGE=./marlowe_csv_stage] [KEEP_STAGE=1] [SKIP_PAS=1] \
#   bash scp_env_csvs_via_local.sh
set -euo pipefail

: "${NEXT_HOST:?set NEXT_HOST=user@next-login-hostname}"
MARLOWE_HOST="${MARLOWE_HOST:-stephhk@login-01.marlowe.stanford.edu}"
NEXT_REPO="${NEXT_REPO:-/next/u/stephhk/sycophancy}"
MARLOWE_REPO="${MARLOWE_REPO:-/projects/m000254/sycophancy}"
STAGE="${STAGE:-./marlowe_csv_stage}"

CSVS=(
    "data/aita_clean.csv"
    "ELEPHANT_datasets/AITA-NTA-FLIP.csv"
)
[[ "${SKIP_PAS:-0}" == "1" ]] || CSVS+=(
    "osfstorage-archive/Data/PAS_results/endorse4_elephant_oeq_pas_claude.csv"
)

echo "[relay] next    : $NEXT_HOST:$NEXT_REPO"
echo "[relay] stage   : $STAGE (local)"
echo "[relay] marlowe : $MARLOWE_HOST:$MARLOWE_REPO"
echo

# --- Hop 1: next -> laptop (scp per file; next is key-based, no Duo). ---
# Skip a file that is already staged at the exact same byte size (lets a re-run
# after a failed push avoid re-pulling the 175M aita_clean.csv).
for rel in "${CSVS[@]}"; do
    local_path="$STAGE/$rel"
    mkdir -p "$(dirname "$local_path")"
    remote_size=$(ssh "$NEXT_HOST" "stat -c%s '$NEXT_REPO/$rel'" 2>/dev/null || echo "")
    if [[ -f "$local_path" && -n "$remote_size" ]]; then
        local_size=$(wc -c < "$local_path" | tr -d ' ')
        if [[ "$local_size" == "$remote_size" ]]; then
            echo "[relay] pull  $rel  (already staged, $local_size bytes — skip)"
            continue
        fi
    fi
    echo "[relay] pull  $rel"
    scp -p "$NEXT_HOST:$NEXT_REPO/$rel" "$local_path"
done

# --- Hop 2: laptop -> Marlowe, ONE sftp batch (subsystem, not remote-exec). ---
# Build a batch that -mkdir's each dir component (leading '-' => ignore "exists")
# then put's each file at its full repo-relative path.
batch=$(mktemp)
for rel in "${CSVS[@]}"; do
    dir="$(dirname "$rel")"
    path="$MARLOWE_REPO"
    IFS='/' read -ra parts <<< "$dir"
    for p in "${parts[@]}"; do
        path="$path/$p"
        echo "-mkdir $path" >> "$batch"
    done
    echo "put $STAGE/$rel $MARLOWE_REPO/$rel" >> "$batch"
done

echo "[relay] push $(printf '%s ' "${CSVS[@]}")-> Marlowe (single sftp session)"
sftp -b "$batch" "$MARLOWE_HOST"
rm -f "$batch"

if [[ "${KEEP_STAGE:-0}" == "1" ]]; then
    echo "[relay] kept local stage at $STAGE"
else
    rm -rf "$STAGE"
    echo "[relay] removed local stage $STAGE (set KEEP_STAGE=1 to keep)"
fi

echo
echo "[relay] done. Verify on Marlowe:"
for rel in "${CSVS[@]}"; do echo "    ls -la $MARLOWE_REPO/$rel"; done
