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
# Usage (on your laptop):
#   NEXT_HOST=you@next-login.stanford.edu \
#   MARLOWE_HOST=you@marlowe-login \
#   [NEXT_REPO=/next/u/stephhk/sycophancy] \
#   [MARLOWE_REPO=/projects/m000254/mitigating_sycophancy] \
#   [STAGE=./marlowe_csv_stage] [KEEP_STAGE=1] [SKIP_PAS=1] \
#   bash scp_env_csvs_via_local.sh
set -euo pipefail

: "${NEXT_HOST:?set NEXT_HOST=user@next-login-hostname}"
: "${MARLOWE_HOST:?set MARLOWE_HOST=user@marlowe-login-hostname}"
NEXT_REPO="${NEXT_REPO:-/next/u/stephhk/sycophancy}"
MARLOWE_REPO="${MARLOWE_REPO:-/projects/m000254/mitigating_sycophancy}"
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

for rel in "${CSVS[@]}"; do
    local_path="$STAGE/$rel"
    mkdir -p "$(dirname "$local_path")"

    # Hop 1: next -> laptop. (Swap to `rsync -avhP` for a resumable pull.)
    echo "[relay] pull  $rel"
    scp -p "$NEXT_HOST:$NEXT_REPO/$rel" "$local_path"

    # Hop 2: laptop -> Marlowe, recreating the repo-relative dir remotely.
    echo "[relay] push  $rel  ($(du -h "$local_path" | cut -f1))"
    ssh "$MARLOWE_HOST" "mkdir -p '$MARLOWE_REPO/$(dirname "$rel")'"
    scp -p "$local_path" "$MARLOWE_HOST:$MARLOWE_REPO/$(dirname "$rel")/"
done

if [[ "${KEEP_STAGE:-0}" == "1" ]]; then
    echo "[relay] kept local stage at $STAGE"
else
    rm -rf "$STAGE"
    echo "[relay] removed local stage $STAGE (set KEEP_STAGE=1 to keep)"
fi

echo
echo "[relay] done. Verify on Marlowe:"
for rel in "${CSVS[@]}"; do echo "    ls -la $MARLOWE_REPO/$rel"; done
