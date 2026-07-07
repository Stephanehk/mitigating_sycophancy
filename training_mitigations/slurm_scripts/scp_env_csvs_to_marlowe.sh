#!/bin/bash
# Copy the three env CSVs the AITA/FLIP/PAS Marlowe DPO runs need to the Marlowe
# project checkout. These files are gitignored (too large / third-party), so
# `git pull` on Marlowe will NOT bring them — run this from a host that can reach
# BOTH the Stanford filesystem (the source repo) AND Marlowe over ssh.
#
# resolve_csv() anchors each path to the repo root, so on Marlowe the CSVs must
# land at $MARLOWE_REPO/<same repo-relative path>. This script recreates that
# layout remotely and copies each file.
#
#   AITA  data/aita_clean.csv                                                (~175M)
#   FLIP  ELEPHANT_datasets/AITA-NTA-FLIP.csv                                (~4.9M)
#   PAS   osfstorage-archive/Data/PAS_results/endorse4_elephant_oeq_pas_claude.csv (~25M,
#         usually already on Marlowe from prior PAS runs — set SKIP_PAS=1 to omit)
#
# Usage:
#   MARLOWE_HOST=you@marlowe-login \
#   [MARLOWE_REPO=/projects/m000254/sycophancy] \
#   [SKIP_PAS=1] \
#   bash training_mitigations/slurm_scripts/scp_env_csvs_to_marlowe.sh
set -euo pipefail

MARLOWE_HOST="${MARLOWE_HOST:-stephhk@login-01.marlowe.stanford.edu}"
MARLOWE_REPO="${MARLOWE_REPO:-/projects/m000254/sycophancy}"

# Source repo root (this script lives two dirs under it).
SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CSVS=(
    "data/aita_clean.csv"
    "ELEPHANT_datasets/AITA-NTA-FLIP.csv"
)
[[ "${SKIP_PAS:-0}" == "1" ]] || CSVS+=(
    "osfstorage-archive/Data/PAS_results/endorse4_elephant_oeq_pas_claude.csv"
)

echo "[scp] source repo : $SRC_ROOT"
echo "[scp] marlowe     : $MARLOWE_HOST:$MARLOWE_REPO"
echo

for rel in "${CSVS[@]}"; do
    src="$SRC_ROOT/$rel"
    [[ -f "$src" ]] || { echo "ERROR: missing source $src" >&2; exit 1; }
    dst_dir="$MARLOWE_REPO/$(dirname "$rel")"
    echo "[scp] $rel  ($(du -h "$src" | cut -f1))"
    ssh "$MARLOWE_HOST" "mkdir -p '$dst_dir'"
    # scp preserves nothing fancy; for the 175M aita_clean a resumable transfer
    # is nicer — swap to: rsync -avhP "$src" "$MARLOWE_HOST:$dst_dir/"
    scp -p "$src" "$MARLOWE_HOST:$dst_dir/"
done

echo
echo "[scp] done. Verify on Marlowe:"
for rel in "${CSVS[@]}"; do echo "    ls -la $MARLOWE_REPO/$rel"; done
