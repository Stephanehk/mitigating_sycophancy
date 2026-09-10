#!/bin/bash
# LOGIN-NODE verdict + metrics for the FINAL AITA / AITA-NTA-FLIP sweep. NO GPU,
# NO sbatch — pure OpenRouter gpt-5-mini. Runs classify_aita_verdicts.py over each
# (env, condition) produced by marlowe_final_gen_submit.sh (num-shards 1): reads
# aita_responses_<tag>_shard0of1.csv, classifies each response's PROSE into
# YTA/NTA/UNCLEAR with gpt-5-mini, writes aita_scored_<tag>_shard0of1.csv, and
# prints FNR/FPR (AITA) or both-NTA sycophancy (FLIP), UNCLEAR/OTHER dropped from
# the denominators. Marlowe compute nodes are air-gapped -> run on a LOGIN node
# after the AITA/FLIP gen tasks finish. Re-runnable (classify checkpoints per shard).
#
#   export OPENROUTER_API_KEY=...   # in ~/.bashrc on Marlowe
#   bash training_mitigations/slurm_scripts/marlowe_final_classify_all.sh [aita|flip|all]
set -uo pipefail
PROJECT=m000254
REPO=/projects/$PROJECT/sycophancy
RUNS_ROOT=/projects/$PROJECT/runs
PY=/projects/$PROJECT/envs/llm126/bin/python
CLS=training_mitigations/eval/classify_aita_verdicts.py
cd "$REPO"
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY (source ~/.bashrc on Marlowe)}"

GROUP="${1:-all}"

# ENV  RUNDIR(rel $RUNS_ROOT)  TAG  NSHARDS
#
# NSHARDS is PER-CONDITION, not per-env: it must match the shard regime the gen
# job actually used, because that is what the filenames encode
# (aita_responses_<tag>_shard{i}of{N}.csv). marlowe_final_gen_submit.sh emits
# AITA=1 / FLIP=2, so those are the defaults here — EXCEPT the FLIP P6 arm, whose
# generation ran on `next` (next_final_gen_flip_P6_perspshift.sbatch) at 5 shards
# to fit the 2000-generation @1536-tok job onto next6's a4000 pairs. Its files are
# therefore *of5, and hardcoding FLIP=2 would make this script report it as "gen
# not ready" forever.
AITA_JOBS=(
  "AITA baseline_final_AITA base_finaldata_temp1.0_seed0 1"
  "AITA baseline_final_AITA base-becritical_finaldata_temp1.0_seed0 1"
  "AITA baseline_final_AITA base-explicit_finaldata_temp1.0_seed0 1"
  "AITA baseline_final_AITA base-sharma_finaldata_temp1.0_seed0 1"
  "AITA baseline_final_AITA base-dontbesyco_finaldata_temp1.0_seed0 1"
  "AITA baseline_final_AITA base-perspshiftassess_finaldata_temp1.0_seed0 1"
  "AITA baseline_final_AITA base-stakeholder_finaldata_temp1.0_seed0 1"
  "AITA baseline_final_AITA base-besycophantic_finaldata_temp1.0_seed0 1"
  "AITA dpo_lora_aita_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final final_finaldata_temp1.0_seed0 1"
)
FLIP_JOBS=(
  "AITA-NTA-FLIP baseline_final_AITA-NTA-FLIP base_finaldata_temp1.0_seed0 2"
  "AITA-NTA-FLIP baseline_final_AITA-NTA-FLIP base-becritical_finaldata_temp1.0_seed0 2"
  "AITA-NTA-FLIP baseline_final_AITA-NTA-FLIP base-explicit_finaldata_temp1.0_seed0 2"
  "AITA-NTA-FLIP baseline_final_AITA-NTA-FLIP base-sharma_finaldata_temp1.0_seed0 2"
  "AITA-NTA-FLIP baseline_final_AITA-NTA-FLIP base-dontbesyco_finaldata_temp1.0_seed0 2"
  "AITA-NTA-FLIP baseline_final_AITA-NTA-FLIP base-perspshiftassess_finaldata_temp1.0_seed0 2"
  "AITA-NTA-FLIP baseline_final_AITA-NTA-FLIP base-stakeholder_finaldata_temp1.0_seed0 2"
  "AITA-NTA-FLIP dpo_lora_flip_qwen3-8b_n500_mixed_P6_perspshift_distill_gp_rpo0p5_final final_finaldata_temp1.0_seed0 5"
)
case "$GROUP" in
  aita) JOBS=("${AITA_JOBS[@]}") ;;
  flip) JOBS=("${FLIP_JOBS[@]}") ;;
  all)  JOBS=("${AITA_JOBS[@]}" "${FLIP_JOBS[@]}") ;;
  *) echo "usage: $0 [aita|flip|all]" >&2; exit 2 ;;
esac

for row in "${JOBS[@]}"; do
  read -r ENV RUN TAG N <<< "$row"
  rundir="$RUNS_ROOT/$RUN"
  case "$ENV" in AITA|AITA-NTA-FLIP) ;; *) echo "bad env $ENV"; continue ;; esac
  [[ "$N" =~ ^[1-9][0-9]*$ ]] || { echo "bad NSHARDS '$N' for $ENV $TAG"; continue; }
  echo "=================================================================="
  ready=1
  for ((s=0; s<N; s++)); do
    [[ -f "$rundir/eval/aita_responses_${TAG}_shard${s}of${N}.csv" ]] || ready=0
  done
  if (( ! ready )); then
    echo "[classify] $ENV $TAG: gen not ready (need $N shard(s) in $rundir/eval) — skipping for now."
    continue
  fi
  echo "[classify] env=$ENV tag=$TAG run=$RUN num_shards=$N"
  "$PY" "$CLS" --run-dir "$rundir" --env "$ENV" --tag "$TAG" --num-shards "$N"
done
echo "[classify] group=$GROUP complete. Scored -> <run>/eval/aita_scored_<tag>_shard{i}of{N}.csv"
