#!/bin/bash
# Login-node verdict + metrics stage for the Marlowe AITA/FLIP eval sweep.
# Runs classify_aita_verdicts.py (gpt-5-mini via OpenRouter) over every
# (env, run-dir, tag) produced by the generation arrays:
#   marlowe_run_eval_aita_flip_baselines.sbatch     -> base / base-becritical
#   marlowe_run_eval_aita_promptbaselines.sbatch    -> base-explicit / -sharma /
#                                                       -dontbesyco / -perspshiftassess
#   marlowe_run_eval_aita_trained.sbatch            -> final (frozenstake, P6)
#
# NOT an sbatch job: this stage is network-bound (no GPU) and Marlowe COMPUTE
# nodes are typically air-gapped, whereas the LOGIN node reaches openrouter.ai.
# Run it directly on a login node after the generation arrays finish:
#   export OPENROUTER_API_KEY=...   # already in ~/.bashrc on Marlowe
#   bash training_mitigations/slurm_scripts/marlowe_classify_all.sh [baselines|trained|all]
#
# It writes aita_scored_<tag>_shard{i}of{N}.csv next to the response shards and
# prints FNR/FPR (AITA) or both-NTA sycophancy (FLIP), UNCLEAR/OTHER dropped from
# the denominators. Re-runnable: classify_aita_verdicts.py checkpoints per shard.

set -uo pipefail
PROJECT=m000254
REPO=/projects/$PROJECT/sycophancy
PY=/projects/$PROJECT/envs/llm126/bin/python
CLS=training_mitigations/eval/classify_aita_verdicts.py
cd "$REPO"

: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY (it is in ~/.bashrc on Marlowe; 'source ~/.bashrc')}"

GROUP="${1:-all}"
TS=_temp1.0_seed0                          # decode tag suffix all gens carry
BASE_AITA=training_mitigations/runs/baseline_AITA
BASE_FLIP=training_mitigations/runs/baseline_AITA-NTA-FLIP
RUNS_ROOT=/projects/$PROJECT/runs

# Each entry: ENV  RUN_DIR  TAG_STEM  NUM_SHARDS. One unsharded shard each
# (--num-shards 1), matching the single-task-per-condition generation.
BASELINE_JOBS=(
    "AITA          $BASE_AITA base                 1"
    "AITA          $BASE_AITA base-becritical      1"
    "AITA          $BASE_AITA base-explicit        1"
    "AITA          $BASE_AITA base-sharma          1"
    "AITA          $BASE_AITA base-dontbesyco      1"
    "AITA          $BASE_AITA base-perspshiftassess 1"
    "AITA-NTA-FLIP $BASE_FLIP base                 1"
    "AITA-NTA-FLIP $BASE_FLIP base-becritical      1"
    "AITA-NTA-FLIP $BASE_FLIP base-explicit        1"
    "AITA-NTA-FLIP $BASE_FLIP base-sharma          1"
    "AITA-NTA-FLIP $BASE_FLIP base-dontbesyco      1"
    "AITA-NTA-FLIP $BASE_FLIP base-perspshiftassess 1"
)
TRAINED_JOBS=(
    "AITA          $RUNS_ROOT/dpo_lora_aita_qwen3-8b_n1000_mixed_P5_nocrit_distill_gp_frozenstake final 1"
    "AITA-NTA-FLIP $RUNS_ROOT/dpo_lora_flip_qwen3-8b_n500_mixed_P5_nocrit_distill_gp_frozenstake  final 1"
    "AITA          $RUNS_ROOT/dpo_lora_aita_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp         final 1"
    "AITA-NTA-FLIP $RUNS_ROOT/dpo_lora_flip_qwen3-8b_n500_mixed_P6_perspshift_distill_gp          final 1"
)

case "$GROUP" in
    baselines) JOBS=("${BASELINE_JOBS[@]}") ;;
    trained)   JOBS=("${TRAINED_JOBS[@]}") ;;
    all)       JOBS=("${BASELINE_JOBS[@]}" "${TRAINED_JOBS[@]}") ;;
    *) echo "usage: $0 [baselines|trained|all]" >&2; exit 2 ;;
esac

for row in "${JOBS[@]}"; do
    read -r ENV RUN STEM N <<< "$row"
    TAG="${STEM}${TS}"
    echo "=================================================================="
    echo "[classify] env=$ENV tag=$TAG shards=$N run=$RUN"
    "$PY" "$CLS" --run-dir "$RUN" --env "$ENV" --tag "$TAG" --num-shards "$N"
done
echo "[classify] all requested groups done."
