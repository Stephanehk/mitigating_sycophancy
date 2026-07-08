#!/bin/bash
# Stage + score the held-out generations for OEQ/PAS runs with the GPT-5-mini
# action-endorsement judge -- the project's DEFAULT evaluation method for these
# two envs (training_mitigations/eval/run_endorse_gpt5mini.py).
#
# Network-bound, no GPU: run on a Marlowe LOGIN node (like marlowe_classify_all.sh).
#
#   bash training_mitigations/slurm_scripts/marlowe_endorse_judge_all.sh oeq_alpha_sweep
#   bash training_mitigations/slurm_scripts/marlowe_endorse_judge_all.sh <group>
#
# Two steps per run, mirroring run_judge_v5_heldout1000.sbatch:
#   1. stage_heldout_inputs.py  concat the 2 response shards -> sycoharm_runs/inputs/<name>.csv
#                               (exactly [sentence,response]; asserts 1000 unique ids)
#   2. run_endorse_gpt5mini.py  score every (sentence,response) with the endorse
#                               template, report action endorsement rate #1/(#0+#1)
#
# CRITICAL: --prompt-version v5. The argparse DEFAULT is v1; every established
# heldout1000 number in this project was produced with v5. Passing nothing would
# silently score with a different prompt and the results would not be comparable.
#
# Needs OPENROUTER_API_KEY (in ~/.bashrc on Marlowe; `source ~/.bashrc`).
set -euo pipefail

PROJECT=m000254
REPO=/projects/$PROJECT/sycophancy
PY=/projects/$PROJECT/envs/llm126/bin/python
RUNS_ROOT=/projects/$PROJECT/runs
cd "$REPO"

: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY (it is in ~/.bashrc on Marlowe; 'source ~/.bashrc')}"

GROUP="${1:-oeq_alpha_sweep}"
TAG=final_final1000_temp1.0_seed0     # the tag the gen job writes (--tag-extra final1000)

# Each entry: RUN_DIR_BASENAME  JUDGE_NAME (the sycoharm_runs/inputs/<name>.csv stem)
OEQ_ALPHA_SWEEP=(
    "dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_perspshift_gp                        oeq_P5perspshift_a0_final1000"
    "dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_gp                        oeq_P6perspshift_a0_final1000"
    "dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p25        oeq_P6perspshift_a025_final1000"
    "dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5         oeq_P6perspshift_a05_final1000"
    "dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_nocrit_gp_frozenstake                oeq_frozenstake_a0_final1000"
    "dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_nocrit_distill_gp_frozenstake_rpo0p25 oeq_frozenstake_a025_final1000"
    "dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_nocrit_distill_gp_frozenstake_rpo0p5  oeq_frozenstake_a05_final1000"
)
# The already-trained alpha=1.0 reference arms (the top of the dose-response curve).
OEQ_ALPHA1_REF=(
    "dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp                oeq_P6perspshift_a1_final1000"
    "dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_nocrit_distill_gp_frozenstake        oeq_frozenstake_a1_final1000"
    "dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_perspshift_distill_gp                oeq_P5perspshift_a1_final1000"
)

case "$GROUP" in
    oeq_alpha_sweep) JOBS=("${OEQ_ALPHA_SWEEP[@]}") ;;
    oeq_alpha1_ref)  JOBS=("${OEQ_ALPHA1_REF[@]}") ;;
    oeq_all)         JOBS=("${OEQ_ALPHA_SWEEP[@]}" "${OEQ_ALPHA1_REF[@]}") ;;
    *) echo "usage: $0 [oeq_alpha_sweep|oeq_alpha1_ref|oeq_all]" >&2; exit 2 ;;
esac

NAMES=()
for row in "${JOBS[@]}"; do
    read -r RD NAME <<< "$row"
    RUN="$RUNS_ROOT/$RD"
    [[ -d "$RUN/eval" ]] || { echo "ERROR: no eval/ in $RUN (generation not run?)" >&2; exit 1; }
    echo "[judge] staging $RD -> inputs/$NAME.csv"
    "$PY" training_mitigations/eval/stage_heldout_inputs.py \
        --run-dir "$RUN" --tag "$TAG" --name "$NAME" --expect 1000
    NAMES+=("$NAME")
done

echo "[judge] scoring ${#NAMES[@]} runs with the v5 gpt-5-mini endorse judge"
# max-cost scales with the number of runs (1000 rows each); the default 5.0 would
# abort partway through a multi-run group.
MAXCOST=$(python3 -c "print(max(5.0, 4.0*${#NAMES[@]}))")
"$PY" training_mitigations/eval/run_endorse_gpt5mini.py \
    --datasets "${NAMES[@]}" \
    --prompt-version v5 \
    --model openai/gpt-5-mini --reasoning minimal \
    --price-in 0.25 --price-out 2.00 --max-cost "$MAXCOST" \
    --workers 8 --checkpoint 100
echo "[judge] done -> sycoharm_runs/endorse_gpt5mini_v5/"
