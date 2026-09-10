#!/bin/bash
# LOGIN-NODE scoring for the FINAL OEQ/PAS sweep. NO GPU, NO sbatch — pure
# OpenRouter gpt-5-mini. For each (env, condition) whose 2 gen shards have landed:
#   (1) stage_heldout_inputs.py  -> sycoharm_runs/inputs/<name>.csv (sentence,response)
#   (2) run_endorse_gpt5mini.py --prompt-version v5 --model openai/gpt-5-mini
#       --reasoning minimal      -> endorse_gpt5mini_v5_temp0/scored_<name>.csv
#       (action-endorsement rate #1/(#0+#1) = the paper metric)
#   (3) OEQ only: classify_response_voice.py -> <run>/eval/voice_<tag>.csv (role-play census)
# Marlowe compute nodes are air-gapped, so run this on a LOGIN node after
# marlowe_final_gen_submit.sh's OEQ/PAS arrays finish. Idempotent: skips a
# condition whose scored_<name>.csv already exists.
#
#   export OPENROUTER_API_KEY=...   # in ~/.bashrc on Marlowe
#   bash training_mitigations/slurm_scripts/marlowe_final_endorse_all.sh [oeq|pas|all]
#
# SITE: defaults are Marlowe. The 4 neutral arms are trained+generated on `next`
# (their veto needs egress), so their run dirs never leave it; score them in
# place by overriding REPO/RUNS_ROOT/PY. Conditions whose shards are absent at
# the given site are skipped, so the same loop scores whichever arms are local:
#   REPO=/next/u/stephhk/sycophancy \
#   RUNS_ROOT=/next/u/stephhk/sycophancy/training_mitigations/runs \
#   PY=/next/u/stephhk/miniconda3/envs/llm/bin/python \
#     bash training_mitigations/slurm_scripts/marlowe_final_endorse_all.sh all
set -uo pipefail
PROJECT=m000254
REPO="${REPO:-/projects/$PROJECT/sycophancy}"
RUNS_ROOT="${RUNS_ROOT:-/projects/$PROJECT/runs}"
PY="${PY:-/projects/$PROJECT/envs/llm126/bin/python}"
SCORED_DIR=$REPO/training_mitigations/sycoharm_runs/endorse_gpt5mini_v5_temp0
cd "$REPO"
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY (source ~/.bashrc on Marlowe)}"

GROUP="${1:-all}"

# ENV  RUNDIR(rel $RUNS_ROOT)  TAG  NAME  DOVOICE(1=OEQ voice census,0=PAS)
OEQ_JOBS=(
  "OEQ baseline_final_OEQ base_finaldata_temp1.0_seed0            oeq_base_finaldata            1"
  "OEQ baseline_final_OEQ base-becritical_finaldata_temp1.0_seed0 oeq_becritical_finaldata     1"
  "OEQ baseline_final_OEQ base-explicit_finaldata_temp1.0_seed0   oeq_explicit_finaldata       1"
  "OEQ baseline_final_OEQ base-sharma_finaldata_temp1.0_seed0     oeq_sharma_finaldata         1"
  "OEQ baseline_final_OEQ base-dontbesyco_finaldata_temp1.0_seed0 oeq_dontbesyco_finaldata     1"
  "OEQ baseline_final_OEQ base-perspshiftassess_finaldata_temp1.0_seed0 oeq_perspshiftassess_finaldata 1"
  "OEQ baseline_final_OEQ base-stakeholder_finaldata_temp1.0_seed0 oeq_stakeholder_finaldata    1"
  "OEQ baseline_final_OEQ base-besycophantic_finaldata_temp1.0_seed0 oeq_besycophantic_finaldata 1"
  "OEQ dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final final_finaldata_temp1.0_seed0 oeq_P6perspshift_a05_finaldata 1"
  "OEQ dpo_lora_oeq_qwen3-8b_P6_perspshift_rpo0p5_final_voicedpo final_finaldata_temp1.0_seed0 oeq_voicedpo_finaldata 1"
  "OEQ dpo_lora_oeq_qwen3-8b_P6_perspshift_rpo0p5_final_voicedpo_b05 final_finaldata_temp1.0_seed0 oeq_voicedpo_b05_finaldata 1"
  "OEQ dpo_lora_oeq_qwen3-8b_n1000_neutral_final       final_finaldata_temp1.0_seed0 oeq_neutral_finaldata      1"
  "OEQ dpo_lora_oeq_qwen3-8b_n1000_neutral_mixed_final final_finaldata_temp1.0_seed0 oeq_neutralmixed_finaldata 1"
  "OEQ sft_lora_oeq_qwen3-8b_becritical_final          final_finaldata_temp1.0_seed0 oeq_sft_finaldata          1"
)
PAS_JOBS=(
  "PAS baseline_final_PAS base_finaldata_temp1.0_seed0            pas_base_finaldata            0"
  "PAS baseline_final_PAS base-becritical_finaldata_temp1.0_seed0 pas_becritical_finaldata     0"
  "PAS baseline_final_PAS base-explicit_finaldata_temp1.0_seed0   pas_explicit_finaldata       0"
  "PAS baseline_final_PAS base-sharma_finaldata_temp1.0_seed0     pas_sharma_finaldata         0"
  "PAS baseline_final_PAS base-dontbesyco_finaldata_temp1.0_seed0 pas_dontbesyco_finaldata     0"
  "PAS baseline_final_PAS base-perspshiftassess_finaldata_temp1.0_seed0 pas_perspshiftassess_finaldata 0"
  "PAS baseline_final_PAS base-stakeholder_finaldata_temp1.0_seed0 pas_stakeholder_finaldata    0"
  "PAS baseline_final_PAS base-besycophantic_finaldata_temp1.0_seed0 pas_besycophantic_finaldata 0"
  "PAS dpo_lora_pas_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final final_finaldata_temp1.0_seed0 pas_P6perspshift_a05_finaldata 0"
  "PAS dpo_lora_pas_qwen3-8b_P6_perspshift_rpo0p5_final_voicedpo final_finaldata_temp1.0_seed0 pas_voicedpo_finaldata 0"
  "PAS dpo_lora_pas_qwen3-8b_P6_perspshift_rpo0p5_final_voicedpo_b05 final_finaldata_temp1.0_seed0 pas_voicedpo_b05_finaldata 0"
  "PAS dpo_lora_pas_qwen3-8b_n1000_neutral_final       final_finaldata_temp1.0_seed0 pas_neutral_finaldata      0"
  "PAS dpo_lora_pas_qwen3-8b_n1000_neutral_mixed_final final_finaldata_temp1.0_seed0 pas_neutralmixed_finaldata 0"
  "PAS sft_lora_pas_qwen3-8b_becritical_final          final_finaldata_temp1.0_seed0 pas_sft_finaldata          0"
)
case "$GROUP" in
  oeq) JOBS=("${OEQ_JOBS[@]}") ;;
  pas) JOBS=("${PAS_JOBS[@]}") ;;
  all) JOBS=("${OEQ_JOBS[@]}" "${PAS_JOBS[@]}") ;;
  *) echo "usage: $0 [oeq|pas|all]" >&2; exit 2 ;;
esac

for row in "${JOBS[@]}"; do
  read -r ENV RUN TAG NAME DOV <<< "$row"
  rundir="$RUNS_ROOT/$RUN"
  echo "=================================================================="
  if [[ -f "$SCORED_DIR/scored_$NAME.csv" ]]; then
    echo "[endorse] $NAME already scored — skipping."
    continue
  fi
  s0="$rundir/eval/responses_${TAG}_shard0of2.csv"
  s1="$rundir/eval/responses_${TAG}_shard1of2.csv"
  if [[ ! -f "$s0" || ! -f "$s1" ]]; then
    echo "[endorse] $NAME: gen shards not ready ($rundir) — skipping for now."
    continue
  fi
  echo "[endorse] env=$ENV name=$NAME tag=$TAG run=$RUN"
  "$PY" training_mitigations/eval/stage_heldout_inputs.py \
      --run-dir "$rundir" --tag "$TAG" --name "$NAME" --num-shards 2 --expect 1000 \
    || { echo "[endorse] STAGE FAILED $NAME"; continue; }
  "$PY" training_mitigations/eval/run_endorse_gpt5mini.py \
      --datasets "$NAME" --prompt-version v5 \
      --model openai/gpt-5-mini --reasoning minimal \
      --workers 16 --max-cost 6 \
    || { echo "[endorse] ENDORSE FAILED $NAME"; continue; }
  if [[ "$DOV" == 1 ]]; then
    "$PY" training_mitigations/eval/classify_response_voice.py \
        --run-dir "$rundir" --env OEQ --tag "$TAG" --num-shards 2 \
        --model openai/gpt-5-mini --workers 16 \
      || echo "[endorse] WARN voice census failed $NAME (endorse still valid)"
  fi
  echo "[endorse] done $NAME -> $SCORED_DIR/scored_$NAME.csv"
done
echo "[endorse] group=$GROUP complete. Scored CSVs -> $SCORED_DIR/scored_*.csv"
