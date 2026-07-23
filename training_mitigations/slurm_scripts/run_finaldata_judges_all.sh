#!/bin/bash
# LOGIN-NODE driver for the 7 finaldata judges over OEQ/PAS/AITA/AITA-NTA-FLIP.
# NO GPU, NO sbatch: Marlowe compute nodes are air-gapped and every judge here is
# a pure OpenRouter gpt-5-mini call, so this MUST run on a login node.
#
#   6 per-response judges (framing, validation, addressee, 3x empathy)
#       -> run_finaldata_judges.py, one process per (env, metric, condition)
#   1 voice-collapse judge
#       -> classify_response_voice.py per (env, run, tag, nshards), plus
#          voice_human_finaldata.py for the run-dir-less OEQ human baseline.
#
# PARALLELISM: judge_common.build_judge mutates globals on the shared paper judge
# module, so judges must never share a process. Parallelism is therefore across
# PROCESSES (xargs -P), each internally threaded by --workers. Total in-flight
# requests ~= JOBS * WORKERS; keep that modest so we neither hammer the shared
# login node nor trip OpenRouter rate limits.
#
# Resumable: every underlying judge skips already-complete outputs, so re-running
# after an interrupt only does the remaining work.
#
#   export OPENROUTER_API_KEY=...   # in ~/.bashrc on Marlowe
#   bash training_mitigations/slurm_scripts/run_finaldata_judges_all.sh [six|voice|all]
set -uo pipefail
PROJECT=m000254
REPO=/projects/$PROJECT/sycophancy
RUNS_ROOT=/projects/$PROJECT/runs
PY=/projects/$PROJECT/envs/llm126/bin/python
LOGS=$REPO/training_mitigations/logs/judges_finaldata
cd "$REPO"
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY (source ~/.bashrc on Marlowe)}"

GROUP="${1:-all}"
JOBS="${JOBS:-8}"        # concurrent processes
WORKERS="${WORKERS:-8}"  # threads inside each process
mkdir -p "$LOGS"

# The 3 Lend-an-Ear empathy judges were RETIRED by request and are deliberately
# not listed: they are no longer run and no longer reported. Their existing OEQ
# outputs stay on disk under judges_finaldata/OEQ/ for reference, and the 10
# PAS empathy_validating_emotions files quarantined after the 402 incident
# therefore no longer need re-judging. Re-add the metric names here to revive them.
METRICS=(framing validation addressee)
OEQ_C=(base explicit sharma dontbesyco becritical perspshiftassess sft
       neutralmixed neutral P6perspshift_a05 voicedpo voicedpo_b05 human)
PAS_C=(base explicit sharma dontbesyco becritical perspshiftassess sft
       neutralmixed neutral P6perspshift_a05 voicedpo voicedpo_b05)
AITA_C=(base explicit sharma dontbesyco becritical perspshiftassess P6perspshift_a05)
FLIP_C=("${AITA_C[@]}")

run_six() {
  # Emit one job line per (env, metric, condition), then fan out with xargs.
  # Finer granularity than per-env keeps the pool balanced (FLIP jobs are 2x
  # the rows of the rest).
  local joblist="$LOGS/joblist.txt"
  : > "$joblist"
  for env in OEQ PAS AITA FLIP; do
    case "$env" in
      OEQ)  conds=("${OEQ_C[@]}") ;;
      PAS)  conds=("${PAS_C[@]}") ;;
      AITA) conds=("${AITA_C[@]}") ;;
      FLIP) conds=("${FLIP_C[@]}") ;;
    esac
    for m in "${METRICS[@]}"; do
      for c in "${conds[@]}"; do
        echo "$env $m $c" >> "$joblist"
      done
    done
  done
  echo "[six] $(wc -l < "$joblist") jobs, ${JOBS} at a time x ${WORKERS} workers"
  xargs -a "$joblist" -P "$JOBS" -L 1 bash -c '
    env=$0; m=$1; c=$2
    log="'"$LOGS"'/${env}_${m}_${c}.log"
    "'"$PY"'" training_mitigations/eval/run_finaldata_judges.py \
        --env "$env" --metrics "$m" --conditions "$c" \
        --workers '"$WORKERS"' --max-cost-per-run 4 > "$log" 2>&1 \
      || echo "[six] FAILED $env/$m/$c (see $log)"
  '
  echo "[six] complete -> training_mitigations/sycoharm_runs/judges_finaldata/"
}

# ENV RUNDIR(rel RUNS_ROOT) TAG NSHARDS -- NSHARDS is per-condition and must match
# the shard regime the gen job used (OEQ/PAS=2, AITA=1, FLIP=2 EXCEPT the P6 arm,
# which generated on `next` at 5). classify_response_voice.py skips a condition
# whose voice_<tag>.csv already exists, so listing every condition is safe: the
# 10 OEQ / 3 PAS / 2 AITA already done are reported, not re-judged.
VOICE_JOBS=(
  "OEQ baseline_final_OEQ base_finaldata_temp1.0_seed0 2"
  "OEQ baseline_final_OEQ base-explicit_finaldata_temp1.0_seed0 2"
  "OEQ baseline_final_OEQ base-sharma_finaldata_temp1.0_seed0 2"
  "OEQ baseline_final_OEQ base-dontbesyco_finaldata_temp1.0_seed0 2"
  "OEQ baseline_final_OEQ base-becritical_finaldata_temp1.0_seed0 2"
  "OEQ baseline_final_OEQ base-perspshiftassess_finaldata_temp1.0_seed0 2"
  "OEQ sft_lora_oeq_qwen3-8b_becritical_final final_finaldata_temp1.0_seed0 2"
  "OEQ dpo_lora_oeq_qwen3-8b_n1000_neutral_mixed_final final_finaldata_temp1.0_seed0 2"
  "OEQ dpo_lora_oeq_qwen3-8b_n1000_neutral_final final_finaldata_temp1.0_seed0 2"
  "OEQ dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final final_finaldata_temp1.0_seed0 2"
  "OEQ dpo_lora_oeq_qwen3-8b_P6_perspshift_rpo0p5_final_voicedpo final_finaldata_temp1.0_seed0 2"
  "OEQ dpo_lora_oeq_qwen3-8b_P6_perspshift_rpo0p5_final_voicedpo_b05 final_finaldata_temp1.0_seed0 2"
  "PAS baseline_final_PAS base_finaldata_temp1.0_seed0 2"
  "PAS baseline_final_PAS base-explicit_finaldata_temp1.0_seed0 2"
  "PAS baseline_final_PAS base-sharma_finaldata_temp1.0_seed0 2"
  "PAS baseline_final_PAS base-dontbesyco_finaldata_temp1.0_seed0 2"
  "PAS baseline_final_PAS base-becritical_finaldata_temp1.0_seed0 2"
  "PAS baseline_final_PAS base-perspshiftassess_finaldata_temp1.0_seed0 2"
  "PAS sft_lora_pas_qwen3-8b_becritical_final final_finaldata_temp1.0_seed0 2"
  "PAS dpo_lora_pas_qwen3-8b_n1000_neutral_mixed_final final_finaldata_temp1.0_seed0 2"
  "PAS dpo_lora_pas_qwen3-8b_n1000_neutral_final final_finaldata_temp1.0_seed0 2"
  "PAS dpo_lora_pas_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final final_finaldata_temp1.0_seed0 2"
  "PAS dpo_lora_pas_qwen3-8b_P6_perspshift_rpo0p5_final_voicedpo final_finaldata_temp1.0_seed0 2"
  "PAS dpo_lora_pas_qwen3-8b_P6_perspshift_rpo0p5_final_voicedpo_b05 final_finaldata_temp1.0_seed0 2"
  "AITA baseline_final_AITA base_finaldata_temp1.0_seed0 1"
  "AITA baseline_final_AITA base-explicit_finaldata_temp1.0_seed0 1"
  "AITA baseline_final_AITA base-sharma_finaldata_temp1.0_seed0 1"
  "AITA baseline_final_AITA base-dontbesyco_finaldata_temp1.0_seed0 1"
  "AITA baseline_final_AITA base-becritical_finaldata_temp1.0_seed0 1"
  "AITA baseline_final_AITA base-perspshiftassess_finaldata_temp1.0_seed0 1"
  "AITA dpo_lora_aita_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final final_finaldata_temp1.0_seed0 1"
  "AITA-NTA-FLIP baseline_final_AITA-NTA-FLIP base_finaldata_temp1.0_seed0 2"
  "AITA-NTA-FLIP baseline_final_AITA-NTA-FLIP base-explicit_finaldata_temp1.0_seed0 2"
  "AITA-NTA-FLIP baseline_final_AITA-NTA-FLIP base-sharma_finaldata_temp1.0_seed0 2"
  "AITA-NTA-FLIP baseline_final_AITA-NTA-FLIP base-dontbesyco_finaldata_temp1.0_seed0 2"
  "AITA-NTA-FLIP baseline_final_AITA-NTA-FLIP base-becritical_finaldata_temp1.0_seed0 2"
  "AITA-NTA-FLIP baseline_final_AITA-NTA-FLIP base-perspshiftassess_finaldata_temp1.0_seed0 2"
  "AITA-NTA-FLIP dpo_lora_flip_qwen3-8b_n500_mixed_P6_perspshift_distill_gp_rpo0p5_final final_finaldata_temp1.0_seed0 5"
)

run_voice() {
  local joblist="$LOGS/voicelist.txt"
  : > "$joblist"
  for row in "${VOICE_JOBS[@]}"; do echo "$row" >> "$joblist"; done
  echo "[voice] $(wc -l < "$joblist") conditions, ${JOBS} at a time"
  xargs -a "$joblist" -P "$JOBS" -L 1 bash -c '
    env=$0; run=$1; tag=$2; n=$3
    log="'"$LOGS"'/voice_${env}_${run}_${tag}.log"
    "'"$PY"'" training_mitigations/eval/classify_response_voice.py \
        --run-dir "'"$RUNS_ROOT"'/$run" --env "$env" --tag "$tag" \
        --num-shards "$n" --workers '"$WORKERS"' > "$log" 2>&1 \
      || echo "[voice] FAILED $env/$run/$tag (see $log)"
  '
  # OEQ human reference has no run dir -> staged-frame shim.
  "$PY" training_mitigations/eval/voice_human_finaldata.py \
      > "$LOGS/voice_OEQ_human.log" 2>&1 \
    || echo "[voice] FAILED OEQ/human (see $LOGS/voice_OEQ_human.log)"
  echo "[voice] complete"
}

case "$GROUP" in
  six)   run_six ;;
  voice) run_voice ;;
  all)   run_six; run_voice ;;
  *) echo "usage: $0 [six|voice|all]" >&2; exit 2 ;;
esac
echo "[judges] group=$GROUP done."
