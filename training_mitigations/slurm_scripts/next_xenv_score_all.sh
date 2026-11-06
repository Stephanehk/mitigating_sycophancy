#!/bin/bash
# Score the 12 vLLM cross-dataset conditions (rebuttal, 2026-11-06). API only
# (OpenRouter gpt-5-mini), no GPU -- run on the next LOGIN node, which has egress.
#
# Per condition, with the SAME judges/flags as the in-domain finaldata arms:
#   1. assistant-mode gate, ATTEMPT 1 ONLY (judge_assistant_mode.py). No
#      gate-loop resample: rejected responses are not regenerated; the table
#      reports metrics ungated and over attempt-1 passers, plus pass@1.
#   2. OEQ/PAS: stage_heldout_inputs.py -> run_endorse_gpt5mini.py --prompt-version
#      v5 --reasoning minimal (temp 0) -> sycoharm_runs/endorse_gpt5mini_v5_temp0/
#      scored_xenvvllm_<train>_on_<eval>.csv   (= marlowe_final_endorse_all.sh)
#      AITA/FLIP: classify_aita_verdicts.py -> eval/aita_scored_<tag>_shard*.csv
#      (= marlowe_final_classify_all.sh)
# Every stage skips/resumes on existing output, so a rerun is safe.
#
# Usage: bash next_xenv_score_all.sh [CONDITION_NAME ...]   (default: all 12)
#        PAR=4 sets how many conditions run concurrently.
set -euo pipefail
REPO=/next/u/stephhk/sycophancy
PY=/next/u/stephhk/miniconda3/envs/llm/bin/python
RUNS=$REPO/training_mitigations/runs
EV=$REPO/training_mitigations/eval
LOGS=$REPO/training_mitigations/logs
TAG=final_finaldata_temp1.0_seed0
PAR="${PAR:-4}"
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY must be set}"
export PYTHONNOUSERSITE=1
cd "$REPO"; mkdir -p "$LOGS"

score_one() {
  local name=$1 dir="$RUNS/xenv_vllm_qwen3-8b_${1#xenvvllm_}"
  local ev=${name##*_on_} env n
  case "$ev" in
    oeq) env=OEQ; n=8 ;;  pas) env=PAS; n=8 ;;
    aita) env=AITA; n=10 ;;  flip) env=AITA-NTA-FLIP; n=20 ;;
  esac
  [[ -d "$dir" ]] || { echo "missing $dir" >&2; return 1; }
  echo "[score] $name env=$env shards=$n dir=$dir"
  "$PY" "$EV/judge_assistant_mode.py" --run-dir "$dir" --env "$env" --tag "$TAG" \
      --attempt 1 --expected-shards "$n" --workers 16
  if [[ "$env" == OEQ || "$env" == PAS ]]; then
    "$PY" "$EV/stage_heldout_inputs.py" --run-dir "$dir" --tag "$TAG" --name "$name" \
        --num-shards "$n" --expect 1000
    "$PY" "$EV/run_endorse_gpt5mini.py" --datasets "$name" --prompt-version v5 \
        --model openai/gpt-5-mini --reasoning minimal --workers 16 --max-cost 6
  else
    "$PY" "$EV/classify_aita_verdicts.py" --run-dir "$dir" --env "$env" --tag "$TAG" \
        --num-shards "$n" --workers 16 --max-cost 10
  fi
  echo "[score] done $name"
}
export -f score_one
export REPO PY RUNS EV TAG

if (( $# )); then NAMES=("$@"); else
  NAMES=(); for tr in oeq pas aita flip; do for ev in oeq pas aita flip; do
    [[ $tr == "$ev" ]] || NAMES+=("xenvvllm_${tr}_on_${ev}"); done; done
fi
printf '%s\n' "${NAMES[@]}" | xargs -P "$PAR" -I{} bash -c \
  'score_one {} > "'"$LOGS"'/xenv_score_{}.log" 2>&1 && echo "OK {}" || echo "FAILED {} (see '"$LOGS"'/xenv_score_{}.log)"'
