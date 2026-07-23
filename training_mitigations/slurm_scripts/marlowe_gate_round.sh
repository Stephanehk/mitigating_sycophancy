#!/usr/bin/env bash
# One round of the assistant-mode gate for a single method, plus the driver loop
# over rounds. See eval/gate_common.py for the full pipeline rationale.
#
# A round is TWO HOPS because generation and judging cannot run on the same host:
# Marlowe compute nodes are air-gapped (HF_HUB_OFFLINE=1, no outbound network)
# while the gate judge calls gpt-5-mini over OpenRouter. So this script runs the
# NETWORKED half (judge + worklist + assemble) and prints the exact offline
# command to run on Marlowe between rounds. It deliberately does not submit the
# regeneration sbatch itself.
#
#   ./marlowe_gate_round.sh ENV RUN_DIR TAG [MAX_ATTEMPTS]
#
# Idempotent: each stage skips when its output exists, so a re-invocation after
# an interrupted round picks up where it stopped.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_marlowe_eval_env.sh"
EVAL="$REPO/training_mitigations/eval"

ENV_NAME="${1:?usage: marlowe_gate_round.sh ENV RUN_DIR TAG [MAX_ATTEMPTS]}"
RUN_DIR="${2:?}"
TAG="${3:?}"
MAX_ATTEMPTS="${4:-5}"

: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY must be set — the gate judge needs it}"

GATE_DIR="$RUN_DIR/eval/gate"

# Which generation driver a regeneration round would use, mirroring the dispatch
# in marlowe_final_gen.sbatch:54-71.
case "$ENV_NAME" in
  OEQ|PAS)            DRIVER="eval_endorsement_heldout.py"; GEN_MAX_TOK=512  ;;
  AITA|AITA-NTA-FLIP) DRIVER="eval_aita_label_heldout.py";   GEN_MAX_TOK=1536 ;;
  *) echo "unknown env: $ENV_NAME" >&2; exit 1 ;;
esac

for (( A=1; A<=MAX_ATTEMPTS; A++ )); do
  echo "=== $ENV_NAME / $TAG — attempt $A ==="

  # Attempts >= 2 need their responses generated offline on Marlowe first.
  if (( A >= 2 )); then
    RESP="$GATE_DIR/$( [[ "$ENV_NAME" == AITA* ]] && echo aita_responses || echo responses )_${TAG}_gateattempt${A}.csv"
    WL="$GATE_DIR/resample_${TAG}_a${A}.json"
    if [[ ! -f "$WL" ]]; then
      echo "[round] no worklist for attempt $A — converged or exhausted. Stopping."
      break
    fi
    if [[ ! -f "$RESP" ]]; then
      cat <<EOF

[round] STOP — attempt $A responses are missing. Run this ON MARLOWE (offline),
        then re-invoke this script to continue:

  \$PY $EVAL/$DRIVER \\
      --run-dir $RUN_DIR --env $ENV_NAME \\
      --use-final-data --final-split test --tag-extra finaldata \\
      --device cuda:0 --seed 0 --num-shards 1 --shard-id 0 \\
      --sample-max-tokens $GEN_MAX_TOK --no-judge \\
      --attempt $A --resample-ids-file $WL \\
      <THE SAME ARM FLAGS THIS TAG WAS GENERATED WITH>

        The arm flags (--base/--adapter, --perspective-shift, --assess-narrative,
        --be-critical, --gen-prefix/--gen-suffix/--gen-tag, ...) must match the
        JOBS entry for this tag in marlowe_final_gen_submit.sh:58-82 exactly —
        a resample drawn from a different arm is not a resample.

EOF
      exit 2
    fi
  fi

  "$PY" "$EVAL/judge_assistant_mode.py" \
      --run-dir "$RUN_DIR" --env "$ENV_NAME" --tag "$TAG" --attempt "$A"

  "$PY" "$EVAL/gate_worklist.py" \
      --run-dir "$RUN_DIR" --env "$ENV_NAME" --tag "$TAG" \
      --completed-attempt "$A" --max-attempts "$MAX_ATTEMPTS"

  # gate_worklist writes no file when it has converged or exhausted; either way
  # there is nothing left to regenerate, so assemble now.
  if [[ ! -f "$GATE_DIR/resample_${TAG}_a$((A+1)).json" ]]; then
    echo "[round] no further worklist — assembling."
    break
  fi
done

"$PY" "$EVAL/assemble_gate.py" \
    --run-dir "$RUN_DIR" --env "$ENV_NAME" --tag "$TAG" \
    --max-attempts "$MAX_ATTEMPTS"
