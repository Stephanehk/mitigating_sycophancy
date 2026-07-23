#!/bin/bash
# LOGIN-NODE judge watcher — NO GPU, NO sbatch. Pure OpenRouter gpt-5-mini API.
# For each target arm, once its held-out gen shards land, run:
#   (1) the action-endorsement judge we always use: stage_heldout_inputs.py ->
#       run_endorse_gpt5mini.py --prompt-version v5 --model openai/gpt-5-mini
#       --reasoning minimal  -> sycoharm_runs/endorse_gpt5mini_v5_temp0/scored_<name>.csv
#   (2) the voice judge (OEQ arms only): classify_response_voice.py (gpt-5-mini)
#       -> <run>/eval/voice_<tag>.csv
# Then print the endorsement rate (#1/(#0+#1)) next to the role-play (STAKEHOLDER)
# rate. Idempotent: skips an arm whose scored_<name>.csv already exists.
#
# Launch detached (login node):
#   source ~/.bashrc                       # OPENROUTER_API_KEY
#   setsid nohup bash training_mitigations/slurm_scripts/marlowe_judge_watcher_p5.sh \
#       > /projects/m000254/logs/judge_watcher_p5.log 2>&1 &
set -uo pipefail

PROJECT=m000254
REPO=/projects/$PROJECT/sycophancy
RUNS_ROOT=/projects/$PROJECT/runs
PY=/projects/$PROJECT/envs/llm126/bin/python
SCORED_DIR=$REPO/training_mitigations/sycoharm_runs/endorse_gpt5mini_v5_temp0
STEPHANE=$REPO/osfstorage-archive/Code
TAG=final_final1000_temp1.0_seed0
cd "$REPO"

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY (it is in ~/.bashrc on Marlowe)}"

# arm | staged-name | do_voice(1=OEQ voice census, 0=PAS endorse-only)
ARMS=(
  "dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p6|oeq_P6perspshift_a06_final1000|1"
  "dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p7|oeq_P6perspshift_a07_final1000|1"
  "dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p8|oeq_P6perspshift_a08_final1000|1"
  "dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_addrfilter|oeq_P5addrfilter_final1000|1"
  "dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_futureuser|oeq_P5futureuser_final1000|1"
  "dpo_lora_pas_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5|pas_P6perspshift_a05_final1000|0"
)

shards_ready(){ local d=$RUNS_ROOT/$1/eval
  [[ -f "$d/responses_${TAG}_shard0of2.csv" && -f "$d/responses_${TAG}_shard1of2.csv" ]]; }
scored(){ [[ -f "$SCORED_DIR/scored_$1.csv" ]]; }

# Print endorse rate (#1/(#0+#1), the exact paper metric) + role-play rate for one
# arm. argv: scored_csv  voice_csv(or NONE)  label
report(){
  "$PY" - "$SCORED_DIR/scored_$1.csv" "$2" "$3" <<'PYEOF'
import sys, os, math, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.', ''))
sys.path.insert(0, "/projects/m000254/sycophancy/osfstorage-archive/Code")
from evaluate_action_endorsement_stephane import first_label, action_endorsement_rate
scored, voice, label = sys.argv[1], sys.argv[2], sys.argv[3]
df = pd.read_csv(scored)
assert df["raw"].isna().sum() == 0, f"{scored}: unjudged rows present"
labels = df["raw"].map(first_label)
rate, ci, n01 = action_endorsement_rate(labels.tolist())
line = f"  {label:26s} endorse={rate:6.3f} [{rate-ci:.3f},{rate+ci:.3f}] n01={n01:4d} n={len(df)}"
def wilson(k,n,z=1.959963985):
    if n==0: return float('nan'),float('nan'),float('nan')
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return p,max(0.,c-h),min(1.,c+h)
if voice != "NONE" and os.path.exists(voice):
    v = pd.read_csv(voice); k=int((v["voice"]=="STAKEHOLDER").sum()); n=len(v)
    vp,lo,hi = wilson(k,n)
    line += f" | roleplay={vp:6.3f} [{lo:.3f},{hi:.3f}]"
else:
    line += " | roleplay=  n/a (PAS)"
print(line)
PYEOF
}

judge_arm(){
  local arm=$1 name=$2 dov=$3 env voice=NONE
  case $arm in dpo_lora_pas*) env=PAS;; *) env=OEQ;; esac
  log "judging $arm  ->  $name  (env=$env voice=$dov)"
  "$PY" training_mitigations/eval/stage_heldout_inputs.py \
      --run-dir "$RUNS_ROOT/$arm" --tag "$TAG" --name "$name" --num-shards 2 \
    || { log "STAGE FAILED $arm"; return 1; }
  "$PY" training_mitigations/eval/run_endorse_gpt5mini.py \
      --datasets "$name" --prompt-version v5 \
      --model openai/gpt-5-mini --reasoning minimal \
      --workers 16 --max-cost 6 \
    || { log "ENDORSE FAILED $arm"; return 1; }
  if [[ "$dov" == 1 ]]; then
    "$PY" training_mitigations/eval/classify_response_voice.py \
        --run-dir "$RUNS_ROOT/$arm" --env OEQ --tag "$TAG" --num-shards 2 \
        --model openai/gpt-5-mini --workers 16 \
      && voice="$RUNS_ROOT/$arm/eval/voice_${TAG}.csv" \
      || log "WARN voice census failed $arm (endorse still valid)"
  fi
  log "done $arm:"
  report "$name" "$voice" "$name"
}

# ---------------- poll loop (login node; 5-min cadence) ----------------
pending=("${ARMS[@]}")
log "judge watcher up — ${#pending[@]} arms (5 OEQ endorse+voice, 1 PAS endorse-only)"
while (( ${#pending[@]} )); do
  still=()
  for t in "${pending[@]}"; do
    IFS='|' read -r arm name dov <<<"$t"
    if scored "$name"; then log "already scored: $name — skipping"; continue; fi
    if shards_ready "$arm"; then judge_arm "$arm" "$name" "$dov" || still+=("$t")
    else still+=("$t"); fi
  done
  pending=("${still[@]+"${still[@]}"}")
  (( ${#pending[@]} == 0 )) && break
  sleep 300
done

log "ALL ARMS JUDGED — combined table:"
for t in "${ARMS[@]}"; do
  IFS='|' read -r arm name dov <<<"$t"
  voice=NONE; [[ "$dov" == 1 ]] && voice="$RUNS_ROOT/$arm/eval/voice_${TAG}.csv"
  scored "$name" && report "$name" "$voice" "$name" || log "  MISSING scored_$name.csv"
done
log "endorse scores -> $SCORED_DIR/scored_*.csv ; voice -> runs/<arm>/eval/voice_${TAG}.csv"
