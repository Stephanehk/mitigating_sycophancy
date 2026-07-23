#!/usr/bin/env bash
# Per-shard watcher for the voice2 held-out generation, run detached so it
# survives the login session. All four (env,shard) slots run CONCURRENTLY;
# each is tracked by its own output file, not by array-job state, because a
# root-scancelled task hides behind a healthy sibling in an array's squeue row.
#
# Handles the two recurring Marlowe failures (see run_voice2_all.sh header):
#   * root `scancel` at ~92 s -> resubmit the same shard
#   * CUDA OOM from a co-tenant (fast FAILED) -> add that node to EXCLUDE
# The gen sbatch skips any shard whose .csv already exists, so resubmission
# resumes rather than repeats.
#
# On completion it prints the no-API repetition/voice detectors for both arms,
# so the result is waiting in the log rather than needing a follow-up run.
set -uo pipefail
SS=/projects/m000254/sycophancy/training_mitigations/slurm_scripts
SP=/tmp/claude-851705398/-projects-m000254/1d43f15b-ba48-4de2-8bfd-1f3b12c01b37/scratchpad
PY=/projects/m000254/envs/llm126/bin/python
LOG=/projects/m000254/logs/watch_voice2_gen.log
EXCLUDE="${EXCLUDE:-n06}"
MAX=5
declare -A STEM=( [OEQ]=oeq [PAS]=pas )
declare -A TRIES
say(){ echo "[$(date -Is)] $*" >> "$LOG"; }
rundir(){ echo "/projects/m000254/runs/dpo_lora_${STEM[$1]}_qwen3-8b_P6_voice2_rpo0p5"; }
rows(){ local f; f="$(rundir "$1")/eval/responses_final_finaldata_temp1.0_seed0_shard${2}of2.csv"
  [[ -f $f ]] || { echo -1; return; }
  "$PY" -c "import pandas as pd,sys;print(len(pd.read_csv(sys.argv[1])))" "$f" 2>/dev/null || echo -1; }

for E in OEQ PAS; do for S in 0 1; do TRIES[$E:$S]=0; done; done
say "=== voice2 gen watcher start (4 shards in parallel, exclude=$EXCLUDE) ==="

while :; do
  sleep 90
  alldone=1
  for E in OEQ PAS; do for S in 0 1; do
    k="$E:$S"; r=$(rows "$E" "$S"); [[ $r -ge 500 ]] && continue
    alldone=0
    jid=$(cat "$SP/v2gen_${E}_${S}" 2>/dev/null)
    if squeue -j "$jid" -h -o "%T" 2>/dev/null | grep -qE "PENDING|RUNNING|COMPLETING|CONFIGURING"; then continue; fi
    st=$(sacct -j "$jid" -X -n -o State 2>/dev/null | sort -u | tr -d ' ' | paste -sd, -)
    say "$k job=$jid left queue state=[$st] rows=$r"
    if [[ "$st" == *FAILED* ]]; then
      nd=$(sacct -j "$jid" -X -n -o NodeList 2>/dev/null | tr -d ' ' | head -1)
      [[ -n "$nd" && "$nd" != None* && ",$EXCLUDE," != *",$nd,"* ]] && { EXCLUDE="$EXCLUDE,$nd"; say "EXCLUDE -> $EXCLUDE"; }
    fi
    TRIES[$k]=$(( ${TRIES[$k]} + 1 ))
    [[ ${TRIES[$k]} -gt $MAX ]] && { say "$k GIVING UP after $MAX"; continue; }
    cd /projects/m000254/sycophancy || exit 1
    n=$(sbatch --parsable --no-requeue --exclude="$EXCLUDE" --job-name="v2gen_${E}${S}" --array=$S \
        --export=ALL,GEN_ENV=$E,GEN_RUNDIR=$(rundir "$E"),GEN_KIND=trained,GEN_STEER=none,GEN_NUM_SHARDS=2 \
        "$SS/marlowe_final_gen.sbatch")
    echo "$n" > "$SP/v2gen_${E}_${S}"
    say "$k RESUBMITTED as $n (try ${TRIES[$k]}/$MAX, exclude=$EXCLUDE)"
  done; done
  [[ $alldone == 1 ]] && break
done

say "ALL SHARDS COMPLETE: OEQ=$(rows OEQ 0)/$(rows OEQ 1)  PAS=$(rows PAS 0)/$(rows PAS 1)"
say "--- detectors (no API calls) ---"
"$PY" - >> "$LOG" 2>&1 <<'PYEOF'
import pandas as pd, re
def loopy(t,n=10):
    w=str(t).split(); w=w[len(w)//2:]
    if len(w)<n*3: return 0.0
    g=[' '.join(w[i:i+n]) for i in range(len(w)-n+1)]
    return 1-len(set(g))/len(g)
V1=re.compile(r"\b(?:we|us) (?:saw|know|were|are|don'?t|didn'?t|disapprove|watched|expect)\b",re.I)
R='/projects/m000254/runs'
ARMS={'OEQ voice2':f'{R}/dpo_lora_oeq_qwen3-8b_P6_voice2_rpo0p5',
      'OEQ P6':f'{R}/dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final',
      'PAS voice2':f'{R}/dpo_lora_pas_qwen3-8b_P6_voice2_rpo0p5',
      'PAS P6':f'{R}/dpo_lora_pas_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final',
      'PAS voicedpo':f'{R}/dpo_lora_pas_qwen3-8b_P6_perspshift_rpo0p5_final_voicedpo'}
print(f"{'arm':<16}{'n':>6}{'rep>0.10':>10}{'we-mark':>10}{'lowercase':>11}{'medlen':>8}")
for k,run in ARMS.items():
    try:
        d=pd.concat([pd.read_csv(f'{run}/eval/responses_final_finaldata_temp1.0_seed0_shard{s}of2.csv')
                     for s in (0,1)],ignore_index=True)
    except Exception as e:
        print(f"{k:<16} (not ready)"); continue
    r=d.response.astype(str)
    print(f"{k:<16}{len(d):>6}{(r.map(loopy)>0.10).mean():>10.1%}"
          f"{r.str.slice(0,900).str.contains(V1).mean():>10.1%}"
          f"{r.map(lambda t: t[:1].islower() and sum(c.isupper() for c in t)/max(len(t),1)<0.005).mean():>11.1%}"
          f"{int(r.str.len().median()):>8}")
PYEOF
say "=== watcher done ==="
