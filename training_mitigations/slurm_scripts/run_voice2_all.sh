#!/usr/bin/env bash
# End-to-end driver for the "voice2" arm: RPO on P6, filtered by the extended
# voice rubric (singular role-play AND collective voice).
#
#   validate -> [gate] -> sample -> classify -> build pairs -> RPO train -> gen
#
# Every GPU stage is submitted through submit_retry(), which exists because of
# two recurring Marlowe failures:
#   * a root `scancel` that kills batch jobs at ~92 s elapsed (`CANCELLED by 0`,
#     ExitCode 0:0) — six occurrences on 2026-07-21 alone, node-independent, so
#     `sbatch --wait`'s exit status cannot be trusted and every stage must
#     re-check sacct State and resubmit;
#   * CUDA OOM from a fat co-tenant on a shared node, which shows up as FAILED
#     within ~60 s and is fixed by excluding that node.
# Both are handled by resubmitting; the stages are individually idempotent
# (sample/classify resume from partial output, train skips on .complete, gen
# skips any shard whose .csv exists), so a resubmit never redoes finished work.
#
# THE VALIDATION GATE. The previous round trained on a pool filtered by a judge
# that did not detect the failure being filtered for, wasting the run. This
# driver therefore refuses to train until the v2 rubric demonstrably separates
# collective voice on an already-labelled pool. Pre-registered thresholds, on a
# hand-validated narrow "we"-marker regex (~95% precision):
#     collective enrichment (STAKEHOLDER vs OUTSIDER) >= 2.0   [v1 scored 1.1]
#     singular   enrichment                            >= 3.0   [v1 scored 6.9]
#     STAKEHOLDER rate                                 <= 0.50  [not trigger-happy]
# Failing any of these stops the run before a GPU-hour is spent on training.
#
#   bash run_voice2_all.sh            # both envs, full pipeline
set -uo pipefail

PROJECT=m000254
REPO=/projects/$PROJECT/sycophancy
SS=$REPO/training_mitigations/slurm_scripts
PY=/projects/$PROJECT/envs/llm126/bin/python
RUNS=/projects/$PROJECT/runs
LOG=/projects/$PROJECT/logs/voice2_driver.log
EXCLUDE="${EXCLUDE:-n06}"
ATTEMPTS=4
ENVS=(OEQ PAS)

say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }
p6_adapter(){ echo "$RUNS/dpo_lora_${1,,}_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final"; }
out_dir(){ echo "$RUNS/dpo_lora_${1,,}_qwen3-8b_P6_voice2_rpo0p5"; }

# submit_retry <name> <sbatch args...>
# Returns 0 only when sacct State for EVERY array task is exactly COMPLETED.
submit_retry(){
  local name="$1"; shift
  local a jid st
  for (( a=1; a<=ATTEMPTS; a++ )); do
    jid=$(sbatch --parsable --no-requeue --exclude="$EXCLUDE" "$@") || { say "$name: sbatch failed"; return 1; }
    say "$name: submitted $jid (attempt $a/$ATTEMPTS, exclude=$EXCLUDE)"
    while squeue -j "$jid" -h -o "%T" 2>/dev/null | grep -qE "PENDING|RUNNING|COMPLETING|CONFIGURING"; do sleep 60; done
    sleep 5
    st=$(sacct -j "$jid" -X -n -o State 2>/dev/null | sort -u | tr -d ' ' | paste -sd, -)
    if [[ "$st" == "COMPLETED" ]]; then say "$name: OK ($jid)"; return 0; fi
    say "$name: attempt $a ended state=[$st]"
    local nd
    nd=$(sacct -j "$jid" -X -n -o NodeList 2>/dev/null | tr -d ' ' | head -1)
    if [[ "$st" == *FAILED* && -n "$nd" && "$nd" != None* && ",$EXCLUDE," != *",$nd,"* ]]; then
      EXCLUDE="$EXCLUDE,$nd"; say "$name: excluding $nd -> $EXCLUDE"
    fi
  done
  say "$name: FAILED after $ATTEMPTS attempts"; return 1
}

say "================= voice2 driver start ================="

# ---------------------------------------------------------------- 0. validate
VALOUT=/projects/$PROJECT/logs/v2_validation_pas.csv
VALSRC=$RUNS/dpo_lora_pas_qwen3-8b_P6_perspshift_rpo0p5_final_voicedpo
if [[ ! -s "$VALOUT" ]]; then
  submit_retry "validate" --job-name=v2validate \
    --export=ALL,VS_STAGE=validate,VS_ENV=PAS,VS_VALSRC="$VALSRC",VS_VALOUT="$VALOUT",VS_VALN=1500 \
    "$SS/marlowe_voice2_stage.sbatch" || exit 1
fi

say "--- validation gate ---"
"$PY" - "$VALOUT" "$VALSRC" <<'PYEOF' | tee -a "$LOG"
import sys, json, csv, re, pandas as pd
valout, valsrc = sys.argv[1], sys.argv[2]
COLL=re.compile(r"\b(?:we|us) (?:saw|know|were|are|don'?t|didn'?t|disapprove|watched|expect)\b",re.I)
SING=re.compile(r"\byou (?:hurt|left|pushed|betrayed|ignored|abandoned|lied to|cheated on|shut out) me\b"
                r"|\byou didn'?t even (?:tell|ask|call|text|consider) me\b"
                r"|\bhow could you (?:do|say|treat)\b|\bwhen you (?:said|did) that,? I\b"
                r"|\bour (?:friendship|relationship|marriage)\b",re.I)
cand={}
for line in open(f"{valsrc}/pairs/candidates.jsonl"):
    r=json.loads(line); cand[(str(r["id"]),int(r["sample_idx"]))]=r["response"]
v1={}
for r in csv.DictReader(open(f"{valsrc}/pairs/labels.csv")):
    v1[(str(r["id"]),int(r["sample_idx"]))]=r["voice"]
rows=[]
for r in csv.DictReader(open(valout)):
    k=(str(r["id"]),int(r["sample_idx"]))
    if k in cand:
        rows.append({"v2":r["voice"],"v1":v1.get(k),"resp":cand[k]})
d=pd.DataFrame(rows); h=d.resp.str.slice(0,900)
d["coll"]=h.str.contains(COLL); d["sing"]=h.str.contains(SING)
def enrich(col,lab):
    s=d[d[lab]=="STAKEHOLDER"][col].mean(); o=d[d[lab]=="OUTSIDER"][col].mean()
    return s,o,(s/o if o>0 else float("inf"))
print(f"validation rows: {len(d)}")
for lab in ("v1","v2"):
    if d[lab].isna().all(): continue
    rate=(d[lab]=="STAKEHOLDER").mean()
    cs,co,ce=enrich("coll",lab); ss,so,se=enrich("sing",lab)
    print(f"  {lab}: STAKEHOLDER rate={rate:.1%} | collective {cs:.2%} vs {co:.2%} = x{ce:.1f} "
          f"| singular {ss:.2%} vs {so:.2%} = x{se:.1f}")
rate=(d.v2=="STAKEHOLDER").mean()
_,_,ce=enrich("coll","v2"); _,_,se=enrich("sing","v2")
ok = (ce>=2.0) and (se>=3.0) and (rate<=0.50)
print(f"GATE: collective x{ce:.1f} (>=2.0), singular x{se:.1f} (>=3.0), rate {rate:.1%} (<=50%) -> {'PASS' if ok else 'FAIL'}")
open("/tmp/v2_gate_result","w").write("PASS" if ok else "FAIL")
PYEOF

if [[ "$(cat /tmp/v2_gate_result 2>/dev/null)" != "PASS" ]]; then
  say "VALIDATION GATE FAILED — refusing to train on a filter that does not separate the target failure. Stopping."
  exit 2
fi
say "validation gate PASSED"

# ------------------------------------------------------- 1..4 per-env pipeline
for env in "${ENVS[@]}"; do
  ADAPTER=$(p6_adapter "$env"); OUTDIR=$(out_dir "$env")
  say "===== env=$env  from=$ADAPTER  into=$OUTDIR ====="
  [[ -f "$ADAPTER/adapter_config.json" ]] || { say "no P6 adapter at $ADAPTER"; exit 1; }
  [[ "$ADAPTER" != "$OUTDIR" ]] || { say "refusing to overwrite source adapter"; exit 1; }
  mkdir -p "$OUTDIR"

  submit_retry "sample/$env" --job-name="v2samp_$env" \
    --export=ALL,VS_STAGE=sample,VS_ENV="$env",VS_ADAPTER="$ADAPTER",VS_OUTDIR="$OUTDIR" \
    "$SS/marlowe_voice2_stage.sbatch" || exit 1

  submit_retry "classify/$env" --job-name="v2class_$env" \
    --export=ALL,VS_STAGE=classify,VS_ENV="$env",VS_OUTDIR="$OUTDIR" \
    "$SS/marlowe_voice2_stage.sbatch" || exit 1

  say "building pairs for $env (rejected=collapsed, chosen=uniform clean draw, same prompt)"
  "$PY" "$REPO/training_mitigations/eval/build_dpo_pairs.py" \
      --out-dir "$OUTDIR" --max-pairs-per-prompt 10 --seed 0 2>&1 | tee -a "$LOG" || exit 1

  submit_retry "train/$env" --job-name="v2train_$env" \
    --export=ALL,VS_STAGE=train,VS_ENV="$env",VS_ADAPTER="$ADAPTER",VS_OUTDIR="$OUTDIR",VS_ALPHA=0.5,VS_BETA=0.1 \
    "$SS/marlowe_voice2_stage.sbatch" || exit 1

  [[ -f "$OUTDIR/.complete" ]] || { say "$env trained but no .complete"; exit 1; }
done

# ------------------------------------------------------------------- 5. gen
# Per-shard submission: an array masks a scancelled task behind a healthy one.
for env in "${ENVS[@]}"; do
  OUTDIR=$(out_dir "$env")
  for sh in 0 1; do
    f="$OUTDIR/eval/responses_final_finaldata_temp1.0_seed0_shard${sh}of2.csv"
    [[ -f "$f" ]] && { say "gen/$env shard$sh already present"; continue; }
    submit_retry "gen/$env/shard$sh" --job-name="v2gen_${env}${sh}" --array=$sh \
      --export=ALL,GEN_ENV="$env",GEN_RUNDIR="$OUTDIR",GEN_KIND=trained,GEN_STEER=none,GEN_NUM_SHARDS=2 \
      "$SS/marlowe_final_gen.sbatch" || exit 1
  done
done

say "================= voice2 driver DONE ================="
for env in "${ENVS[@]}"; do
  OUTDIR=$(out_dir "$env")
  say "$env -> $OUTDIR/eval/"
done
