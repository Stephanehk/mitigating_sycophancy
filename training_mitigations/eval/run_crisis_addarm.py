"""Add one arm to the crisis-safety comparison, reusing the cached gate.

Scores the three safety dimensions for the DPO neutral_mixed arm on the crisis
subset (severity >= 2) defined by the already-computed gate_<task>.csv, for PAS
and OEQ. Writes safety_<task>_neutralmixed.csv alongside the others.
"""
import glob
import json
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import judge_common
import judge_crisis_gate as G
import judge_crisis_safety as S

JUDGE_KW = dict(
    use_openrouter=True, deployment="openai/gpt-5-mini",
    reasoning_effort="minimal", send_max_tokens=False, temperature=0,
    workers=16, checkpoint_every=200, price_in=0.25, price_out=2.0,
)
RUNS = "/projects/m000254/runs"
OUT = os.path.join(RUNS, "_crisis_safety")

ARM = "neutralmixed"
SRC = {
    "PAS": f"{RUNS}/dpo_lora_pas_qwen3-8b_n1000_neutral_mixed_final",
    "OEQ": f"{RUNS}/dpo_lora_oeq_qwen3-8b_n1000_neutral_mixed_final",
}
BASE = {"PAS": f"{RUNS}/baseline_final_PAS", "OEQ": f"{RUNS}/baseline_final_OEQ"}


def load(pat):
    d = {}
    for f in glob.glob(pat):
        for l in open(f):
            r = json.loads(l)
            d[r["id"]] = r
    return d


def truncated(resp):
    return not str(resp).rstrip().endswith((".", "!", "?", '"', "”", ")", "]"))


for task in ["PAS", "OEQ"]:
    gdf = pd.read_csv(f"{OUT}/gate_{task}.csv")
    sev = gdf.set_index("id")[f"{G.METRIC}_response"]
    sev = sev.mask(sev.astype(str).str.startswith("ERROR")) \
        .map(judge_common.make_digit_parser(G.LO, G.HI))
    base = load(f"{BASE[task]}/eval/responses_base_finaldata_*.jsonl")
    arm = load(f"{SRC[task]}/eval/responses_final_finaldata_*.jsonl")
    crisis_ids = [i for i in sorted(base) if i in arm
                  and sev.get(i) is not None and sev.get(i) >= 2]
    base_df = pd.DataFrame({
        "id": crisis_ids,
        "sentence": [base[i]["prompt"] for i in crisis_ids],
        "response": [arm[i]["response"] for i in crisis_ids],
        "truncated": [truncated(arm[i]["response"]) for i in crisis_ids],
        "severity": [int(sev.get(i)) for i in crisis_ids],
    })
    raw = {}
    for metric, (lo, hi, tmpl, direction) in S.DIMENSIONS.items():
        print(f"[safety] {task}/{ARM}/{metric} on {len(base_df)} rows", flush=True)
        scored = judge_common.score(
            base_df[["id", "sentence", "response"]].copy(), "response",
            f"{OUT}/tmp_{task}_{ARM}_{metric}.csv", metric, lo, hi, template=tmpl,
            reverse=(direction == "proscriptive"),
            system_prompt=S.SYSTEM_PROMPT, **JUDGE_KW)
        raw[metric] = scored[f"{metric}_response"].values
    final = base_df.copy()
    for metric, vals in raw.items():
        final[f"{metric}_response"] = vals
    final.to_csv(f"{OUT}/safety_{task}_{ARM}.csv", index=False)
    print(f"wrote {OUT}/safety_{task}_{ARM}.csv (n={len(final)})", flush=True)

print("DONE")
