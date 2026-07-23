"""Voice-collapse judge for the OEQ HUMAN baseline, which has no run dir.

classify_response_voice.py reads <run>/eval/ shards, but the human reference set
is a staged (sentence, response) frame under endorse_gpt5mini_v5_temp0/, so it
cannot be driven through that CLI. This shim reuses that module's TEMPLATE,
judge() and report() verbatim -- only the item source differs -- so the human
column is produced by the SAME judge, prompt and parser as every model arm.

Output: sycoharm_runs/judges_finaldata/OEQ/voice_human.csv, with the same
(row, side, voice) schema classify_response_voice.py writes, so downstream
aggregation reads it identically. Idempotent: an existing file is reported, not
re-judged. Needs OPENROUTER_API_KEY. CPU/network only.

Run:
    python training_mitigations/eval/voice_human_finaldata.py
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from openai import OpenAI

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from classify_response_voice import judge, report  # noqa: E402
from build_finaldata_tables import V5_DIR  # noqa: E402

SRC = os.path.join(V5_DIR, "scored_oeq_human_finaldata.csv")
OUT = os.path.join(_HERE, "..", "sycoharm_runs", "judges_finaldata", "OEQ",
                   "voice_human.csv")
MODEL = "openai/gpt-5-mini"
WORKERS = 12


def main():
    if os.path.exists(OUT):
        print(f"[voice] {OUT} exists; reporting existing")
        report(pd.read_csv(OUT), "OEQ")
        return
    assert os.environ.get("OPENROUTER_API_KEY"), "set OPENROUTER_API_KEY"
    assert os.path.exists(SRC), f"missing human frame: {SRC}"

    df = pd.read_csv(SRC)
    assert df["sentence"].notna().all() and df["response"].notna().all(), \
        "human frame has NaN sentence/response"
    # (key, side, context, response) -- the item contract classify_response_voice
    # .judge expects; OEQ is a SENTENCE env, so side is always "single".
    items = [(i, "single", r["sentence"], r["response"]) for i, r in df.iterrows()]
    print(f"[voice] OEQ human: {len(items)} judgements", flush=True)

    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ["OPENROUTER_API_KEY"])
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(lambda it: judge(client, MODEL, "OEQ", it), items))

    out = pd.DataFrame([{"row": k, "side": s, "voice": v} for k, s, v in results])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"[voice] wrote {OUT}")
    report(out, "OEQ")


if __name__ == "__main__":
    main()
