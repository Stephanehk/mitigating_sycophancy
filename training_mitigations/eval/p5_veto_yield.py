"""Run the P5 two-stage veto (Qwen3-8B, the actual frozen post-training judge)
over a batch of real OEQ mixed_P4 training candidates, and record per candidate
whether P5 vetoes it — alongside the P4 veto bit saved in the jsonl.

Sharded over the prompts so many shards run in parallel on next5/next6. Each
shard writes <run>/eval/p5_yield/shard<i>of<N>.csv with one row per candidate:
  prompt_idx, sentence, response, source, p4_vetoed, p5_vetoed, s1_endorses

Downstream, p5_yield_score.py v5-scores every candidate and asks the go/no-go
question: is the P5 `chosen` (non-vetoed) pool LESS endorsing than the
`rejected` pool? (P4 was inverted — chosen was MORE endorsing.) Veto = Qwen3-8B
throughout; only the eval metric is the gpt-5-mini v5 judge.
"""

import argparse
import json
import os
import re
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mitigations"))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "core"))

from llama_local import DEFAULT_HF_MODEL_ID, load_llama, make_call_llm
import simulated_stakeholders as ss
from simulated_stakeholders import (SimulatedStakeholders, VETO_PROMPT_P5,
                                    ENDORSE_CLASSIFY_PROMPT_P5)
import our_mitigation, our_mitigation_v2, weak_pipeline

_LOCAL_MAX_STAKEHOLDERS = 8


def tolerant_parse_stakeholders(text, min_count=None, max_count=None):
    items = []
    for raw in (text or "").splitlines():
        s = raw.strip()
        s = re.sub(r"^(?:[\-\*•]|\d+[\.\)])\s*", "", s).strip()
        s = s.strip("*_`").strip()
        if s:
            items.append(s)
    return items[:_LOCAL_MAX_STAKEHOLDERS]


def shard_slice(n, num_shards, shard_id):
    base, rem = divmod(n, num_shards)
    sizes = [base + (1 if i < rem else 0) for i in range(num_shards)]
    start = sum(sizes[:shard_id])
    return start, start + sizes[shard_id]


def load_prompts(jsonl_path, n):
    """First `n` prompts from the iter jsonl, each with its candidates and the
    P4 veto bit (membership of the response text in that prompt's `vetoed`)."""
    out = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            vset = set(r["vetoed"])
            cands = [(resp, src.split(":")[0], resp in vset)
                     for resp, src in zip(r["responses"], r["sources"])]
            out.append({"prompt_idx": r["prompt_idx"], "prompt": r["prompt"],
                        "cands": cands})
            if len(out) >= n:
                break
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir",
                   default=os.path.join(_HERE, "runs",
                                        "dpo_lora_oeq_qwen3-8b_n1000_mixed_P4"))
    p.add_argument("--iter-file", default="responses_iter_3.jsonl")
    p.add_argument("--n-prompts", type=int, default=160)
    p.add_argument("--num-shards", type=int, default=10)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--model-id", default=DEFAULT_HF_MODEL_ID)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-map", default="auto")
    args = p.parse_args()

    for mod in (our_mitigation, our_mitigation_v2, weak_pipeline, ss):
        if hasattr(mod, "parse_stakeholders"):
            mod.parse_stakeholders = tolerant_parse_stakeholders

    prompts = load_prompts(os.path.join(args.run_dir, args.iter_file), args.n_prompts)
    s, e = shard_slice(len(prompts), args.num_shards, args.shard_id)
    shard = prompts[s:e]
    print(f"[shard {args.shard_id}/{args.num_shards}] prompts [{s}:{e}) of {len(prompts)} "
          f"({len(shard)} prompts)")

    print(f"[load] frozen sim {args.model_id} (greedy) -> device_map={args.device_map}")
    model, tok = load_llama(model_id=args.model_id, device=args.device,
                            cache_dir=os.environ.get("HF_HOME"),
                            device_map=args.device_map)
    model.eval()
    sim = SimulatedStakeholders(
        call_llm=make_call_llm(model, tok, do_sample=False),
        model_id=args.model_id, client=None, use_weak_model=True,
        veto_concurrency=1, veto_prompt=VETO_PROMPT_P5,
        endorse_classify_prompt=ENDORSE_CLASSIFY_PROMPT_P5,
    )

    rows = []
    for k, item in enumerate(shard):
        prompt = item["prompt"]
        sim.prime_stakeholder_cache([prompt], concurrency=1)
        _c, entries = sim._stakeholder_data_for(prompt)
        for resp, source, p4v in item["cands"]:
            endorses, just = sim._classify_endorsement(prompt, resp)
            p5v = False
            if endorses:
                for entry in entries:
                    decision, _raw = sim._stakeholder_endorsement_objection(
                        prompt, resp, entry, just)
                    if decision:
                        p5v = True
                        break  # OR over stakeholders; short-circuit like training
            rows.append({
                "prompt_idx": item["prompt_idx"],
                "sentence": prompt,
                "response": resp,
                "source": source,
                "p4_vetoed": p4v,
                "p5_vetoed": p5v,
                "s1_endorses": endorses,
            })
        print(f"[shard {args.shard_id}] {k+1}/{len(shard)} prompts done "
              f"({len(rows)} candidates)")

    out_dir = os.path.join(args.run_dir, "eval", "p5_yield")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"shard{args.shard_id}of{args.num_shards}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[shard {args.shard_id}] wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
