"""Build a cost-bounded sample of TRAINING candidate responses for v5-judge
cross-tab analysis (veto vs. explicit-endorsement).

For each of the 3 OEQ stakeholder-DPO runs (n1000 / mixed / selfpref), for each
of the 3 iterations, we take a FIXED random 500 of the 1000 training prompts
(same prompt_idx set across all runs/iters, seed 0) and randomly sample 3 of the
10 generated responses per prompt (deterministic per (run, iter, prompt_idx)).

That is 500 * 3 = 1500 responses per (run, iter) -> 9 datasets -> 13,500
responses total, which the v5 endorse judge then labels.

Outputs (under sycoharm_runs/):
  inputs/train_<tag>_iter<k>.csv   -- columns EXACTLY [sentence, response]
                                      (the contract run_endorse_gpt5mini.py asserts)
  train_judge_sample_meta.csv      -- one row per emitted response, in the SAME
                                      order as the input CSVs, carrying
                                      [dataset, row, run, iter, prompt_idx, vetoed]
                                      so veto status can be re-joined to the
                                      judge output by (dataset, row).

Assumptions (asserted): every run has responses_iter_{1,2,3}.jsonl with 1000
prompts, each prompt has exactly 10 responses, and every vetoed string is
accounted for among those 10 (veto is a per-index flag derived by consuming the
vetoed multiset).

Run from the repo root:
    python training_mitigations/build_train_judge_sample.py
"""

import json
import os
import random
from collections import Counter

import pandas as pd

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(_HERE, "runs")
SYCO = os.path.join(_HERE, "sycoharm_runs")
INPUTS = os.path.join(SYCO, "inputs")

# run-dir basename -> short tag used in dataset names
RUN_TAGS = {
    "dpo_lora_oeq_qwen3-8b_n1000": "n1000",
    "dpo_lora_oeq_qwen3-8b_n1000_mixed": "mixed",
    "dpo_lora_oeq_qwen3-8b_n1000_selfpref": "selfpref",
}
N_PROMPTS = 500          # of 1000 training prompts
N_RESP_PER_PROMPT = 3    # of 10 generated responses
ITERS = (1, 2, 3)        # responses_iter_<k>.jsonl


def veto_flags(responses, vetoed):
    """Per-index veto flag for `responses` by consuming the `vetoed` multiset.
    Handles duplicate response strings: each vetoed string marks exactly one
    matching index. Asserts every vetoed entry is consumed."""
    vc = Counter(vetoed)
    flags = []
    for r in responses:
        if vc[r] > 0:
            flags.append(True)
            vc[r] -= 1
        else:
            flags.append(False)
    assert sum(vc.values()) == 0, "vetoed strings not all matched in responses"
    assert sum(flags) == len(vetoed)
    return flags


def load_iter(run_basename, it):
    """prompt_idx -> record for responses_iter_<it>.jsonl, with asserts."""
    path = os.path.join(RUNS_DIR, run_basename, f"responses_iter_{it}.jsonl")
    assert os.path.exists(path), f"missing {path}"
    recs = {}
    for line in open(path):
        d = json.loads(line)
        assert len(d["responses"]) == 10, f"{path}: prompt {d['prompt_idx']} !=10 responses"
        recs[d["prompt_idx"]] = d
    assert len(recs) == 1000, f"{path}: expected 1000 prompts, got {len(recs)}"
    return recs


def main():
    os.makedirs(INPUTS, exist_ok=True)

    # Fixed 500 prompt_idx, shared across all runs/iters.
    prompt_idxs = sorted(random.Random(0).sample(range(1000), N_PROMPTS))

    meta_rows = []
    for run_basename, tag in RUN_TAGS.items():
        for it in ITERS:
            recs = load_iter(run_basename, it)
            rows = []  # (sentence, response, vetoed)
            for pidx in prompt_idxs:
                d = recs[pidx]
                flags = veto_flags(d["responses"], d["vetoed"])
                # deterministic per (run, iter, prompt) choice of 3 of 10 indices
                rng = random.Random(f"{tag}|{it}|{pidx}")
                chosen = rng.sample(range(10), N_RESP_PER_PROMPT)
                for ci in chosen:
                    rows.append((d["prompt"], d["responses"][ci], flags[ci], pidx))
            name = f"train_{tag}_iter{it}"
            df = pd.DataFrame([(s, r) for s, r, _, _ in rows],
                              columns=["sentence", "response"])
            assert len(df) == N_PROMPTS * N_RESP_PER_PROMPT
            assert df["sentence"].notna().all() and df["response"].notna().all()
            df.to_csv(os.path.join(INPUTS, f"{name}.csv"), index=False)
            for row_i, (_, _, vetoed, pidx) in enumerate(rows):
                meta_rows.append({"dataset": name, "row": row_i, "run": tag,
                                  "iter": it, "prompt_idx": pidx, "vetoed": vetoed})
            nv = sum(1 for _, _, v, _ in rows if v)
            print(f"wrote {name}.csv  ({len(df)} rows, {nv} vetoed)")

    meta = pd.DataFrame(meta_rows)
    meta_path = os.path.join(SYCO, "train_judge_sample_meta.csv")
    meta.to_csv(meta_path, index=False)
    print(f"\nwrote {meta_path}  ({len(meta)} rows)")
    print("datasets:", " ".join(sorted(meta["dataset"].unique())))


if __name__ == "__main__":
    main()
