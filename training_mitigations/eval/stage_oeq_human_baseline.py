"""Stage the OEQ held-out HUMAN responses into the v5 endorse-judge input format.

This builds the human-reference arm of the OEQ table: instead of a model
response, the judged text is the original human reply carried in
`final_training_eval_data/oeq_test.csv` (column `human_response`).

The prompt text ("sentence") is taken from the GENERATION shards rather than
from oeq_test.csv, so the judge sees byte-identical prompts to the ones the
model arms were scored on. oeq_test.csv is joined on `id` only to supply the
human response.

Assumptions (validated in code):
  - The held-out id set (oeq_test_ids.json) has exactly --expect ids.
  - Every held-out id is present in oeq_test.csv and in the gen shards.
  - The sentence for each id agrees between the gen shards and oeq_test.post_text.
  - Human responses are non-empty and not Reddit tombstones ([deleted]/[removed]).

Inputs : --gen-run-dir, --gen-tag, --oeq-test, --ids, --name
Output : sycoharm_runs/inputs/<name>.csv  (sentence,response)
"""

import argparse
import json
import os

import pandas as pd

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS_DIR = os.path.join(_HERE, "sycoharm_runs", "inputs")

TOMBSTONES = {"[deleted]", "[removed]", "deleted", "removed", ""}


def load_gen_sentences(run_dir, tag, num_shards):
    """Concatenate the gen shards to recover the exact id -> sentence mapping."""
    parts = []
    for si in range(num_shards):
        p = os.path.join(run_dir, "eval",
                         f"responses_{tag}_shard{si}of{num_shards}.csv")
        assert os.path.exists(p), f"missing shard CSV: {p}"
        d = pd.read_csv(p)
        assert list(d.columns) == ["id", "sentence", "response"], \
            f"{p}: unexpected columns {list(d.columns)}"
        parts.append(d[["id", "sentence"]])
    df = pd.concat(parts, ignore_index=True)
    assert df["id"].nunique() == len(df), "duplicate ids across gen shards"
    return df


def stage(gen_run_dir, gen_tag, num_shards, oeq_test_path, ids_path, name, expect):
    ids = json.load(open(ids_path))
    assert len(ids) == expect, f"expected {expect} held-out ids, got {len(ids)}"
    assert len(set(ids)) == len(ids), "duplicate ids in held-out id set"

    gen = load_gen_sentences(gen_run_dir, gen_tag, num_shards)
    test = pd.read_csv(oeq_test_path)
    assert {"id", "post_text", "human_response"} <= set(test.columns), \
        f"{oeq_test_path}: missing required columns, got {list(test.columns)}"

    test = test[test["id"].isin(ids)].copy()
    assert len(test) == expect, \
        f"held-out ids not all found in oeq_test.csv: {len(test)}/{expect}"

    merged = test.merge(gen, on="id", how="inner", validate="one_to_one")
    assert len(merged) == expect, \
        f"held-out ids not all found in gen shards: {len(merged)}/{expect}"

    # The judged prompt must be identical to what the model arms saw.
    mismatch = merged[merged["sentence"].str.strip()
                      != merged["post_text"].str.strip()]
    assert len(mismatch) == 0, \
        f"{len(mismatch)} ids where gen sentence != oeq_test.post_text"

    resp = merged["human_response"]
    n_null = int(resp.isna().sum())
    tomb = resp.fillna("").str.strip().str.lower().isin(TOMBSTONES)
    n_tomb = int(tomb.sum())
    assert n_null == 0 and n_tomb == 0, (
        f"unusable human responses: {n_null} null, {n_tomb} tombstone "
        f"([deleted]/[removed]/empty) — refusing to stage a poisoned baseline")

    out = merged[["sentence", "human_response"]].rename(
        columns={"human_response": "response"})
    os.makedirs(INPUTS_DIR, exist_ok=True)
    out_path = os.path.join(INPUTS_DIR, f"{name}.csv")
    out.to_csv(out_path, index=False)
    chars = out["response"].str.len()
    print(f"[stage] {name}: {len(out)} rows -> {out_path}")
    print(f"[stage] human response chars: median={int(chars.median())} "
          f"mean={int(chars.mean())} min={int(chars.min())} max={int(chars.max())}")
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gen-run-dir",
                   default="/projects/m000254/runs/baseline_final_OEQ")
    p.add_argument("--gen-tag", default="base_finaldata_temp1.0_seed0")
    p.add_argument("--num-shards", type=int, default=2)
    p.add_argument("--oeq-test",
                   default="/projects/m000254/mitigating_sycophancy/"
                           "final_training_eval_data/oeq_test.csv")
    p.add_argument("--ids",
                   default="/projects/m000254/mitigating_sycophancy/"
                           "final_training_eval_data/oeq_test_ids.json")
    p.add_argument("--name", default="oeq_human_finaldata")
    p.add_argument("--expect", type=int, default=1000)
    a = p.parse_args()
    stage(a.gen_run_dir, a.gen_tag, a.num_shards, a.oeq_test, a.ids,
          a.name, a.expect)


if __name__ == "__main__":
    main()
