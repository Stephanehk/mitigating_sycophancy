"""Stage held-out generation outputs into the v5 endorse-judge input format.

For one trained run, this concatenates the two disjoint response shards written
by eval_endorsement_heldout.py (--no-judge, --num-shards 2) over the 1000-prompt
held-out set, drops the `id` column, and writes a single
`sycoharm_runs/inputs/<name>.csv` with EXACTLY the columns
['sentence', 'response'] that run_endorse_gpt5mini.py asserts.

Assumptions (validated in code):
  - Both shard CSVs exist under <run_dir>/eval/ and carry columns
    id,sentence,response (the --no-judge responses table).
  - The two shards are disjoint and together cover the full held-out set, so the
    concatenated row count equals --expect (default 1000) with unique ids.

Inputs : --run-dir, --tag (default final_final1000_temp1.0_seed0), --name
Output : sycoharm_runs/inputs/<name>.csv  (sentence,response)
"""

import argparse
import os

import pandas as pd

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS_DIR = os.path.join(_HERE, "sycoharm_runs", "inputs")


def stage(run_dir, tag, name, num_shards, expect):
    """Concatenate the per-shard responses CSVs and write the judge input CSV.
    Returns the staged DataFrame (sentence,response)."""
    eval_dir = os.path.join(run_dir, "eval")
    parts = []
    for si in range(num_shards):
        p = os.path.join(
            eval_dir, f"responses_{tag}_shard{si}of{num_shards}.csv")
        assert os.path.exists(p), f"missing shard CSV: {p}"
        d = pd.read_csv(p)
        assert list(d.columns) == ["id", "sentence", "response"], \
            f"{p}: unexpected columns {list(d.columns)}"
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    assert df["id"].nunique() == len(df), "duplicate ids across shards"
    assert len(df) == expect, f"{name}: expected {expect} rows, got {len(df)}"
    out = df[["sentence", "response"]]
    os.makedirs(INPUTS_DIR, exist_ok=True)
    out_path = os.path.join(INPUTS_DIR, f"{name}.csv")
    out.to_csv(out_path, index=False)
    print(f"[stage] {name}: {len(out)} rows -> {out_path}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--name", required=True,
                   help="output stem under sycoharm_runs/inputs/")
    p.add_argument("--tag", default="final_final1000_temp1.0_seed0",
                   help="response-file tag preceding _shard{i}of{n}")
    p.add_argument("--num-shards", type=int, default=2)
    p.add_argument("--expect", type=int, default=1000)
    args = p.parse_args()
    stage(args.run_dir, args.tag, args.name, args.num_shards, args.expect)


if __name__ == "__main__":
    main()
