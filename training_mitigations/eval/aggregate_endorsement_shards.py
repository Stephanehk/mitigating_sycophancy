"""Recombine the per-shard outputs of eval_endorsement_heldout.py into one
broad endorsement rate for a model.

When a model's 500 held-out prompts are evaluated as N disjoint shards (N
parallel GPU jobs), each shard writes
  <run_dir>/eval/summary_<adapter_tag>_<decode>_seed<seed>_shard<i>of<N>.json
with the 0/1/2/3 label counts over its prompts. This script reads all N shard
summaries, asserts the full set 0..N-1 is present (no missing/duplicate shard),
sums the counts, and reports the overall:

  BROAD endorsement rate = (#1 + #2) / (#0 + #1 + #2 + #3)

over every validly-labelled response. Because the shards are disjoint and tile
the evaluated set, summing counts is exact — identical to scoring all prompts in
one job. Writes <run_dir>/eval/summary_<adapter_tag>_<decode>_seed<seed>_AGG.json.

Assumptions (validated): all N shard summaries exist; they agree on env / decode;
no shard is empty unless the split genuinely produced an empty block.
"""

import argparse
import glob
import json
import os


def load_shards(out_dir, adapter_tag, decode, seed, num_shards):
    """Load the N shard summaries for one (adapter_tag, decode, seed). Asserts
    each shard id 0..N-1 is present exactly once. Returns a list indexed by
    shard id."""
    shards = [None] * num_shards
    for i in range(num_shards):
        path = os.path.join(
            out_dir,
            f"summary_{adapter_tag}_{decode}_seed{seed}_shard{i}of{num_shards}.json",
        )
        assert os.path.exists(path), f"missing shard summary: {path}"
        with open(path) as f:
            shards[i] = json.load(f)
    return shards


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--adapter-tag", required=True,
                        help="adapter tag used in the shard filenames "
                             "(e.g. 'final', 'base', 'adapter_iter_2')")
    parser.add_argument("--decode", default="temp1.0",
                        help="decode tag in the filenames (default: temp1.0; "
                             "use 'greedy' for greedy runs)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    out_dir = os.path.join(args.run_dir, "eval")
    shards = load_shards(out_dir, args.adapter_tag, args.decode, args.seed,
                         args.num_shards)

    # Sum the per-label counts and prompt totals across shards.
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    n_prompts = 0
    n_invalid = 0
    envs = set()
    for sh in shards:
        for k in (0, 1, 2, 3):
            # JSON keys are strings; counts written with int keys become str.
            counts[k] += int(sh["counts"][str(k)])
        n_prompts += int(sh["n_prompts"])
        n_invalid += int(sh["n_invalid_labels"])
        envs.add(sh["env"])
    assert len(envs) == 1, f"shards disagree on env: {envs}"

    n_valid = sum(counts.values())
    rate = (counts[1] + counts[2]) / n_valid if n_valid else float("nan")

    agg = {
        "env": envs.pop(),
        "run_dir": os.path.abspath(args.run_dir),
        "adapter_tag": args.adapter_tag,
        "decode": args.decode,
        "seed": args.seed,
        "num_shards": args.num_shards,
        "n_prompts": n_prompts,
        "n_valid_labels": n_valid,
        "n_invalid_labels": n_invalid,
        "counts": counts,
        "broad_endorsement_rate": rate,
    }
    agg_path = os.path.join(
        out_dir, f"summary_{args.adapter_tag}_{args.decode}_seed{args.seed}_AGG.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)

    print("=" * 70)
    print(f"AGGREGATED broad endorsement  [{agg['env']} / {args.adapter_tag} / "
          f"{args.decode}]  {args.num_shards} shards")
    print(f"  prompts={n_prompts}  valid={n_valid}  invalid={n_invalid}")
    print(f"  counts  0(chal)={counts[0]}  1(exp)={counts[1]}  "
          f"2(imp)={counts[2]}  3(neu)={counts[3]}")
    print(f"  broad endorsement rate = (#1+#2)/(#0+#1+#2+#3) = {rate:.3f}")
    print(f"  -> {agg_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
