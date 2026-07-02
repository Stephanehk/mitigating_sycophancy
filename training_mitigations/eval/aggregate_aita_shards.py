"""Recombine the per-shard outputs of eval_aita_label_heldout.py into one
metric for a model+condition (AITA FNR/FPR, or AITA-NTA-FLIP consistency).

When a model's held-out set is evaluated as N disjoint shards (N parallel GPU
jobs), each shard writes
  <run_dir>/eval/aita_scored_<adapter_tag>_<decode>_seed<seed>_shard<i>of<N>.csv
holding its per-row parsed labels. Because the shards are disjoint and tile the
evaluated set, CONCATENATING the shard rows reproduces the full single-job set
exactly, so recomputing the metric on the concatenation is identical to scoring
all rows in one job.

Writes:
  <run_dir>/eval/aita_scored_<adapter_tag>_<decode>_seed<seed>_AGG.csv
  <run_dir>/eval/summary_aita_<adapter_tag>_<decode>_seed<seed>_AGG.json

Assumptions (validated): all N shard CSVs exist; each shard id 0..N-1 appears
exactly once; ids are unique across the concatenation (no overlap).
"""

import argparse
import json
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO_ROOT, "mitigations"))

from explore_mitigations import compute_fnr_fpr, compute_aita_nta_flip_metrics


def load_shard_frames(out_dir, adapter_tag, decode, seed, num_shards):
    """Load and concatenate the N shard scored CSVs for one (adapter_tag,
    decode, seed). Asserts each shard id 0..N-1 is present exactly once and that
    the concatenated `id` column is unique (disjoint shards)."""
    frames = []
    for i in range(num_shards):
        path = os.path.join(
            out_dir,
            f"aita_scored_{adapter_tag}_{decode}_seed{seed}_shard{i}of{num_shards}.csv",
        )
        assert os.path.exists(path), f"missing shard scored csv: {path}"
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    assert df["id"].is_unique, (
        "ids are not unique across shards — shards overlap or were "
        "double-counted; aggregation would be wrong"
    )
    return df


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--env", required=True, choices=("AITA", "AITA-NTA-FLIP"))
    parser.add_argument("--adapter-tag", required=True,
                        help="adapter tag used in the shard filenames "
                             "(e.g. 'base', 'base-becritical', 'final')")
    parser.add_argument("--decode", default="temp1.0",
                        help="decode tag in the filenames (default: temp1.0; "
                             "use 'greedy' for greedy runs)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    out_dir = os.path.join(args.run_dir, "eval")
    df = load_shard_frames(out_dir, args.adapter_tag, args.decode, args.seed,
                           args.num_shards)

    if args.env == "AITA":
        metrics = compute_fnr_fpr(df)
    else:
        metrics = compute_aita_nta_flip_metrics(df)

    base = f"{args.adapter_tag}_{args.decode}_seed{args.seed}_AGG"
    agg_csv = os.path.join(out_dir, f"aita_scored_{base}.csv")
    df.to_csv(agg_csv, index=False)

    agg = {
        "env": args.env,
        "run_dir": os.path.abspath(args.run_dir),
        "adapter_tag": args.adapter_tag,
        "decode": args.decode,
        "seed": args.seed,
        "num_shards": args.num_shards,
        "n_rows": len(df),
        "metrics": metrics,
    }
    agg_path = os.path.join(out_dir, f"summary_aita_{base}.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)

    print("=" * 70)
    print(f"AGGREGATED  [{args.env} / {args.adapter_tag} / {args.decode}]  "
          f"{args.num_shards} shards  N={len(df)}")
    if args.env == "AITA":
        print(f"  FNR (sycophancy) = {metrics['fnr']:.4f}  "
              f"({metrics['fn']}/{metrics['n_yta_truth']})")
        print(f"  FPR              = {metrics['fpr']:.4f}  "
              f"({metrics['fp']}/{metrics['n_nta_truth']})")
        print(f"  dropped (OTHER)  = {metrics['n_dropped_other']}")
    else:
        print(f"  sycophancy_score_rate (both NTA) = "
              f"{metrics['sycophancy_score_rate']:.4f}  "
              f"({metrics['n_sycophancy']}/{metrics['n_total']})")
        print(f"  critical_score_rate   (both YTA) = "
              f"{metrics['critical_score_rate']:.4f}  "
              f"({metrics['n_critical']}/{metrics['n_total']})")
        print(f"  dropped (unparsed pair) = {metrics['n_dropped_other']}")
    print(f"  -> {agg_csv}")
    print(f"  -> {agg_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
