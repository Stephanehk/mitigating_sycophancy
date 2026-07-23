"""Decide which prompts still need a resample after a gate round.

Reads every gate_<tag>_a{1..k}.csv written so far and emits the worklist for
attempt k+1: the (id, side) keys that have been REJECTED at every attempt so
far. A key that has passed at any attempt is resolved -- it never reappears,
because rejection sampling stops at the first success.

Emits nothing and reports "converged" when no keys remain, which is the signal
to stop the round loop early rather than burning the full MAX_ATTEMPTS.

No network, no GPU -- runs anywhere between the two halves of a round.
"""

import argparse
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from gate_common import MAX_ATTEMPTS, attempt_paths, write_worklist


def load_gate_rounds(run_dir, tag, upto):
    """Concatenate the gate labels from attempts 1..upto.

    A missing attempt file is fatal rather than skipped: silently treating an
    absent round as "nothing rejected" would resolve keys that were never
    actually judged.
    """
    frames = []
    for a in range(1, upto + 1):
        p = attempt_paths(run_dir, tag, a)["gate"]
        assert os.path.exists(p), f"missing gate labels for attempt {a}: {p}"
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    df["side"] = df["side"].fillna("").astype(str)
    df["id"] = df["id"].astype(str)
    return df


def unresolved(df):
    """Keys rejected at every attempt in `df` (i.e. max gate label is 0)."""
    best = df.groupby(["id", "side"])["gate"].max().reset_index()
    return best[best["gate"] == 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--completed-attempt", type=int, required=True,
                    help="the highest attempt whose gate labels exist")
    ap.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    args = ap.parse_args()

    df = load_gate_rounds(args.run_dir, args.tag, args.completed_attempt)
    rem = unresolved(df)
    n_keys = len(df.groupby(["id", "side"]))

    print(f"[worklist] {args.tag} after attempt {args.completed_attempt}: "
          f"{n_keys - len(rem)}/{n_keys} resolved, {len(rem)} still rejected")

    if len(rem) == 0:
        print("[worklist] CONVERGED — every key has a passing response; stop the loop")
        return
    if args.completed_attempt >= args.max_attempts:
        print(f"[worklist] EXHAUSTED at attempt {args.max_attempts} — {len(rem)} keys "
              f"will be dropped and tallied as gate failures by assemble_gate.py")
        return

    nxt = args.completed_attempt + 1
    keys = [{"id": r.id, "side": r.side} for r in rem.itertuples(index=False)]
    path = attempt_paths(args.run_dir, args.tag, nxt)["worklist"]
    payload = write_worklist(path, args.env, args.tag, nxt, keys)
    print(f"[worklist] attempt {nxt}: {payload['n_keys']} keys over "
          f"{payload['n_ids']} ids -> {path}")


if __name__ == "__main__":
    main()
