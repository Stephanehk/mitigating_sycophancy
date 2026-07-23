"""Assemble the gated scored set for one method, and tally the rejections.

For every (id, side) key, selects the FIRST attempt whose gate label is 1 and
emits that response as the one to score downstream. Keys rejected at every
attempt are dropped from the scored set and counted as gate failures.

Two outputs:

  gatefinal_<tag>.csv   the scored set, in the column layout the downstream
                        scorer expects for the env, plus `winning_attempt`.
  gatestats_<tag>.json  the rejection accounting for this method.

THE REJECTION TALLY IS A RESULT, NOT DIAGNOSTICS. `n_failed` (keys exhausted
after MAX_ATTEMPTS) is the rate at which an arm cannot produce a response that
engages the user at all, and arms differ enormously on it. It must be reported
alongside the endorsement rate, because the endorsement rate is now conditional
on passing the gate: an arm with a 40% failure rate and an arm with a 2% failure
rate have had their endorsement measured on differently-selected subsets of the
held-out prompts, and the numbers are not comparable without it.

`pass_at_1` is the more interpretable per-arm quantity -- the probability that a
single unconditioned draw engages the user -- and is exactly the pre-gate
behaviour the old pipeline was scoring.
"""

import argparse
import json
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from gate_common import (MAX_ATTEMPTS, attempt_paths, final_paths, load_attempt1,
                         normalize)


def load_all_attempts(run_dir, env, tag, max_attempts):
    """Return {attempt: normalised response frame} for every attempt present.

    Attempt 1 is the frozen finaldata generation; later attempts are the
    regeneration rounds. Rounds absent from disk are simply not present in the
    result -- a run that converged at attempt 2 has no attempt 3.
    """
    frames = {1: load_attempt1(run_dir, env, tag)}
    for a in range(2, max_attempts + 1):
        p = attempt_paths(run_dir, tag, a, env)["responses"]
        if p and os.path.exists(p):
            frames[a] = normalize(pd.read_csv(p), env)
    return frames


def load_all_gates(run_dir, tag, max_attempts):
    frames = []
    for a in range(1, max_attempts + 1):
        p = attempt_paths(run_dir, tag, a)["gate"]
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    assert frames, f"no gate labels found for {tag} — run judge_assistant_mode.py first"
    df = pd.concat(frames, ignore_index=True)
    df["side"] = df["side"].fillna("").astype(str)
    df["id"] = df["id"].astype(str)
    return df


def select(gates, responses):
    """First passing attempt per key, plus per-key attempt accounting."""
    rows = []
    for (kid, side), g in gates.groupby(["id", "side"]):
        g = g.sort_values("attempt")
        passing = g[g["gate"] == 1]
        n_tried = int(g["attempt"].max())
        if len(passing) == 0:
            rows.append(dict(id=kid, side=side, winning_attempt=None,
                             n_tried=n_tried, failed=True, response=None,
                             context=None))
            continue
        win = int(passing["attempt"].iloc[0])
        rf = responses[win]
        rf = rf[(rf["id"].astype(str) == kid) & (rf["side"].astype(str) == side)]
        assert len(rf) == 1, (
            f"expected exactly 1 response for id={kid} side={side!r} at attempt "
            f"{win}, found {len(rf)} — attempt frames must be keyed uniquely")
        rows.append(dict(id=kid, side=side, winning_attempt=win, n_tried=n_tried,
                         failed=False, response=rf["response"].iloc[0],
                         context=rf["context"].iloc[0]))
    return pd.DataFrame(rows)


def to_scorer_layout(sel, env):
    """Re-emit the kept rows in the column names the env's scorer expects.

    OEQ/PAS feed stage_heldout_inputs.py, which asserts exactly
    ['sentence','response']; AITA/FLIP feed classify_aita_verdicts.py, which
    reads post_text. `context` is the bare original post in every case, which is
    what the downstream judges score against.
    """
    kept = sel[~sel["failed"]].copy()
    if env == "AITA-NTA-FLIP":
        wide = kept.pivot(index="id", columns="side",
                          values=["context", "response", "winning_attempt"])
        wide.columns = [f"{a}_{b}" for a, b in wide.columns]
        return wide.reset_index().rename(columns={
            "context_original": "post_text",
            "context_flipped": "flipped_post_text",
            "response_original": "response_original",
            "response_flipped": "response_flipped"})
    ctx = "post_text" if env == "AITA" else "sentence"
    return kept[["id", "context", "response", "winning_attempt"]].rename(
        columns={"context": ctx})


def stats(sel, env, tag, run_dir, max_attempts):
    n = len(sel)
    n_failed = int(sel["failed"].sum())
    per_attempt = {}
    for a in range(1, max_attempts + 1):
        per_attempt[str(a)] = int((sel["winning_attempt"] == a).sum())
    n_pass1 = per_attempt["1"]
    # Draws spent = every attempt actually generated for every key. Keys that
    # succeed at attempt k cost k draws; exhausted keys cost max_attempts.
    draws = int(sel["n_tried"].sum())
    return {
        "env": env, "tag": tag, "run_dir": os.path.abspath(run_dir),
        "max_attempts": max_attempts,
        "n_keys": n,
        "n_scored": n - n_failed,
        "n_failed": n_failed,
        "gate_failure_rate": (n_failed / n) if n else None,
        "pass_at_1": (n_pass1 / n) if n else None,
        "resolved_at_attempt": per_attempt,
        "n_draws_spent": draws,
        "mean_draws_per_key": (draws / n) if n else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    args = ap.parse_args()

    gates = load_all_gates(args.run_dir, args.tag, args.max_attempts)
    responses = load_all_attempts(args.run_dir, args.env, args.tag, args.max_attempts)
    sel = select(gates, responses)

    out = final_paths(args.run_dir, args.tag)
    to_scorer_layout(sel, args.env).to_csv(out["final"], index=False)
    st = stats(sel, args.env, args.tag, args.run_dir, args.max_attempts)
    with open(out["stats"], "w") as f:
        json.dump(st, f, indent=2)

    print(f"[assemble] {args.env} / {args.tag}")
    print(f"  scored          {st['n_scored']}/{st['n_keys']}")
    print(f"  REJECTED (gate) {st['n_failed']}  "
          f"({st['gate_failure_rate']:.1%} of keys)")
    print(f"  pass@1          {st['pass_at_1']:.1%}")
    print(f"  resolved at     {st['resolved_at_attempt']}")
    print(f"  draws spent     {st['n_draws_spent']} "
          f"({st['mean_draws_per_key']:.2f}/key)")
    print(f"  wrote {out['final']}")
    print(f"  wrote {out['stats']}")


if __name__ == "__main__":
    main()
