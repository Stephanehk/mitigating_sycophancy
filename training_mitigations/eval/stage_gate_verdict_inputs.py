"""Bridge the STAGED gate path into the consolidated frame classify_gate_verdicts reads.

WHY THIS EXISTS. The two gate implementations write different artifacts, and the
verdict stage only understands one of them:

  in-loop  (eval_aita_label_heldout.py --gate-loop): rounds 2..N run inside a
           single job and emit ONE consolidated
           `aita_responses_<tag>_gateloop.csv` + `summary_aita_<tag>_gateloop.json`
           holding just the rescued keys. classify_gate_verdicts.py globs for
           that summary and reads that csv. The 8B AITA/FLIP arms took this path.

  staged   (marlowe_gate_round.sh / the weekend drip-feeder): each round is its
           own job, emitting `aita_responses_<tag>_gateattempt<A>.csv` +
           `gate_<tag>_a<A>.csv`. assemble_gate.py reads those directly (this is
           the path explode_gateloop.py does NOT apply to -- explode goes the
           other way, in-loop -> per-attempt). The 32B AITA/FLIP arms took this
           path, so classify_gate_verdicts.py finds nothing for them at all.

This reconstructs the in-loop frame from `gatefinal_<tag>.csv`, which
assemble_gate.py has already reduced to the FIRST gate-passing attempt per
(id, side) key. Run assemble_gate.py first.

WHICH KEYS COUNT AS RESCUED. A key is rescued iff its winning_attempt >= 2, i.e.
its attempt-1 draw failed the gate and a later draw passed. Deriving this from
assemble's per-(id, side) winning_attempt -- rather than from the gate loop's
resample stamps -- is what avoids the AITA-NTA-FLIP double-count: gate_loop.py
resamples BOTH sides of a pair when EITHER side fails, so a side whose attempt-1
response already passed still gets restamped. Its winning_attempt stays 1 here,
so it is correctly left to the frozen labels and never re-judged.
(build_5judge_gated_tables.py applies the same correction downstream.)

Keys exhausted after MAX_ATTEMPTS are absent from gatefinal entirely and so are
absent here -- they are a reported gate failure, not a verdict.

Outputs, per arm, only when at least one key was rescued:
  <run>/eval/gate/aita_responses_<tag>_gateloop.csv
  <run>/eval/gate/summary_aita_<tag>_gateloop.json   (env + adapter_tag)

Usage:
    python eval/stage_gate_verdict_inputs.py --run-dir <run> --env AITA --tag <tag>
"""

import argparse
import json
import os

import pandas as pd

SIDES = {"AITA": [("single", "winning_attempt", "response", "gate_passed")],
         "AITA-NTA-FLIP": [
             ("original", "winning_attempt_original", "response_original",
              "gate_passed_original"),
             ("flipped", "winning_attempt_flipped", "response_flipped",
              "gate_passed_flipped")]}


def build(run_dir, env, tag):
    gd = os.path.join(run_dir, "eval", "gate")
    gf = os.path.join(gd, f"gatefinal_{tag}.csv")
    if not os.path.exists(gf):
        print(f"[stage-verdict] {tag}: no gatefinal — run assemble_gate.py first")
        return None
    g = pd.read_csv(gf)

    spec = SIDES[env]
    missing = [c for _, wa, resp, _ in spec for c in (wa, resp)
               if c not in g.columns]
    if missing:
        raise AssertionError(f"{gf}: missing {missing} (has {list(g.columns)})")

    # A row is carried over iff at least one of its sides was rescued; per-side
    # gate_passed then marks WHICH sides the classifier should judge.
    keep = pd.Series(False, index=g.index)
    for _, wa_col, _, _ in spec:
        keep |= g[wa_col].fillna(0).astype(float) >= 2
    out = g[keep].copy()
    if out.empty:
        print(f"[stage-verdict] {tag}: no rescued keys — nothing to judge")
        return None
    for _, wa_col, _, gate_col in spec:
        out[gate_col] = (out[wa_col].fillna(0).astype(float) >= 2).astype(int)

    out_csv = os.path.join(gd, f"aita_responses_{tag}_gateloop.csv")
    out.to_csv(out_csv, index=False)

    # adapter_tag is the SHORT tag (no _finaldata_temp1.0_seed0): it is what
    # classify_gate_verdicts names its output with, and the 8B files it must sit
    # alongside are aita_scored_<short>_gateloop.csv.
    short = tag.split("_finaldata")[0]
    summ = os.path.join(gd, f"summary_aita_{tag}_gateloop.json")
    with open(summ, "w") as f:
        json.dump({"env": env, "run_dir": os.path.abspath(run_dir),
                   "adapter_tag": short, "n_rows": int(len(out)),
                   "source": "staged gate rounds via stage_gate_verdict_inputs.py"},
                  f, indent=2)
    per_side = {side: int(out[gate_col].sum()) for side, _, _, gate_col in spec}
    print(f"[stage-verdict] {tag}: {len(out)} rescued rows {per_side} -> {out_csv}")
    return out_csv


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--env", required=True, choices=tuple(SIDES))
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    build(a.run_dir, a.env, a.tag)


if __name__ == "__main__":
    main()
