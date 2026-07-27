"""Explode a --gate-loop run's consolidated output into the per-attempt files
that assemble_gate.py reads.

WHY THIS EXISTS. The two gate implementations write different artifacts. The
staged path (marlowe_gate_round.sh) emits one `responses_<tag>_gateattempt<A>.csv`
and one `gate_<tag>_a<A>.csv` per round, and assemble_gate.py reads exactly those.
The in-loop path (eval_endorsement_heldout.py --gate-loop) runs rounds 2..N inside
a single job and emits only the consolidated `responses_<tag>_gateloop.csv` with
`winning_attempt`/`gate_passed`/`n_tried` columns. Running assemble_gate.py against
an in-loop run without this shim silently finds no attempt>=2 labels and counts
every rescued key as a GATE FAILURE -- inflating n_filtered and shrinking the
scored set, with no error raised.

Attempt 1 is never touched: `gate_<tag>_a1.csv` is the frozen login-node judging
and is the input this run was built from.

Usage:
    python eval/explode_gateloop.py --run-dir <run> --env OEQ --tag <tag>
"""

import argparse
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from gate_common import AITA_ENVS, MAX_ATTEMPTS, attempt_paths, gate_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    a = ap.parse_args()

    stem = "aita_responses" if a.env in AITA_ENVS else "responses"
    src = os.path.join(gate_dir(a.run_dir), f"{stem}_{a.tag}_gateloop.csv")
    if not os.path.exists(src):
        sys.exit(f"no gate-loop output at {src} — did the resample job run?")
    df = pd.read_csv(src)
    for c in ("id", "winning_attempt", "gate_passed", "n_tried"):
        assert c in df.columns, f"{src} missing {c}; got {list(df.columns)}"
    df["side"] = df["side"] if "side" in df.columns else ""
    df["side"] = df["side"].fillna("").astype(str)

    # One gate row per (key, attempt) actually drawn in this loop: rounds 2..n_tried.
    # A key that passed scores 1 on its winning attempt and 0 on the rounds before
    # it; a key that never passed scores 0 on every round it was given.
    gate_rows = []
    for _, r in df.iterrows():
        passed = bool(r["gate_passed"])
        win = int(r["winning_attempt"]) if passed and pd.notna(r["winning_attempt"]) else None
        for att in range(2, int(r["n_tried"]) + 1):
            gate_rows.append({"id": r["id"], "side": r["side"], "attempt": att,
                              "gate": 1 if (win is not None and att == win) else 0,
                              "raw": "", "ptok": "", "ctok": ""})
    g = pd.DataFrame(gate_rows)

    wrote = []
    for att in sorted(set(g["attempt"])) if len(g) else []:
        p = attempt_paths(a.run_dir, a.tag, int(att), a.env)["gate"]
        g[g["attempt"] == att].to_csv(p, index=False)
        wrote.append(os.path.basename(p))

    # Responses are only ever read for a key's WINNING attempt, so a round with no
    # winners needs no responses file; writing one anyway would be harmless but
    # would misrepresent what was kept.
    ctx = "post_text" if a.env in AITA_ENVS else "sentence"
    winners = df[df["gate_passed"].astype(bool)]
    for att in sorted(set(winners["winning_attempt"].dropna().astype(int))):
        if att < 2:
            continue
        p = attempt_paths(a.run_dir, a.tag, int(att), a.env)["responses"]
        sub = winners[winners["winning_attempt"] == att]
        assert ctx in sub.columns, f"{src} missing {ctx}; got {list(sub.columns)}"
        sub[["id", ctx, "response"]].to_csv(p, index=False)
        wrote.append(os.path.basename(p))

    n_fail = int((~df["gate_passed"].astype(bool)).sum())
    print(f"[explode] {a.env} / {a.tag}: {len(df)} resampled keys, "
          f"{len(df) - n_fail} rescued, {n_fail} exhausted")
    print(f"[explode] wrote {', '.join(wrote) if wrote else '(nothing)'}")


if __name__ == "__main__":
    main()
