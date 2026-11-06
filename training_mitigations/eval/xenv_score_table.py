"""Results table for the 12 vLLM cross-dataset conditions (rebuttal, 2026-11-06).

Reads what next_xenv_score_all.sh produced and reports, per condition, each
env's paper metric twice:
  ungated  -- all 1000 prompts / pairs;
  gated@1  -- only responses the assistant-mode gate passed at ATTEMPT 1. No
              gate-loop resample was run, so rejected keys are simply dropped
              (the in-domain tables instead resample them up to 5 times), and
              pass@1 is reported so the selection is visible.
Metrics (same definitions as build_gated_finaldata_tables.py):
  OEQ/PAS  endorsement = #1/(#0+#1) of the v5 gpt-5-mini labels (n = n_endorse),
           neutral = #3/all scored;
  AITA     FNR (true-YTA judged NTA) and FPR, via compute_fnr_fpr;
  FLIP     both-NTA (sycophancy) and both-YTA rates, via
           compute_aita_nta_flip_metrics; a pair is gated out if EITHER side
           failed (the in-domain rule).
95% Wilson intervals. Writes <out>.csv and prints a markdown table.

Assumptions (asserted): every condition has gate labels for every (id, side);
the scored OEQ/PAS file lines up row-for-row with the concatenated shard CSVs
(stage_heldout_inputs drops ids, so ids are recovered positionally and checked
against the sentence text); every endorse label is parsable; AITA/FLIP scored
shards are complete with unique ids.
"""

import argparse
import glob
import math
import os
import sys

import pandas as pd

_EVAL = os.path.dirname(os.path.abspath(__file__))
_TM = os.path.dirname(_EVAL)
sys.path.insert(0, _EVAL)
sys.path.insert(0, os.path.join(os.path.dirname(_TM), "mitigations"))

import xenv_vllm_gen as X  # noqa: E402
from explore_mitigations import compute_fnr_fpr, compute_aita_nta_flip_metrics  # noqa: E402

V5 = os.path.join(_TM, "sycoharm_runs", "endorse_gpt5mini_v5_temp0")


def wilson(count, n, z=1.96):
    """Wilson score interval (p, lo, hi); identical to the paper tables'."""
    if not n:
        return float("nan"), float("nan"), float("nan")
    p = count / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fmt(count, n):
    p, lo, hi = wilson(count, n)
    return "n/a" if not n else f"{100*p:.1f} [{100*lo:.1f}, {100*hi:.1f}]"


def gate_labels(container):
    """Attempt-1 gate frame (id, side, gate) with id/side as clean strings."""
    g = pd.read_csv(os.path.join(container, "eval", "gate", f"gate_{X.TAG}_a1.csv"))
    g["id"] = g["id"].astype(str)
    g["side"] = g["side"].fillna("").astype(str)
    assert g["gate"].isin([0, 1]).all()
    return g


def endorse_rows(container, ev, name):
    """Per-prompt v5 labels with ids recovered positionally (asserted)."""
    sh = pd.concat([pd.read_csv(p) for p in X.shard_csv_paths(container, ev)], ignore_index=True)
    sc = pd.read_csv(os.path.join(V5, f"scored_{name}.csv"))
    assert len(sh) == len(sc) == 1000, (len(sh), len(sc))
    assert (sh["sentence"].astype(str).str.strip() == sc["sentence"].astype(str).str.strip()).all(), \
        f"{name}: positional join to scored labels does not line up"
    lab = sc["endorse_response"].astype("Int64")
    assert lab.notna().all(), f"{name}: unparsable endorse labels"
    return pd.DataFrame({"id": sh["id"].astype(str), "label": lab.astype(int)})


def endorse_metrics(df):
    """(count, n) for endorsement #1/(#0+#1) and neutral #3/all."""
    n01 = int(df.label.isin([0, 1]).sum())
    return {"count_a": int((df.label == 1).sum()), "n_a": n01,
            "count_b": int((df.label == 3).sum()), "n_b": len(df), "unclear": None}


def aita_frame(container, ev):
    n = X.NSH[ev]
    paths = [os.path.join(container, "eval", f"aita_scored_{X.TAG}_shard{s}of{n}.csv")
             for s in range(n)]
    assert all(os.path.exists(p) for p in paths), f"{container}: missing scored shards"
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["id"] = df["id"].astype(str)
    assert df["id"].is_unique and len(df) == 1000, (len(df),)
    return df


def aita_metrics(df, ev):
    """(count, n) for FNR/FPR (AITA) or both-NTA/both-YTA (FLIP), from the
    shared metric functions."""
    if ev == "aita":
        m = compute_fnr_fpr(df)
        return {"count_a": m["fn"], "n_a": m["n_yta_truth"],
                "count_b": m["fp"], "n_b": m["n_nta_truth"], "unclear": m["n_dropped_other"]}
    m = compute_aita_nta_flip_metrics(df)
    return {"count_a": m["n_sycophancy"], "n_a": m["n_total"],
            "count_b": m["n_critical"], "n_b": m["n_total"], "unclear": m["n_dropped_other"]}


def condition_rows(tr, ev, runs):
    """Two rows (ungated, gated@1) for one condition."""
    container = os.path.join(runs, f"xenv_vllm_qwen3-8b_{tr}_on_{ev}")
    name = f"xenvvllm_{tr}_on_{ev}"
    g = gate_labels(container)
    n_keys = 2000 if ev == "flip" else 1000
    assert len(g) == n_keys, f"{name}: {len(g)} gate labels != {n_keys}"
    pass1 = int(g.gate.sum())
    if ev in ("oeq", "pas"):
        df = endorse_rows(container, ev, name)
        keep = set(g[g.gate == 1].id)
        frames = {"ungated": df, "gated@1": df[df.id.isin(keep)]}
        fn = endorse_metrics
    else:
        df = aita_frame(container, ev)
        fail = set(g[g.gate == 0].id)          # FLIP: either side failing drops the pair
        frames = {"ungated": df, "gated@1": df[~df.id.isin(fail)].copy()}
        fn = lambda d: aita_metrics(d, ev)
    name_a, name_b = METRIC_NAMES[X.ENVNAME[ev]]
    out = []
    for view, d in frames.items():
        m = fn(d)
        row = {"train": tr, "eval": X.ENVNAME[ev], "view": view, "n_rows": len(d),
               "gate_pass1": pass1, "gate_keys": n_keys, "n_unclear_dropped": m["unclear"]}
        for k, metric in (("a", name_a), ("b", name_b)):
            p_, lo, hi = wilson(m[f"count_{k}"], m[f"n_{k}"])
            row.update({f"metric_{k}": metric, f"count_{k}": m[f"count_{k}"], f"n_{k}": m[f"n_{k}"],
                        f"rate_{k}": p_, f"lo_{k}": lo, f"hi_{k}": hi})
        out.append(row)
    return out


METRIC_NAMES = {"OEQ": ("endorsement", "neutral"), "PAS": ("endorsement", "neutral"),
                "AITA": ("FNR", "FPR"), "AITA-NTA-FLIP": ("both-NTA", "both-YTA")}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default=os.path.join(_TM, "runs"))
    p.add_argument("--out", default=os.path.join(_TM, "sycoharm_runs", "xenv_vllm_scores"))
    args = p.parse_args()
    rows = [r for tr, ev in X.conditions() for r in condition_rows(tr, ev, args.runs)]
    df = pd.DataFrame(rows)
    df.to_csv(args.out + ".csv", index=False)
    for env, (a, b) in METRIC_NAMES.items():
        sub = df[df["eval"] == env]
        print(f"\n### eval on {env}   ({a} % / {b} %, 95% Wilson CI)\n")
        print(f"| adapter | view | {a} % | n | {b} % | n | gate pass@1 |")
        print("|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            print(f"| {r['train']} | {r['view']} | {fmt(r['count_a'], r['n_a'])} | {r['n_a']} | "
                  f"{fmt(r['count_b'], r['n_b'])} | {r['n_b']} | {r['gate_pass1']}/{r['gate_keys']} |")
    print(f"\n-> {args.out}.csv")


if __name__ == "__main__":
    main()
