"""Plot-ready MMLU data for the rebuttal (2026-11-06): numeric versions of the
mmlu_mcnemar.py table, plus per-subject accuracy.

Writes, under --out-dir:
  mmlu_summary.csv     model, n, n_correct, acc, acc_lo, acc_hi (95% Wilson),
                       delta_vs_base, b (base right / model wrong), c (reverse),
                       p_mcnemar (exact, two-sided), p_holm (across the 4 adapters)
  mmlu_by_subject.csv  model, subject, n, n_correct, acc

Reuses mmlu_mcnemar's loader, McNemar test and Holm correction verbatim, so
these numbers are the printed table's. Asserts every model has all 14,042
questions and the same question set as base.
"""

import argparse
import os
import sys

import pandas as pd
from scipy.stats import binomtest

_EVAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EVAL)

from mmlu_mcnemar import NAMES, N_EXPECTED, holm, load_correct  # noqa: E402
from xenv_score_table import wilson  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dir holding base/ oeq/ pas/ aita/ flip/")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data = {n: load_correct(os.path.join(args.root, n)) for n in NAMES}
    for n, d in data.items():
        assert len(d) == N_EXPECTED, f"{n}: {len(d)} questions != {N_EXPECTED}"
        assert d.keys() == data["base"].keys(), f"{n}: question set differs from base"
    base = data["base"]
    base_acc = sum(base.values()) / len(base)

    rows, ps = [], []
    for n in NAMES:
        cur = data[n]
        k = sum(cur.values())
        acc, lo, hi = wilson(k, len(cur))
        row = {"model": n, "n": len(cur), "n_correct": k, "acc": acc, "acc_lo": lo, "acc_hi": hi,
               "delta_vs_base": acc - base_acc, "b": None, "c": None,
               "p_mcnemar": None, "p_holm": None}
        if n != "base":
            b = sum(1 for q in cur if base[q] == 1 and cur[q] == 0)
            c = sum(1 for q in cur if base[q] == 0 and cur[q] == 1)
            row.update(b=b, c=c, p_mcnemar=binomtest(b, b + c, 0.5).pvalue)
            ps.append(row["p_mcnemar"])
        rows.append(row)
    for row, p_adj in zip(rows[1:], holm(ps)):
        row["p_holm"] = p_adj
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "mmlu_summary.csv"), index=False)

    subj = []
    for n in NAMES:
        per = {}
        for (s, _), v in data[n].items():
            t = per.setdefault(s, [0, 0])
            t[0] += 1
            t[1] += v
        for s, (cnt, k) in sorted(per.items()):
            subj.append({"model": n, "subject": s[len("mmlu_"):], "n": cnt,
                         "n_correct": k, "acc": k / cnt})
    sdf = pd.DataFrame(subj)
    assert sdf.groupby("model").subject.nunique().eq(57).all(), "expected 57 subjects per model"
    sdf.to_csv(os.path.join(args.out_dir, "mmlu_by_subject.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"-> {args.out_dir}/mmlu_summary.csv, mmlu_by_subject.csv")


if __name__ == "__main__":
    main()
