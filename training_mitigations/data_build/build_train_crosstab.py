"""Cross-tab the v5 endorse-judge labels against stakeholder veto status on the
sampled TRAINING responses (built by build_train_judge_sample.py, judged by
run_endorse_gpt5mini.py --tag train).

For each (run, iter) it reports:
  - the v5 label distribution (0 challenge / 1 explicit-affirm / 2 implicit /
    3 neutral) and the overall veto rate,
  - frac of EXPLICIT-ENDORSE (label 1) responses that are vetoed,
  - frac of VETOED responses that are explicit-endorse (label 1),
each with a 95% Wilson CI (z=1.96).

The judge preserves input row order, so scored_<name>.csv row i corresponds to
the meta row i for that dataset (joined on (dataset, row)).

Run from the repo root (after the judge finishes):
    python training_mitigations/build_train_crosstab.py
"""

import math
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, "..", "osfstorage-archive", "Code"))
from evaluate_action_endorsement_stephane import first_label  # noqa: E402

SYCO = os.path.join(_HERE, "sycoharm_runs")
SCORED_DIR = os.path.join(SYCO, "endorse_gpt5mini_v5_temp0_train")
META = os.path.join(SYCO, "train_judge_sample_meta.csv")
RUNS = ["n1000", "mixed", "selfpref"]
ITERS = [1, 2, 3]


def wilson(c, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = c / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return ctr - half, ctr + half


def frac(c, n):
    lo, hi = wilson(c, n)
    return f"{(c / n if n else float('nan')):.3f} [{lo:.3f}, {hi:.3f}] ({c}/{n})"


def main():
    meta = pd.read_csv(META)
    out = []
    for run in RUNS:
        for it in ITERS:
            name = f"train_{run}_iter{it}"
            sp = os.path.join(SCORED_DIR, f"scored_{name}.csv")
            assert os.path.exists(sp), f"missing {sp}"
            sc = pd.read_csv(sp).reset_index(drop=True)
            m = meta[meta["dataset"] == name].sort_values("row").reset_index(drop=True)
            assert len(sc) == len(m), f"{name}: scored {len(sc)} vs meta {len(m)}"
            lab = [first_label(x) for x in sc["raw"]]
            assert all(x is not None for x in lab), f"{name}: unparsable labels"
            df = pd.DataFrame({"lab": lab, "vetoed": m["vetoed"].values})
            n = len(df)
            counts = {k: int((df["lab"] == k).sum()) for k in range(4)}
            n_vet = int(df["vetoed"].sum())
            endorse = df["lab"] == 1                       # explicit endorse
            n_end = int(endorse.sum())
            end_and_vet = int((endorse & df["vetoed"]).sum())
            out.append({
                "run": run, "iter": it, "n": n,
                "labels 0/1/2/3": "/".join(str(counts[k]) for k in range(4)),
                "veto_rate": frac(n_vet, n),
                "frac_endorse_vetoed": frac(end_and_vet, n_end),
                "frac_vetoed_endorse": frac(end_and_vet, n_vet),
            })
    res = pd.DataFrame(out)
    pd.set_option("display.max_colwidth", None, "display.width", 200)
    print(res.to_string(index=False))
    res.to_csv(os.path.join(SYCO, "train_veto_endorse_crosstab.csv"), index=False)
    print(f"\nwrote {os.path.join(SYCO, 'train_veto_endorse_crosstab.csv')}")


if __name__ == "__main__":
    main()
