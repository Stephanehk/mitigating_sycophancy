"""Assemble the OEQ and PAS held-out results tables (12 methods, +human for OEQ)
over the fixed 1000-prompt held-out sets, with 95% CIs per cell.

Sections per table:
  (a) action endorsement rate (#1/(#0+#1), paper metric) and neutral rate (#3/N)
  (b) the 4 quality judges (addressee + 3 Lend-an-Ear empathy sub-components)
  (c) the ELEPHANT framing and validation scorers

CI conventions (documented so the table is auditable):
  - endorsement rate: mean +/- 1.96*SEM over the {0,1} label vector, exactly the
    repo's action_endorsement_rate helper. n reported = #0+#1 (the 0/1 subset).
  - neutral rate / addressee / framing / validation (binary proportions):
    95% Wilson score interval (z=1.96), the repo's wilson_ci convention.
  - the 3 empathy sub-scores (ordinal 1-5 -> [0,1] means): mean +/- 1.96*SEM.

All response frames resolve through run_quality_judges_pas.CONDITIONS_BY_ENV so
they are byte-identical to the files the judges scored. Every referenced file is
asserted to exist and to have zero unparsable rows.

Run:
    python training_mitigations/eval/build_results_tables.py
"""

import math
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_RUNS = os.path.join(_HERE, "..", "sycoharm_runs")
_OSF = os.path.join(_HERE, "..", "..", "osfstorage-archive", "Code")
sys.path.insert(0, _OSF)

from run_quality_judges_pas import CONDITIONS_BY_ENV, V5_DIR  # noqa: E402
from evaluate_action_endorsement_stephane import (  # noqa: E402
    first_label, action_endorsement_rate)

# requested methods, in order; display label + condition key
METHODS = [
    ("Qwen3-8B (base)",                     "base"),
    ("+ Cheng et al. prompt",               "explicit"),
    ("+ Sharma et al. prompt",              "sharma"),
    ("+ “Don’t be sycophantic”", "dontbesyco"),
    ("+ “Be critical”",           "becritical"),
    ("+ perspective-shift",                 "perspshift"),
    ("SFT “be critical”",         "becritical_sft"),
    ("Neutral-mixed DPO",                   "neutral_mixed"),
    ("Neutral DPO",                         "dpo_neutral"),
    ("mixed_P5",                            "mixed_P5"),
    ("mixed_P5_nocrit_distill_gp_frozenstake", "mixed_P5_nocrit_distill_gp_frozenstake"),
    ("mixed_P6_perspshift_distill_gp",      "mixed_P6_perspshift_distill_gp"),
]
HUMAN = ("Human baseline", "human")

QJUDGES = [
    ("addressee",                            "addressee"),
    ("empathy: validating_emotions",         "empathy_validating_emotions"),
    ("empathy: demonstrating_understanding", "empathy_demonstrating_understanding"),
    ("empathy: dismissing_emotions (rev)",   "empathy_dismissing_emotions"),
]
EMETRICS = [("framing", "framing"), ("validation", "validation")]


def wilson_ci(count, n, z=1.96):
    """95% Wilson score interval (center, lo, hi) for count/n. Matches
    summarize_sycoharm_fractions.wilson_ci."""
    p = count / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, center - half, center + half


def mean_sem(values, z=1.96):
    """(mean, lo, hi) for a mean +/- z*SEM over a numeric vector."""
    v = np.asarray(values, dtype=float)
    m = v.mean()
    half = z * v.std(ddof=1) / math.sqrt(len(v))
    return m, m - half, m + half


def endorse_labels(env, cond):
    """Concatenated integer endorse labels for one condition. Asserts every
    referenced scored file exists and has zero unparsable labels."""
    files = CONDITIONS_BY_ENV[env][cond]
    labs = []
    for f in files:
        p = os.path.join(V5_DIR, f)
        assert os.path.exists(p), f"missing endorse frame: {p}"
        col = pd.read_csv(p)["endorse_response"]
        parsed = col.map(first_label)
        assert parsed.notna().all(), f"{p}: {int(parsed.isna().sum())} unparsable"
        labs.extend(int(x) for x in parsed)
    return labs


def quality_scores(env, metric, cond):
    """The 0-1 score vector for one (quality judge, condition). Asserts the file
    exists, is the full held-out size, and has no missing score."""
    p = os.path.join(_RUNS, f"quality_judges_{env.lower()}", f"{metric}_{cond}.csv")
    assert os.path.exists(p), f"missing quality file: {p}"
    s = pd.read_csv(p)[f"{metric}_score_response"]
    assert s.notna().all(), f"{p}: {int(s.isna().sum())} unparsable"
    return s.to_numpy(dtype=float)


def elephant_labels(env, metric, cond):
    """The 0/1 label vector for one (ELEPHANT metric, condition)."""
    p = os.path.join(_RUNS, f"elephant_scorers_{env.lower()}", f"{metric}_{cond}.csv")
    assert os.path.exists(p), f"missing elephant file: {p}"
    s = pd.read_csv(p)[f"{metric}_response"]
    assert s.notna().all(), f"{p}: {int(s.isna().sum())} unparsable"
    return s.to_numpy(dtype=float)


def fmt(center, lo, hi):
    return f"{center:.3f} [{lo:.3f}, {hi:.3f}]"


def build_env(env):
    rows = []
    methods = METHODS + ([HUMAN] if env == "OEQ" else [])
    for label, cond in methods:
        labs = endorse_labels(env, cond)
        n = len(labs)
        # (a) endorsement: paper helper (mean +/- 1.96 SEM over {0,1})
        rate, ci, n01 = action_endorsement_rate(labs)
        endorse_cell = f"{rate:.3f} [{rate-ci:.3f}, {rate+ci:.3f}]"
        # (a) neutral rate: #3/N, Wilson
        n3 = sum(1 for x in labs if x == 3)
        neutral_cell = fmt(*wilson_ci(n3, n))
        row = {"method": label, "N": n, "n(0/1)": n01,
               "endorse_rate": endorse_cell, "neutral_rate": neutral_cell}
        # (b) quality judges
        for jlabel, jmetric in QJUDGES:
            if jmetric == "addressee":
                v = quality_scores(env, jmetric, cond)
                row[jlabel] = fmt(*wilson_ci(int(v.sum()), len(v)))  # binary -> Wilson
            else:
                row[jlabel] = fmt(*mean_sem(quality_scores(env, jmetric, cond)))
        # (c) ELEPHANT framing + validation (binary -> Wilson)
        for elabel, emetric in EMETRICS:
            v = elephant_labels(env, emetric, cond)
            row[elabel] = fmt(*wilson_ci(int(v.sum()), len(v)))
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(df, env):
    cols = list(df.columns)
    lines = [f"### {env} — held-out (n reported per column)",
             "| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    for env in ("OEQ", "PAS"):
        df = build_env(env)
        out = os.path.join(_RUNS, f"results_table_{env.lower()}.csv")
        df.to_csv(out, index=False)
        print("\n\n" + to_markdown(df, env))
        print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
