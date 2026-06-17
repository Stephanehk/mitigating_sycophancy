"""Rebuild sycoharm_runs/fractions_summary.csv from the scored_<name>.csv
files produced by judge_sycophancy.py.

For each dataset it reports, per judge level (critical/neutral/mild/
substantial) plus the combined mild_plus_substantial bucket: the count, the
fraction of all n rows, and a 95% Wilson CI with z = 1.96 (hand-rolled; the
original ad-hoc summary used 1.96 rather than scipy's 1.95996..., and this
reproduces its values bit-for-bit).

Assumptions (asserted): every scored file exists for the datasets listed in
SOURCES, has a sycoharm_score_response column with no NaN (no unparsable /
content-filtered judge outputs), and scores are exactly k/3 for k in 0..3.

Run from anywhere:
    python training_mitigations/summarize_sycoharm_fractions.py
"""

import math
import os

import pandas as pd

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(_HERE, "sycoharm_runs")
OUT = os.path.join(RUNS, "fractions_summary.csv")

LEVEL_NAMES = ["critical", "neutral", "mild", "substantial"]

# dataset name (scored_<name>.csv) -> human-readable source, in output order
SOURCES = {
    "heldout_pas": "heldout Qwen3-8B base",
    "heldout_oeq": "heldout Qwen3-8B base",
    "heldout_pas_becritical": "heldout Qwen3-8B base + 'Be critical.' prompt",
    "heldout_oeq_becritical": "heldout Qwen3-8B base + 'Be critical.' prompt",
    "heldout_pas_dontbesyco": "heldout Qwen3-8B base + 'Don't be sycophantic.' prompt",
    "heldout_oeq_dontbesyco": "heldout Qwen3-8B base + 'Don't be sycophantic.' prompt",
    "heldout_pas_trained": "heldout Qwen3-8B stakeholder-DPO (stakeholders-postraining-v1)",
    "heldout_oeq_trained": "heldout Qwen3-8B stakeholder-DPO (stakeholders-postraining-v1)",
    "pas_deepseek": "saved responses from Myra's paper",
    "pas_gpt4o": "saved responses from Myra's paper",
    "oeq_deepseek": "saved responses from Myra's paper",
    "oeq_gpt4o": "saved responses from Myra's paper",
    # Freshly generated baseline responses (single user turn, no system prompt)
    # on the same 500-prompt PAS/OEQ sets, judged with the same sycoharm judge.
    "pas_gpt5": "newly generated, gpt-5.4 (baseline)",
    "oeq_gpt5": "newly generated, gpt-5.4 (baseline)",
    "pas_gpt5mini": "newly generated, gpt-5.4-mini (baseline)",
    "oeq_gpt5mini": "newly generated, gpt-5.4-mini (baseline)",
    "pas_sonnet": "newly generated, claude-sonnet-4-6 (baseline)",
    "oeq_sonnet": "newly generated, claude-sonnet-4-6 (baseline)",
    "pas_deepseekv32": "newly generated, deepseek-v3.2 (baseline)",
    "oeq_deepseekv32": "newly generated, deepseek-v3.2 (baseline)",
}


def wilson_ci(count, n, z=1.96):
    """95% Wilson score interval for count successes out of n."""
    p = count / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def bucket_stats(count, n):
    """Fraction of n plus 95% Wilson CI for a level count."""
    lo, hi = wilson_ci(count, n)
    return {"count": count, "frac": count / n, "ci_lo": lo, "ci_hi": hi}


def summarize(name, source):
    """One summary row for scored_<name>.csv: per-level counts/fracs/CIs and
    the combined mild_plus_substantial bucket."""
    path = os.path.join(RUNS, f"scored_{name}.csv")
    assert os.path.exists(path), f"missing {path}"
    scores = pd.read_csv(path)["sycoharm_score_response"]
    assert scores.notna().all(), f"{name}: {scores.isna().sum()} unparsable rows"
    levels = (scores * 3).round().astype(int)
    assert levels.isin(range(4)).all(), f"{name}: scores outside 0..3/3"
    n = len(levels)

    row = {"dataset": name, "n": n, "source": source}
    for lvl, lvl_name in enumerate(LEVEL_NAMES):
        for k, v in bucket_stats(int((levels == lvl).sum()), n).items():
            row[f"{lvl_name}_{k}"] = v
    for k, v in bucket_stats(int((levels >= 2).sum()), n).items():
        row[f"mild_plus_substantial_{k}"] = v
    return row


def main():
    df = pd.DataFrame([summarize(name, src) for name, src in SOURCES.items()])
    df.to_csv(OUT, index=False)
    print(df[["dataset", "n", "critical_frac", "neutral_frac", "mild_frac",
              "substantial_frac", "mild_plus_substantial_frac"]]
          .to_string(index=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
