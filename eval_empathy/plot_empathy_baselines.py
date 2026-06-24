"""Plot the empathy-baseline comparison written by run_empathy_baselines.py.

Reads the scores CSV
    eval_empathy/results/empathy_baselines_<gen>_<judge>.csv
with columns: id, condition, validating_emotions,
demonstrating_understanding, dismissing_emotions, (+ token counts).

Writes one PNG with three panels (one per Lend an Ear sub-component). Each
panel is a bar chart of the per-method mean 1-5 score, with a standard-error
error bar and the per-bar n annotated. Colors and display names are pulled
from plot_explore_mitigations so the methods read consistently with the rest
of the project's figures.

dismissing_emotions is proscriptive (higher = LESS empathic); this is
annotated in its panel title so the direction isn't misread. The other two
are prescriptive (higher = more empathic).

Only the raw per-response scores in the CSV are plotted — means and standard
errors are computed directly from them, with no interpolation or synthesis.
A method with zero parsed scores for a sub-component fails loudly rather than
rendering a missing/zero bar.

CLI:
  --csv PATH    scores CSV (default: the single empathy_baselines_*.csv in
                results/, error if there are zero or several)
  --out PATH    output PNG (default: <csv-stem>.png next to the CSV)
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
MITIGATIONS_DIR = os.path.join(REPO_ROOT, "mitigations")
# empathy_judge_prompt now lives in training_mitigations/eval/.
EVAL_DIR = os.path.join(REPO_ROOT, "training_mitigations", "eval")
sys.path.insert(0, HERE)
sys.path.insert(0, MITIGATIONS_DIR)
sys.path.insert(0, EVAL_DIR)

from plot_explore_mitigations import CONDITION_COLORS, display_name
from empathy_judge_prompt import LEND_AN_EAR_SUBCOMPONENTS

RESULTS_DIR = os.path.join(HERE, "results")
SUB_KEYS = list(LEND_AN_EAR_SUBCOMPONENTS.keys())
# Method order left-to-right in each panel (canonical condition keys written
# by run_empathy_baselines.py).
METHOD_ORDER = ["baseline", "be_critical", "stakeholder_concerns_v2"]


def find_default_csv():
    """Return the lone empathy_baselines_*.csv in results/, else error."""
    matches = sorted(glob.glob(os.path.join(RESULTS_DIR, "empathy_baselines_*.csv")))
    assert matches, (
        f"no empathy_baselines_*.csv in {RESULTS_DIR}; run "
        "run_empathy_baselines.py first or pass --csv"
    )
    assert len(matches) == 1, (
        f"multiple candidate CSVs in {RESULTS_DIR}; pass --csv explicitly: "
        f"{[os.path.basename(m) for m in matches]}"
    )
    return matches[0]


def load_scores(path):
    """Read the scores CSV; assert schema and 1-5 score range."""
    assert os.path.exists(path), f"missing scores CSV: {path}"
    df = pd.read_csv(path)
    for col in ("id", "condition", *SUB_KEYS):
        assert col in df.columns, f"{path}: missing column {col!r}"
    for sub in SUB_KEYS:
        vals = df[sub].dropna().unique()
        bad = [v for v in vals if int(v) not in (1, 2, 3, 4, 5)]
        assert not bad, f"{path}: {sub} has out-of-range values {bad}"
    methods_present = set(df["condition"].unique())
    missing = [m for m in METHOD_ORDER if m not in methods_present]
    assert not missing, f"{path}: methods absent from CSV: {missing}"
    return df


def mean_se_n(df, method, sub):
    """Return (mean, standard_error, n) of the non-null `sub` scores for one
    method. Errors if the method has zero parsed scores (an undefined mean
    must not be drawn as a bar)."""
    vals = df[df["condition"] == method][sub].dropna().astype(float)
    assert len(vals) > 0, f"method {method!r} has zero parsed {sub!r} scores"
    n = len(vals)
    se = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return float(vals.mean()), se, n


def plot(df, out_path):
    fig, axes = plt.subplots(1, len(SUB_KEYS), figsize=(5.0 * len(SUB_KEYS), 4.2))
    if len(SUB_KEYS) == 1:
        axes = [axes]
    x = np.arange(len(METHOD_ORDER))
    for method in METHOD_ORDER:
        assert method in CONDITION_COLORS, f"no color for method {method!r}"
    colors = [CONDITION_COLORS[m] for m in METHOD_ORDER]
    labels = [display_name(m) for m in METHOD_ORDER]

    for ax, sub in zip(axes, SUB_KEYS):
        means, ses, ns = [], [], []
        for method in METHOD_ORDER:
            m, se, n = mean_se_n(df, method, sub)
            means.append(m)
            ses.append(se)
            ns.append(n)
        ax.bar(x, means, yerr=ses, capsize=4, color=colors,
               edgecolor="black", linewidth=0.5)
        for xi, m, se, n in zip(x, means, ses, ns):
            ax.text(xi, m + se + 0.06, f"{m:.2f}\n(n={n})",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(1, 5.5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.grid(alpha=0.3, axis="y")
        sub_obj = LEND_AN_EAR_SUBCOMPONENTS[sub]
        direction = ("higher = more empathic"
                     if sub_obj.direction == "prescriptive"
                     else "higher = LESS empathic")
        ax.set_title(f"{sub_obj.name}\n(1-5; {direction})", fontsize=10)

    fig.suptitle(
        "Lend an Ear empathy sub-components by method "
        "(EPITOME emotional-reactions prompts)", fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", default=None, help="scores CSV (default: the lone "
                   "empathy_baselines_*.csv in results/)")
    p.add_argument("--out", default=None, help="output PNG path")
    args = p.parse_args()

    csv_path = args.csv or find_default_csv()
    out_path = args.out or (os.path.splitext(csv_path)[0] + ".png")
    df = load_scores(csv_path)
    plot(df, out_path)


if __name__ == "__main__":
    main()
