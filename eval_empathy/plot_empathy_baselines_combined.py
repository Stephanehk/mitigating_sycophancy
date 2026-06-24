"""Combined empathy figure across generation models, in plots_combined/ style.

Reads the per-model scores CSVs written by run_empathy_baselines.py
    eval_empathy/results/empathy_baselines_<gen-slug>_<judge-slug>.csv
and lays them out as a 3-row x N-model-column grid:
  - rows    = the three Lend an Ear sub-components (SUB_KEYS)
  - columns = the generation models in GEN_MODELS order
Each cell is a per-method bar chart of the mean 1-5 score (with a
standard-error bar and per-bar n), colored by the shared
plot_explore_mitigations.CONDITION_COLORS palette so methods read
consistently with the rest of the project's figures.

dismissing_emotions is proscriptive (higher = LESS empathic); annotated in
the row y-label. The other two are prescriptive. Only raw per-response scores
are plotted (means/SEs computed directly from them via the shared helpers in
plot_empathy_baselines.py); a model x method x sub cell with zero parsed
scores fails loudly rather than rendering an empty bar.

CLI:
  --judge-slug SLUG   judge slug in the CSV filenames (default: gpt-5.4-mini)
  --out PATH          output PNG (default: results/empathy_baselines_combined.png)
"""

import argparse
import os
import sys

import numpy as np
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
from plot_empathy_baselines import (
    RESULTS_DIR, SUB_KEYS, METHOD_ORDER, load_scores, mean_se_n,
)

# Generation models as (filename-slug, column display name), left to right.
# Slugs match model_slug() of the --model passed to run_empathy_baselines.py.
GEN_MODELS = [
    ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ("gpt-5.4", "GPT-5.4"),
    ("deepseek-v3.2", "DeepSeek-V3.2"),
]


def csv_path(gen_slug, judge_slug):
    return os.path.join(
        RESULTS_DIR, f"empathy_baselines_{gen_slug}_{judge_slug}.csv",
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--judge-slug", default="gpt-5.4-mini",
                   help="judge slug in the CSV filenames (default: gpt-5.4-mini)")
    p.add_argument("--out", default=os.path.join(
        RESULTS_DIR, "empathy_baselines_combined.png"))
    args = p.parse_args()

    # Load every model's scores up front so a missing CSV fails before drawing.
    per_model = []
    for gen_slug, display in GEN_MODELS:
        path = csv_path(gen_slug, args.judge_slug)
        per_model.append((display, load_scores(path)))

    for method in METHOD_ORDER:
        assert method in CONDITION_COLORS, f"no color for method {method!r}"
    colors = [CONDITION_COLORS[m] for m in METHOD_ORDER]
    labels = [display_name(m) for m in METHOD_ORDER]
    x = np.arange(len(METHOD_ORDER))

    n_rows, n_cols = len(SUB_KEYS), len(GEN_MODELS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.6 * n_cols, 3.4 * n_rows),
        sharex="col", sharey="row", squeeze=False,
    )

    for col_idx, (display, df) in enumerate(per_model):
        for row_idx, sub in enumerate(SUB_KEYS):
            ax = axes[row_idx][col_idx]
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
                        ha="center", va="bottom", fontsize=7)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
            ax.set_ylim(1, 5.6)
            ax.set_yticks([1, 2, 3, 4, 5])
            ax.grid(alpha=0.3, axis="y")
            if row_idx == 0:
                ax.set_title(display, fontsize=11)
            if col_idx == 0:
                sub_obj = LEND_AN_EAR_SUBCOMPONENTS[sub]
                direction = ("higher = more empathic"
                             if sub_obj.direction == "prescriptive"
                             else "higher = LESS empathic")
                ax.set_ylabel(f"{sub_obj.name}\n(1-5; {direction})", fontsize=9)

    fig.suptitle(
        "Lend an Ear empathy sub-components by method and generation model "
        "(EPITOME emotional-reactions prompts; judge: "
        f"{args.judge_slug})", fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
