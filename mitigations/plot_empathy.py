"""Combined empathy-score plot per env.

Reads the per-(env, family) outputs of `judge_empathy.py`:
    mitigations/results/empathy_<env>_<family>_judge-<judge-slug>.csv
columns: id, condition, validating_emotions, demonstrating_understanding,
         dismissing_emotions, judge_input_tokens, judge_output_tokens

For each env, writes one combined PNG to
    mitigations/plots_combined/empathy_<env>.png
laid out as a 3-row × N-family-column grid. Rows correspond to the three
LEND_AN_EAR_SUBCOMPONENTS, columns to the families in COMBINED_FAMILIES.
Each cell is a bar chart of the per-condition mean score (1-5 Likert),
colored using the shared CONDITION_COLORS palette from
plot_explore_mitigations.py so the same condition reads as the same color
across this figure and the rest of the combined plots.

Sub-component `dismissing_emotions` is proscriptive (lower is better);
this is annotated in the row's y-label so the reader doesn't have to
remember the direction. The other two are prescriptive (higher is better).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from plot_explore_mitigations import (
    CONDITION_COLORS, COMBINED_FAMILIES, COMBINED_OUT_DIR,
    OEQ_CONDITION_ORDER, PAS_CONDITION_ORDER, _filter_order,
    display_name,
)
from empathy_judge_prompt import LEND_AN_EAR_SUBCOMPONENTS

SUB_KEYS = list(LEND_AN_EAR_SUBCOMPONENTS.keys())
RESULTS_DIR = os.path.join(HERE, "results")
SUPPORTED_ENVS = ("OEQ", "PAS")


def env_condition_order(env):
    """Per-env canonical condition order (reuses the explore_mitigations
    orderings so empathy figures match the other combined plots)."""
    if env == "OEQ":
        return OEQ_CONDITION_ORDER
    assert env == "PAS"
    return PAS_CONDITION_ORDER


def empathy_csv_path(env, family, judge_slug):
    return os.path.join(
        RESULTS_DIR,
        f"empathy_{env}_{family}_judge-{judge_slug}.csv",
    )


def load_empathy(path):
    """Read one judge_empathy.py CSV. Asserts the three sub-component score
    columns are present and that every non-null value is in {1,2,3,4,5}.
    """
    assert os.path.exists(path), f"missing empathy CSV: {path}"
    df = pd.read_csv(path)
    for col in ("id", "condition", *SUB_KEYS):
        assert col in df.columns, f"{path}: missing column {col!r}"
    for sub in SUB_KEYS:
        vals = df[sub].dropna().unique()
        bad = [v for v in vals if int(v) not in (1, 2, 3, 4, 5)]
        assert not bad, f"{path}: {sub} has out-of-range values {bad}"
    return df


def mean_per_condition(df, condition_order):
    """Return DataFrame indexed by condition with columns SUB_KEYS, each the
    mean of the parsed (non-null) scores for that (condition, sub).

    Asserts every condition has at least one parsed score per sub so a NaN
    in the plot can't be silently passed off as a missing bar.
    """
    rows = []
    for cond in condition_order:
        sub_df = df[df["condition"] == cond]
        assert len(sub_df) > 0, f"empathy plot: no rows for condition {cond!r}"
        rec = {"condition": cond, "n_total": len(sub_df)}
        for sub in SUB_KEYS:
            scores = sub_df[sub].dropna()
            assert len(scores) > 0, (
                f"empathy plot: condition {cond!r} has zero parsed {sub!r} "
                "scores; mean is undefined"
            )
            rec[sub] = float(scores.mean())
            rec[f"{sub}_n"] = int(len(scores))
        rows.append(rec)
    return pd.DataFrame(rows).set_index("condition")


def _draw_empathy_cell(ax, means, sub_key):
    """Render the per-condition bar chart for one (family, sub) cell."""
    conds = list(means.index)
    heights = means[sub_key].tolist()
    x = np.arange(len(conds))
    for cond in conds:
        assert cond in CONDITION_COLORS, f"no color for condition {cond!r}"
    for xi, h, cond in zip(x, heights, conds):
        ax.bar(xi, h, color=CONDITION_COLORS[cond], edgecolor="black",
               linewidth=0.5, label=display_name(cond))
    for xi, v in zip(x, heights):
        ax.text(xi, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([display_name(c) for c in conds],
                       rotation=30, ha="right", fontsize=7)
    ax.set_ylim(1, 5.4)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.grid(alpha=0.3, axis="y")


def plot_combined_empathy(env, families, judge_slug, out_path):
    """Build the 3-row × N-family-col grid figure for one env."""
    n_rows = len(SUB_KEYS)
    n_cols = len(families)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.0 * n_cols, 3.4 * n_rows),
        sharex="col", sharey="row",
    )
    # Normalize axes shape so axes[r][c] always works.
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    per_family = []
    for family, display in families:
        path = empathy_csv_path(env, family, judge_slug)
        df = load_empathy(path)
        order = _filter_order(env_condition_order(env), df,
                              f"{display}/{env}/empathy")
        means = mean_per_condition(df, order)
        per_family.append((family, display, means))

    for col_idx, (family, display, means) in enumerate(per_family):
        for row_idx, sub_key in enumerate(SUB_KEYS):
            ax = axes[row_idx][col_idx]
            _draw_empathy_cell(ax, means, sub_key)
            if row_idx == 0:
                ax.set_title(display)
            if col_idx == 0:
                sub_obj = LEND_AN_EAR_SUBCOMPONENTS[sub_key]
                direction = (
                    "higher = more empathic"
                    if sub_obj.direction == "prescriptive"
                    else "higher = LESS empathic"
                )
                ax.set_ylabel(f"{sub_obj.name}\n(1-5; {direction})")

    # Shared legend at the figure level (pull from first cell).
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right",
               bbox_to_anchor=(1.0, 0.5), fontsize=9, title="condition")
    fig.suptitle(
        f"{env} - Lend an Ear empathy sub-components per mitigation "
        f"(judge: {judge_slug})"
    )
    fig.tight_layout(rect=[0, 0, 0.86, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--envs", nargs="+", default=list(SUPPORTED_ENVS),
                   choices=SUPPORTED_ENVS,
                   help="envs to plot (default: OEQ PAS)")
    p.add_argument("--judge-slug", default="gpt-5.4-mini",
                   help="judge slug used in the empathy CSV filenames "
                        "(default: gpt-5.4-mini, matching the "
                        "judge_empathy.py default of openai/gpt-5.4-mini)")
    args = p.parse_args()

    os.makedirs(COMBINED_OUT_DIR, exist_ok=True)
    for env in args.envs:
        out = os.path.join(COMBINED_OUT_DIR, f"empathy_{env.lower()}.png")
        plot_combined_empathy(env, COMBINED_FAMILIES, args.judge_slug, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
