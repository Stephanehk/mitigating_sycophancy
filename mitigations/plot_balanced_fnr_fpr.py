"""Horizontal grouped bar plots of AITA FNR (sycophancy) and FPR (criticalness)
across the balanced n=1000 NTA/YTA runs, one bar group per model and one bar per
mitigation method.

Two figures are written:
  - fnr_by_model.png : x-axis = FNR (sycophancy)
  - fpr_by_model.png : x-axis = FPR (criticalness)

Each model row carries three bars (no prompt / be critical / prompt-stakeholders)
in distinct colors, with binomial standard-error whiskers.

Data source: the per-run checkpoint JSONL files (uniform across complete and
partial runs). Metrics reuse `core.metrics.compute_fnr_fpr` so the numbers match
the harness's own summary exactly. Reduced-n cells (a partial run whose condition
has fewer than the full 500/500 classified rows) are annotated with their actual
denominator so the figure is honest about coverage.

Assumptions (all asserted below):
  - Every checkpoint has rows for all three plotted conditions.
  - Each (model, condition) cell has at least one YTA-truth and one NTA-truth
    classified row (else FNR/FPR is undefined and we fail loud).
Plots ONLY the computed raw rates — no smoothing, no interpolation.
"""

import os
import sys
import math
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.metrics import compute_fnr_fpr
from plot_explore_mitigations import CONDITION_COLORS

RESULTS_DIR = "mitigations/results"
OUT_DIR = "mitigations/plots_balanced_fnr_fpr"

# Conditions in the order the bars should appear within each model group, with
# the exact display labels requested.
CONDITIONS = ["baseline", "be_critical", "stakeholder_concerns_v2"]
LABELS = {
    "baseline": "no prompt",
    "be_critical": "be critical",
    "stakeholder_concerns_v2": "prompt-stakeholders",
}

# All seven balanced n=1000 runs, with their checkpoint filename and a display
# name. Order top-to-bottom on the y-axis follows this list.
MODELS = [
    ("DeepSeek-V3.2",
     "explore_mitigations_results_AITA_deepseek-v3.2_gpt-5.4-mini_balanced1k.checkpoint.jsonl"),
    ("Llama-3.3-70B-Instruct",
     "explore_mitigations_results_AITA_llama-3.3-70b-instruct_gpt-5.4-mini_balanced1k.checkpoint.jsonl"),
    ("Qwen3-235B-A22B-2507",
     "explore_mitigations_results_AITA_qwen3-235b-a22b-2507_gpt-5.4-mini_balanced1k.checkpoint.jsonl"),
    ("GPT-4o",
     "explore_mitigations_results_AITA_gpt-4o-2024-11-20_gpt-5.4-mini.checkpoint.jsonl"),
    ("DeepSeek-V3-0324",
     "explore_mitigations_results_AITA_deepseek-chat-v3-0324_gpt-5.4-mini.checkpoint.jsonl"),
    ("Llama-4-Scout-17B-16E-Instruct",
     "explore_mitigations_results_AITA_llama-4-scout_gpt-5.4-mini.checkpoint.jsonl"),
    ("Qwen2.5-7B-Instruct",
     "explore_mitigations_results_AITA_qwen-2.5-7b-instruct_gpt-5.4-mini.checkpoint.jsonl"),
]


def load_checkpoint_df(path):
    """Load a checkpoint JSONL into a DataFrame with the columns the metric
    needs. Fails loud if the file or required columns are missing.
    """
    assert os.path.exists(path), f"missing checkpoint: {path}"
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    for col in ("condition", "pred", "is_asshole"):
        assert col in df.columns, f"{path} missing column {col!r}"
    return df


def standard_error(p, n):
    """Binomial standard error of a proportion: sqrt(p(1-p)/n)."""
    assert n > 0, "standard error needs n > 0"
    return math.sqrt(p * (1.0 - p) / n)


def collect_metrics():
    """Return a dict model_display -> condition -> metric dict (from
    compute_fnr_fpr) augmented with FNR/FPR standard errors. Asserts every
    plotted condition is present for every model.
    """
    out = {}
    for display, fname in MODELS:
        df = load_checkpoint_df(os.path.join(RESULTS_DIR, fname))
        present = set(df["condition"].unique())
        missing = [c for c in CONDITIONS if c not in present]
        assert not missing, f"{display}: checkpoint missing conditions {missing}"
        per_cond = {}
        for cond in CONDITIONS:
            sub = df[df["condition"] == cond]
            m = compute_fnr_fpr(sub)
            m["se_fnr"] = standard_error(m["fnr"], m["n_yta_truth"])
            m["se_fpr"] = standard_error(m["fpr"], m["n_nta_truth"])
            per_cond[cond] = m
        out[display] = per_cond
    return out


def make_plot(metrics, rate_key, se_key, n_key, xlabel, title, out_path):
    """Draw one horizontal grouped-bar figure for a single rate (FNR or FPR).

    rate_key/se_key/n_key select which metric, its standard error, and the
    relevant denominator (for the reduced-n annotation). Bars are grouped by
    model on the y-axis, three colored bars per group with SE whiskers.
    """
    models = list(metrics.keys())
    n_models = len(models)
    n_conds = len(CONDITIONS)
    bar_h = 0.8 / n_conds
    # Center the three bars around each model's integer y position.
    offsets = [(i - (n_conds - 1) / 2.0) * bar_h for i in range(n_conds)]

    fig, ax = plt.subplots(figsize=(9, 1.2 * n_models + 1.5))

    for ci, cond in enumerate(CONDITIONS):
        ys = [mi + offsets[ci] for mi in range(n_models)]
        vals = [metrics[m][cond][rate_key] for m in models]
        errs = [metrics[m][cond][se_key] for m in models]
        ax.barh(
            ys, vals, height=bar_h, xerr=errs,
            color=CONDITION_COLORS[cond], edgecolor="black", linewidth=0.5,
            error_kw=dict(ecolor="black", lw=0.8, capsize=2),
            label=LABELS[cond], zorder=3,
        )
        # Annotate each bar: rate value, plus the denominator when it is below
        # the full 500 (i.e. a partial run) so reduced coverage is visible.
        for y, m in zip(ys, models):
            cell = metrics[m][cond]
            v = cell[rate_key]
            n = cell[n_key]
            txt = f"{v:.3f}"
            if n < 500:
                txt += f" (n={n})"
            # Place the label just past the value's error whisker.
            ax.text(v + cell[se_key] + 0.005, y, txt,
                    va="center", ha="left", fontsize=7)

    ax.set_yticks(range(n_models))
    ax.set_yticklabels(models)
    ax.invert_yaxis()  # first model in MODELS appears at the top
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(0, max(0.05, ax.get_xlim()[1]) * 1.18)
    ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=0)
    # Legend outside the axes (right) so it never overlaps a bar or its label.
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.95,
              title="method")
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def make_pareto_scatter(metrics, out_path):
    """One figure, one subplot per model: scatter of the three methods in
    (FPR, FNR) space with 95% confidence-interval whiskers on each point, over
    a family of iso-cost "Pareto" reference lines FNR + FPR = const (slope -1;
    lower-left is better). A colored iso-cost line is drawn through each
    method's point so its total error level reads directly.

    Intervals are the normal-approximation (Wald) 95% CI for a proportion,
    p +/- 1.96 * sqrt(p(1-p)/n), with n the relevant classified denominator
    (n_nta_truth for FPR on x, n_yta_truth for FNR on y).
    """
    Z = 1.96  # 95% normal-approximation interval
    models = list(metrics.keys())
    n = len(models)
    ncols = 3
    nrows = math.ceil(n / ncols)

    # Shared axis limits across all subplots so models are visually comparable.
    all_x = [metrics[m][c]["fpr"] for m in models for c in CONDITIONS]
    all_y = [metrics[m][c]["fnr"] for m in models for c in CONDITIONS]
    lim = max(max(all_x), max(all_y)) * 1.12
    lim = min(1.0, lim) if max(max(all_x), max(all_y)) > 0.9 else lim

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows),
                             squeeze=False)

    for idx, m in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        # Family of faint iso-cost lines FNR+FPR = c (the "Pareto" indifference
        # lines): along each, trading FPR for FNR leaves total error unchanged.
        for c in np.arange(0.1, 2 * lim + 1e-9, 0.1):
            ax.plot([max(0, c - lim), min(lim, c)],
                    [min(lim, c), max(0, c - lim)],
                    color="0.8", lw=0.7, zorder=1)

        for cond in CONDITIONS:
            cell = metrics[m][cond]
            x, y = cell["fpr"], cell["fnr"]
            color = CONDITION_COLORS[cond]
            # Iso-cost line through this point, colored to the method.
            tot = x + y
            ax.plot([max(0, tot - lim), min(lim, tot)],
                    [min(lim, tot), max(0, tot - lim)],
                    color=color, lw=1.0, alpha=0.6, zorder=2)
            # 95% CI whiskers on both axes.
            ax.errorbar(x, y, xerr=Z * cell["se_fpr"], yerr=Z * cell["se_fnr"],
                        fmt="none", ecolor="black", lw=0.8, capsize=2, zorder=3)
            ax.scatter([x], [y], s=70, color=color, edgecolor="black",
                       linewidth=0.6, zorder=4, label=LABELS[cond])

        ax.set_title(m, fontsize=10)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        if idx % ncols == 0:
            ax.set_ylabel("FNR (sycophancy)")
        if idx // ncols == nrows - 1:
            ax.set_xlabel("FPR (criticalness)")

    # Hide any unused axes.
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    # One shared legend (method swatches + the ring meaning).
    handles = [Patch(facecolor=CONDITION_COLORS[c], edgecolor="black",
                     label=LABELS[c]) for c in CONDITIONS]
    handles.append(plt.Line2D([0], [0], color="black", lw=0.8,
                              label="95% CI (both axes)"))
    handles.append(plt.Line2D([0], [0], color="0.8", lw=1.0,
                              label="iso-cost: FNR+FPR = const"))
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.95)
    fig.suptitle("AITA method trade-off per model: FNR (sycophancy) vs "
                 "FPR (criticalness)\nlower-left is better; diagonals are "
                 "equal-total-error iso-cost lines; whiskers are 95% CIs", y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.99))
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    metrics = collect_metrics()

    # Audit dump: print every plotted number so the figure is reproducible by
    # eye from the console output.
    print("model".ljust(32), "condition".ljust(24),
          "FNR(±SE)".ljust(18), "FPR(±SE)".ljust(18), "n_yta/n_nta")
    for m, per in metrics.items():
        for cond in CONDITIONS:
            c = per[cond]
            print(m.ljust(32), cond.ljust(24),
                  f"{c['fnr']:.3f}±{c['se_fnr']:.3f}".ljust(18),
                  f"{c['fpr']:.3f}±{c['se_fpr']:.3f}".ljust(18),
                  f"{c['n_yta_truth']}/{c['n_nta_truth']}")

    make_plot(metrics, "fnr", "se_fnr", "n_yta_truth",
              "FNR (sycophancy)",
              "AITA action endorsement — FNR (sycophancy) by model and method",
              os.path.join(OUT_DIR, "fnr_by_model.png"))
    make_plot(metrics, "fpr", "se_fpr", "n_nta_truth",
              "FPR (criticalness)",
              "AITA — FPR (criticalness) by model and method",
              os.path.join(OUT_DIR, "fpr_by_model.png"))
    make_pareto_scatter(metrics,
                        os.path.join(OUT_DIR, "fnr_fpr_pareto_by_model.png"))


if __name__ == "__main__":
    main()
