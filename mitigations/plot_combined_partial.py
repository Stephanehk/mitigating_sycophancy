"""Combined cross-model plots for the in-progress deepseek-v3.2 /
llama-3.3-70b-instruct / qwen3-235b-a22b-2507 sweep with the gpt-5.4-mini
judge.

Differences from `plot_explore_mitigations.py --combined`:
  - Reads each per-(env, model) checkpoint .jsonl directly instead of the
    end-of-run CSV (the runs were stopped mid-flight when the OpenRouter
    balance hit \$0; the CSVs were never written). Each result row in the
    checkpoint is the same dict `run_one` produced, so the per-env
    `*_rates_per_condition` functions in `plot_explore_mitigations` work
    unchanged on the assembled DataFrame.
  - The plotted family list is fixed to the three models in this sweep,
    each paired with the gpt-5.4-mini judge.
  - Only the three conditions actually being run are kept: baseline,
    be_critical, stakeholder_concerns_v2. Anything else in the checkpoint
    is ignored.
  - Each plotted point / bar is annotated with the sample size that fed
    into its rate. The denominator differs per plot kind:
      AITA               n_yta_truth + n_nta_truth (split shown)
      AITA-NTA-FLIP bar  n_total
      OEQ / PAS scatter  n_{0,1} (label-0/1 only; the rest are dropped)
      PAS bar            n_{0,1}
  - The OEQ human-diff panel is skipped because the runs do not include
    the `human_reference` condition.

Outputs into `mitigations/plots_combined_mini/`.
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot_explore_mitigations import (
    CONDITION_COLORS, display_name,
    aita_nta_flip_rates_per_condition,
    oeq_label_counts_per_condition,
    paper_ci_halfwidths, _overlap_sizes,
)


def aita_fnr_per_condition(df, condition_order):
    """FNR-only per-condition rates (no FPR — see module docstring).

    Returns DataFrame[condition, fnr, n_yta_truth, n_nta_truth, n_other].
    FNR = #(is_asshole=1 ∩ pred=NTA) / #(is_asshole=1 ∩ pred∈{YTA,NTA}).
    """
    rows = []
    for cond in condition_order:
        sub = df[df["condition"] == cond]
        assert len(sub) > 0, f"AITA: no rows for {cond!r}"
        parsed = sub[sub["pred"].isin(["YTA", "NTA"])]
        n_yta = int((parsed["is_asshole"] == 1).sum())
        n_nta = int((parsed["is_asshole"] == 0).sum())
        assert n_yta > 0, f"AITA: no parsed YTA-truth rows for {cond!r}"
        fn = int(((parsed["is_asshole"] == 1) & (parsed["pred"] == "NTA")).sum())
        rows.append({
            "condition": cond,
            "fnr": fn / n_yta,
            "n_yta_truth": n_yta,
            "n_nta_truth": n_nta,
            "n_other": int((sub["pred"] == "OTHER").sum()),
        })
    return pd.DataFrame(rows)


JUDGE_SLUG = "gpt-5.4-mini"
FAMILIES = [
    ("deepseek-v3.2", "DeepSeek v3.2"),
    ("llama-3.3-70b-instruct", "Llama-3.3-70B-Instruct"),
    ("qwen3-235b-a22b-2507", "Qwen3-235B-A22B-2507"),
]
CONDITIONS = ["baseline", "be_critical", "stakeholder_concerns_v2"]
ENVS = ["AITA", "AITA-NTA-FLIP", "OEQ", "PAS"]
OUT_DIR = "mitigations/plots_combined_mini"


def checkpoint_path(env, model_slug):
    """Path to the per-(env, model) checkpoint .jsonl for the in-progress runs."""
    return (
        "mitigations/results/explore_mitigations_results_"
        f"{env}_{model_slug}_{JUDGE_SLUG}.checkpoint.jsonl"
    )


def load_checkpoint_df(env, model_slug):
    """Read the checkpoint into a DataFrame, filtered to CONDITIONS only.

    Each line in the checkpoint is one dict from `run_one`, so the resulting
    DataFrame has exactly the columns the per-env `*_rates_per_condition`
    helpers expect (id, condition, plus the env-specific fields).
    """
    path = checkpoint_path(env, model_slug)
    assert os.path.exists(path), f"missing checkpoint: {path}"
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["condition"] not in CONDITIONS:
                continue
            rows.append(r)
    assert rows, f"checkpoint has no rows in CONDITIONS={CONDITIONS}: {path}"
    return pd.DataFrame(rows)


def filter_order(df, env_label):
    """Return CONDITIONS filtered to those actually present in this run."""
    actual = set(df["condition"].dropna().unique())
    unexpected = actual - set(CONDITIONS)
    assert not unexpected, (
        f"{env_label}: unexpected conditions in checkpoint: {sorted(unexpected)}"
    )
    return [c for c in CONDITIONS if c in actual]


def annotate_point(ax, x, y, n_str, color):
    """Place a tiny `n=...` label next to a scatter point. Offset just to
    the right of the marker so it doesn't overlap the marker's edge. Uses the
    point's color (slightly darkened) so multi-point clusters stay legible.
    """
    ax.annotate(
        n_str, xy=(x, y), xytext=(6, 4), textcoords="offset points",
        fontsize=7, color=color, alpha=0.95,
    )


def scatter_conditions_with_n(ax, xs, ys, conds, ns, xerr=None, yerr=None):
    """Like `_scatter_conditions` in plot_explore_mitigations but annotates
    each marker with its sample size. Same overlap-sizing logic so colors
    stay visible when two conditions land on the same coordinate.
    """
    sizes = _overlap_sizes(xs, ys)
    for i, (x, y, cond, size, n) in enumerate(zip(xs, ys, conds, sizes, ns)):
        assert cond in CONDITION_COLORS, f"no color for {cond!r}"
        color = CONDITION_COLORS[cond]
        if xerr is not None or yerr is not None:
            xe = None if xerr is None else np.array([[xerr[0][i]], [xerr[1][i]]])
            ye = None if yerr is None else np.array([[yerr[0][i]], [yerr[1][i]]])
            ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="none", ecolor=color,
                        elinewidth=1.0, capsize=3, alpha=0.6, zorder=2)
        ax.scatter(x, y, s=size, zorder=3, color=color,
                   edgecolor="black", linewidth=0.5, label=display_name(cond))
        annotate_point(ax, x, y, f"n={n}", color)


# -------- AITA: FPR vs FNR scatter --------

def draw_aita_fnr_panel(ax, rates):
    """Bar of FNR per condition with 95% CI + n annotation.

    FNR-only (no FPR) because the balanced sampler emits all YTA-truth
    rows before any NTA-truth rows, and the runs were stopped before any
    NTA-truth rows completed — so the NTA-truth bucket is empty. FNR is
    the action-endorsement rate on AITA (paper §6.1.2 metric).
    """
    conds = rates["condition"].tolist()
    heights = rates["fnr"].tolist()
    ns = rates["n_yta_truth"].tolist()
    x = np.arange(len(conds))
    for cond in conds:
        assert cond in CONDITION_COLORS, f"no color for {cond!r}"
    for xi, h, cond in zip(x, heights, conds):
        ax.bar(xi, h, color=CONDITION_COLORS[cond], edgecolor="black",
               linewidth=0.5, label=display_name(cond))
    yerr = paper_ci_halfwidths(heights, ns)
    ax.errorbar(x, heights, yerr=np.array(yerr), fmt="none", ecolor="black",
                elinewidth=1.0, capsize=3, zorder=4)
    upper = yerr[1]
    for xi, v, u, n in zip(x, heights, upper, ns):
        ax.text(xi, v + u + 0.01, f"{v:.2f}\nn_yta={n}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([display_name(c) for c in conds],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("FNR (NTA pred on YTA-truth)")
    ax.grid(alpha=0.3, axis="y")


def plot_combined_aita(out_path):
    """One AITA FNR bar panel per family, shared y axis, with per-bar ns."""
    n = len(FAMILIES)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 6.0), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (slug, display) in zip(axes, FAMILIES):
        df = load_checkpoint_df("AITA", slug)
        order = filter_order(df, f"{display}/AITA")
        rates = aita_fnr_per_condition(df, order)
        draw_aita_fnr_panel(ax, rates)
        ax.set_title(f"AITA — {display}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.0, 0.5),
               fontsize=10, title="condition")
    fig.suptitle("AITA — FNR per mitigation (lower is less sycophantic). "
                 "FPR omitted: runs stopped before NTA-truth posts were sampled.")
    fig.tight_layout(rect=[0, 0, 0.85, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------- AITA-NTA-FLIP: bar (sycophancy rate) + scatter (sycophancy vs critical) --------

def draw_aita_flip_bar_panel(ax, rates):
    """Bar of sycophancy_score_rate per condition with 95% CI + n annotation."""
    conds = rates["condition"].tolist()
    heights = rates["sycophancy_score_rate"].tolist()
    ns = rates["n_total"].tolist()
    x = np.arange(len(conds))
    for cond in conds:
        assert cond in CONDITION_COLORS, f"no color for {cond!r}"
    for xi, h, cond in zip(x, heights, conds):
        ax.bar(xi, h, color=CONDITION_COLORS[cond], edgecolor="black",
               linewidth=0.5, label=display_name(cond))
    yerr = paper_ci_halfwidths(heights, ns)
    ax.errorbar(x, heights, yerr=np.array(yerr), fmt="none", ecolor="black",
                elinewidth=1.0, capsize=3, zorder=4)
    upper = yerr[1]
    for xi, v, u, n in zip(x, heights, upper, ns):
        ax.text(xi, v + u + 0.01, f"{v:.2f}\nn={n}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([display_name(c) for c in conds],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("sycophancy rate (both NTA)")
    ax.grid(alpha=0.3, axis="y")


def draw_aita_flip_scatter_panel(ax, rates):
    """critical_score_rate vs sycophancy_score_rate scatter, with per-point n."""
    xs = rates["critical_score_rate"].tolist()
    ys = rates["sycophancy_score_rate"].tolist()
    conds = rates["condition"].tolist()
    ns = rates["n_total"].tolist()
    n_strs = [f"n={n}" for n in ns]
    xerr = paper_ci_halfwidths(xs, ns)
    yerr = paper_ci_halfwidths(ys, ns)
    scatter_conditions_with_n(ax, xs, ys, conds, n_strs, xerr=xerr, yerr=yerr)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("critical rate (both YTA / N)")
    ax.set_ylabel("sycophancy rate (both NTA / N)")
    ax.grid(alpha=0.3)


def plot_combined_aita_flip_bar(out_path):
    n = len(FAMILIES)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 6.0), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (slug, display) in zip(axes, FAMILIES):
        df = load_checkpoint_df("AITA-NTA-FLIP", slug)
        order = filter_order(df, f"{display}/AITA-NTA-FLIP")
        rates = aita_nta_flip_rates_per_condition(df, order)
        draw_aita_flip_bar_panel(ax, rates)
        ax.set_title(f"AITA-NTA-FLIP — {display}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.0, 0.5),
               fontsize=10, title="condition")
    fig.suptitle("AITA-NTA-FLIP — sycophancy rate (lower is better)")
    fig.tight_layout(rect=[0, 0, 0.85, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined_aita_flip_scatter(out_path):
    n = len(FAMILIES)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 6.5), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (slug, display) in zip(axes, FAMILIES):
        df = load_checkpoint_df("AITA-NTA-FLIP", slug)
        order = filter_order(df, f"{display}/AITA-NTA-FLIP")
        rates = aita_nta_flip_rates_per_condition(df, order)
        draw_aita_flip_scatter_panel(ax, rates)
        ax.set_title(f"AITA-NTA-FLIP — {display}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.0, 0.5),
               fontsize=10, title="condition")
    fig.suptitle("AITA-NTA-FLIP — sycophancy vs critical "
                 "(lower-right = critical-without-sycophancy)")
    fig.tight_layout(rect=[0, 0, 0.85, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------- OEQ: {0,1}-only endorsement vs unendorsement scatter --------

def draw_endorse_pareto_panel(ax, counts, env_label):
    """Same as `_draw_endorse_pareto_panel` but with per-point n annotations
    (no human_reference dependence — we just plot whatever conditions exist).
    """
    xs, ys, conds, ns = [], [], [], []
    for cond, row in counts.iterrows():
        denom = row[0] + row[1]
        assert denom > 0, (
            f"{env_label}: condition {cond!r} has zero label-0/1 rows; "
            f"endorsement_rate is undefined"
        )
        xs.append(row[0] / denom)
        ys.append(row[1] / denom)
        conds.append(cond)
        ns.append(denom)
    ax.plot([0, 1], [1, 0], linestyle="--", color="grey", linewidth=1,
            label="endorsement + unendorsement = 1", zorder=1)
    xerr = paper_ci_halfwidths(xs, ns)
    yerr = paper_ci_halfwidths(ys, ns)
    n_strs = [f"n={n}" for n in ns]
    scatter_conditions_with_n(ax, xs, ys, conds, n_strs, xerr=xerr, yerr=yerr)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("unendorsement rate")
    ax.set_ylabel("endorsement rate")
    ax.grid(alpha=0.3)


def plot_combined_oeq(out_path):
    n = len(FAMILIES)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 6.5), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (slug, display) in zip(axes, FAMILIES):
        df = load_checkpoint_df("OEQ", slug)
        order = filter_order(df, f"{display}/OEQ")
        counts = oeq_label_counts_per_condition(df, order)
        draw_endorse_pareto_panel(ax, counts, f"{display}/OEQ")
        ax.set_title(f"OEQ — {display}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.0, 0.5),
               fontsize=10, title="condition")
    fig.suptitle("OEQ — endorsement vs unendorsement over labels {0,1} "
                 "(paper §6.1.2; n is the label-0/1 count)")
    fig.tight_layout(rect=[0, 0, 0.85, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------- PAS: bar of endorsement (sycophancy) rate --------

def draw_pas_bar_panel(ax, counts, env_label):
    """Bar of endorsement_rate per condition with 95% CI + n annotation.

    endorsement_rate = #(label==1) / #(label in {0,1}).
    """
    conds = list(counts.index)
    heights, ns = [], []
    for cond in conds:
        row = counts.loc[cond]
        denom = row[0] + row[1]
        assert denom > 0, (
            f"{env_label}: condition {cond!r} has zero label-0/1 rows"
        )
        heights.append(row[1] / denom)
        ns.append(denom)
    x = np.arange(len(conds))
    for cond in conds:
        assert cond in CONDITION_COLORS, f"no color for {cond!r}"
    for xi, h, cond in zip(x, heights, conds):
        ax.bar(xi, h, color=CONDITION_COLORS[cond], edgecolor="black",
               linewidth=0.5, label=display_name(cond))
    yerr = paper_ci_halfwidths(heights, ns)
    ax.errorbar(x, heights, yerr=np.array(yerr), fmt="none", ecolor="black",
                elinewidth=1.0, capsize=3, zorder=4)
    upper = yerr[1]
    for xi, v, u, n in zip(x, heights, upper, ns):
        ax.text(xi, v + u + 0.01, f"{v:.2f}\nn={n}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([display_name(c) for c in conds],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("sycophancy rate (endorsement)")
    ax.grid(alpha=0.3, axis="y")


def plot_combined_pas(out_path):
    n = len(FAMILIES)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 6.0), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (slug, display) in zip(axes, FAMILIES):
        df = load_checkpoint_df("PAS", slug)
        order = filter_order(df, f"{display}/PAS")
        counts = oeq_label_counts_per_condition(df, order)
        draw_pas_bar_panel(ax, counts, f"{display}/PAS")
        ax.set_title(f"PAS — {display}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.0, 0.5),
               fontsize=10, title="condition")
    fig.suptitle("PAS — sycophancy rate (lower is better; "
                 "n is the label-0/1 count)")
    fig.tight_layout(rect=[0, 0, 0.85, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_combined_aita(os.path.join(OUT_DIR, "aita_combined.png"))
    print(f"wrote {OUT_DIR}/aita_combined.png")
    plot_combined_aita_flip_bar(os.path.join(OUT_DIR, "aita-nta-flip_combined.png"))
    print(f"wrote {OUT_DIR}/aita-nta-flip_combined.png")
    plot_combined_aita_flip_scatter(os.path.join(OUT_DIR, "aita-nta-flip_scatter_combined.png"))
    print(f"wrote {OUT_DIR}/aita-nta-flip_scatter_combined.png")
    plot_combined_oeq(os.path.join(OUT_DIR, "oeq_combined.png"))
    print(f"wrote {OUT_DIR}/oeq_combined.png")
    plot_combined_pas(os.path.join(OUT_DIR, "pas_combined.png"))
    print(f"wrote {OUT_DIR}/pas_combined.png")


if __name__ == "__main__":
    main()
