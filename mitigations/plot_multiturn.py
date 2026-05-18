"""Plot per-condition ToF, NoF, and turn-by-turn alignment rate from
explore_mitigations_multiturn.py result CSVs.

Inputs (written by `explore_mitigations_multiturn.py` into mitigations/results/):
  mitigations/results/explore_mitigations_multiturn_<env>_<gen>_<judge>.csv

Each input row is one (id, condition) rollout with columns:
  response_1..5, label_1..5, tof, nof, gen_*_tokens, judge_*_tokens

Outputs (per env, per family):
  mitigations/plots_multiturn_<gen>_<judge>/<env>_tof_nof.png
    Two-panel bar chart: mean ToF (higher = more resistant) and mean NoF
    (lower = more stable) per condition.
  mitigations/plots_multiturn_<gen>_<judge>/<env>_alignment_curve.png
    Per-turn alignment rate (fraction of items with label==1) per condition,
    with N=5 turn-x-axis. One line per condition.

All values come straight from the CSV — no smoothing, no interpolation. If a
turn has zero non-null labels (e.g. judge parse failed on every item), we
assert rather than substitute. Run with --family <gen>_<judge> to control
which CSV is loaded; --envs defaults to both `ethical` and `fp`.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_explore_mitigations import display_name


# Condition order mirrors `get_conditions_multiturn` (OEQ minus
# stakeholder_concerns and human_reference). Kept in sync so plots are
# visually comparable to the single-turn OEQ figures.
CONDITION_ORDER = [
    "baseline",
    "no_syco",
    "be_critical",
    "no_syco_critical",
    "avoid_validation",
    "no_syco_critical_avoid_validation",
    "repeat_question",
    "sharma_honesty",
    "explicit",
]

# Reused from plot_explore_mitigations.CONDITION_COLORS so a single condition
# reads as the same color across single-turn and multi-turn figures.
CONDITION_COLORS = {
    "baseline": "#1f77b4",
    "no_syco": "#ff7f0e",
    "be_critical": "#2ca02c",
    "no_syco_critical": "#d62728",
    "repeat_question": "#9467bd",
    "avoid_validation": "#e377c2",
    "no_syco_critical_avoid_validation": "#7f7f7f",
    "sharma_honesty": "#bcbd22",
    "explicit": "#5254a3",
}

DEFAULT_FAMILY = "claude-sonnet-4-6_gpt-5.4"
DEFAULT_ENVS = ("ethical", "fp")


def csv_for(env, family):
    """Return the result CSV path for `(env, family)`."""
    return (
        f"mitigations/results/explore_mitigations_multiturn_"
        f"{env}_{family}.csv"
    )


def out_dir_for(family):
    """Return the per-family plot output directory."""
    return f"mitigations/plots_multiturn_{family}"


def load_results(env, family):
    """Load the per-rollout CSV for one (env, family).

    Asserts the file exists, all label columns are present, and at least one
    row has a finite ToF (a 100%-missing result usually means the judge model
    failed catastrophically — fail loudly so the figure isn't quietly empty).
    """
    path = csv_for(env, family)
    assert os.path.exists(path), f"missing results CSV: {path}"
    df = pd.read_csv(path)
    for col in ("condition", "tof", "nof",
                "label_1", "label_2", "label_3", "label_4", "label_5"):
        assert col in df.columns, f"{path} missing column {col!r}"
    assert df["tof"].notna().any(), (
        f"all `tof` values are NaN in {path} — judge likely failed on every row"
    )
    return df


def conditions_present(df):
    """Return the subset of `CONDITION_ORDER` actually present in `df`.

    Mitigations get added/removed over time; this keeps the plot order stable
    while gracefully skipping conditions the run didn't include.
    """
    have = set(df["condition"].dropna().unique())
    return [c for c in CONDITION_ORDER if c in have]


def plot_tof_nof(df, env, out_path):
    """Two-panel bar chart: mean ToF and mean NoF per condition.

    Y-axis ranges:
      ToF panel: [0, 5] (5 turns max).
      NoF panel: [0, 4] (max 4 transitions in a 5-turn sequence).
    Bar heights are the mean across all rollouts for that condition; we add
    a small text annotation with N above each bar so a thin-evidence condition
    isn't mistaken for a robust effect.
    """
    conds = conditions_present(df)
    assert conds, f"no recognised conditions in df for env={env!r}"
    means_tof = []
    means_nof = []
    ns = []
    for c in conds:
        sub = df[df["condition"] == c]
        ns.append(len(sub))
        means_tof.append(sub["tof"].mean())
        means_nof.append(sub["nof"].mean())

    fig, axes = plt.subplots(1, 2, figsize=(max(8, 1.1 * len(conds) * 2), 4.5))
    x = np.arange(len(conds))
    colors = [CONDITION_COLORS.get(c, "#888888") for c in conds]

    axes[0].bar(x, means_tof, color=colors)
    for xi, v, n in zip(x, means_tof, ns):
        axes[0].text(xi, v + 0.1, f"n={n}", ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([display_name(c) for c in conds], rotation=30, ha="right", fontsize=8)
    axes[0].set_ylim(0, 5.3)
    axes[0].set_ylabel("mean ToF  (higher = more resistant)")
    axes[0].set_title(f"{env}: mean Turn-of-Flip per condition")
    axes[0].axhline(5.0, color="grey", linewidth=0.5, linestyle="--")

    axes[1].bar(x, means_nof, color=colors)
    for xi, v, n in zip(x, means_nof, ns):
        axes[1].text(xi, v + 0.05, f"n={n}", ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([display_name(c) for c in conds], rotation=30, ha="right", fontsize=8)
    axes[1].set_ylim(0, 4.3)
    axes[1].set_ylabel("mean NoF  (lower = more stable)")
    axes[1].set_title(f"{env}: mean Number-of-Flips per condition")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_alignment_curve(df, env, out_path):
    """Per-turn alignment rate (label==1) by condition.

    For each condition and each turn t in 1..5, plot:
      fraction of rollouts whose label_t == 1
    over rollouts whose label_t is non-null. Asserts there's at least one
    non-null label per (condition, turn) cell so the figure can't silently
    hide a 100%-judge-fail case.
    """
    conds = conditions_present(df)
    assert conds, f"no recognised conditions in df for env={env!r}"
    turns = [1, 2, 3, 4, 5]

    fig, ax = plt.subplots(figsize=(8, 5))
    for c in conds:
        sub = df[df["condition"] == c]
        rates = []
        for t in turns:
            col = f"label_{t}"
            valid = sub[sub[col].notna()]
            assert len(valid) > 0, (
                f"condition={c!r} turn={t} has zero non-null {col} values "
                f"(env={env!r}) — judge likely failed every call"
            )
            rates.append((valid[col] == 1).mean())
        ax.plot(turns, rates, marker="o", color=CONDITION_COLORS.get(c, "#888888"),
                label=display_name(c))
    ax.set_xticks(turns)
    ax.set_xlabel("turn")
    ax.set_ylabel("alignment rate  (fraction with label=1)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{env}: per-turn alignment rate")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", default=DEFAULT_FAMILY,
                        help="<gen-slug>_<judge-slug> segment (default: claude-sonnet-4-6_gpt-5.4)")
    parser.add_argument("--envs", nargs="+", default=list(DEFAULT_ENVS),
                        help="which envs to plot (default: ethical fp)")
    args = parser.parse_args()

    out_dir = out_dir_for(args.family)
    os.makedirs(out_dir, exist_ok=True)

    for env in args.envs:
        df = load_results(env, args.family)
        plot_tof_nof(df, env, os.path.join(out_dir, f"{env}_tof_nof.png"))
        plot_alignment_curve(df, env,
                             os.path.join(out_dir, f"{env}_alignment_curve.png"))


if __name__ == "__main__":
    main()
