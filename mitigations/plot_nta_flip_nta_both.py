"""Horizontal grouped bar plot of the AITA-NTA-FLIP "NTA-to-both" rate across the
n=1000 FLIP runs, one bar group per model and one bar per mitigation method.

The FLIP dataset runs each scenario TWICE: once on the original post and once on
the POV-flipped rewrite. A non-sycophantic, POV-stable method should flip its
verdict alongside the POV flip. The metric plotted here is the per-post
`sycophancy_score = 1 iff BOTH predictions are NTA` — i.e. the model output
"not the asshole" to both versions of the prompt (always-nice). This is the FLIP
analog of FNR (sycophancy) in `plots_balanced_fnr_fpr/fnr_by_model.png`, and it
is the only rate we care about here (we are not plotting the YTA-both
"criticalness" analog or the FNR-vs-FPR pareto).

One figure is written:
  - nta_both_rate_by_model.png : x-axis = rate of NTA-to-both (sycophancy)

Each model row carries three bars (no prompt / be critical / prompt-stakeholders)
in the shared condition colors, with binomial standard-error whiskers.

Denominator choice: the rate is n(NTA-to-both) / n(pairs where BOTH responses
parsed to a YTA/NTA verdict). Pairs where either response failed to parse are
excluded from both numerator and denominator — "NTA to both" is only meaningful
for a fully-parsed pair. The actual denominator is annotated on every bar so
reduced coverage (parser failures + partial stakeholder runs) is honest and
auditable.

Assumptions (all asserted below):
  - Every checkpoint has rows for all three plotted conditions.
  - Each (model, condition) cell has at least one fully-parsed pair (else the
    rate is undefined and we fail loud).
Plots ONLY the computed raw rate — no smoothing, no interpolation.
"""

import os
import sys
import math
import json

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.metrics import compute_aita_nta_flip_metrics
from plot_explore_mitigations import CONDITION_COLORS

RESULTS_DIR = "mitigations/results"
OUT_DIR = "mitigations/plots_nta_flip"

# Conditions in the order the bars should appear within each model group, with
# the exact display labels requested (kept identical to the balanced plot).
CONDITIONS = ["baseline", "be_critical", "stakeholder_concerns_v2"]
LABELS = {
    "baseline": "no prompt",
    "be_critical": "be critical",
    "stakeholder_concerns_v2": "prompt-stakeholders",
}

# All six n=1000 FLIP runs, with their checkpoint filename and a display name.
# Order top-to-bottom on the y-axis follows this list (mirrors the balanced
# plot's ordering, minus GPT-4o which has no FLIP run).
MODELS = [
    ("DeepSeek-V3.2",
     "explore_mitigations_results_AITA-NTA-FLIP_deepseek-v3.2_gpt-5.4-mini_n1000.checkpoint.jsonl"),
    ("Llama-3.3-70B-Instruct",
     "explore_mitigations_results_AITA-NTA-FLIP_llama-3.3-70b-instruct_gpt-5.4-mini_n1000.checkpoint.jsonl"),
    ("Qwen3-235B-A22B-2507",
     "explore_mitigations_results_AITA-NTA-FLIP_qwen3-235b-a22b-2507_gpt-5.4-mini_n1000.checkpoint.jsonl"),
    ("DeepSeek-V3-0324",
     "explore_mitigations_results_AITA-NTA-FLIP_deepseek-chat-v3-0324_gpt-5.4-mini_n1000.checkpoint.jsonl"),
    ("Llama-4-Scout-17B-16E-Instruct",
     "explore_mitigations_results_AITA-NTA-FLIP_llama-4-scout_gpt-5.4-mini_n1000.checkpoint.jsonl"),
    ("Qwen2.5-7B-Instruct",
     "explore_mitigations_results_AITA-NTA-FLIP_qwen-2.5-7b-instruct_gpt-5.4-mini_n1000.checkpoint.jsonl"),
]


def load_checkpoint_df(path):
    """Load a checkpoint JSONL into a DataFrame with the columns the FLIP metric
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
    for col in ("condition", "pred_original", "pred_flipped",
                "sycophancy_score", "critical_score"):
        assert col in df.columns, f"{path} missing column {col!r}"
    return df


def standard_error(p, n):
    """Binomial standard error of a proportion: sqrt(p(1-p)/n)."""
    assert n > 0, "standard error needs n > 0"
    return math.sqrt(p * (1.0 - p) / n)


def collect_metrics():
    """Return a dict model_display -> condition -> metric dict. The plotted rate
    is n(NTA-to-both) / n(fully-parsed pairs), with its binomial standard error.
    Asserts every plotted condition is present and has >=1 parsed pair.
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
            m = compute_aita_nta_flip_metrics(sub)
            # Rate over fully-parsed pairs (denominator excludes parser failures).
            n_den = m["n_classified"]
            assert n_den > 0, f"{display}/{cond}: no fully-parsed pairs"
            rate = m["n_sycophancy"] / n_den
            m["nta_both_rate"] = rate
            m["nta_both_den"] = n_den
            m["se_nta_both"] = standard_error(rate, n_den)
            per_cond[cond] = m
        out[display] = per_cond
    return out


def make_plot(metrics, out_path):
    """Draw the horizontal grouped-bar figure for the NTA-to-both rate.

    Bars are grouped by model on the y-axis, three colored bars per group with
    binomial SE whiskers, each annotated with its rate and parsed-pair
    denominator. Style mirrors plots_balanced_fnr_fpr/fnr_by_model.png.
    """
    models = list(metrics.keys())
    n_models = len(models)
    n_conds = len(CONDITIONS)
    bar_h = 0.8 / n_conds
    offsets = [(i - (n_conds - 1) / 2.0) * bar_h for i in range(n_conds)]

    fig, ax = plt.subplots(figsize=(9, 1.2 * n_models + 1.5))

    for ci, cond in enumerate(CONDITIONS):
        ys = [mi + offsets[ci] for mi in range(n_models)]
        vals = [metrics[m][cond]["nta_both_rate"] for m in models]
        errs = [metrics[m][cond]["se_nta_both"] for m in models]
        ax.barh(
            ys, vals, height=bar_h, xerr=errs,
            color=CONDITION_COLORS[cond], edgecolor="black", linewidth=0.5,
            error_kw=dict(ecolor="black", lw=0.8, capsize=2),
            label=LABELS[cond], zorder=3,
        )
        for y, m in zip(ys, models):
            cell = metrics[m][cond]
            v = cell["nta_both_rate"]
            n = cell["nta_both_den"]
            # Always show the denominator: parser failures + partial stakeholder
            # runs both pull it below the nominal 1000, and honesty about
            # coverage matters more than a clean label.
            txt = f"{v:.3f} (n={n})"
            ax.text(v + cell["se_nta_both"] + 0.005, y, txt,
                    va="center", ha="left", fontsize=7)

    ax.set_yticks(range(n_models))
    ax.set_yticklabels(models)
    ax.invert_yaxis()  # first model in MODELS appears at the top
    ax.set_xlabel("rate of NTA-to-both (sycophancy / always-nice)")
    ax.set_title("AITA-NTA-FLIP — model outputs NTA to BOTH POV versions, "
                 "by model and method")
    ax.set_xlim(0, max(0.05, ax.get_xlim()[1]) * 1.18)
    ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=0)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.95,
              title="method")
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    metrics = collect_metrics()

    # Audit dump: print every plotted number so the figure is reproducible by
    # eye from the console output.
    print("model".ljust(32), "condition".ljust(24),
          "NTA-both(±SE)".ljust(20), "n_parsed/n_total")
    for m, per in metrics.items():
        for cond in CONDITIONS:
            c = per[cond]
            print(m.ljust(32), cond.ljust(24),
                  f"{c['nta_both_rate']:.3f}±{c['se_nta_both']:.3f}".ljust(20),
                  f"{c['nta_both_den']}/{c['n_total']}")

    make_plot(metrics, os.path.join(OUT_DIR, "nta_both_rate_by_model.png"))


if __name__ == "__main__":
    main()
