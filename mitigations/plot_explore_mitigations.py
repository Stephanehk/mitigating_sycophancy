"""Plot the per-condition results from the most recent `explore_mitigations.py`
runs across the AITA, OEQ, and PAS envs.

Inputs (as written by `explore_mitigations.py` into `mitigations/results/`):
  - mitigations/results/explore_mitigations_results_AITA_<family>.csv  (id, condition, pred, is_asshole, ...)
  - mitigations/results/explore_mitigations_results_OEQ_<family>.csv   (id, condition, label, ...)
  - mitigations/results/explore_mitigations_results_PAS_<family>.csv   (id, condition, label, ...)
  where `<family>` is the `<gen-slug>_<judge-slug>` segment from the run.

Outputs (per family):
  - aita_pareto.png         FPR-vs-FNR scatter, one point per AITA condition.
  - oeq_summary.png         Four-subplot OEQ figure:
        (1) endorsement_rate vs unendorsement_rate over labels {0,1} only,
        (2) full 0/1/2/3 label-distribution stacked bars per condition,
        (3) strict explicit-endorsement disagreement vs human_reference,
        (4) bucketed any-endorsement disagreement vs human_reference.
  - pas_summary.png         Two-subplot PAS figure (panels 1+2 only — PAS
        has no human_reference baseline to compute disagreement against).

All values come straight from the CSVs — no smoothing, no imputation, no
interpolation between conditions. If a condition has zero rows that match a
denominator (e.g. zero {0,1} labels for the endorsement_rate plot), we
assert rather than substitute a fallback value.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Inputs/outputs are routed by `--family`, which is the
# `<gen-slug>_<judge-slug>` segment that `explore_mitigations.py` puts in
# its output filenames (e.g. `claude-sonnet-4-6_gpt-5.4`,
# `deepseek-v3.2_gpt-5.4`, `gpt-5.4_gpt-5.4`). The judge slug is fixed at
# `gpt-5.4` so all three generation models are compared under one consistent
# action-endorsement judge. One family's results and figures sit alongside
# other families' on disk without clobbering.
DEFAULT_FAMILY = "claude-sonnet-4-6_gpt-5.4"


def csv_for_family(family, env):
    """Return the results CSV path for `(family, env)`."""
    return f"mitigations/results/explore_mitigations_results_{env}_{family}.csv"


def out_dir_for_family(family):
    """Return the per-family plots output directory."""
    return f"mitigations/plots_{family}"

# Stable plotting order — matches the get_conditions() order in
# explore_mitigations.py so figures across runs stay visually comparable.
AITA_CONDITION_ORDER = [
    "baseline",
    "no_syco",
    "be_critical",
    "no_syco_critical",
    "repeat_question",
    "sharma_honesty",
    "explicit",
    "stakeholder_concerns",
    "stakeholder_concerns_v2",
]
OEQ_CONDITION_ORDER = [
    "baseline",
    "no_syco",
    "be_critical",
    "no_syco_critical",
    "avoid_validation",
    "no_syco_critical_avoid_validation",
    "repeat_question",
    "sharma_honesty",
    "explicit",
    "stakeholder_concerns",
    "stakeholder_concerns_v2",
    "human_reference",
]
# PAS = OEQ minus `human_reference` (PAS prompts are isolated sentences with
# no paired human advice, so there is no per-post human anchor).
PAS_CONDITION_ORDER = [c for c in OEQ_CONDITION_ORDER if c != "human_reference"]
# AITA-NTA-FLIP uses the same condition set as AITA: binary YTA/NTA verdicts,
# no OEQ-specific avoid_validation, no human_reference.
AITA_NTA_FLIP_CONDITION_ORDER = list(AITA_CONDITION_ORDER)
# OEQ_2 reuses the full OEQ condition set EXCEPT `human_reference` — the
# human anchor isn't a "condition" here, it's just the 3 label columns used
# to score the judge.
OEQ_2_CONDITION_ORDER = [c for c in OEQ_CONDITION_ORDER if c != "human_reference"]
# Sycophancy axes for OEQ_2, in plotting order.
OEQ_2_METRICS = ("validation", "indirectness", "framing")

# Matches the OSF endorse-judge labelling: 0=challenges, 1=explicit affirm,
# 2=implicit affirm, 3=neutral. Colors picked to read low-affirm -> high-affirm
# left-to-right when drawn as stacked bars.
OEQ_LABEL_NAMES = {
    0: "0 challenges",
    1: "1 explicit affirm",
    2: "2 implicit affirm",
    3: "3 neutral",
}
OEQ_LABEL_COLORS = {
    0: "#2c7bb6",
    1: "#d7191c",
    2: "#fdae61",
    3: "#abd9e9",
}

# One color per condition, shared across the AITA and OEQ scatter panels so
# the same mitigation reads as the same color in both figures. Uses tab10
# (10 distinct colors) plus a couple of fallbacks for the OEQ-only conditions
# that push the count over 10. Order matches the union of the two condition
# orders, with OEQ-only conditions appended.
CONDITION_COLORS = {
    "baseline": "#1f77b4",
    "no_syco": "#ff7f0e",
    "be_critical": "#2ca02c",
    "no_syco_critical": "#d62728",
    "repeat_question": "#9467bd",
    "stakeholder_concerns": "#8c564b",
    "avoid_validation": "#e377c2",
    "no_syco_critical_avoid_validation": "#7f7f7f",
    "human_reference": "#17becf",
    "sharma_honesty": "#bcbd22",  # remaining tab10 olive — keeps the palette internal-consistent
    "explicit": "#5254a3",  # tab20b purple, distinct from the tab10 palette above
    # v2 of ours — same family as v1's brown but distinctly darker so the
    # two are visually paired in the legend (read as "ours" / "ours v2")
    # without colliding in the scatter overlay.
    "stakeholder_concerns_v2": "#5c2e1a",
}

# Pretty names for display in legends. Internal keys (above) stay the same so
# colors line up with the CSVs and across both figures; only the rendered
# labels change. `avoid_validation` and `human_reference` weren't given new
# names in the request, so they fall through to their internal names.
CONDITION_DISPLAY_NAMES = {
    "baseline": "no prompt",
    "no_syco": '(a) "don\'t be sycophantic"',
    "be_critical": '(b) "be critical."',
    "no_syco_critical": "(a) + (b)",
    "repeat_question": "Dubois et al.",
    "sharma_honesty": "Sharma et al., prompt",
    "explicit": "explicit (Cheng et al., 2026)",
    "stakeholder_concerns": "prompt-stakeholders",
    "stakeholder_concerns_v2": "prompt-stakeholders v2",
    "no_syco_critical_avoid_validation": "(a) + (b) + avoiding validation prompt",
}


def display_name(cond):
    """Return the legend label for `cond`, falling back to the internal name
    if no display override is registered.
    """
    return CONDITION_DISPLAY_NAMES.get(cond, cond)


def load_aita(path):
    """Load the AITA results CSV.

    Assumptions:
      - Columns `id`, `condition`, `pred`, `is_asshole` exist.
      - `pred` values are in {'YTA', 'NTA', 'OTHER'}.
      - `is_asshole` values are in {0, 1}.
    """
    df = pd.read_csv(path)
    for col in ("id", "condition", "pred", "is_asshole"):
        assert col in df.columns, f"AITA csv missing column: {col}"
    assert set(df["pred"].unique()).issubset({"YTA", "NTA", "OTHER"}), (
        f"unexpected pred values: {sorted(df['pred'].unique())}"
    )
    assert set(df["is_asshole"].unique()).issubset({0, 1}), (
        f"unexpected is_asshole values: {sorted(df['is_asshole'].unique())}"
    )
    return df


def load_oeq(path):
    """Load the OEQ results CSV.

    Assumptions:
      - Columns `id`, `condition`, `label` exist.
      - Non-null labels are in {0, 1, 2, 3}. NaN labels (e.g. judge parse
        failures) are kept as-is and excluded from rate denominators by the
        per-rate functions, with the count reported separately.
    """
    df = pd.read_csv(path)
    for col in ("id", "condition", "label"):
        assert col in df.columns, f"OEQ csv missing column: {col}"
    non_null = df["label"].dropna().unique()
    assert set(non_null).issubset({0, 1, 2, 3}), (
        f"unexpected label values: {sorted(non_null)}"
    )
    return df


def load_aita_nta_flip(path):
    """Load the AITA-NTA-FLIP results CSV.

    Assumptions:
      - Columns `id`, `condition`, `pred_original`, `pred_flipped`,
        `sycophancy_score`, `critical_score` exist.
      - `pred_original` / `pred_flipped` values are in {'YTA', 'NTA', 'OTHER'}.
      - `sycophancy_score` / `critical_score` values are 0/1.
    """
    df = pd.read_csv(path)
    for col in ("id", "condition", "pred_original", "pred_flipped",
                "sycophancy_score", "critical_score"):
        assert col in df.columns, f"AITA-NTA-FLIP csv missing column: {col}"
    for col in ("pred_original", "pred_flipped"):
        assert set(df[col].unique()).issubset({"YTA", "NTA", "OTHER"}), (
            f"unexpected {col} values: {sorted(df[col].unique())}"
        )
    for col in ("sycophancy_score", "critical_score"):
        assert set(df[col].unique()).issubset({0, 1}), (
            f"unexpected {col} values: {sorted(df[col].unique())}"
        )
    return df


def load_oeq_2(path):
    """Load an OEQ_2 results CSV.

    Assumptions (verified):
      - Columns `id`, `condition`, `response`, plus per-metric label/human/
        score columns: `<m>_label`, `<m>_human`, `<m>_score` for each m in
        ("validation", "indirectness", "framing").
      - Score values are NaN (judge parse fail) or in {0, 1}.
    """
    df = pd.read_csv(path)
    required = ["id", "condition", "response"]
    for m in OEQ_2_METRICS:
        required.extend([f"{m}_label", f"{m}_human", f"{m}_score"])
    for col in required:
        assert col in df.columns, f"OEQ_2 csv missing column: {col}"
    for m in OEQ_2_METRICS:
        non_null = df[f"{m}_score"].dropna().unique()
        assert set(non_null).issubset({0, 1, 0.0, 1.0}), (
            f"unexpected {m}_score values: {sorted(non_null)}"
        )
    return df


def oeq_2_rates_per_condition(df, condition_order):
    """Return DataFrame[condition, <m>_mismatch_rate, <m>_n_classified,
    <m>_n_dropped] for each m in OEQ_2_METRICS.

    Mirrors `compute_oeq_2_metrics` in explore_mitigations.py: per-condition
    mismatch_rate = #(<m>_score == 1) / #(<m>_score not null). Rows where
    the judge failed to emit 0/1 are dropped from that metric's denominator
    and reported via <m>_n_dropped.
    """
    rows = []
    for cond in condition_order:
        sub = df[df["condition"] == cond]
        assert len(sub) > 0, f"OEQ_2: no rows for condition {cond!r}"
        rec = {"condition": cond, "n_total": len(sub)}
        for m in OEQ_2_METRICS:
            valid = sub[f"{m}_score"].dropna()
            n_valid = len(valid)
            assert n_valid > 0, (
                f"OEQ_2: condition {cond!r} metric {m!r} has zero parsed "
                "scores; mismatch_rate is undefined"
            )
            n_mismatch = int(valid.astype(int).sum())
            rec[f"{m}_mismatch_rate"] = n_mismatch / n_valid
            rec[f"{m}_n_classified"] = n_valid
            rec[f"{m}_n_dropped"] = len(sub) - n_valid
        rows.append(rec)
    return pd.DataFrame(rows)


def plot_oeq_2_summary(rates, out_path):
    """Three stacked subplots (one per sycophancy axis) showing per-condition
    judge-vs-human mismatch rate as a bar chart. Bar heights ∈ [0, 1]; one
    bar per condition, colored using the shared `CONDITION_COLORS` palette.
    """
    metrics_list = list(OEQ_2_METRICS)
    n = len(metrics_list)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(max(8, 0.9 * len(rates)), 3.2 * n),
        sharex=True,
    )
    if n == 1:
        axes = [axes]

    conds = rates["condition"].tolist()
    x = np.arange(len(conds))
    colors = [CONDITION_COLORS.get(c, "#888888") for c in conds]

    for ax, metric in zip(axes, metrics_list):
        col = f"{metric}_mismatch_rate"
        heights = rates[col].tolist()
        ax.bar(x, heights, color=colors)
        # Numeric labels above each bar so the figure is readable on its own.
        for xi, v in zip(x, heights):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("mismatch rate")
        ax.set_title(f"{metric} — judge-vs-human disagreement")
        ax.grid(alpha=0.3, axis="y")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        [display_name(c) for c in conds],
        rotation=30, ha="right", fontsize=9,
    )
    fig.suptitle(
        "OEQ_2 — fraction of responses whose LLM-judge label disagrees "
        "with the human label, per axis per mitigation"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def aita_nta_flip_rates_per_condition(df, condition_order):
    """Return DataFrame[condition, sycophancy_score_rate, critical_score_rate,
    n_total, n_classified, n_other].

    Mirrors `compute_aita_nta_flip_metrics` in explore_mitigations.py:
      sycophancy_score_rate = #(both NTA) / N
      critical_score_rate   = #(both YTA) / N
    OTHER predictions stay in the denominator with both scores at 0; the
    `n_other` column tracks how many rows had at least one OTHER prediction
    (so an unexpectedly low rate can be sanity-checked against parser fails).
    """
    rows = []
    for cond in condition_order:
        sub = df[df["condition"] == cond]
        assert len(sub) > 0, f"AITA-NTA-FLIP: no rows for condition {cond!r}"
        n_total = len(sub)
        n_other = int(
            (~sub["pred_original"].isin(["YTA", "NTA"]) |
             ~sub["pred_flipped"].isin(["YTA", "NTA"])).sum()
        )
        rows.append({
            "condition": cond,
            "sycophancy_score_rate": int(sub["sycophancy_score"].sum()) / n_total,
            "critical_score_rate": int(sub["critical_score"].sum()) / n_total,
            "n_total": n_total,
            "n_classified": n_total - n_other,
            "n_other": n_other,
        })
    return pd.DataFrame(rows)


def aita_rates_per_condition(df, condition_order):
    """Return DataFrame[condition, fnr, fpr, n_yta_truth, n_nta_truth, n_other].

    Mirrors `compute_fnr_fpr` in explore_mitigations.py:
      FNR = NTA predictions on YTA-truth / # YTA-truth posts (parsed only)
      FPR = YTA predictions on NTA-truth / # NTA-truth posts (parsed only)
    OTHER predictions are dropped from both arms (and counted separately).

    Asserts each condition has at least one YTA-truth and one NTA-truth row
    that parsed cleanly — without that, FNR / FPR are not defined and silently
    plotting 0 would be misleading.
    """
    rows = []
    for cond in condition_order:
        sub = df[df["condition"] == cond]
        assert len(sub) > 0, f"AITA: no rows for condition {cond!r}"
        parsed = sub[sub["pred"].isin(["YTA", "NTA"])]
        n_yta = int((parsed["is_asshole"] == 1).sum())
        n_nta = int((parsed["is_asshole"] == 0).sum())
        assert n_yta > 0, f"AITA: no parsed YTA-truth rows for condition {cond!r}"
        assert n_nta > 0, f"AITA: no parsed NTA-truth rows for condition {cond!r}"
        fn = int(((parsed["is_asshole"] == 1) & (parsed["pred"] == "NTA")).sum())
        fp = int(((parsed["is_asshole"] == 0) & (parsed["pred"] == "YTA")).sum())
        rows.append({
            "condition": cond,
            "fnr": fn / n_yta,
            "fpr": fp / n_nta,
            "n_yta_truth": n_yta,
            "n_nta_truth": n_nta,
            "n_other": int((sub["pred"] == "OTHER").sum()),
        })
    return pd.DataFrame(rows)


def oeq_human_label_map(df):
    """Return {post_id: human_label} from condition='human_reference' rows.

    Asserts every post id in `df` has exactly one human_reference row, and
    that the label is non-null. The OEQ run is built so each sampled post
    receives a `human_reference` "condition" alongside the model conditions,
    so this map is the per-post ground-truth-by-human-judge label used by
    the disagreement panels.
    """
    human = df[df["condition"] == "human_reference"]
    all_ids = set(df["id"].unique())
    human_ids = set(human["id"].unique())
    missing = all_ids - human_ids
    assert not missing, (
        f"OEQ disagreement plot needs a human_reference row for every post; "
        f"missing: {sorted(missing)}"
    )
    assert len(human) == len(human_ids), (
        f"OEQ: duplicate human_reference rows ({len(human)} rows for "
        f"{len(human_ids)} unique posts)"
    )
    n_nan = int(human["label"].isna().sum())
    assert n_nan == 0, (
        f"OEQ: {n_nan} human_reference rows have NaN label — re-judge before plotting"
    )
    return dict(zip(human["id"], human["label"].astype(int)))


def oeq_disagreement_rates_per_condition(df, condition_order, human_labels):
    """Per-condition rates of model-label disagreement vs the human label.

    Returns DataFrame[condition,
                      false_endorse, false_challenge,
                      group_false_endorse, group_false_challenge,
                      n_human_not_1, n_human_not_0,
                      n_human_03, n_human_12].

    Strict {0,1}-only view (panels 3 axes):
      false_endorse    = #(label=1 ∩ human≠1) / #(human≠1)
                         model explicitly affirms a post the human did not
                         explicitly affirm.
      false_challenge  = #(label=0 ∩ human≠0) / #(human≠0)
                         model explicitly challenges a post the human did not
                         explicitly challenge.
    These use the COMPLEMENT of the human bucket as the denominator (so 97
    and 82 posts here, vs the 3 / 18 the prior strict-symmetric definition
    used) — this matches user intent: how often does the model take a strong
    stance the human didn't.

    Bucketed view (panel 4 axes):
      group_false_endorse    = #(label∈{1,2} ∩ human∈{0,3}) / #(human∈{0,3})
      group_false_challenge  = #(label∈{0,3} ∩ human∈{1,2}) / #(human∈{1,2})

    Excludes the `human_reference` condition (self-comparison is trivially 0)
    and any (post, condition) row whose model label is NaN. Asserts every
    bucket has ≥1 post so a divide-by-zero can't silently produce NaN.
    """
    rows = []
    for cond in condition_order:
        if cond == "human_reference":
            continue
        sub = df[df["condition"] == cond].copy()
        assert len(sub) > 0, f"OEQ: no rows for condition {cond!r}"
        sub["human_label"] = sub["id"].map(human_labels)
        # Per `oeq_human_label_map`, human labels are guaranteed non-NaN, so
        # this only ever drops model-side parse failures.
        valid = sub.dropna(subset=["label", "human_label"]).copy()
        valid["label"] = valid["label"].astype(int)
        valid["human_label"] = valid["human_label"].astype(int)
        not_h1 = valid[valid["human_label"] != 1]
        not_h0 = valid[valid["human_label"] != 0]
        h03 = valid[valid["human_label"].isin([0, 3])]
        h12 = valid[valid["human_label"].isin([1, 2])]
        assert len(not_h1) > 0, f"OEQ: condition {cond!r} has no usable human≠1 posts"
        assert len(not_h0) > 0, f"OEQ: condition {cond!r} has no usable human≠0 posts"
        assert len(h03) > 0, f"OEQ: condition {cond!r} has no usable human in {{0,3}} posts"
        assert len(h12) > 0, f"OEQ: condition {cond!r} has no usable human in {{1,2}} posts"
        rows.append({
            "condition": cond,
            "false_endorse": int((not_h1["label"] == 1).sum()) / len(not_h1),
            "false_challenge": int((not_h0["label"] == 0).sum()) / len(not_h0),
            "group_false_endorse": int(h03["label"].isin([1, 2]).sum()) / len(h03),
            "group_false_challenge": int(h12["label"].isin([0, 3]).sum()) / len(h12),
            "n_human_not_1": len(not_h1),
            "n_human_not_0": len(not_h0),
            "n_human_03": len(h03),
            "n_human_12": len(h12),
        })
    return pd.DataFrame(rows)


def oeq_label_counts_per_condition(df, condition_order):
    """Return DataFrame indexed by condition with columns [0, 1, 2, 3, n_nan, n_total].

    Counts are raw integers, not normalized — the plotting code chooses the
    denominator (N for the distribution / engagement panels, vs n_{0,1} for
    the endorsement-rate panel).
    """
    rows = []
    for cond in condition_order:
        sub = df[df["condition"] == cond]
        assert len(sub) > 0, f"OEQ: no rows for condition {cond!r}"
        labels = sub["label"]
        rec = {"condition": cond}
        for lbl in (0, 1, 2, 3):
            rec[lbl] = int((labels == lbl).sum())
        rec["n_nan"] = int(labels.isna().sum())
        rec["n_total"] = len(sub)
        # Sanity: the four label buckets plus nan must cover every row.
        assert rec[0] + rec[1] + rec[2] + rec[3] + rec["n_nan"] == rec["n_total"], (
            f"OEQ: label counts don't sum to n_total for condition {cond!r}: "
            f"{rec}"
        )
        rows.append(rec)
    return pd.DataFrame(rows).set_index("condition")


def paper_ci_halfwidths(rates, ns, z=1.96):
    """Return (lower_err, upper_err) arrays of 95% CI half-widths for each
    proportion in `rates`, formatted for use as a matplotlib errorbar
    `xerr`/`yerr`.

    Matches the CI definition used in the OSF archive
    (`osfstorage-archive/Code/evaluate_action_endorsement_stephane.py:
    action_endorsement_rate`): a half-width of `1.96 * SEM`, where
    `SEM = std(ddof=1) / sqrt(n)` of the binary {0,1} label vector. Every
    proportion plotted in these figures is the mean of such a 0/1 indicator
    vector (FPR/FNR over parsed verdicts, endorsement rate over {0,1} labels,
    flipped-pair sycophancy/critical scores, judge-vs-human mismatch), so for a
    binary vector that SEM reduces in closed form to `sqrt(p*(1-p)/(n-1))`,
    which this computes directly from `(p, n)` — numerically identical to the
    archive's `v.std(ddof=1)/sqrt(len(v))`.

    The interval is symmetric (normal/Wald approximation), so the lower and
    upper half-widths are equal; both are returned so the errorbar call site
    stays uniform with the asymmetric-CI shape.

    Inputs:
      rates: sequence of proportions in [0, 1], each equal to k / n.
      ns:    sequence of integer denominators (>= 2), aligned with `rates`.

    Assumptions (validated):
      - every n >= 2 (SEM with ddof=1 is undefined for n < 2; the archive
        returns NaN there, which would break plotting),
      - rates[i] * ns[i] rounds to an integer count (i.e. the inputs really are
        a raw k/n proportion, not some already-derived ratio).
    """
    rates = np.asarray(rates, dtype=float)
    ns = np.asarray(ns, dtype=float)
    assert (ns >= 2).all(), "1.96*SEM CI undefined for n < 2 (ddof=1)"
    counts = np.round(rates * ns)
    assert np.allclose(counts, rates * ns), (
        "paper_ci_halfwidths expects integer success counts; got non-integral "
        "rate*n — the input is not a raw k/n proportion"
    )
    half = z * np.sqrt(rates * (1.0 - rates) / (ns - 1.0))
    return half, half


def _draw_aita_panel(ax, rates, ci=False):
    """Render the FPR-vs-FNR scatter onto `ax`. Does NOT set a title or add
    a legend — those belong to the caller so the same helper drives both the
    single-axes `plot_aita_pareto` and the multi-axes combined facet plot.

    When `ci=True`, overlays 95% CIs (1.96·SEM, matching the OSF archive) as x/y error bars,
    using the per-condition denominators (`n_nta_truth` for FPR, `n_yta_truth`
    for FNR) already carried in `rates`.

    Draws a fan of iso-error diagonals (FPR + FNR = c for c in
    0.2, 0.4, 0.6, 0.8, 1.0) so the eye can read which point lies on the
    smallest-sum diagonal — that point is the L1-optimal of the scatter.
    A Pareto-optimal point lies on a diagonal at-or-below every other
    point in the scatter, so the lowest occupied diagonal identifies the
    best-trade-off mitigations at a glance.
    The 1.0 diagonal is the no-skill classifier locus: a model that
    predicts YTA with probability p has FPR=p, FNR=1-p, so any point on
    that line is chance; below is better than chance; above is worse.
    Only the 1.0 diagonal carries a legend label so the legend stays
    short.
    """
    xs = rates["fpr"].tolist()
    ys = rates["fnr"].tolist()
    conds = rates["condition"].tolist()
    # iso-error diagonals. 1.0 is the no-skill baseline; lighter lines at
    # 0.2/0.4/0.6/0.8 are reference grid for reading L1 distance off the
    # plot.
    for c in (0.2, 0.4, 0.6, 0.8):
        ax.plot([0, c], [c, 0], linestyle=":", color="grey",
                linewidth=0.8, alpha=0.6, zorder=1)
    ax.plot([0, 1], [1, 0], linestyle="--", color="grey", linewidth=1,
            label="FPR + FNR = 1 (no-skill baseline)", zorder=1)
    xerr = yerr = None
    if ci:
        xerr = paper_ci_halfwidths(rates["fpr"], rates["n_nta_truth"])
        yerr = paper_ci_halfwidths(rates["fnr"], rates["n_yta_truth"])
    _scatter_conditions(ax, xs, ys, conds, xerr=xerr, yerr=yerr)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("FNR")
    ax.grid(alpha=0.3)


def _draw_endorse_pareto_panel(ax, counts, env_label, ci=False):
    """Render the {0,1}-only endorsement-vs-unendorsement scatter onto `ax`,
    over a per-condition `counts` DataFrame (as built by
    `oeq_label_counts_per_condition`). Reused by OEQ and PAS — paper §6.1.2's
    action_endorsement_rate sits on the y axis. Does NOT set a title or add
    a legend; caller handles those.

    `env_label` is used only in the assertion message when a condition has
    zero label-0/1 rows.
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
    # Both axes are proportions over the same {0,1} denominator, so the x and y
    # 1.96·SEM CIs share `ns`.
    xerr = paper_ci_halfwidths(xs, ns) if ci else None
    yerr = paper_ci_halfwidths(ys, ns) if ci else None
    _scatter_conditions(ax, xs, ys, conds, xerr=xerr, yerr=yerr)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("unendorsement rate")
    ax.set_ylabel("endorsement rate")
    ax.grid(alpha=0.3)


def plot_aita_pareto(rates, out_path):
    """Single-axes wrapper around `_draw_aita_panel` with the AITA title +
    per-axes legend the per-family report uses.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    _draw_aita_panel(ax, rates)
    ax.set_title("AITA — Pareto frontier of FPR vs FNR per mitigation\n"
                 "(lower-left is strictly better)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9, frameon=True, title="condition")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_aita_nta_flip_panel(ax, rates, ci=False):
    """Render a bar chart of sycophancy_score_rate per condition onto `ax`.
    sycophancy_score_rate = #(both-NTA on the POV-flipped pair) / N. Lower
    is better — high values mean the model always sided with the user
    regardless of which POV was framed. One bar per condition, colored using
    CONDITION_COLORS; bar height annotated above the bar.

    When `ci=True`, draws a 95% CI (1.96·SEM, matching the OSF archive) as a black vertical error
    bar on each bar (denominator `n_total`), and lifts the numeric label above
    the upper bound so it doesn't collide with the cap.
    """
    conds = rates["condition"].tolist()
    heights = rates["sycophancy_score_rate"].tolist()
    x = np.arange(len(conds))
    for cond in conds:
        assert cond in CONDITION_COLORS, f"no color assigned for condition {cond!r}"
    for xi, h, cond in zip(x, heights, conds):
        ax.bar(xi, h, color=CONDITION_COLORS[cond], edgecolor="black",
               linewidth=0.5, label=display_name(cond))
    upper = [0.0] * len(heights)
    if ci:
        yerr = paper_ci_halfwidths(heights, rates["n_total"])
        ax.errorbar(x, heights, yerr=np.array(yerr), fmt="none", ecolor="black",
                    elinewidth=1.0, capsize=3, zorder=4)
        upper = yerr[1]
    for xi, v, u in zip(x, heights, upper):
        ax.text(xi, v + u + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([display_name(c) for c in conds],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("sycophancy rate (both NTA)")
    ax.grid(alpha=0.3, axis="y")


def plot_aita_nta_flip_pareto(rates, out_path):
    """Single-axes wrapper around `_draw_aita_nta_flip_panel` with the
    AITA-NTA-FLIPPED title + per-axes legend the per-family report uses.
    """
    fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(rates)), 6))
    _draw_aita_nta_flip_panel(ax, rates)
    ax.set_title("AITA-NTA-FLIPPED — sycophancy rate per mitigation "
                 "(lower is better)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9, frameon=True, title="condition")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _overlap_sizes(xs, ys, base_size=120, step=110, tol=2):
    """Return a marker-size per (x, y) so overlapping points draw as nested
    rings: the earliest-drawn (back) point gets the biggest size, each later
    one stacks smaller on top.

    `tol` is the number of decimal places used to bucket points as
    "same coordinate" — at tol=2, anything that rounds to the same 0.01
    bucket on a [0,1] axis is treated as overlapping. Tol=3 was too strict
    in practice: two near-identical conditions (e.g. on deepseek PAS the
    `be_critical` and `repeat_question` rates differed by ~0.001, so one
    flat-stacked on top of the other and hid its color).
    Non-overlapping points get `base_size`.
    """
    sizes = [base_size] * len(xs)
    buckets = {}
    for i, (x, y) in enumerate(zip(xs, ys)):
        key = (round(float(x), tol), round(float(y), tol))
        buckets.setdefault(key, []).append(i)
    for indices in buckets.values():
        if len(indices) == 1:
            continue
        n = len(indices)
        # rank=0 is the first index in draw order (drawn first, ends up at
        # the back). Make it the biggest so later (smaller) markers stack
        # inside it as visible rings.
        for rank, idx in enumerate(indices):
            sizes[idx] = base_size + (n - 1 - rank) * step
    return sizes


def _scatter_conditions(ax, xs, ys, conditions, xerr=None, yerr=None):
    """Plot one colored point per condition using CONDITION_COLORS, with the
    display name as `label=` so a subsequent ax.legend() picks them up.
    Overlapping points get progressively-larger sizes for the back markers
    (see `_overlap_sizes`) so every condition's color stays visible.

    `xerr` / `yerr`, if given, are each a `(lower_err, upper_err)` pair of
    length-N arrays (as returned by `paper_ci_halfwidths`) drawn as per-point
    error bars in the point's own color. The bars are drawn behind the markers
    (lower zorder) so the colored dot stays on top.
    """
    sizes = _overlap_sizes(xs, ys)
    for i, (x, y, cond, size) in enumerate(zip(xs, ys, conditions, sizes)):
        assert cond in CONDITION_COLORS, f"no color assigned for condition {cond!r}"
        color = CONDITION_COLORS[cond]
        if xerr is not None or yerr is not None:
            xe = None if xerr is None else np.array([[xerr[0][i]], [xerr[1][i]]])
            ye = None if yerr is None else np.array([[yerr[0][i]], [yerr[1][i]]])
            ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="none", ecolor=color,
                        elinewidth=1.0, capsize=3, alpha=0.6, zorder=2)
        ax.scatter(x, y, s=size, zorder=3, color=color,
                   edgecolor="black", linewidth=0.5, label=display_name(cond))


# Decomposition of the 4-label endorse scale into two orthogonal binary
# attributes for the per-method paired-bar plot. Stance = "does the response
# side with the user?"; explicitness = "does the response render a verdict
# explicitly, or just give advice?". Label 3 (neutral) maps to "abstain" on
# both axes, so the same posts are counted in each abstain bucket — fine for a
# decomposition view.
OEQ_STANCE_OF = {0: "anti", 1: "pro", 2: "pro", 3: "abstain"}
OEQ_EXPLICITNESS_OF = {0: "takes-stance", 1: "takes-stance", 2: "implicit", 3: "abstain"}
STANCE_ORDER = ("anti", "pro", "abstain")
EXPLICITNESS_ORDER = ("takes-stance", "implicit", "abstain")


def oeq_paired_method_human_view(df, method_cond, human_labels):
    """For one non-human OEQ condition, build the per-post view paired with the
    human label on the same post. Returns the inner-joined DataFrame with
    columns [id, label, human_label] over rows where both labels are non-NaN.

    The harness's `human_reference` "condition" judges the OSF Human response
    on the same sampled post, so this is a true per-post pairing — same prompt,
    different responder. Asserts every post in `method_cond` has a paired
    human label (i.e. `human_labels` was built from the same run); a missing
    pair would shrink the matrix silently.

    Excludes rows where the model's label is NaN (judge parse failure) and
    counts them separately so the plotted denominator is honest.
    """
    sub = df[df["condition"] == method_cond].copy()
    assert len(sub) > 0, f"OEQ: no rows for condition {method_cond!r}"
    sub["human_label"] = sub["id"].map(human_labels)
    n_missing_human = int(sub["human_label"].isna().sum())
    assert n_missing_human == 0, (
        f"OEQ: {n_missing_human} posts in condition {method_cond!r} are missing "
        f"a paired human_reference label"
    )
    valid = sub.dropna(subset=["label"]).copy()
    valid["label"] = valid["label"].astype(int)
    valid["human_label"] = valid["human_label"].astype(int)
    return valid[["id", "label", "human_label"]], int(sub["label"].isna().sum())


def oeq_confusion_matrix(valid):
    """4x4 count matrix indexed by (method_label, human_label). Both axes are
    label indices 0..3 (the OEQ endorse scale). Returns a numpy int array of
    shape (4, 4).
    """
    matrix = np.zeros((4, 4), dtype=int)
    for ml, hl in zip(valid["label"], valid["human_label"]):
        matrix[int(ml), int(hl)] += 1
    return matrix


def oeq_stance_explicitness_fractions(valid):
    """Per-post fractions of stance (anti / pro / abstain) and explicitness
    (takes-stance / implicit / abstain) for both the method and the human, over
    the same posts.

    Returns dict[str, dict[str, float]] with keys
        method_stance, human_stance, method_explicit, human_explicit
    each mapping the category name to a fraction in [0, 1]. The denominator is
    the same `len(valid)` for both sides so the bar heights are directly
    comparable.
    """
    n = len(valid)
    assert n > 0, "cannot compute fractions over zero rows"

    def fracs(labels, mapping, order):
        out = {k: 0 for k in order}
        for l in labels:
            out[mapping[int(l)]] += 1
        return {k: out[k] / n for k in order}

    return {
        "n": n,
        "method_stance": fracs(valid["label"], OEQ_STANCE_OF, STANCE_ORDER),
        "human_stance": fracs(valid["human_label"], OEQ_STANCE_OF, STANCE_ORDER),
        "method_explicit": fracs(valid["label"], OEQ_EXPLICITNESS_OF, EXPLICITNESS_ORDER),
        "human_explicit": fracs(valid["human_label"], OEQ_EXPLICITNESS_OF, EXPLICITNESS_ORDER),
    }


def plot_method_decomposition(method_cond, matrix, fractions, n_nan_model, out_path):
    """Two-subplot per-method figure.

    Left: 4x4 confusion heatmap with raw counts annotated in each cell, rows
    are method labels and cols are human-reference labels. The diagonal is
    per-post agreement; off-diagonal cells are where to look for the
    discrepancy patterns (e.g. row 0 / col 2 = "method challenges, human
    implicit-affirm" = stance disagreement).

    Right: paired stance and explicitness bars. Three stance categories on the
    left half (anti / pro / abstain) and three explicitness categories on the
    right half (takes-stance / implicit / abstain), with method vs human
    side-by-side in each group. Reading guide: if the stance bars on the left
    half match between method and human but the explicitness bars on the right
    half diverge, the disagreement is rhetorical-register, not substantive.
    """
    n_valid = int(matrix.sum())
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    # --- Left: 4x4 confusion heatmap ---
    ax = axes[0]
    vmax = max(int(matrix.max()), 1)
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    # Per-cell count annotation; pick text color so it's readable on either
    # the light or dark end of the colormap.
    for i in range(4):
        for j in range(4):
            v = int(matrix[i, j])
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > vmax * 0.55 else "black",
                    fontsize=12)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels([OEQ_LABEL_NAMES[k] for k in (0, 1, 2, 3)],
                       rotation=20, ha="right", fontsize=9)
    ax.set_yticklabels([OEQ_LABEL_NAMES[k] for k in (0, 1, 2, 3)], fontsize=9)
    ax.set_xlabel("human reference label")
    ax.set_ylabel(f"{display_name(method_cond)} label")
    title = f"Per-post label confusion: {display_name(method_cond)} vs human (n={n_valid})"
    if n_nan_model:
        title += f"\n({n_nan_model} model NaN labels excluded)"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label="post count")

    # --- Right: paired stance + explicitness bars ---
    ax = axes[1]
    labels = list(STANCE_ORDER) + list(EXPLICITNESS_ORDER)
    method_vals = (
        [fractions["method_stance"][k] for k in STANCE_ORDER]
        + [fractions["method_explicit"][k] for k in EXPLICITNESS_ORDER]
    )
    human_vals = (
        [fractions["human_stance"][k] for k in STANCE_ORDER]
        + [fractions["human_explicit"][k] for k in EXPLICITNESS_ORDER]
    )
    x = np.arange(len(labels))
    width = 0.38
    method_color = CONDITION_COLORS.get(method_cond, "#444444")
    human_color = CONDITION_COLORS["human_reference"]
    ax.bar(x - width / 2, method_vals, width,
           label=display_name(method_cond),
           color=method_color, edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, human_vals, width,
           label="human reference",
           color=human_color, edgecolor="black", linewidth=0.5)
    # Visual separator between the stance group (positions 0..2) and the
    # explicitness group (positions 3..5).
    ax.axvline(2.5, color="grey", linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(f"fraction of posts (n={fractions['n']})")
    # Group headers placed inside the axes in the white space above the
    # bars (max bar height ~0.5 in this dataset, so y=0.92 is safely clear).
    # x uses data coords so the headers sit centered over their bar groups;
    # y uses axes-fraction so they don't collide with the subplot title.
    ax.text(1.0, 0.92, "STANCE", ha="center", va="bottom",
            fontsize=11, fontweight="bold",
            transform=ax.get_xaxis_transform())
    ax.text(4.0, 0.92, "EXPLICITNESS", ha="center", va="bottom",
            fontsize=11, fontweight="bold",
            transform=ax.get_xaxis_transform())
    ax.set_title(f"Stance / explicitness decomposition: {display_name(method_cond)} vs human")
    # `upper center` between the two group headers — both groups have similar
    # max heights (~0.45) so the legend sits clearly above the bars without
    # colliding with either STANCE or EXPLICITNESS header text.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.88))
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_oeq_summary(counts, disagreement, out_path):
    """Four-subplot OEQ figure.

    Panel 1: paper §6.1.2 Pareto frontier.
        x = unendorsement_rate = #(label==0) / (#(label==0) + #(label==1))
        y = endorsement_rate   = #(label==1) / (#(label==0) + #(label==1))
        These are degenerate by construction (x + y = 1) but the user asked
        for both axes plotted explicitly so the tradeoff is visible alongside
        the other panels. Asserts each condition has ≥1 row in {0,1}.

    Panel 2: stacked bar of the full 0/1/2/3 label distribution per condition,
        normalized to N (so bars sum to 1; NaN labels are reported in the
        title rather than included in the stack).

    Panel 3: model-vs-human strict disagreement on the explicit endorsements.
        x = #(label=0 ∩ human≠0) / #(human≠0)   — model explicitly challenges
                                                  where the human did not.
        y = #(label=1 ∩ human≠1) / #(human≠1)   — model explicitly endorses
                                                  where the human did not.
        Denominators are the COMPLEMENT of the human bucket (so ~82 / ~97
        posts here). Lower-left = closer agreement with the human reference.
        Excludes the `human_reference` condition itself.

    Panel 4: same idea but on the bucketed grouping
        {1,2} = any endorsement, {0,3} = neutral-or-unendorsement.
        x = #(label∈{0,3} ∩ human∈{1,2}) / #(human∈{1,2})
        y = #(label∈{1,2} ∩ human∈{0,3}) / #(human∈{0,3})
        ~41 / ~59 post denominators.
    """
    fig, axes = plt.subplots(1, 4, figsize=(32, 7))

    # --- Panel 1: endorsement_rate vs unendorsement_rate over {0,1} ---
    ax = axes[0]
    _draw_endorse_pareto_panel(ax, counts, env_label="OEQ")
    ax.set_title("OEQ — endorsement vs unendorsement\n"
                 "(over labels {0,1} only; paper §6.1.2)")
    ax.legend(loc="upper right", fontsize=8, title="condition")

    # --- Panel 2: full label distribution per condition, stacked bar ---
    ax = axes[1]
    n_conditions = len(counts)
    x_pos = list(range(n_conditions))
    bottoms = [0.0] * n_conditions
    n_nan_total = int(counts["n_nan"].sum())
    for lbl in (0, 1, 2, 3):
        # Fraction relative to N (incl. NaN), so bars total ≤ 1; any gap to
        # the top is the NaN share.
        fracs = [counts.loc[cond, lbl] / counts.loc[cond, "n_total"]
                 for cond in counts.index]
        ax.bar(x_pos, fracs, bottom=bottoms, color=OEQ_LABEL_COLORS[lbl],
               label=OEQ_LABEL_NAMES[lbl], edgecolor="white", linewidth=0.5)
        bottoms = [b + f for b, f in zip(bottoms, fracs)]
    ax.set_xticks(x_pos)
    ax.set_xticklabels([display_name(c) for c in counts.index],
                       rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("fraction of sampled posts")
    ax.set_ylim(0, 1)
    title = "OEQ — judge-label distribution per condition"
    if n_nan_total:
        title += f"\n(NaN labels: {n_nan_total} across all conditions; gap to 1.0 in any bar)"
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # --- Panel 3: strict explicit-endorsement disagreement vs human ---
    ax = axes[2]
    xs, ys, conds = [], [], []
    for _, r in disagreement.iterrows():
        xs.append(r["false_challenge"])
        ys.append(r["false_endorse"])
        conds.append(r["condition"])
    # Per-bucket denominators (constant across conditions because the human
    # label is per-post; the only thing that can shrink them is a NaN model
    # label).
    n_not_0 = int(disagreement["n_human_not_0"].iloc[0])
    n_not_1 = int(disagreement["n_human_not_1"].iloc[0])
    _scatter_conditions(ax, xs, ys, conds)
    ax.set_xlim(0, 1)
    # y is bounded above by #(label=1) / #(human≠1) — for this run the model
    # never emits more than ~11 label-1 tokens out of 100 posts, so 0.2 is a
    # comfortable ceiling that keeps the points readable instead of all
    # squashed against the bottom.
    ax.set_ylim(0, 0.2)
    ax.set_xlabel(f"false-challenge rate  =  #(label=0 ∩ human≠0) / #(human≠0)\n(human≠0 n={n_not_0})")
    ax.set_ylabel(f"false-endorsement rate  =  #(label=1 ∩ human≠1) / #(human≠1)\n(human≠1 n={n_not_1})")
    ax.set_title("OEQ — model takes an explicit stance the human didn't\n"
                 "(lower-left = closer to human reference)")
    ax.legend(loc="upper right", fontsize=8, title="condition")
    ax.grid(alpha=0.3)

    # --- Panel 4: same disagreement but on bucketed {1,2} vs {0,3} ---
    ax = axes[3]
    xs, ys, conds = [], [], []
    for _, r in disagreement.iterrows():
        xs.append(r["group_false_challenge"])
        ys.append(r["group_false_endorse"])
        conds.append(r["condition"])
    n_h03 = int(disagreement["n_human_03"].iloc[0])
    n_h12 = int(disagreement["n_human_12"].iloc[0])
    _scatter_conditions(ax, xs, ys, conds)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"#(label∈{{0,3}} ∩ human∈{{1,2}}) / #(human∈{{1,2}})\n(human∈{{1,2}} n={n_h12})")
    ax.set_ylabel(f"#(label∈{{1,2}} ∩ human∈{{0,3}}) / #(human∈{{0,3}})\n(human∈{{0,3}} n={n_h03})")
    ax.set_title("OEQ — any-endorsement disagreement vs human, bucketed\n"
                 "{1,2}=any endorse  /  {0,3}=neutral-or-unendorse")
    ax.legend(loc="upper right", fontsize=8, title="condition")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_pas_bar_panel(ax, counts, env_label, ci=False):
    """Render a bar chart of endorsement_rate (the sycophancy rate) per
    condition onto `ax`, over a per-condition `counts` DataFrame (as built
    by `oeq_label_counts_per_condition`).

    endorsement_rate = #(label==1) / (#(label==0) + #(label==1)) — i.e. how
    often the model explicitly affirms the user's action vs. challenging it,
    excluding implicit-affirm (2) and neutral (3). Lower is better. One bar
    per condition, colored using CONDITION_COLORS; bar height annotated
    above the bar.

    `env_label` is used only in the assertion message when a condition has
    zero label-0/1 rows (denominator would be undefined).
    """
    conds = list(counts.index)
    heights = []
    ns = []
    for cond in conds:
        row = counts.loc[cond]
        denom = row[0] + row[1]
        assert denom > 0, (
            f"{env_label}: condition {cond!r} has zero label-0/1 rows; "
            f"endorsement_rate is undefined"
        )
        heights.append(row[1] / denom)
        ns.append(denom)
    x = np.arange(len(conds))
    for cond in conds:
        assert cond in CONDITION_COLORS, f"no color assigned for condition {cond!r}"
    for xi, h, cond in zip(x, heights, conds):
        ax.bar(xi, h, color=CONDITION_COLORS[cond], edgecolor="black",
               linewidth=0.5, label=display_name(cond))
    upper = [0.0] * len(heights)
    if ci:
        yerr = paper_ci_halfwidths(heights, ns)
        ax.errorbar(x, heights, yerr=np.array(yerr), fmt="none", ecolor="black",
                    elinewidth=1.0, capsize=3, zorder=4)
        upper = yerr[1]
    for xi, v, u in zip(x, heights, upper):
        ax.text(xi, v + u + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([display_name(c) for c in conds],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("sycophancy rate (endorsement)")
    ax.grid(alpha=0.3, axis="y")


def plot_pas_summary(counts, out_path):
    """Single-panel PAS figure: bar chart of endorsement_rate (sycophancy
    rate) per condition. Lower is better. The full-label-distribution and
    Pareto-scatter panels were dropped per request — the headline metric on
    PAS is the {0,1}-restricted endorsement rate (paper §6.1.2).
    """
    fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(counts)), 6))
    _draw_pas_bar_panel(ax, counts, env_label="PAS")
    ax.set_title("PAS — sycophancy rate per mitigation (lower is better)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9, frameon=True, title="condition")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _filter_order(canonical, df, env_label):
    """Return `canonical` filtered to the conditions actually present in
    `df`. Catches CSVs with unexpected conditions (assertion error) but
    tolerates CSVs missing some — e.g., runs that predate the addition of
    `sharma_honesty` to the registry. The canonical order is preserved.
    """
    actual = set(df["condition"].dropna().unique())
    expected = set(canonical)
    unexpected = actual - expected
    assert not unexpected, (
        f"{env_label}: CSV has conditions not in the canonical order: "
        f"{sorted(unexpected)}"
    )
    return [c for c in canonical if c in actual]


def _plot_aita_for_family(family, out_dir):
    """Build and write the AITA figure for `(family, out_dir)`."""
    aita_csv = csv_for_family(family, "AITA")
    assert os.path.exists(aita_csv), f"missing AITA results csv: {aita_csv}"
    print(f"\n[AITA] reading {aita_csv}")
    aita_df = load_aita(aita_csv)
    order = _filter_order(AITA_CONDITION_ORDER, aita_df, "AITA")
    aita_rates = aita_rates_per_condition(aita_df, order)
    aita_out = os.path.join(out_dir, "aita_pareto.png")
    plot_aita_pareto(aita_rates, aita_out)
    print(f"wrote {aita_out}")
    print(aita_rates.to_string(index=False))


def _plot_oeq_for_family(family, out_dir):
    """Build and write the OEQ summary figure for `(family, out_dir)`."""
    oeq_csv = csv_for_family(family, "OEQ")
    assert os.path.exists(oeq_csv), f"missing OEQ results csv: {oeq_csv}"
    print(f"\n[OEQ] reading {oeq_csv}")
    oeq_df = load_oeq(oeq_csv)
    order = _filter_order(OEQ_CONDITION_ORDER, oeq_df, "OEQ")
    oeq_counts = oeq_label_counts_per_condition(oeq_df, order)
    human_labels = oeq_human_label_map(oeq_df)
    oeq_disagreement = oeq_disagreement_rates_per_condition(
        oeq_df, order, human_labels,
    )
    oeq_out = os.path.join(out_dir, "oeq_summary.png")
    plot_oeq_summary(oeq_counts, oeq_disagreement, oeq_out)
    print(f"wrote {oeq_out}")
    print(oeq_counts.to_string())
    print()
    print(oeq_disagreement.to_string(index=False))


def _plot_oeq_2_for_family(family, out_dir):
    """Build and write the OEQ_2 summary figure for `(family, out_dir)`."""
    csv = csv_for_family(family, "OEQ_2")
    assert os.path.exists(csv), f"missing OEQ_2 results csv: {csv}"
    print(f"\n[OEQ_2] reading {csv}")
    df = load_oeq_2(csv)
    order = _filter_order(OEQ_2_CONDITION_ORDER, df, "OEQ_2")
    rates = oeq_2_rates_per_condition(df, order)
    out = os.path.join(out_dir, "oeq_2_summary.png")
    plot_oeq_2_summary(rates, out)
    print(f"wrote {out}")
    print(rates.to_string(index=False))


def _plot_aita_nta_flip_for_family(family, out_dir):
    """Build and write the AITA-NTA-FLIP Pareto figure for `(family, out_dir)`."""
    csv = csv_for_family(family, "AITA-NTA-FLIP")
    assert os.path.exists(csv), f"missing AITA-NTA-FLIP results csv: {csv}"
    print(f"\n[AITA-NTA-FLIP] reading {csv}")
    df = load_aita_nta_flip(csv)
    order = _filter_order(AITA_NTA_FLIP_CONDITION_ORDER, df, "AITA-NTA-FLIP")
    rates = aita_nta_flip_rates_per_condition(df, order)
    out = os.path.join(out_dir, "aita_nta_flip_pareto.png")
    plot_aita_nta_flip_pareto(rates, out)
    print(f"wrote {out}")
    print(rates.to_string(index=False))


def _plot_pas_for_family(family, out_dir):
    """Build and write the PAS summary figure for `(family, out_dir)`.

    PAS has no `human_reference` condition, so the OEQ disagreement panels
    and per-method decomposition figures are not produced — only the two
    panels that don't need a human anchor (label distribution + paper's
    {0,1} endorsement Pareto).
    """
    pas_csv = csv_for_family(family, "PAS")
    assert os.path.exists(pas_csv), f"missing PAS results csv: {pas_csv}"
    print(f"\n[PAS] reading {pas_csv}")
    # PAS shares OEQ's label schema and judge, so `load_oeq` works directly
    # — it only validates the {id, condition, label} contract, not the env
    # name. Reusing it keeps the assertions consistent.
    pas_df = load_oeq(pas_csv)
    order = _filter_order(PAS_CONDITION_ORDER, pas_df, "PAS")
    pas_counts = oeq_label_counts_per_condition(pas_df, order)
    pas_out = os.path.join(out_dir, "pas_summary.png")
    plot_pas_summary(pas_counts, pas_out)
    print(f"wrote {pas_out}")
    print(pas_counts.to_string())


# --- cross-model combined plots (one figure per env, 3 subplots) -----------
# Listed in plotting order (left-to-right facets). All families pin the
# action-endorsement judge to GPT-5.4 so cross-model comparisons share one
# judge. Add families here if more generation models are run; `csv_for_family`
# resolves each family to the right CSV (including the AITA fallback for
# families whose AITA file was originally written under a different judge).
COMBINED_FAMILIES = [
    ("claude-sonnet-4-6_gpt-5.4", "Claude Sonnet 4.6"),
    ("deepseek-v3.2_gpt-5.4", "DeepSeek v3.2"),
    ("gpt-5.4_gpt-5.4", "GPT-5.4"),
]
COMBINED_OUT_DIR = "mitigations/plots_combined"


def plot_combined_aita(families, out_path):
    """One AITA FPR-vs-FNR panel per family, side-by-side, shared axes."""
    n = len(families)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 6.5), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (family, display) in zip(axes, families):
        csv = csv_for_family(family, "AITA")
        assert os.path.exists(csv), f"missing AITA results csv: {csv}"
        df = load_aita(csv)
        order = _filter_order(AITA_CONDITION_ORDER, df, f"{display}/AITA")
        rates = aita_rates_per_condition(df, order)
        _draw_aita_panel(ax, rates, ci=True)
        ax.set_title(f"AITA - {display}")
    # One shared legend at the figure level (right side) instead of per-axes.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.0, 0.5),
               fontsize=10, title="condition")
    fig.suptitle("AITA - Parreto frontier is best")
    fig.tight_layout(rect=[0, 0, 0.86, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined_endorse(env, condition_order, families, out_path):
    """One endorsement-vs-unendorsement scatter panel per family for OEQ,
    side-by-side, shared axes. The {0,1}-only Pareto from paper §6.1.2 —
    same data as panel 1 of the per-family OEQ summary figure. The turquoise
    `human_reference` point is the per-post human ground truth; conditions
    that sit closer to it (in either x or y) are more human-aligned.
    """
    assert env == "OEQ", f"plot_combined_endorse: only OEQ is rendered here (got {env!r})"
    n = len(families)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 6.5), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (family, display) in zip(axes, families):
        csv = csv_for_family(family, env)
        assert os.path.exists(csv), f"missing {env} results csv: {csv}"
        df = load_oeq(csv)
        order = _filter_order(condition_order, df, f"{display}/{env}")
        counts = oeq_label_counts_per_condition(df, order)
        _draw_endorse_pareto_panel(ax, counts, env_label=f"{display}/{env}", ci=True)
        ax.set_title(f"{env} - {display}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.0, 0.5),
               fontsize=10, title="condition")
    fig.suptitle("OEQ - Closer to human reference (turquoise) is better")
    fig.tight_layout(rect=[0, 0, 0.86, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_oeq_human_diff_panel(ax, counts, env_label, ci=False):
    """Render a horizontal bar of endorsement_rate per non-human condition,
    with a dotted red vertical line marking the `human_reference` condition's
    rate (no bar drawn for the human reference itself). Each bar is labeled
    above it with the signed difference from the human reference (e.g.
    `+0.23` or `-0.05`).

    endorsement_rate = #(label==1) / (#(label==0) + #(label==1)) — same metric
    as `_draw_pas_bar_panel` / `_draw_endorse_pareto_panel` so this panel is
    directly comparable to the existing OEQ combined views. Asserts the
    `human_reference` condition is present and that every (method) condition
    has at least one {0,1} label.
    """
    assert "human_reference" in counts.index, (
        f"{env_label}: human_reference condition missing from counts"
    )
    human_row = counts.loc["human_reference"]
    human_denom = human_row[0] + human_row[1]
    assert human_denom > 0, (
        f"{env_label}: human_reference has zero label-0/1 rows; "
        "endorsement_rate is undefined"
    )
    human_rate = human_row[1] / human_denom

    method_conds = [c for c in counts.index if c != "human_reference"]
    rates = []
    ns = []
    for cond in method_conds:
        row = counts.loc[cond]
        denom = row[0] + row[1]
        assert denom > 0, (
            f"{env_label}: condition {cond!r} has zero label-0/1 rows; "
            "endorsement_rate is undefined"
        )
        rates.append(row[1] / denom)
        ns.append(denom)
    rates = np.array(rates)
    diffs = rates - human_rate

    y = np.arange(len(method_conds))
    for yi, cond, rate in zip(y, method_conds, rates):
        assert cond in CONDITION_COLORS, f"no color assigned for condition {cond!r}"
        ax.barh(yi, rate, color=CONDITION_COLORS[cond], edgecolor="black",
                linewidth=0.5, label=display_name(cond))

    # 95% CI (1.96·SEM, OSF-archive method) on each method's endorsement_rate (horizontal error
    # bar). The human-reference line itself is a point estimate over the same
    # {0,1} denominator; the per-method bars are what's being compared to it.
    upper = np.zeros(len(rates))
    if ci:
        xerr = paper_ci_halfwidths(rates, ns)
        ax.errorbar(rates, y, xerr=np.array(xerr), fmt="none", ecolor="black",
                    elinewidth=1.0, capsize=3, zorder=4)
        upper = xerr[1]

    ax.axvline(
        human_rate, color="red", linestyle=":", linewidth=2,
        label=f"human reference ({human_rate:.2f})",
    )

    # +x / -x label above each bar (slightly above the bar's vertical position,
    # past the right end of the bar / its error bar). Two decimal places so the
    # sign is always readable.
    for yi, rate, diff, u in zip(y, rates, diffs, upper):
        sign = "+" if diff >= 0 else "-"
        ax.text(
            rate + u + 0.01, yi - 0.35,
            f"{sign}{abs(diff):.2f}",
            va="bottom", ha="left", fontsize=8,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([display_name(c) for c in method_conds])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("endorsement rate")
    ax.grid(alpha=0.3, axis="x")


def plot_combined_oeq_human_diff(families, out_path):
    """One horizontal-bar panel per family: endorsement_rate per non-human
    condition, with the `human_reference` rate drawn as a dotted red vertical
    line and the signed diff from human reference labeled above each bar.
    """
    n = len(families)
    fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 6.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (family, display) in zip(axes, families):
        csv = csv_for_family(family, "OEQ")
        assert os.path.exists(csv), f"missing OEQ results csv: {csv}"
        df = load_oeq(csv)
        order = _filter_order(OEQ_CONDITION_ORDER, df, f"{display}/OEQ")
        counts = oeq_label_counts_per_condition(df, order)
        _draw_oeq_human_diff_panel(ax, counts, env_label=f"{display}/OEQ", ci=True)
        ax.set_title(f"OEQ - {display}")
    # Shared figure-level legend (includes the human-reference dotted-line
    # entry from each axes). Pull from the first axes — all axes share the
    # same set of conditions and the same red-line label format.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.0, 0.5),
               fontsize=10, title="condition")
    fig.suptitle("OEQ - endorsement rate per mitigation vs. human reference")
    fig.tight_layout(rect=[0, 0, 0.86, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined_pas(families, out_path):
    """One sycophancy-rate bar panel per family for PAS, side-by-side,
    shared y axis. Bar = endorsement_rate over the {0,1} label subset
    (paper §6.1.2's `action_endorsement_rate`). Lower is better.
    """
    n = len(families)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 6.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (family, display) in zip(axes, families):
        csv = csv_for_family(family, "PAS")
        assert os.path.exists(csv), f"missing PAS results csv: {csv}"
        df = load_oeq(csv)
        order = _filter_order(PAS_CONDITION_ORDER, df, f"{display}/PAS")
        counts = oeq_label_counts_per_condition(df, order)
        _draw_pas_bar_panel(ax, counts, env_label=f"{display}/PAS", ci=True)
        ax.set_title(f"PAS - {display}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.0, 0.5),
               fontsize=10, title="condition")
    fig.suptitle("PAS - lower is better")
    fig.tight_layout(rect=[0, 0, 0.86, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined_oeq_2(families, out_path):
    """3 (metrics) x N (families) grid of OEQ_2 mismatch-rate bar charts.

    Rows: validation / indirectness / framing (in `OEQ_2_METRICS` order).
    Cols: one per family in `families`.
    All cells share the y-axis (mismatch rate in [0,1]) and each row also
    shares the x-axis (canonical condition order). Family display name is
    the column title; metric is the y-label of the leftmost cell in each row.
    """
    metrics_list = list(OEQ_2_METRICS)
    n_rows = len(metrics_list)
    n_cols = len(families)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(8, 0.9 * len(OEQ_2_CONDITION_ORDER)) * n_cols / 1.5,
                 3.2 * n_rows),
        sharex="col", sharey="row",
    )
    # Normalize axes shape so axes[r][c] always works.
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    # Pre-compute rates + condition order per family so we can render every
    # cell from one place.
    per_family = []
    for family, display in families:
        csv = csv_for_family(family, "OEQ_2")
        assert os.path.exists(csv), f"missing OEQ_2 results csv: {csv}"
        df = load_oeq_2(csv)
        order = _filter_order(OEQ_2_CONDITION_ORDER, df, f"{display}/OEQ_2")
        rates = oeq_2_rates_per_condition(df, order)
        per_family.append((family, display, rates))

    for col_idx, (family, display, rates) in enumerate(per_family):
        conds = rates["condition"].tolist()
        x = np.arange(len(conds))
        colors = [CONDITION_COLORS.get(c, "#888888") for c in conds]
        for row_idx, metric in enumerate(metrics_list):
            ax = axes[row_idx][col_idx]
            heights = rates[f"{metric}_mismatch_rate"].tolist()
            ax.bar(x, heights, color=colors)
            # 95% CI (1.96·SEM, OSF-archive method) on the judge-vs-human
            # mismatch rate; denominator = per-condition count of parsed (0/1)
            # labels.
            yerr = paper_ci_halfwidths(heights, rates[f"{metric}_n_classified"])
            ax.errorbar(x, heights, yerr=np.array(yerr), fmt="none",
                        ecolor="black", elinewidth=0.8, capsize=2, zorder=4)
            for xi, v, u in zip(x, heights, yerr[1]):
                ax.text(xi, v + u + 0.01, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7)
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3, axis="y")
            if row_idx == 0:
                ax.set_title(display)
            if col_idx == 0:
                ax.set_ylabel(f"{metric}\nmismatch rate")
            if row_idx == n_rows - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(
                    [display_name(c) for c in conds],
                    rotation=30, ha="right", fontsize=8,
                )

    fig.suptitle(
        "OEQ_2 — fraction of responses whose LLM-judge label disagrees "
        "with the human label, per axis per mitigation per model"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_aita_nta_flip_scatter_panel(ax, rates, ci=False):
    """Render a critical_score_rate (x) vs sycophancy_score_rate (y) scatter,
    one point per condition. Both rates come from the same flipped-pair
    accounting in `aita_nta_flip_rates_per_condition`:
      sycophancy_score_rate = #(both NTA) / N      (always sides with user)
      critical_score_rate   = #(both YTA) / N      (always sides against user)
    Lower-right = critical without being sycophantic. Upper-left = sycophantic
    without being critical.
    """
    xs = rates["critical_score_rate"].tolist()
    ys = rates["sycophancy_score_rate"].tolist()
    conds = rates["condition"].tolist()
    # Both rates share the same per-condition denominator N (= n_total).
    xerr = paper_ci_halfwidths(xs, rates["n_total"]) if ci else None
    yerr = paper_ci_halfwidths(ys, rates["n_total"]) if ci else None
    _scatter_conditions(ax, xs, ys, conds, xerr=xerr, yerr=yerr)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("critical rate  (#(both YTA) / N)")
    ax.set_ylabel("sycophancy rate  (#(both NTA) / N)")
    ax.grid(alpha=0.3)


def plot_combined_aita_nta_flip_scatter(families, out_path):
    """One sycophancy-vs-critical scatter panel per family, side-by-side,
    shared axes. Same per-condition rates as `plot_combined_aita_nta_flip`,
    rendered as a 2D scatter so the tradeoff between the two failure modes is
    visible per mitigation.
    """
    n = len(families)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 6.5), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (family, display) in zip(axes, families):
        csv = csv_for_family(family, "AITA-NTA-FLIP")
        assert os.path.exists(csv), f"missing AITA-NTA-FLIP results csv: {csv}"
        df = load_aita_nta_flip(csv)
        order = _filter_order(AITA_NTA_FLIP_CONDITION_ORDER, df, f"{display}/AITA-NTA-FLIP")
        rates = aita_nta_flip_rates_per_condition(df, order)
        _draw_aita_nta_flip_scatter_panel(ax, rates, ci=True)
        ax.set_title(f"AITA-NTA-FLIPPED - {display}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.0, 0.5),
               fontsize=10, title="condition")
    fig.suptitle("AITA-NTA-FLIPPED - sycophancy vs critical per mitigation "
                 "(lower-right is critical-without-sycophancy)")
    fig.tight_layout(rect=[0, 0, 0.86, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined_aita_nta_flip(families, out_path):
    """One sycophancy-rate bar panel per family, side-by-side, shared y axis.
    Bar = sycophancy_score_rate = #(both-NTA on the POV-flipped pair) / N.
    Lower is better.
    """
    n = len(families)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 6.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (family, display) in zip(axes, families):
        csv = csv_for_family(family, "AITA-NTA-FLIP")
        assert os.path.exists(csv), f"missing AITA-NTA-FLIP results csv: {csv}"
        df = load_aita_nta_flip(csv)
        order = _filter_order(AITA_NTA_FLIP_CONDITION_ORDER, df, f"{display}/AITA-NTA-FLIP")
        rates = aita_nta_flip_rates_per_condition(df, order)
        _draw_aita_nta_flip_panel(ax, rates, ci=True)
        ax.set_title(f"AITA-NTA-FLIPPED - {display}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.0, 0.5),
               fontsize=10, title="condition")
    fig.suptitle("AITA-NTA-FLIPPED - Lower is better")
    fig.tight_layout(rect=[0, 0, 0.86, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _run_combined(envs):
    """Write the cross-model combined figures for each env in `envs`."""
    os.makedirs(COMBINED_OUT_DIR, exist_ok=True)
    fams = [f for f, _ in COMBINED_FAMILIES]
    print(f"combined: families={fams}, envs={envs} → {COMBINED_OUT_DIR}/")
    for env in envs:
        out = os.path.join(COMBINED_OUT_DIR, f"{env.lower()}_combined.png")
        if env == "AITA":
            plot_combined_aita(COMBINED_FAMILIES, out)
        elif env == "AITA-NTA-FLIP":
            plot_combined_aita_nta_flip(COMBINED_FAMILIES, out)
        elif env == "OEQ_2":
            plot_combined_oeq_2(COMBINED_FAMILIES, out)
        elif env == "PAS":
            plot_combined_pas(COMBINED_FAMILIES, out)
        else:
            plot_combined_endorse(env, OEQ_CONDITION_ORDER, COMBINED_FAMILIES, out)
        print(f"wrote {out}")
        if env == "OEQ":
            # Companion to the {0,1} Pareto: per-family horizontal-bar view
            # of endorsement_rate per method, with the human_reference rate
            # as a dotted red vertical line and signed diffs labeled per bar.
            diff_out = os.path.join(COMBINED_OUT_DIR, "oeq_human_diff_combined.png")
            plot_combined_oeq_human_diff(COMBINED_FAMILIES, diff_out)
            print(f"wrote {diff_out}")
        if env == "AITA-NTA-FLIP":
            # Companion to the sycophancy-rate bar chart: 2D scatter of
            # critical_score_rate (x) vs sycophancy_score_rate (y) so the
            # tradeoff between the two flipped-pair failure modes is visible.
            scatter_out = os.path.join(
                COMBINED_OUT_DIR, "aita-nta-flip_scatter_combined.png",
            )
            plot_combined_aita_nta_flip_scatter(COMBINED_FAMILIES, scatter_out)
            print(f"wrote {scatter_out}")


def main():
    """Build per-env figures from the most recent CSV outputs.

    Assumes the script is run from the repo root (the same cwd
    `explore_mitigations.py` writes its outputs from). Asserts the input
    files exist rather than silently producing empty plots.

    `--family` selects which run's CSVs to read (and where to write plots).
    It is the `<gen-slug>_<judge-slug>` segment that
    `explore_mitigations.py` writes into its output filenames — e.g.
    `claude-sonnet-4-6_gpt-5.4`, `deepseek-v3.2_gpt-5.4`. Defaults pin the
    judge slug to `gpt-5.4` for cross-model consistency.

    `--envs` controls which envs to plot. Default `AITA OEQ` preserves the
    historical behavior; pass `PAS` (or `AITA OEQ PAS`) to plot PAS.
    """
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", default=DEFAULT_FAMILY,
                   help=f"provider family whose results to plot (default: "
                        f"{DEFAULT_FAMILY}). The `<gen-slug>_<judge-slug>` "
                        f"segment from `explore_mitigations.py`'s output "
                        f"filenames; must match the `--model` / "
                        f"`--judge_model` pair that produced the CSVs. "
                        f"Reads `_<family>`-suffixed inputs and writes to "
                        f"`plots_<family>/`.")
    p.add_argument("--envs", nargs="+", default=["AITA", "OEQ"],
                   choices=["AITA", "OEQ", "PAS", "AITA-NTA-FLIP", "OEQ_2"],
                   help="envs to plot (default: AITA OEQ). Pass PAS, "
                        "AITA-NTA-FLIP, OEQ_2 (or any combination) to "
                        "include those figures.")
    p.add_argument("--combined", action="store_true",
                   help="instead of per-family figures, write one cross-model "
                        "figure per env into `plots_combined/`. Each figure "
                        "is a row of subplots, one per family in "
                        "COMBINED_FAMILIES. AITA: the FPR-vs-FNR panel. "
                        "OEQ / PAS: only the {0,1} endorsement-vs-unendorsement "
                        "Pareto (panel 1 of the per-family summary). "
                        "`--family` is ignored in this mode.")
    args = p.parse_args()

    if args.combined:
        _run_combined(args.envs)
        return

    out_dir = out_dir_for_family(args.family)
    os.makedirs(out_dir, exist_ok=True)
    print(f"plotting family={args.family}, envs={args.envs} → {out_dir}/")

    plotters = {
        "AITA": _plot_aita_for_family,
        "OEQ":  _plot_oeq_for_family,
        "PAS":  _plot_pas_for_family,
        "AITA-NTA-FLIP": _plot_aita_nta_flip_for_family,
        "OEQ_2": _plot_oeq_2_for_family,
    }
    for env in args.envs:
        plotters[env](args.family, out_dir)


if __name__ == "__main__":
    main()
