"""Assemble the FINAL-DATASET (`finaldata` tag) held-out results tables for
OEQ, PAS, AITA and AITA-NTA-FLIP over the frozen 1000-item held-out sets.

Scope: this builder covers ONLY the `finaldata` decode tag — the frozen
final-paper run set. It deliberately does not read any `final1000` (tuning-era)
file, because those are a DIFFERENT held-out split and are not comparable
row-for-row. See build_results_tables.py / build_aita_flip_tables.py for the
tuning-era tables.

Method -> file maps are transcribed from the two drivers that actually produced
the data, so the tables cannot silently drift from the runs:
  - OEQ/PAS : slurm_scripts/marlowe_final_endorse_all.sh   (scored_<name>.csv)
  - AITA/FLIP: slurm_scripts/marlowe_final_classify_all.sh  (run, tag, NSHARDS)

Metrics (all computed with the repo's own helpers, never reimplemented here):
  - OEQ/PAS : action endorsement rate = #1/(#0+#1) via action_endorsement_rate.
              This is the paper's headline metric. Its denominator is the {0,1}
              subset, which the JUDGE selects -- so n(0/1) varies by arm and is
              reported per row.
  - AITA    : compute_fnr_fpr. FNR = NTA-pred / YTA-truth (the sycophancy axis,
              higher = more sycophantic); FPR = YTA-pred / NTA-truth.
  - FLIP    : compute_aita_nta_flip_metrics. both-NTA = sycophancy (always-nice),
              both-YTA = critical (always-harsh), over a stable N-pair denominator.

CI conventions:
  - endorsement rate: mean +/- 1.96*SEM over the {0,1} vector (the paper helper's
    own convention, so the number matches the notebook).
  - every other rate is a binary proportion -> 95% Wilson score interval.

NSHARDS is per-condition, not per-env: the FLIP P6 arm generated at 5 shards on
`next` while the FLIP baselines generated at 2 on Marlowe. load_concat asserts
the shard set tiles 0..N-1 exactly once with unique ids, so a partial or mixed
shard regime fails loudly instead of silently changing a denominator.

Run:
    python training_mitigations/eval/build_finaldata_tables.py
"""

import glob
import math
import os
import re
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_TM = os.path.dirname(_HERE)
_REPO = os.path.dirname(_TM)
sys.path.insert(0, os.path.join(_REPO, "mitigations"))
sys.path.insert(0, os.path.join(_REPO, "osfstorage-archive", "Code"))

from explore_mitigations import compute_fnr_fpr, compute_aita_nta_flip_metrics  # noqa: E402
from evaluate_action_endorsement_stephane import (  # noqa: E402
    first_label, action_endorsement_rate)

RUNS_ROOT = "/projects/m000254/runs"
V5_DIR = os.path.join(_TM, "sycoharm_runs", "endorse_gpt5mini_v5_temp0")
OUT_DIR = os.path.join(_TM, "sycoharm_runs")

# ---------------------------------------------------------------------------
# Method maps. Order is the order requested for the paper tables.
# ---------------------------------------------------------------------------

# label -> scored_<name>.csv basename, from marlowe_final_endorse_all.sh
OEQ_METHODS = [
    ("Qwen3-8B (base)",              "scored_oeq_base_finaldata.csv"),
    ("+ Cheng et al. prompt",        "scored_oeq_explicit_finaldata.csv"),
    ("+ Sharma et al. prompt",       "scored_oeq_sharma_finaldata.csv"),
    ("+ “Don’t be sycophantic”", "scored_oeq_dontbesyco_finaldata.csv"),
    ("+ “Be critical”",     "scored_oeq_becritical_finaldata.csv"),
    ("+ perspective-shift",          "scored_oeq_perspshiftassess_finaldata.csv"),
    ("+ stakeholder-objective prompt", "scored_oeq_stakeholder_finaldata.csv"),
    ("SFT “be critical”",   "scored_oeq_sft_finaldata.csv"),
    ("Neutral-mixed DPO",            "scored_oeq_neutralmixed_finaldata.csv"),
    ("Neutral DPO",                  "scored_oeq_neutral_finaldata.csv"),
    ("P6_persp, α0.5",          "scored_oeq_P6perspshift_a05_finaldata.csv"),
    ("P6_persp + voice-DPO",         "scored_oeq_voicedpo_finaldata.csv"),
    ("P6_persp + voice-DPO (b0.5)",  "scored_oeq_voicedpo_b05_finaldata.csv"),
    ("Human baseline",               "scored_oeq_human_finaldata.csv"),
    # POSITIVE CONTROL, not a mitigation -- ordered last so it never reads as one.
    ("+ be-sycophantic (control)",   "scored_oeq_besycophantic_finaldata.csv"),
]
PAS_METHODS = [
    ("Qwen3-8B (base)",              "scored_pas_base_finaldata.csv"),
    ("+ Cheng et al. prompt",        "scored_pas_explicit_finaldata.csv"),
    ("+ Sharma et al. prompt",       "scored_pas_sharma_finaldata.csv"),
    ("+ “Don’t be sycophantic”", "scored_pas_dontbesyco_finaldata.csv"),
    ("+ “Be critical”",     "scored_pas_becritical_finaldata.csv"),
    ("+ perspective-shift",          "scored_pas_perspshiftassess_finaldata.csv"),
    ("+ stakeholder-objective prompt", "scored_pas_stakeholder_finaldata.csv"),
    ("SFT “be critical”",   "scored_pas_sft_finaldata.csv"),
    ("Neutral-mixed DPO",            "scored_pas_neutralmixed_finaldata.csv"),
    ("Neutral DPO",                  "scored_pas_neutral_finaldata.csv"),
    ("P6_persp, α0.5",          "scored_pas_P6perspshift_a05_finaldata.csv"),
    ("P6_persp + voice-DPO",         "scored_pas_voicedpo_finaldata.csv"),
    ("P6_persp + voice-DPO (b0.5)",  "scored_pas_voicedpo_b05_finaldata.csv"),
    # POSITIVE CONTROL, not a mitigation -- ordered last so it never reads as one.
    ("+ be-sycophantic (control)",   "scored_pas_besycophantic_finaldata.csv"),
]

# label -> (run subdir, tag, nshards), from marlowe_final_classify_all.sh
_AITA_TAG = "{}_finaldata_temp1.0_seed0"
AITA_METHODS = [
    ("Qwen3-8B (base)",              ("baseline_final_AITA", _AITA_TAG.format("base"), 1)),
    ("+ Cheng et al. prompt",        ("baseline_final_AITA", _AITA_TAG.format("base-explicit"), 1)),
    ("+ Sharma et al. prompt",       ("baseline_final_AITA", _AITA_TAG.format("base-sharma"), 1)),
    ("+ “Don’t be sycophantic”",
        ("baseline_final_AITA", _AITA_TAG.format("base-dontbesyco"), 1)),
    ("+ “Be critical”",    ("baseline_final_AITA", _AITA_TAG.format("base-becritical"), 1)),
    ("+ perspective-shift",          ("baseline_final_AITA", _AITA_TAG.format("base-perspshiftassess"), 1)),
    ("+ stakeholder-objective prompt",
        ("baseline_final_AITA", _AITA_TAG.format("base-stakeholder"), 1)),
    ("P6_persp, α0.5",
        ("dpo_lora_aita_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final",
         _AITA_TAG.format("final"), 1)),
    # POSITIVE CONTROL, not a mitigation -- ordered last so it never reads as one.
    # Only AITA, not FLIP: the control was never generated on the FLIP env, so
    # adding it to FLIP_METHODS would fail load_concat's shard assert.
    ("+ be-sycophantic (control)",
        ("baseline_final_AITA", _AITA_TAG.format("base-besycophantic"), 1)),
]
FLIP_METHODS = [
    ("Qwen3-8B (base)",              ("baseline_final_AITA-NTA-FLIP", _AITA_TAG.format("base"), 2)),
    ("+ Cheng et al. prompt",        ("baseline_final_AITA-NTA-FLIP", _AITA_TAG.format("base-explicit"), 2)),
    ("+ Sharma et al. prompt",       ("baseline_final_AITA-NTA-FLIP", _AITA_TAG.format("base-sharma"), 2)),
    ("+ “Don’t be sycophantic”",
        ("baseline_final_AITA-NTA-FLIP", _AITA_TAG.format("base-dontbesyco"), 2)),
    ("+ “Be critical”",    ("baseline_final_AITA-NTA-FLIP", _AITA_TAG.format("base-becritical"), 2)),
    ("+ perspective-shift",          ("baseline_final_AITA-NTA-FLIP", _AITA_TAG.format("base-perspshiftassess"), 2)),
    ("+ stakeholder-objective prompt",
        ("baseline_final_AITA-NTA-FLIP", _AITA_TAG.format("base-stakeholder"), 2)),
    # generated on `next` at 5 shards (a4000 pairs, 2000 gens @1536 tok), not 2.
    ("P6_persp, α0.5",
        ("dpo_lora_flip_qwen3-8b_n500_mixed_P6_perspshift_distill_gp_rpo0p5_final",
         _AITA_TAG.format("final"), 5)),
]


def wilson_ci(count, n, z=1.96):
    """95% Wilson score interval (p, lo, hi) for count/n. Matches the repo's
    wilson_ci / summarize_sycoharm_fractions convention."""
    assert n > 0, "empty denominator"
    p = count / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, center - half, center + half


def fmt(p, lo, hi):
    """Render a rate and its CI, clipped to [0,1] (a Wilson bound never leaves
    [0,1] by construction; the clip only guards the SEM-based endorse CI)."""
    return f"{p:.3f} [{max(0.0, lo):.3f}, {min(1.0, hi):.3f}]"


def endorse_labels(basename):
    """Integer endorse labels for one scored OEQ/PAS condition.

    Assumes the scored file exists and every endorse_response cell carries a
    parsable 0-3 label; both are asserted so a partial judge run cannot quietly
    shrink a denominator. Returns a list of ints.
    """
    p = os.path.join(V5_DIR, basename)
    assert os.path.exists(p), f"missing endorse frame: {p}"
    col = pd.read_csv(p)["endorse_response"]
    parsed = col.map(first_label)
    assert parsed.notna().all(), f"{p}: {int(parsed.isna().sum())} unparsable labels"
    return [int(x) for x in parsed]


def load_concat(run_subdir, tag, nshards):
    """Concatenate the complete scored shard set for one AITA/FLIP condition.

    Assumes the shard files tile 0..nshards-1 exactly once and carry unique ids;
    both are asserted, as is the on-disk shard count matching `nshards` (the
    FLIP P6 arm is *of5 while its baselines are *of2, so a per-env constant would
    be wrong). Returns the concatenated frame.
    """
    d = os.path.join(RUNS_ROOT, run_subdir, "eval")
    pat = os.path.join(d, f"aita_scored_{tag}_shard*of*.csv")
    files = glob.glob(pat)
    assert files, f"no scored shards match {pat}"
    ns = set()
    idx = {}
    for f in files:
        m = re.search(r"_shard(\d+)of(\d+)\.csv$", f)
        i, n = int(m.group(1)), int(m.group(2))
        ns.add(n)
        idx[i] = f
    assert len(ns) == 1, f"inconsistent shard counts {ns} for {pat}"
    n = ns.pop()
    assert n == nshards, f"{pat}: on-disk shard count {n} != expected {nshards}"
    assert set(idx) == set(range(n)), \
        f"incomplete shard set for {pat}: have {sorted(idx)} of {n}"
    df = pd.concat([pd.read_csv(idx[i]) for i in range(n)], ignore_index=True)
    assert df["id"].is_unique, f"overlapping ids across shards for {pat}"
    return df


def build_endorse(methods):
    """Endorsement-rate table for OEQ or PAS: one row per method with the paper
    metric, its 95% CI, the judge-selected {0,1} denominator, and the neutral
    (label 3) share that the metric discards."""
    rows = []
    for label, basename in methods:
        labs = endorse_labels(basename)
        rate, ci, n01 = action_endorsement_rate(labs)
        n3 = sum(1 for x in labs if x == 3)
        n2 = sum(1 for x in labs if x == 2)
        rows.append({
            "method": label,
            "endorsement rate [95% CI]": f"{rate:.3f} [{max(0.0, rate-ci):.3f}, {min(1.0, rate+ci):.3f}]",
            "n (0/1 subset)": n01,
            "N (generated)": len(labs),
            "n neutral (3)": n3,
            "n implicit (2)": n2,
        })
    return pd.DataFrame(rows)


def build_aita():
    """AITA FNR/FPR table: FNR is the sycophancy axis, each with its own
    truth-class denominator and a Wilson CI."""
    rows = []
    for label, key in AITA_METHODS:
        df = load_concat(*key)
        m = compute_fnr_fpr(df)
        rows.append({
            "method": label,
            "FNR (sycophancy) [95% CI]": fmt(*wilson_ci(m["fn"], m["n_yta_truth"])),
            "n (YTA-truth)": m["n_yta_truth"],
            "FPR [95% CI]": fmt(*wilson_ci(m["fp"], m["n_nta_truth"])),
            "n (NTA-truth)": m["n_nta_truth"],
            "N (generated)": m["n_total"],
            "n classified": m["n_classified"],
            "dropped (UNCLEAR/OTHER)": m["n_dropped_other"],
        })
    return pd.DataFrame(rows)


def build_flip():
    """AITA-NTA-FLIP table: both-NTA (sycophancy) and both-YTA (critical) pair
    fractions over the stable N-pair denominator, each with a Wilson CI."""
    rows = []
    for label, key in FLIP_METHODS:
        df = load_concat(*key)
        m = compute_aita_nta_flip_metrics(df)
        n = m["n_total"]
        rows.append({
            "method": label,
            "both-NTA (sycophancy) [95% CI]": fmt(*wilson_ci(m["n_sycophancy"], n)),
            "both-YTA (critical) [95% CI]": fmt(*wilson_ci(m["n_critical"], n)),
            "N (pairs)": n,
            "n classified": m["n_classified"],
            "dropped (UNCLEAR/OTHER)": m["n_dropped_other"],
        })
    return pd.DataFrame(rows)


def md(df, title):
    """Render a frame as a GitHub markdown table under an h3 title."""
    cols = list(df.columns)
    out = [f"### {title}", "| " + " | ".join(cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def main():
    specs = [
        ("oeq", build_endorse(OEQ_METHODS),
         "OEQ — final-dataset 1000-prompt held-out (action endorsement rate; higher = more sycophantic)"),
        ("pas", build_endorse(PAS_METHODS),
         "PAS — final-dataset 1000-prompt held-out (action endorsement rate; higher = more sycophantic)"),
        ("aita", build_aita(),
         "AITA — final-dataset 1000-prompt held-out (FNR = sycophancy axis)"),
        ("flip", build_flip(),
         "AITA-NTA-FLIP — final-dataset 1000-pair held-out (both-NTA = sycophancy)"),
    ]
    for name, df, title in specs:
        out = os.path.join(OUT_DIR, f"results_finaldata_{name}.csv")
        df.to_csv(out, index=False)
        print("\n" + md(df, title))
        print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
