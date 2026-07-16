"""Assemble AITA (FNR/FPR) and AITA-NTA-FLIP held-out results tables over the
same method set used for the OEQ/PAS tables, for every method that has eval data.

For each (method, env) it globs the per-shard aita_scored_*.csv files, asserts
the shard set 0..N-1 is complete and ids are unique, concatenates (which exactly
reproduces the single-job set), and computes the env's metric with the repo's
own functions:
  - AITA          -> compute_fnr_fpr:  FNR=fn/#YTA-truth, FPR=fp/#NTA-truth
  - AITA-NTA-FLIP -> compute_aita_nta_flip_metrics: both-NTA sycophancy rate and
                     both-YTA critical rate (FLIP has no per-item truth label, so
                     FNR/FPR is undefined there).
Every reported rate gets a 95% Wilson interval over its own denominator.

FNR / sycophancy_score_rate are the sycophancy axes (higher = more sycophantic).

Run:
    python training_mitigations/eval/build_aita_flip_tables.py
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
from explore_mitigations import compute_fnr_fpr, compute_aita_nta_flip_metrics  # noqa

RUNS = os.path.join(_TM, "runs")

# method display label -> (run subdir, adapter_tag, decode)
AITA_MAP = [
    ("Qwen3-8B (base)",        ("baseline_AITA", "base", "temp1.0")),
    ("+ Cheng et al. prompt",  ("baseline_AITA", "base-explicit", "temp1.0")),
    ("+ Sharma et al. prompt", ("baseline_AITA", "base-sharma", "temp1.0")),
    ("+ “Don’t be sycophantic”", ("baseline_AITA", "base-dontbesyco", "temp1.0")),
    ("+ “Be critical”",        ("baseline_AITA", "base-becritical", "temp1.0")),
    ("+ perspective-shift",    ("baseline_AITA", "base-perspshift", "temp1.0")),
    ("SFT “be critical”",      ("sft_lora_aita_qwen3-8b_n1000_becritical", "final", "temp1.0")),
    ("mixed_P5",               ("dpo_lora_aita_qwen3-8b_n1000_mixed_P5", "final", "final1000_temp1.0")),
    ("mixed_P5_nocrit_distill_gp_frozenstake",
        ("dpo_lora_aita_qwen3-8b_n1000_mixed_P5_nocrit_distill_gp_frozenstake", "final", "final1000_temp1.0")),
    ("mixed_P6_perspshift_distill_gp",
        ("dpo_lora_aita_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp", "final", "final1000_temp1.0")),
]
FLIP_MAP = [
    ("Qwen3-8B (base)",        ("baseline_AITA-NTA-FLIP", "base", "temp1.0")),
    ("+ Cheng et al. prompt",  ("baseline_AITA-NTA-FLIP", "base-explicit", "temp1.0")),
    ("+ Sharma et al. prompt", ("baseline_AITA-NTA-FLIP", "base-sharma", "temp1.0")),
    ("+ “Don’t be sycophantic”", ("baseline_AITA-NTA-FLIP", "base-dontbesyco", "temp1.0")),
    ("+ “Be critical”",        ("baseline_AITA-NTA-FLIP", "base-becritical", "temp1.0")),
    ("+ perspective-shift",    ("baseline_AITA-NTA-FLIP", "base-perspshift", "temp1.0")),
    ("SFT “be critical”",      ("sft_lora_flip_qwen3-8b_n500_becritical", "final", "temp1.0")),
    ("mixed_P5",               ("dpo_lora_flip_qwen3-8b_n500_mixed_P5", "final", "final1000_temp1.0")),
    ("mixed_P5_nocrit_distill_gp_frozenstake",
        ("dpo_lora_flip_qwen3-8b_n500_mixed_P5_nocrit_distill_gp_frozenstake", "final", "final1000_temp1.0")),
    ("mixed_P6_perspshift_distill_gp",
        ("dpo_lora_flip_qwen3-8b_n500_mixed_P6_perspshift_distill_gp", "final", "final1000_temp1.0")),
    # DIFFERENT EVAL SET from every row above: this arm is the frozen final run,
    # held out on final_training_eval_data/AITA-NTA-FLIP_test (decode tag
    # `finaldata`), whereas the rows above are the tuning-era `final1000`
    # held-out. The two splits overlap but are not identical, so this row is NOT
    # a like-for-like comparison against them — hence the explicit label. It is
    # comparable only to other `finaldata` rows (see marlowe_final_classify_all.sh).
    ("mixed_P6_perspshift_distill_gp_rpo0p5 [FINAL DATA — different split]",
        ("dpo_lora_flip_qwen3-8b_n500_mixed_P6_perspshift_distill_gp_rpo0p5_final",
         "final", "finaldata_temp1.0")),
]
OUR = {"mixed_P5", "mixed_P5_nocrit_distill_gp_frozenstake",
       "mixed_P6_perspshift_distill_gp"}


def wilson(count, n, z=1.96):
    """(p, lo, hi) 95% Wilson score interval; matches summarize_sycoharm_fractions."""
    p = count / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, center - half, center + half


def load_concat(run_subdir, tag, decode):
    """Glob the complete shard set for one (run,tag,decode), assert it tiles
    0..N-1 with unique ids, and return the concatenated frame."""
    d = os.path.join(RUNS, run_subdir, "eval")
    pat = os.path.join(d, f"aita_scored_{tag}_{decode}_seed0_shard*of*.csv")
    files = glob.glob(pat)
    assert files, f"no shards match {pat}"
    ns = set()
    idx = {}
    for f in files:
        m = re.search(r"_shard(\d+)of(\d+)\.csv$", f)
        i, N = int(m.group(1)), int(m.group(2))
        ns.add(N)
        idx[i] = f
    assert len(ns) == 1, f"inconsistent shard counts {ns} for {pat}"
    N = ns.pop()
    assert set(idx) == set(range(N)), \
        f"incomplete shard set for {pat}: have {sorted(idx)} of {N}"
    df = pd.concat([pd.read_csv(idx[i]) for i in range(N)], ignore_index=True)
    assert df["id"].is_unique, f"overlapping ids across shards for {pat}"
    return df


def fmt(p, lo, hi):
    return f"{p:.3f} [{max(0.0,lo):.3f}, {min(1.0,hi):.3f}]"


def build_aita():
    rows = []
    for label, key in AITA_MAP:
        df = load_concat(*key)
        m = compute_fnr_fpr(df)
        rows.append({
            "method": label,
            "N": m["n_total"],
            "n_classified": m["n_classified"],
            "dropped": m["n_dropped_other"],
            "FNR (sycophancy)": fmt(*wilson(m["fn"], m["n_yta_truth"])),
            "n(YTA-truth)": m["n_yta_truth"],
            "FPR": fmt(*wilson(m["fp"], m["n_nta_truth"])),
            "n(NTA-truth)": m["n_nta_truth"],
        })
    return pd.DataFrame(rows)


def build_flip():
    rows = []
    for label, key in FLIP_MAP:
        df = load_concat(*key)
        m = compute_aita_nta_flip_metrics(df)
        n = m["n_total"]
        rows.append({
            "method": label,
            "N (pairs)": n,
            "dropped": m["n_dropped_other"],
            "sycophancy (both-NTA)": fmt(*wilson(m["n_sycophancy"], n)),
            "critical (both-YTA)": fmt(*wilson(m["n_critical"], n)),
        })
    return pd.DataFrame(rows)


def md(df, title):
    cols = list(df.columns)
    out = [f"### {title}", "| " + " | ".join(cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def main():
    aita = build_aita()
    flip = build_flip()
    aita.to_csv(os.path.join(_TM, "sycoharm_runs", "results_table_aita.csv"), index=False)
    flip.to_csv(os.path.join(_TM, "sycoharm_runs", "results_table_flip.csv"), index=False)
    print(md(aita, "AITA — 1000-prompt held-out (FNR/FPR; higher FNR = more sycophantic)"))
    print()
    print(md(flip, "AITA-NTA-FLIP — 1000-pair held-out (both-NTA sycophancy / both-YTA critical)"))


if __name__ == "__main__":
    main()
