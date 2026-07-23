"""Paper tables for the finaldata held-out sets, AFTER the assistant-mode gate.

Four tables: OEQ, PAS (endorsement + neutral), AITA (FNR/FPR), AITA-NTA-FLIP
(both-NTA / both-YTA). Every rate carries a 95% Wilson score interval and the
denominator that produced it.

WHAT "AFTER THE GATE" MEANS
--------------------------
Each response was labelled by the assistant-mode gate judge at attempt 1
(`gate_<tag>_a1.csv`). Keys labelled 0 were resampled up to 5 times by
gate_loop.py; the first passing draw replaces the attempt-1 response and was
judged fresh (endorse: `endorse_gpt5mini_v5_temp0_gateloop/`; AITA/FLIP:
`aita_scored_<tag>_gateloop.csv`). Keys that never passed are DROPPED from every
rate and reported in `n_filtered`.

So each row's scored set is:
    attempt-1 passers (frozen labels)  +  rescued keys (gate-loop labels)
and `n_filtered` is the count that no draw rescued.

THE DENOMINATORS ARE NOT THE SAME ACROSS COLUMNS OR ROWS
--------------------------------------------------------
- endorsement rate = #1/(#0+#1), the paper's metric: its denominator EXCLUDES
  labels 2 and 3, and is therefore model-selected -- an arm that answers
  neutrally more often has a smaller n. `n_endorse` is that denominator.
- neutral rate = #3/(#0+#1+#2+#3), over every gated response. `n_scored`.
- Arms differ in `n_filtered`, so rows are measured on differently-selected
  subsets of the same 1000 prompts. That is the point of showing the column, and
  rows with a large `n_filtered` are not cleanly comparable to rows without one.

The OEQ human baseline was never gated (it is not a model arm); its gate columns
are reported as n/a and its rates are over the full frozen set.

Run:  python training_mitigations/eval/build_gated_finaldata_tables.py
"""

import glob
import math
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_TM = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_TM), "mitigations"))

from explore_mitigations import (  # noqa: E402
    compute_fnr_fpr, compute_aita_nta_flip_metrics)

RUNS = "/projects/m000254/runs"
V5 = os.path.join(_TM, "sycoharm_runs", "endorse_gpt5mini_v5_temp0")
V5G = V5 + "_gateloop"
TAG = "{}_finaldata_temp1.0_seed0"

# (display, frozen scored basename, run subdir, adapter_tag, n_shards)
# adapter_tag None -> arm was never gated (human baseline).
OEQ = [
    ("Qwen3-8B (base)", "scored_oeq_base_finaldata.csv", "baseline_final_OEQ", "base", 2),
    ("+ Cheng et al. prompt", "scored_oeq_explicit_finaldata.csv", "baseline_final_OEQ", "base-explicit", 2),
    ("+ Sharma et al. prompt", "scored_oeq_sharma_finaldata.csv", "baseline_final_OEQ", "base-sharma", 2),
    ("+ “Don’t be sycophantic”", "scored_oeq_dontbesyco_finaldata.csv", "baseline_final_OEQ", "base-dontbesyco", 2),
    ("+ “Be critical”", "scored_oeq_becritical_finaldata.csv", "baseline_final_OEQ", "base-becritical", 2),
    ("+ perspective-shift", "scored_oeq_perspshiftassess_finaldata.csv", "baseline_final_OEQ", "base-perspshiftassess", 2),
    ("SFT “be critical”", "scored_oeq_sft_finaldata.csv", "sft_lora_oeq_qwen3-8b_becritical_final", "final", 2),
    ("Neutral-mixed DPO", "scored_oeq_neutralmixed_finaldata.csv", "dpo_lora_oeq_qwen3-8b_n1000_neutral_mixed_final", "final", 2),
    ("Neutral DPO", "scored_oeq_neutral_finaldata.csv", "dpo_lora_oeq_qwen3-8b_n1000_neutral_final", "final", 2),
    ("P6_persp, α0.5", "scored_oeq_P6perspshift_a05_finaldata.csv", "dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final", "final", 2),
    ("Human baseline", "scored_oeq_human_finaldata.csv", None, None, None),
]
PAS = [
    ("Qwen3-8B (base)", "scored_pas_base_finaldata.csv", "baseline_final_PAS", "base", 2),
    ("+ Cheng et al. prompt", "scored_pas_explicit_finaldata.csv", "baseline_final_PAS", "base-explicit", 2),
    ("+ Sharma et al. prompt", "scored_pas_sharma_finaldata.csv", "baseline_final_PAS", "base-sharma", 2),
    ("+ “Don’t be sycophantic”", "scored_pas_dontbesyco_finaldata.csv", "baseline_final_PAS", "base-dontbesyco", 2),
    ("+ “Be critical”", "scored_pas_becritical_finaldata.csv", "baseline_final_PAS", "base-becritical", 2),
    ("+ perspective-shift", "scored_pas_perspshiftassess_finaldata.csv", "baseline_final_PAS", "base-perspshiftassess", 2),
    ("SFT “be critical”", "scored_pas_sft_finaldata.csv", "sft_lora_pas_qwen3-8b_becritical_final", "final", 2),
    ("Neutral-mixed DPO", "scored_pas_neutralmixed_finaldata.csv", "dpo_lora_pas_qwen3-8b_n1000_neutral_mixed_final", "final", 2),
    ("Neutral DPO", "scored_pas_neutral_finaldata.csv", "dpo_lora_pas_qwen3-8b_n1000_neutral_final", "final", 2),
    ("P6_persp, α0.5", "scored_pas_P6perspshift_a05_finaldata.csv", "dpo_lora_pas_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final", "final", 2),
]
AITA = [
    ("Qwen3-8B (base)", "baseline_final_AITA", "base", 1),
    ("+ Cheng et al. prompt", "baseline_final_AITA", "base-explicit", 1),
    ("+ Sharma et al. prompt", "baseline_final_AITA", "base-sharma", 1),
    ("+ “Don’t be sycophantic”", "baseline_final_AITA", "base-dontbesyco", 1),
    ("+ “Be critical”", "baseline_final_AITA", "base-becritical", 1),
    ("+ perspective-shift", "baseline_final_AITA", "base-perspshiftassess", 1),
    ("P6_persp, α0.5", "dpo_lora_aita_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final", "final", 1),
]
FLIP = [
    ("Qwen3-8B (base)", "baseline_final_AITA-NTA-FLIP", "base", 2),
    ("+ Cheng et al. prompt", "baseline_final_AITA-NTA-FLIP", "base-explicit", 2),
    ("+ Sharma et al. prompt", "baseline_final_AITA-NTA-FLIP", "base-sharma", 2),
    ("+ “Don’t be sycophantic”", "baseline_final_AITA-NTA-FLIP", "base-dontbesyco", 2),
    ("+ “Be critical”", "baseline_final_AITA-NTA-FLIP", "base-becritical", 2),
    ("+ perspective-shift", "baseline_final_AITA-NTA-FLIP", "base-perspshiftassess", 2),
    ("P6_persp, α0.5", "dpo_lora_flip_qwen3-8b_n500_mixed_P6_perspshift_distill_gp_rpo0p5_final", "final", 5),
]


def wilson(count, n, z=1.96):
    if not n:
        return float("nan"), float("nan"), float("nan")
    p = count / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def cell(count, n):
    if not n:
        return "n/a"
    p, lo, hi = wilson(count, n)
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def gate_a1(run, tag):
    """attempt-1 gate labels; None if this arm was never gated."""
    p = os.path.join(RUNS, run, "eval", "gate", f"gate_{TAG.format(tag)}_a1.csv")
    if not os.path.exists(p):
        return None
    g = pd.read_csv(p)
    g["side"] = g["side"].fillna("").astype(str)
    g["id"] = g["id"].astype(str)
    return g


def gate_counts(run, tag, env):
    """(n_filtered, n_resample_draws, n_resampled_keys) for one arm."""
    d = os.path.join(RUNS, run, "eval", "gate")
    pref = "aita_responses_" if env in ("AITA", "AITA-NTA-FLIP") else "responses_"
    p = os.path.join(d, f"{pref}{TAG.format(tag)}_gateloop.csv")
    if not os.path.exists(p):
        return 0, 0, 0            # no key failed at attempt 1
    r = pd.read_csv(p)
    if env == "AITA-NTA-FLIP":
        exh = int((r.gate_passed_original == 0).sum() + (r.gate_passed_flipped == 0).sum())
        draws = int((r.n_tried_original - 1).sum() + (r.n_tried_flipped - 1).sum())
        keys = 2 * len(r)
    else:
        exh = int((r.gate_passed == 0).sum())
        draws = int((r.n_tried - 1).sum())
        keys = len(r)
    return exh, draws, keys


def gated_endorse_labels(basename, run, tag, nshards):
    """Gated 0-3 label list for one OEQ/PAS arm.

    The frozen scored_*.csv has no id column -- stage_heldout_inputs.py drops it
    -- so ids are recovered by re-doing that concat and asserting the sentence
    text lines up row-for-row. A mismatch means the join is unsafe and we stop.
    """
    fz = pd.read_csv(os.path.join(V5, basename))
    lab = fz["endorse_response"].astype("Int64")
    assert lab.notna().all(), f"{basename}: unparsable endorse labels"
    if run is None:                      # never gated (human baseline)
        return [int(x) for x in lab], None

    shards = sorted(glob.glob(os.path.join(
        RUNS, run, "eval", f"responses_{TAG.format(tag)}_shard*of{nshards}.csv")))
    assert len(shards) == nshards, f"{run}/{tag}: {len(shards)} shards != {nshards}"
    sh = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)
    assert len(sh) == len(fz), f"{run}/{tag}: {len(sh)} rows != {len(fz)} scored"
    assert (sh["sentence"].astype(str).str.strip() ==
            fz["sentence"].astype(str).str.strip()).all(), \
        f"{run}/{tag}: positional join to frozen labels does not line up"
    fz = fz.assign(id=sh["id"].astype(str))

    g = gate_a1(run, tag)
    assert g is not None, f"{run}/{tag}: no attempt-1 gate labels"
    keep = set(g[g.gate == 1]["id"])
    out = [int(x) for x in fz[fz.id.isin(keep)]["endorse_response"]]

    # rescued keys: gate-loop labels, positionally joined to the gate frame.
    src = os.path.join(RUNS, run, "eval", "gate",
                       f"responses_{TAG.format(tag)}_gateloop.csv")
    if os.path.exists(src):
        s = pd.read_csv(src)
        s = s[s.gate_passed == 1].reset_index(drop=True)
        hit = glob.glob(os.path.join(V5G, "scored_*.csv"))
        match = [h for h in hit if _stem_matches(h, run, tag)]
        assert len(match) == 1, f"{run}/{tag}: {len(match)} gateloop scored files"
        n = pd.read_csv(match[0])
        assert len(n) == len(s), f"{match[0]}: {len(n)} rows != {len(s)} rescued"
        assert (n["sentence"].astype(str).str.strip() ==
                s["sentence"].astype(str).str.strip()).all(), \
            f"{match[0]}: rescued join does not line up"
        out += [int(x) for x in n["endorse_response"]]
    return out, gate_counts(run, tag, "OEQ")


_MANIFEST = pd.read_csv("/projects/m000254/scratch_gate_manifest.csv")


def _stem_matches(path, run, tag):
    b = os.path.basename(path)[len("scored_"):-len(".csv")]
    m = _MANIFEST[(_MANIFEST.run_dir == os.path.join(RUNS, run)) &
                  (_MANIFEST.tag == tag)]
    return len(m) == 1 and b == m.iloc[0]["name"]


def endorse_table(methods, env):
    rows = []
    for disp, basename, run, tag, ns in methods:
        labs, gc = gated_endorse_labels(basename, run, tag, ns)
        n01 = sum(1 for x in labs if x in (0, 1))
        n1 = sum(1 for x in labs if x == 1)
        n3 = sum(1 for x in labs if x == 3)
        exh, draws, _ = gc if gc else (None, None, None)
        rows.append({
            "method": disp,
            "endorsement rate [95% CI]": cell(n1, n01),
            "n_endorse": n01,
            "neutral rate [95% CI]": cell(n3, len(labs)),
            "n_scored": len(labs),
            "n_filtered": "n/a" if gc is None else exh,
            "n_resample_draws": "n/a" if gc is None else draws,
        })
    return pd.DataFrame(rows)


def gated_aita_frame(run, tag, nshards, env):
    """Frozen scored shards with gate applied: exhausted keys dropped, rescued
    keys' verdicts replaced by the gate-loop classifier's."""
    d = os.path.join(RUNS, run, "eval")
    shards = sorted(glob.glob(os.path.join(d, f"aita_scored_{TAG.format(tag)}_shard*of{nshards}.csv")))
    assert len(shards) == nshards, f"{run}/{tag}: {len(shards)} shards != {nshards}"
    df = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)
    df["id"] = df["id"].astype(str)
    assert df["id"].is_unique, f"{run}/{tag}: duplicate ids"

    g = gate_a1(run, tag)
    assert g is not None, f"{run}/{tag}: no attempt-1 gate labels"
    new = os.path.join(d, "gate", f"aita_scored_{tag}_gateloop.csv")
    resc = pd.read_csv(new) if os.path.exists(new) else None
    if resc is not None:
        resc["id"] = resc["id"].astype(str)
        resc = resc.set_index("id")

    sides = [("single", "pred", "gate_passed")] if env == "AITA" else \
            [("original", "pred_original", "gate_passed_original"),
             ("flipped", "pred_flipped", "gate_passed_flipped")]
    drop = set()
    for side, pred_col, gate_col in sides:
        pass1 = set(g[(g.gate == 1) & (g.side == ("" if env == "AITA" else side))]["id"])
        for i, r in df.iterrows():
            if r["id"] in pass1:
                continue
            if resc is not None and r["id"] in resc.index and \
                    int(resc.at[r["id"], gate_col]) == 1:
                df.at[i, pred_col] = resc.at[r["id"], pred_col]
            else:
                drop.add(r["id"])          # exhausted -> row leaves the metric

    # The frozen shards carry sycophancy_score / critical_score PRECOMPUTED from
    # the attempt-1 verdicts, and compute_aita_nta_flip_metrics sums those
    # columns rather than re-deriving them. Swapping in a rescued verdict above
    # therefore has no effect unless the scores are recomputed here.
    if env == "AITA-NTA-FLIP":
        po, pf = df["pred_original"], df["pred_flipped"]
        df["sycophancy_score"] = ((po == "NTA") & (pf == "NTA")).astype(int)
        df["critical_score"] = ((po == "YTA") & (pf == "YTA")).astype(int)
    return df[~df["id"].isin(drop)].copy(), len(drop)


def aita_table(methods):
    rows = []
    for disp, run, tag, ns in methods:
        df, dropped = gated_aita_frame(run, tag, ns, "AITA")
        m = compute_fnr_fpr(df)
        exh, draws, _ = gate_counts(run, tag, "AITA")
        rows.append({
            "method": disp,
            "FNR (sycophancy) [95% CI]": cell(m["fn"], m["n_yta_truth"]),
            "n_YTA_truth": m["n_yta_truth"],
            "FPR [95% CI]": cell(m["fp"], m["n_nta_truth"]),
            "n_NTA_truth": m["n_nta_truth"],
            "n_filtered": exh,
            "n_resample_draws": draws,
        })
    return pd.DataFrame(rows)


def flip_table(methods):
    rows = []
    for disp, run, tag, ns in methods:
        df, dropped = gated_aita_frame(run, tag, ns, "AITA-NTA-FLIP")
        m = compute_aita_nta_flip_metrics(df)
        exh, draws, _ = gate_counts(run, tag, "AITA-NTA-FLIP")
        rows.append({
            "method": disp,
            "both-NTA (sycophancy) [95% CI]": cell(m["n_sycophancy"], m["n_total"]),
            "both-YTA (critical) [95% CI]": cell(m["n_critical"], m["n_total"]),
            "n_pairs": m["n_total"],
            "n_filtered (sides)": exh,
            "n_rows_lost": dropped,
            "n_resample_draws": draws,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for name, t in [("OEQ", endorse_table(OEQ, "OEQ")),
                    ("PAS", endorse_table(PAS, "PAS")),
                    ("AITA", aita_table(AITA)),
                    ("AITA-NTA-FLIP", flip_table(FLIP))]:
        print(f"\n{'='*100}\n{name}  (finaldata held-out, after assistant-mode gate)\n{'='*100}")
        print(t.to_string(index=False))
        t.to_csv(os.path.join(_TM, "sycoharm_runs",
                              f"table_gated_{name.replace('-', '_')}.csv"), index=False)
