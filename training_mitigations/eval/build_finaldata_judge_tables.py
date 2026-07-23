"""Assemble the 7-judge quality/ELEPHANT/voice table for the FINAL-DATASET
(`finaldata` tag) conditions, per env.

Reads what run_finaldata_judges.py (6 per-response judges) and
classify_response_voice.py / voice_human_finaldata.py (voice) have written, and
reports every env whose judge outputs are complete. An env or column that has
not been judged yet is reported as missing rather than silently omitted, so a
partially-finished sweep can never look like a finished one.

Columns and their CI conventions (documented so the table is auditable):
  framing, validation, addressee   binary 0/1 -> 95% Wilson score interval
  voice (STAKEHOLDER rate)         binary -> Wilson, over usable judgements only

The 3 Lend-an-Ear empathy judges were RETIRED by request: they are neither run
nor reported. Their already-computed OEQ outputs remain on disk under
judges_finaldata/OEQ/ if that decision is ever revisited.

Polarity notes (do not read every column as "higher = better"):
  framing      0 = CHALLENGES the question's premise. 0 is the NON-sycophantic
               side, so a LOWER framing rate is the less-sycophantic direction.
  validation   1 = emotionally validating. Not a sycophancy axis on its own.
  voice        STAKEHOLDER = the response role-plays someone inside the user's
               story instead of answering as an outsider. Higher = more collapse.

Every rate is asserted to have zero unparsable rows before it is reported: after
the ERROR-cell fix in judge_common.expand_score a failed API call scores NaN, so
a NaN here means the condition is incomplete and must be re-judged, not averaged
over.

Run:
    python training_mitigations/eval/build_finaldata_judge_tables.py
    python training_mitigations/eval/build_finaldata_judge_tables.py --envs OEQ
"""

import argparse
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_TM = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from build_finaldata_tables import (  # noqa: E402
    OEQ_METHODS, PAS_METHODS, AITA_METHODS, FLIP_METHODS, RUNS_ROOT, wilson_ci, fmt)
from run_finaldata_judges import build_conditions, frame_for  # noqa: E402

JUDGE_DIR = os.path.join(_TM, "sycoharm_runs", "judges_finaldata")
OUT_DIR = os.path.join(_TM, "sycoharm_runs")

BINARY = ["framing", "validation", "addressee"]
SHORT = {"framing": "framing (0=challenges)", "validation": "validation",
         "addressee": "addressee"}

LABELS = {"OEQ": [l for l, _ in OEQ_METHODS], "PAS": [l for l, _ in PAS_METHODS],
          "AITA": [l for l, _ in AITA_METHODS], "FLIP": [l for l, _ in FLIP_METHODS]}

_TAG = "{}_finaldata_temp1.0_seed0"
# env -> slug -> (run subdir, tag) for the voice judge's per-run output. The OEQ
# human reference has no run dir; voice_human_finaldata.py writes it under
# judges_finaldata/OEQ/voice_human.csv instead, handled in voice_rate().
VOICE_MAP = {
    "OEQ": {
        "base": ("baseline_final_OEQ", _TAG.format("base")),
        "explicit": ("baseline_final_OEQ", _TAG.format("base-explicit")),
        "sharma": ("baseline_final_OEQ", _TAG.format("base-sharma")),
        "dontbesyco": ("baseline_final_OEQ", _TAG.format("base-dontbesyco")),
        "becritical": ("baseline_final_OEQ", _TAG.format("base-becritical")),
        "perspshiftassess": ("baseline_final_OEQ", _TAG.format("base-perspshiftassess")),
        "sft": ("sft_lora_oeq_qwen3-8b_becritical_final", _TAG.format("final")),
        "neutralmixed": ("dpo_lora_oeq_qwen3-8b_n1000_neutral_mixed_final", _TAG.format("final")),
        "neutral": ("dpo_lora_oeq_qwen3-8b_n1000_neutral_final", _TAG.format("final")),
        "P6perspshift_a05": (
            "dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final",
            _TAG.format("final")),
    },
    "PAS": {
        "base": ("baseline_final_PAS", _TAG.format("base")),
        "explicit": ("baseline_final_PAS", _TAG.format("base-explicit")),
        "sharma": ("baseline_final_PAS", _TAG.format("base-sharma")),
        "dontbesyco": ("baseline_final_PAS", _TAG.format("base-dontbesyco")),
        "becritical": ("baseline_final_PAS", _TAG.format("base-becritical")),
        "perspshiftassess": ("baseline_final_PAS", _TAG.format("base-perspshiftassess")),
        "sft": ("sft_lora_pas_qwen3-8b_becritical_final", _TAG.format("final")),
        "neutralmixed": ("dpo_lora_pas_qwen3-8b_n1000_neutral_mixed_final", _TAG.format("final")),
        "neutral": ("dpo_lora_pas_qwen3-8b_n1000_neutral_final", _TAG.format("final")),
        "P6perspshift_a05": (
            "dpo_lora_pas_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final",
            _TAG.format("final")),
    },
    "AITA": {
        "base": ("baseline_final_AITA", _TAG.format("base")),
        "explicit": ("baseline_final_AITA", _TAG.format("base-explicit")),
        "sharma": ("baseline_final_AITA", _TAG.format("base-sharma")),
        "dontbesyco": ("baseline_final_AITA", _TAG.format("base-dontbesyco")),
        "becritical": ("baseline_final_AITA", _TAG.format("base-becritical")),
        "perspshiftassess": ("baseline_final_AITA", _TAG.format("base-perspshiftassess")),
        "P6perspshift_a05": (
            "dpo_lora_aita_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final",
            _TAG.format("final")),
    },
    "FLIP": {
        "base": ("baseline_final_AITA-NTA-FLIP", _TAG.format("base")),
        "explicit": ("baseline_final_AITA-NTA-FLIP", _TAG.format("base-explicit")),
        "sharma": ("baseline_final_AITA-NTA-FLIP", _TAG.format("base-sharma")),
        "dontbesyco": ("baseline_final_AITA-NTA-FLIP", _TAG.format("base-dontbesyco")),
        "becritical": ("baseline_final_AITA-NTA-FLIP", _TAG.format("base-becritical")),
        "perspshiftassess": ("baseline_final_AITA-NTA-FLIP", _TAG.format("base-perspshiftassess")),
        "P6perspshift_a05": (
            "dpo_lora_flip_qwen3-8b_n500_mixed_P6_perspshift_distill_gp_rpo0p5_final",
            _TAG.format("final")),
    },
}
VOICE_LABELS = ("STAKEHOLDER", "OUTSIDER", "UNCLEAR")


def judge_scores(env, metric, slug, expected):
    """-> the [0,1] score vector for one (env, metric, condition), or None if the
    judge has not run yet.

    Asserts the file is the full expected length and has no NaN score: post-fix a
    NaN means a failed API call, so averaging over it would under-report silently.
    """
    p = os.path.join(JUDGE_DIR, env, f"{metric}_{slug}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    col = f"{metric}_score_response"
    assert col in df.columns, f"{p}: no {col} (mid-run checkpoint?)"
    assert len(df) == expected, f"{p}: {len(df)} rows, expected {expected}"
    assert df[col].notna().all(), \
        f"{p}: {int(df[col].isna().sum())} unparsable -- re-judge this condition"
    return df[col].to_numpy(dtype=float)


def voice_rate(env, slug):
    """-> (k, n) STAKEHOLDER count over usable judgements, or None if not run.

    Only rows with a real label count; ERR/PARSE_FAIL are excluded and reported
    by the caller, since they shrink the denominator rather than bias the rate.
    """
    if slug == "human":
        p = os.path.join(JUDGE_DIR, "OEQ", "voice_human.csv")
    else:
        entry = VOICE_MAP[env].get(slug)
        if entry is None:
            return None
        run, tag = entry
        p = os.path.join(RUNS_ROOT, run, "eval", f"voice_{tag}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    ok = df[df["voice"].isin(VOICE_LABELS)]
    return int((ok["voice"] == "STAKEHOLDER").sum()), len(ok)


def build_env(env):
    """-> (frame, list of missing "metric x slug" strings) for one env."""
    conds = build_conditions(env)
    labels = LABELS[env]
    assert len(conds) == len(labels), f"{env}: label/condition mismatch"
    rows, missing = [], []
    for (slug, src), label in zip(conds, labels):
        n_expected = len(frame_for(env, src))
        row = {"method": label, "N": n_expected}
        for metric in BINARY:
            v = judge_scores(env, metric, slug, n_expected)
            if v is None:
                missing.append(f"{metric} x {slug}")
                row[SHORT[metric]] = "—"
            else:
                row[SHORT[metric]] = fmt(*wilson_ci(int(v.sum()), len(v)))
        vr = voice_rate(env, slug)
        if vr is None:
            missing.append(f"voice x {slug}")
            row["voice: STAKEHOLDER"] = "—"
        else:
            k, n = vr
            row["voice: STAKEHOLDER"] = fmt(*wilson_ci(k, n))
            if n != n_expected:
                row["voice: STAKEHOLDER"] += f" (n={n})"
        rows.append(row)
    return pd.DataFrame(rows), missing


def md(df, title):
    cols = list(df.columns)
    out = [f"### {title}", "| " + " | ".join(cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", nargs="+", default=["OEQ", "PAS", "AITA", "FLIP"])
    args = p.parse_args()
    for env in args.envs:
        df, missing = build_env(env)
        out = os.path.join(OUT_DIR, f"results_finaldata_judges_{env.lower()}.csv")
        df.to_csv(out, index=False)
        print("\n" + md(df, f"{env} — final-dataset held-out: 7 judges (95% CIs)"))
        if missing:
            print(f"\n[INCOMPLETE] {env}: {len(missing)} not yet judged -> "
                  + ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""))
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
