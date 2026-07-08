"""Joint summary of the OEQ rpo-alpha sweep: action-endorsement rate NEXT TO the
role-play rate, for every arm, so neither number is read on its own.

WHY BOTH: the v5 endorse prompt scopes "challenge" to the USER's OWN actions
(run_endorse_gpt5mini.py:175 -- v5 exists precisely because Qwen aimed its criticism
at the other party and v4 mislabeled that ~30% of the time). A response that
role-plays a stakeholder and berates the user therefore parses as a genuine
challenge (label 0), pushing the endorsement rate DOWN. That is indistinguishable
from a real sycophancy reduction unless you also know how often the policy stopped
answering as an assistant at all. On AITA/PAS the perspshift arms role-play 30.7% /
37.8% of the time. A low endorsement rate paired with a high role-play rate is an
artifact, not a win.

Reads:
  sycoharm_runs/endorse_gpt5mini_v5/scored_<name>.csv   (col `raw` -> first_label)
  <run_dir>/eval/voice_final_final1000_temp1.0_seed0.csv (col `voice`)

Both are optional per-arm; missing pieces print as n/a rather than aborting.
"""

import math
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "osfstorage-archive", "Code"))
# Reuse the paper's metric verbatim -- do NOT reimplement #1/(#0+#1).
from evaluate_action_endorsement_stephane import (  # noqa: E402
    first_label, action_endorsement_rate)

RUNS_ROOT = "/projects/m000254/runs"
SCORED_DIR = os.path.join(_REPO, "sycoharm_runs", "endorse_gpt5mini_v5")
TAG = "final_final1000_temp1.0_seed0"

# (arm run-dir basename, judge name, alpha, recipe) -- mirrors marlowe_endorse_judge_all.sh
ARMS = [
    ("dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_perspshift_gp",                         "oeq_P5perspshift_a0_final1000",    "0",    "P5_perspshift"),
    ("dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_gp",                         "oeq_P6perspshift_a0_final1000",    "0",    "P6_perspshift"),
    ("dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p25",         "oeq_P6perspshift_a025_final1000",  "0.25", "P6_perspshift"),
    ("dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5",          "oeq_P6perspshift_a05_final1000",   "0.5",  "P6_perspshift"),
    ("dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_nocrit_distill_gp_frozenstake_rpo0p25", "oeq_frozenstake_a025_final1000",   "0.25", "frozenstake"),
    ("dpo_lora_oeq_qwen3-8b_n1000_mixed_P5_nocrit_distill_gp_frozenstake_rpo0p5",  "oeq_frozenstake_a05_final1000",    "0.5",  "frozenstake"),
]


def wilson(k, n, z=1.959963985):
    if not n:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def endorse(name):
    path = os.path.join(SCORED_DIR, f"scored_{name}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "raw" not in df.columns:
        return None
    labels = df["raw"].map(first_label)
    rate, ci, n01 = action_endorsement_rate(labels.tolist())
    return rate, ci, n01, len(df)


def voice(arm):
    path = os.path.join(RUNS_ROOT, arm, "eval", f"voice_{TAG}.csv")
    if not os.path.exists(path):
        return None
    v = pd.read_csv(path)
    v = v[v.voice.isin(["OUTSIDER", "STAKEHOLDER", "UNCLEAR"])]
    if not len(v):
        return None
    k = int((v.voice == "STAKEHOLDER").sum())
    return wilson(k, len(v)) + (len(v),)


def main():
    print("\n" + "=" * 112)
    print("OEQ rpo-alpha sweep — action endorsement (v5 judge) vs role-play rate")
    print("=" * 112)
    print(f"{'recipe':<16s} {'alpha':>5s}  {'endorse #1/(#0+#1)':>26s} {'n01':>5s}  "
          f"{'role-play (STAKEHOLDER)':>26s} {'n':>5s}")
    print("-" * 112)
    last = None
    for arm, name, alpha, recipe in ARMS:
        if last and recipe != last:
            print()
        last = recipe
        e = endorse(name)
        v = voice(arm)
        es = f"{e[0]:.3f} [{e[0]-e[1]:.3f}, {e[0]+e[1]:.3f}]" if e else "n/a"
        en = f"{e[2]}" if e else "-"
        vs = f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" if v else "n/a"
        vn = f"{v[3]}" if v else "-"
        print(f"{recipe:<16s} {alpha:>5s}  {es:>26s} {en:>5s}  {vs:>26s} {vn:>5s}")
    print("=" * 112)
    print("READ THE TWO COLUMNS TOGETHER. A drop in endorsement rate accompanied by a")
    print("rise in role-play is the policy abandoning the assistant role, not less")
    print("sycophancy: v5 scopes 'challenge' to the user's own actions, so a")
    print("stakeholder-voiced grievance aimed at the user scores as label 0.")
    print("Neither judge has a gold set. See classify_response_voice.py docstring.")


if __name__ == "__main__":
    main()
