"""Summarize ArmoRM multi-objective scores: per arm, the mean gated `score` plus
the mean of each of the 19 reward dimensions, per dataset, ranked by gated score.

`trunc_at_cap` (fraction of responses at the generation token cap: 512 OEQ/PAS,
1536 AITA) is reported next to the means because reward models penalize
mid-sentence cutoffs; any arm ordering that tracks it is suspect. Standalone (no
torch) so it runs in any env. Reads runs/<rundir>/eval/rm_armorm_<tag>.csv.
"""
import csv, os, statistics, sys

csv.field_size_limit(10**9)
RUNS = "/projects/m000254/runs"
CAP = {"OEQ": 512, "PAS": 512, "AITA": 1536}
DIMS = [
    "helpsteer-helpfulness", "helpsteer-correctness", "helpsteer-coherence",
    "helpsteer-complexity", "helpsteer-verbosity", "ultrafeedback-overall_score",
    "ultrafeedback-instruction_following", "ultrafeedback-truthfulness",
    "ultrafeedback-honesty", "ultrafeedback-helpfulness", "beavertails-is_safe",
    "prometheus-score", "argilla-overall_quality", "argilla-judge_lm",
    "code-complexity", "code-style", "code-explanation",
    "code-instruction-following", "code-readability",
]
BASE_STEERS = ["base", "base-becritical", "base-explicit",
               "base-sharma", "base-dontbesyco", "base-perspshiftassess",
               "base-besycophantic"]
ENVS = {
    "OEQ": ("baseline_final_OEQ", [
        "dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final",
        "dpo_lora_oeq_qwen3-8b_n1000_neutral_final",
        "dpo_lora_oeq_qwen3-8b_n1000_neutral_mixed_final",
        "sft_lora_oeq_qwen3-8b_becritical_final"]),
    "PAS": ("baseline_final_PAS", [
        "dpo_lora_pas_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final",
        "dpo_lora_pas_qwen3-8b_n1000_neutral_final",
        "dpo_lora_pas_qwen3-8b_n1000_neutral_mixed_final",
        "sft_lora_pas_qwen3-8b_becritical_final"]),
    "AITA": ("baseline_final_AITA", [
        "dpo_lora_aita_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final"]),
}
SHORT = {"base": "base", "base-becritical": "becritical", "base-explicit": "explicit",
         "base-sharma": "sharma", "base-dontbesyco": "dontbesyco",
         "base-perspshiftassess": "perspshiftassess",
         "base-besycophantic": "besycophantic (CTRL)"}


def short(label):
    if label in SHORT:
        return SHORT[label]
    for k, v in [("P6_perspshift", "P6_rpo0p5 (OURS)"), ("neutral_mixed", "neutral_mixed"),
                 ("neutral", "neutral"), ("sft_lora", "becritical_sft")]:
        if k in label:
            return f"{v} (DPO)" if "dpo" in label else f"{v} (SFT)"
    return label


def conditions(env):
    base, trained = ENVS[env]
    return [(s, base, s) for s in BASE_STEERS] + [(r, r, "final") for r in trained]


def load(rundir, tag):
    f = f"{RUNS}/{rundir}/eval/rm_armorm_{tag}.csv"
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        return list(csv.DictReader(fh))


# Compact display: gated score + the 5 HelpSteer dims (the interpretable core).
SHOW = ["score", "helpsteer-helpfulness", "helpsteer-correctness",
        "helpsteer-verbosity", "helpsteer-complexity"]

for env in ["OEQ", "PAS", "AITA"]:
    cap = CAP[env]
    print(f"\n===== {env} (ArmoRM) =====")
    hdr = f"{'method':<22} {'n':>5} " + " ".join(f"{c.split('-')[-1][:9]:>9}" for c in SHOW) + f" {'trunc':>7}"
    print(hdr)
    rows = []
    for label, rundir, tag in conditions(env):
        rs = load(rundir, tag)
        if not rs:
            continue
        n = len(rs)
        means = {c: statistics.mean(float(r[c]) for r in rs) for c in SHOW}
        trunc = statistics.mean(1.0 if int(r["n_resp_tok"]) >= cap else 0.0 for r in rs)
        rows.append((short(label), n, means, trunc))
    for name, n, means, trunc in sorted(rows, key=lambda x: -x[2]["score"]):
        cells = " ".join(f"{means[c]:>9.3f}" for c in SHOW)
        print(f"{name:<22} {n:>5} {cells} {trunc*100:>6.1f}%")

print("\n(full 19-dim means per arm are in each rm_armorm_<tag>.csv; this table "
      "shows gated score + HelpSteer core. 'trunc' = frac at gen cap.)")
