"""Compare the voice-DPO arm against the P6 arm it continues.

Prints one table per env covering the three requested evaluations plus the
confound checks that decide whether a movement is real:

  (a) action-endorsement rate      scored_<name>_finaldata.csv  (gpt-5-mini)
  (b) addressee (J1) + STAKEHOLDER  judges_finaldata/<ENV>/addressee_<slug>.csv
                                    <run>/eval/voice_<tag>.csv
  (c) ArmoRM mean score             <run>/eval/rm_armorm_final.csv

  confounds  response length, 512-truncation rate, markdown-vs-plain register

Why the confounds are not optional. The endorsement metric tracks markdown-vs-
plain register at r=-0.70 and its denominator is model-selected, so a register
shift moves the headline number for reasons unrelated to sycophancy. And DPO's
best-known failure mode is satisfying the objective by changing LENGTH rather
than the behaviour, which is live here because the rejected side of every pair
is a role-collapse or message-critique response and those may be systematically
longer. A large length delta means the result needs a different explanation.

Also prints the pair-construction manifest, which says how much of the train set
the round actually saw (prompts with no failures, or no clean response to pair
against, contribute nothing).

Missing inputs are reported as "-" rather than crashing, so this is safe to run
mid-chain.

Usage:
    python eval/summarize_voicedpo.py --envs OEQ PAS
"""

import argparse
import csv
import json
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = "/projects/m000254/runs"
SCORED = os.path.join(_HERE, "sycoharm_runs", "endorse_gpt5mini_v5_temp0")
JUDGES = os.path.join(_HERE, "sycoharm_runs", "judges_finaldata")

csv.field_size_limit(10**9)

TAG = "final_finaldata_temp1.0_seed0"

# env -> [(column label, run dir, endorse name, judge slug)]
ARMS = {
    "OEQ": [
        ("P6_persp a0.5",
         "dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final",
         "oeq_P6perspshift_a05_finaldata", "P6perspshift_a05"),
        ("+ voice-DPO",
         "dpo_lora_oeq_qwen3-8b_P6_perspshift_rpo0p5_final_voicedpo",
         "oeq_voicedpo_finaldata", "voicedpo"),
        ("+ voice-DPO b0.5",
         "dpo_lora_oeq_qwen3-8b_P6_perspshift_rpo0p5_final_voicedpo_b05",
         "oeq_voicedpo_b05_finaldata", "voicedpo_b05"),
    ],
    "PAS": [
        ("P6_persp a0.5",
         "dpo_lora_pas_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final",
         "pas_P6perspshift_a05_finaldata", "P6perspshift_a05"),
        ("+ voice-DPO",
         "dpo_lora_pas_qwen3-8b_P6_perspshift_rpo0p5_final_voicedpo",
         "pas_voicedpo_finaldata", "voicedpo"),
        ("+ voice-DPO b0.5",
         "dpo_lora_pas_qwen3-8b_P6_perspshift_rpo0p5_final_voicedpo_b05",
         "pas_voicedpo_b05_finaldata", "voicedpo_b05"),
    ],
}


def _rows(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def endorse_rate(name):
    """#1 / (#0 + #1) -- the paper metric. Denominator excludes other labels."""
    rows = _rows(os.path.join(SCORED, f"scored_{name}.csv"))
    if rows is None:
        return None
    vals = []
    for r in rows:
        v = (r.get("endorse_response") or "").strip()
        if v in ("0", "1"):
            vals.append(int(v))
    if not vals:
        return None
    return sum(vals) / len(vals)


def addressee_rate(env, slug):
    rows = _rows(os.path.join(JUDGES, env, f"addressee_{slug}.csv"))
    if rows is None:
        return None
    col = next((c for c in rows[0] if c.startswith("addressee_score")), None)
    if col is None:
        return None
    vals = [float(r[col]) for r in rows if (r.get(col) or "").strip() != ""]
    return sum(vals) / len(vals) if vals else None


def stakeholder_rate(rundir):
    rows = _rows(os.path.join(RUNS, rundir, "eval", f"voice_{TAG}.csv"))
    if rows is None:
        return None
    labs = [r["voice"] for r in rows if r.get("voice")]
    if not labs:
        return None
    return sum(1 for v in labs if v == "STAKEHOLDER") / len(labs)


def armorm_mean(rundir):
    rows = _rows(os.path.join(RUNS, rundir, "eval", "rm_armorm_final.csv"))
    if rows is None:
        return None
    vals = [float(r["score"]) for r in rows if (r.get("score") or "").strip() != ""]
    return sum(vals) / len(vals) if vals else None


def response_stats(rundir):
    """-> (mean chars, truncation rate, markdown rate) over the 2 gen shards.

    Truncation is inferred from the absence of terminal punctuation: the gen job
    caps at 512 tokens and does not record finish_reason, so a response that
    stops mid-sentence is the observable signature.
    """
    rows = []
    for s in range(2):
        r = _rows(os.path.join(RUNS, rundir, "eval",
                               f"responses_{TAG}_shard{s}of2.csv"))
        if r:
            rows.extend(r)
    if not rows:
        return None, None, None
    texts = [r["response"] for r in rows if r.get("response")]
    if not texts:
        return None, None, None
    mean_len = sum(len(t) for t in texts) / len(texts)
    trunc = sum(1 for t in texts if not t.rstrip().endswith((".", "!", "?", '"', ")")))
    md = sum(1 for t in texts if ("**" in t or "\n- " in t or "\n* " in t
                                  or "\n#" in t or "\n1. " in t))
    return mean_len, trunc / len(texts), md / len(texts)


def fmt(v, pct=False, dp=3):
    if v is None:
        return "-"
    return f"{v:.1%}" if pct else f"{v:.{dp}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", nargs="+", default=["OEQ", "PAS"])
    args = ap.parse_args()

    for env in args.envs:
        arms = ARMS.get(env)
        if not arms:
            print(f"[summary] unknown env {env}"); continue

        print("=" * 78)
        print(f"  {env}")
        print("=" * 78)

        # Pair construction: how much of the train set actually contributed.
        newdir = arms[-1][1]
        man_path = os.path.join(RUNS, newdir, "pairs", "manifest.json")
        if os.path.exists(man_path):
            with open(man_path) as f:
                m = json.load(f)
            print(f"  pairs built        : {m.get('n_pairs')}")
            print(f"  from prompts       : {m.get('n_prompts_contributing')}"
                  f" / {m.get('n_prompts_seen')}")
            print(f"    no failure       : {m.get('n_prompts_dropped_no_failure')}")
            print(f"    no clean sample  : {m.get('n_prompts_dropped_no_clean')}")
            print(f"  label counts       : {m.get('label_counts')}")
            cl = (m.get("chosen_length") or {}).get("mean_chars")
            rl = (m.get("rejected_length") or {}).get("mean_chars")
            if cl and rl:
                print(f"  chosen/rejected len: {cl} / {rl} chars "
                      f"({(rl - cl) / cl:+.1%} rejected vs chosen)")
            print()
        else:
            print(f"  (no pair manifest yet at {man_path})\n")

        hdr = f"  {'metric':<28}" + "".join(f"{a[0]:>16}" for a in arms) + f"{'delta':>12}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        def line(name, vals, pct=False, dp=3, lower_better=True):
            cells = "".join(f"{fmt(v, pct, dp):>16}" for v in vals)
            d = "-"
            if vals[0] is not None and vals[-1] is not None:
                delta = vals[-1] - vals[0]
                d = f"{delta:+.1%}" if pct else f"{delta:+.{dp}f}"
            print(f"  {name:<28}{cells}{d:>12}")

        line("(b) STAKEHOLDER rate",
             [stakeholder_rate(a[1]) for a in arms], pct=True)
        line("(b) addressee (J1) rate",
             [addressee_rate(env, a[3]) for a in arms], pct=True)
        line("(a) endorsement rate",
             [endorse_rate(a[2]) for a in arms], pct=True)
        line("(c) ArmoRM mean",
             [armorm_mean(a[1]) for a in arms], dp=4)

        stats = [response_stats(a[1]) for a in arms]
        print()
        line("    mean response chars", [s[0] for s in stats], dp=0)
        line("    truncated (no end punct)", [s[1] for s in stats], pct=True)
        line("    markdown register", [s[2] for s in stats], pct=True)
        print()
        print("  Confound reading: a large length or register shift means the")
        print("  endorsement delta needs a different explanation before it can")
        print("  be read as a sycophancy change.")
        print()


if __name__ == "__main__":
    main()
