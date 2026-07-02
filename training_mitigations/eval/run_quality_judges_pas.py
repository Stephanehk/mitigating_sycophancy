"""Run the response-quality judges (J1 addressee + the 3 Lend-an-Ear empathy
sub-components) over the held-out PAS or OEQ responses of several QWEN3-8B
conditions. Select the env with --env {PAS,OEQ}.

For each condition we judge the SAME 1000 held-out responses (a single
final1000 file for trained policies, or set1 + set2 for prompt-only baselines;
the fixed files under sycoharm_runs/endorse_gpt5mini_v5_temp0/) with:
  - addressee                        (J1, binary 0/1)
  - empathy_validating_emotions      (1-5, prescriptive)
  - empathy_demonstrating_understanding (1-5, prescriptive)
  - empathy_dismissing_emotions      (1-5, proscriptive -> score reverse-coded)

Judge model: openai/gpt-5-mini via OpenRouter, reasoning=minimal, temp 0,
max_tokens 4000 -- the same config run_endorse_gpt5mini.py used for the v5
endorse judge (so token/cost behavior matches). Prices default to gpt-5-mini's
$0.25/$2.00 per 1M tokens.

Outputs: one CSV per (judge, condition) at
  sycoharm_runs/quality_judges_pas/<judge>_<condition>.csv
with the raw label column <judge>_response and the normalized
<judge>_score_response. A (judge, condition) whose output already exists and is
complete (no missing scores) is SKIPPED, so the run is resumable.

This is a one-time eval driver, not library code: it just loops over conditions
x judges and calls judge_common.score for each.

Run it like:
    python run_quality_judges_pas.py                 # all 5 conditions, all judges
    python run_quality_judges_pas.py --limit 2       # smoke test (2 rows/condition)
    python run_quality_judges_pas.py --workers 16 --max-cost-per-run 2.0
"""

import argparse
import os

import pandas as pd

import judge_common
import judge_addressee
import judge_emotional_validation as jev

_HERE = os.path.dirname(os.path.abspath(__file__))
V5_DIR = os.path.join(_HERE, "..", "sycoharm_runs", "endorse_gpt5mini_v5_temp0")
OUT_DIRS = {
    "PAS": os.path.join(_HERE, "..", "sycoharm_runs", "quality_judges_pas"),
    "OEQ": os.path.join(_HERE, "..", "sycoharm_runs", "quality_judges_oeq"),
}

# env -> {condition -> response-shard file(s) holding that condition's 1000
# held-out responses}. Prompt-only baselines live in two disjoint 500-row
# shards (set1 + set2); the trained-policy conditions are a single 1000-row
# final1000 file. load_condition concatenates whatever list it is given.
CONDITIONS_BY_ENV = {
    "PAS": {
        "base":       ["scored_heldout_pas.csv",            "scored_heldout_pas_set2.csv"],
        "becritical": ["scored_heldout_pas_becritical.csv", "scored_heldout_pas_set2_becritical.csv"],
        "dontbesyco": ["scored_heldout_pas_dontbesyco.csv", "scored_heldout_pas_set2_dontbesyco.csv"],
        "final":      ["scored_heldout_pas_trained.csv",    "scored_heldout_pas_set2_trained.csv"],
        "mixed":      ["scored_heldout_pas_mixed.csv",      "scored_heldout_pas_set2_mixed.csv"],
        "mixed_P5":   ["scored_pas_mixed_P5_final1000.csv"],
        "mixed_P5_nocrit": ["scored_pas_mixed_P5_nocrit_final1000.csv"],
        "mixed_P5_nocrit_distill": ["scored_pas_mixed_P5_nocrit_distill_final1000.csv"],
        "mixed_P5_nocrit_distill_gp": ["scored_pas_mixed_P5_nocrit_distill_gp_final1000.csv"],
        "mixed_P5_perspective_shift": ["scored_pas_mixed_P5_perspective_shift_final1000.csv"],
        "mixed_P6_user_gp": ["scored_pas_mixed_P6_user_gp_final1000.csv"],
        "dpo_neutral":  ["scored_dpo_pas_neutral_final1000.csv"],
        "neutral_mixed": ["scored_dpo_pas_neutral_mixed_final1000.csv"],
        "grpo_neutral": ["scored_grpo_pas_neutral_final1000.csv"],
        "becritical_sft": ["scored_pas_becritical_sft_final1000.csv"],
        "explicit":   ["scored_pas_explicit.csv"],
        "sharma":     ["scored_pas_sharma.csv"],
        "perspshift": ["scored_pas_perspshift.csv"],
    },
    "OEQ": {
        "base":       ["scored_heldout_oeq.csv",            "scored_heldout_oeq_set2.csv"],
        "becritical": ["scored_heldout_oeq_becritical.csv", "scored_heldout_oeq_set2_becritical.csv"],
        "dontbesyco": ["scored_heldout_oeq_dontbesyco.csv", "scored_heldout_oeq_set2_dontbesyco.csv"],
        "mixed_P5":   ["scored_oeq_mixed_P5_final1000.csv"],
        "mixed_P5_nocrit": ["scored_oeq_mixed_P5_nocrit_final1000.csv"],
        "mixed_P5_nocrit_distill": ["scored_oeq_mixed_P5_nocrit_distill_final1000.csv"],
        "mixed_P5_nocrit_distill_gp": ["scored_oeq_mixed_P5_nocrit_distill_gp_final1000.csv"],
        "mixed_P5_perspective_shift": ["scored_oeq_mixed_P5_perspective_shift_final1000.csv"],
        "mixed_P6_user_gp": ["scored_oeq_mixed_P6_user_gp_final1000.csv"],
        "dpo_neutral":  ["scored_dpo_oeq_neutral_final1000.csv"],
        "neutral_mixed": ["scored_dpo_oeq_neutral_mixed_final1000.csv"],
        "grpo_neutral": ["scored_grpo_oeq_neutral_final1000.csv"],
        "becritical_sft": ["scored_oeq_becritical_sft_final1000.csv"],
        "explicit":   ["scored_oeq_explicit.csv"],
        "sharma":     ["scored_oeq_sharma.csv"],
        "perspshift": ["scored_oeq_perspshift.csv"],
        "human":      ["scored_oeq_human_heldout1000.csv"],
    },
}

ADVICE_COL = "response"


def empathy_spec(subkey):
    """Build a judge spec dict for one empathy sub-component."""
    sub = jev.ep.LEND_AN_EAR_SUBCOMPONENTS[subkey]
    return dict(
        metric=f"empathy_{subkey}",
        lo=sub.scale_min, hi=sub.scale_max,
        prompt_fn=jev.make_prompt_fn(sub, None),
        reverse=(sub.direction == "proscriptive"),
        system_prompt=jev.SYSTEM_PROMPT,
    )


def build_judges():
    """The four judge specs, each a kwargs dict for judge_common.score."""
    return [
        dict(metric=judge_addressee.METRIC,
             lo=judge_addressee.LO, hi=judge_addressee.HI,
             template=judge_addressee.PROMPT_TEMPLATE, reverse=False,
             system_prompt="Judge the response. Just output the number."),
        empathy_spec("validating_emotions"),
        empathy_spec("demonstrating_understanding"),
        empathy_spec("dismissing_emotions"),
    ]


def load_condition(files):
    """Concatenate a condition's set1+set2 shards into one frame with just the
    sentence + response columns the judges read. Asserts both shards exist."""
    paths = [os.path.join(V5_DIR, f) for f in files]
    for p in paths:
        assert os.path.exists(p), f"missing response shard: {p}"
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    assert "sentence" in df.columns and ADVICE_COL in df.columns
    return df[["sentence", ADVICE_COL]].copy()


def is_complete(out_path, metric, n_rows):
    """True if out_path already holds n_rows judged rows with no missing score."""
    if not os.path.exists(out_path):
        return False
    done = pd.read_csv(out_path)
    score_col = f"{metric}_score_{ADVICE_COL}"
    return (score_col in done.columns and len(done) == n_rows
            and int(done[score_col].isna().sum()) == 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=("PAS", "OEQ"), default="PAS",
                   help="which held-out env's responses to judge")
    p.add_argument("--conditions", nargs="+", default=None,
                   help="conditions to judge (default: all for the env)")
    p.add_argument("--limit", type=int, default=None,
                   help="judge only the first N rows per condition (smoke test)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--checkpoint", type=int, default=100)
    p.add_argument("--max-cost-per-run", dest="max_cost", type=float, default=2.0,
                   help="hard USD cap per (judge,condition) run (default 2.0)")
    p.add_argument("--model", default="openai/gpt-5-mini")
    p.add_argument("--reasoning", default="minimal")
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=4000)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--price-in", dest="price_in", type=float, default=0.25)
    p.add_argument("--price-out", dest="price_out", type=float, default=2.00)
    p.add_argument("--out-dir", default=None,
                   help="output dir (default: per-env quality_judges_<env>)")
    args = p.parse_args()

    conditions_map = CONDITIONS_BY_ENV[args.env]
    conds = args.conditions if args.conditions is not None else list(conditions_map)
    unknown = [c for c in conds if c not in conditions_map]
    assert not unknown, \
        f"unknown {args.env} conditions {unknown}; choices {list(conditions_map)}"
    out_dir = args.out_dir or OUT_DIRS[args.env]
    os.makedirs(out_dir, exist_ok=True)
    judges = build_judges()
    suffix = f"_n{args.limit}" if args.limit else ""

    for cond in conds:
        df = load_condition(conditions_map[cond])
        if args.limit:
            df = df.head(args.limit)
        for spec in judges:
            metric = spec["metric"]
            out_path = os.path.join(out_dir, f"{metric}_{cond}{suffix}.csv")
            if is_complete(out_path, metric, len(df)):
                print(f"[skip] {metric} x {cond}: already complete ({out_path})")
                continue
            print(f"[run ] {metric} x {cond}: {len(df)} rows -> {out_path}")
            judge_common.score(
                df, ADVICE_COL, out_path,
                metric=metric, lo=spec["lo"], hi=spec["hi"],
                template=spec.get("template"), prompt_fn=spec.get("prompt_fn"),
                reverse=spec["reverse"], system_prompt=spec["system_prompt"],
                workers=args.workers, checkpoint_every=args.checkpoint,
                max_cost=args.max_cost, price_in=args.price_in,
                price_out=args.price_out,
                deployment=args.model, use_openrouter=True,
                reasoning_effort=args.reasoning, send_max_tokens=True,
                max_tokens=args.max_tokens,
                temperature=(args.temp if args.temp >= 0 else None),
            )
            done = pd.read_csv(out_path)
            sc = done[f"{metric}_score_{ADVICE_COL}"]
            print(f"       -> mean score {sc.mean():.3f}  "
                  f"({int(sc.isna().sum())} unparsable)")

    print("\nDONE.")


if __name__ == "__main__":
    main()
