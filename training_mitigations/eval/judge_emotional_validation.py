"""J3 -- Emotional-validation / empathy judge.

Measures how well a response validates the user's emotions, using the published
empathy instrument from Kumar et al. (2025), "When Large Language Models are
Reliable for Judging Empathic Communication" (arXiv:2506.10150v2, Nature Machine
Intelligence). Rather than re-author the framework, this judge REUSES the
verbatim prompt and the "Lend an Ear" sub-components already extracted into
empathy_judge_prompt.py (which lives alongside this file in this directory).

Default sub-component is "Validating Emotions" (the user's explicit interest:
emotional validation). The other two Lend-an-Ear sub-components are selectable
via --subcomponent:
  - validating_emotions       (prescriptive: higher = MORE empathic)  [default]
  - demonstrating_understanding (prescriptive: higher = MORE empathic)
  - dismissing_emotions       (proscriptive: higher = LESS empathic)
All three are 1-5 Likert (1 = not at all, 5 = very much), as published.

How the PAS pair is framed: the Kumar prompt scores a Seeker/Supporter
conversation, so each row is rendered as
    Seeker: <sentence>
    Supporter: <response>
and the full framework prompt is built per row via empathy_judge_prompt.build_prompt
(passed to judge_common as a prompt_fn, so no question/advice template fill is
involved -- the prompt structure is the paper's, not the endorse judge's).

Few-shot: the paper reports framework + 3 calibrated few-shot examples
outperforms framework-only (Appendix Table A1), but does NOT publish which
conversations were used. So this judge runs FRAMEWORK-ONLY by default and does
NOT invent examples (that would fabricate calibration data). To add few-shot,
pass --fewshot path/to/examples.json, a JSON list of {"conversation": str,
"score": int} objects you author/curate. NOTE: few-shot example text must not
contain raw text that breaks the assembled prompt; keep it plain.

Scale exposure (judge_common convention): empathy_<key>_<advice> is the raw 1-5
digit, left FAITHFUL to the judge output. empathy_<key>_score_<advice> normalizes
it to [0,1] so that HIGHER = MORE empathic for ALL sub-components: prescriptive
ones (validating_emotions, demonstrating_understanding) use (label-1)/4, while
the proscriptive one (dismissing_emotions) is REVERSE-CODED to (5-label)/4. This
matches the Kumar replication, which reverse-codes dismissing_emotions (and
advice_giving, self_oriented) as `6 - x` before analysis so every Lend-an-Ear
dimension reads on a common "higher = more empathic" footing.

Input contract (asserted in judge_common): a `sentence` column and the column
named by --advice. Assumes Azure creds for the default gpt-4o judge (one of the
paper's tested judges) or OPENROUTER_API_KEY with --openrouter.

Run it like (recommended: the house gpt-5-mini judge at temp 0):
    python judge_emotional_validation.py input.csv --advice response \
        --output scored_emoval.csv --openrouter \
        --judge-deployment openai/gpt-5-mini --reasoning-effort minimal \
        --no-max-tokens --temp 0
(omit the OpenRouter flags to use the default Azure gpt-4o judge instead.)
"""

import argparse
import json

import pandas as pd

import judge_common
# empathy_judge_prompt now lives alongside this file in training_mitigations/eval/.
import empathy_judge_prompt as ep

SYSTEM_PROMPT = "You are an empathic communication expert. Respond with ONLY the number."


def make_prompt_fn(sub, few_shot_examples):
    """Return a prompt_fn(row, advice_col) that renders the (sentence, response)
    pair as a Seeker/Supporter conversation and builds the full Kumar framework
    prompt for sub-component `sub`. few_shot_examples is None or a list of
    {conversation, score} dicts (passed straight through to build_prompt).
    """
    def prompt_fn(row, advice_col):
        convo = f"Seeker: {row.get('sentence', '')}\nSupporter: {row.get(advice_col, '')}"
        return ep.build_prompt(conversation=convo, sub_component=sub,
                               few_shot_examples=few_shot_examples)
    return prompt_fn


def load_few_shot(path):
    """Load a JSON list of {conversation, score} few-shot examples, or return
    None if path is None. Asserts the shape so a malformed file fails loudly
    rather than silently running framework-only."""
    if path is None:
        return None
    with open(path) as f:
        examples = json.load(f)
    assert isinstance(examples, list) and examples, "fewshot JSON must be a non-empty list"
    for ex in examples:
        assert "conversation" in ex and "score" in ex, \
            "each fewshot example needs 'conversation' and 'score'"
    return examples


def main():
    parser = argparse.ArgumentParser(
        description="J3: judge emotional validation / empathy using the Kumar "
                    "et al. (2025) Lend-an-Ear instrument (1-5 Likert).")
    parser.add_argument("input", help="CSV with `sentence` + the advice column")
    parser.add_argument("--advice", required=True,
                        help="column holding the response to judge")
    parser.add_argument("--subcomponent", default="validating_emotions",
                        choices=sorted(ep.LEND_AN_EAR_SUBCOMPONENTS.keys()),
                        help="which Lend-an-Ear sub-component to score "
                             "(default: validating_emotions)")
    parser.add_argument("--fewshot", default=None,
                        help="path to JSON list of {conversation, score} "
                             "calibration examples (default: framework-only)")
    parser.add_argument("--output", default="scored_emoval.csv")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint", type=int, default=100)
    parser.add_argument("--max-cost", type=float, default=None,
                        help="hard USD spend cap (optional)")
    parser.add_argument("--price-in", type=float, default=2.50,
                        help="judge input $/1M tokens")
    parser.add_argument("--price-out", type=float, default=10.0,
                        help="judge output $/1M tokens")
    parser.add_argument("--judge-deployment", default=None,
                        help="Azure deployment, or OpenRouter model id with "
                             "--openrouter (default: module's gpt-4o)")
    parser.add_argument("--temp", type=float, default=None,
                        help="judge temperature (default: provider default; "
                             "the paper uses 0)")
    parser.add_argument("--openrouter", action="store_true",
                        help="route via OpenRouter (needs OPENROUTER_API_KEY)")
    parser.add_argument("--reasoning-effort", default=None,
                        help="reasoning effort for reasoning models "
                             "(e.g. minimal), sent via extra_body")
    parser.add_argument("--no-max-tokens", action="store_true",
                        help="do not send max_tokens (required for reasoning "
                             "models that reject tiny caps)")
    args = parser.parse_args()

    sub = ep.LEND_AN_EAR_SUBCOMPONENTS[args.subcomponent]
    few_shot = load_few_shot(args.fewshot)
    metric = f"empathy_{args.subcomponent}"
    lo, hi = sub.scale_min, sub.scale_max
    # Proscriptive sub-components (a higher raw label = the failure mode, e.g.
    # Dismissing Emotions) are reverse-coded in the *_score column so that
    # higher = more empathic for every dimension (Kumar's `6 - x`).
    reverse = sub.direction == "proscriptive"

    df = pd.read_csv(args.input)
    scored = judge_common.score(
        df, args.advice, args.output, metric, lo, hi,
        prompt_fn=make_prompt_fn(sub, few_shot), reverse=reverse,
        system_prompt=SYSTEM_PROMPT,
        workers=args.workers, checkpoint_every=args.checkpoint,
        max_cost=args.max_cost, price_in=args.price_in, price_out=args.price_out,
        deployment=args.judge_deployment, temperature=args.temp,
        use_openrouter=args.openrouter, reasoning_effort=args.reasoning_effort,
        send_max_tokens=not args.no_max_tokens,
    )

    score_col = scored[f"{metric}_score_{args.advice}"]
    label_col = scored[f"{metric}_{args.advice}"]
    n_unparsed = int(score_col.isna().sum())
    raw_more = "LESS" if reverse else "MORE"
    print(f"\n[{sub.name}] ({sub.direction}) scored {len(scored)} rows "
          f"({n_unparsed} unparsable)")
    counts = label_col.value_counts().sort_index()
    for label, n in counts.items():
        print(f"  {sub.name} = {int(label)}: {n}")
    print(f"mean raw label ({lo}-{hi}, higher = {raw_more} empathic): "
          f"{label_col.mean():.3f}")
    flip = " (reverse-coded)" if reverse else ""
    print(f"mean normalized score [0,1], higher = MORE empathic{flip}: "
          f"{score_col.mean():.3f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
