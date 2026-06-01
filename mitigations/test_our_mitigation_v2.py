"""Inspect the v2 multi-stakeholder concerns-and-judge mitigation on a
single AITA, OEQ, or PAS post.

Same idea as `test_our_mitigation.py`, but drives `our_mitigation_v2`
so the new stakeholder-filter step (change (5)) is visible. The output
walks the four steps in order:

  STEP 1   identify stakeholders  (candidate list)
  STEP 1.5 filter stakeholders    (prune; prints kept vs dropped)
  STEP 2   per-stakeholder concerns + supports  (survivors only)
  STEP 3   judge synthesis        (YTA/NTA for AITA, advice for OEQ/PAS)

For OEQ/PAS it then runs the same action-endorsement judge that
`explore_mitigations.py` uses, so the single-post label is comparable
to the harness metric. Use `--post_id` to target a specific post (e.g.
one of the audit cases), or `--seed` to sample one at random.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from explore_mitigations import (
    AITA_DATA_CSV,
    OEQ_DATA_CSV,
    PAS_DATA_CSV,
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MAX_TOKENS,
    judge_endorsement,
    load_aita_clean,
    load_oeq,
    load_pas,
    make_client,
    model_family,
    model_pricing,
)
from our_mitigation_v2 import run_mitigation


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--env_name", choices=["AITA", "OEQ", "PAS"], required=True,
        help="evaluation environment: AITA (binary YTA/NTA), OEQ "
             "(open-ended advice on a full Reddit post), or PAS (advice "
             "on a single problematic-action sentence).",
    )
    parser.add_argument(
        "--post_id", default=None,
        help="run on this specific post id (e.g. pas_665784d9). Overrides "
             "--seed. Must be present in the env's data.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="random seed for selecting which post to load when --post_id "
             "is not given (default: 0)",
    )
    parser.add_argument("--model", default=DEFAULT_CLAUDE_MODEL,
                        help="generation model used by the mitigation")
    parser.add_argument("--judge_model", default=DEFAULT_JUDGE_MODEL,
                        help="action-endorsement judge model (OEQ and PAS only)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()

    data_csv = {
        "AITA": AITA_DATA_CSV,
        "OEQ":  OEQ_DATA_CSV,
        "PAS":  PAS_DATA_CSV,
    }[args.env_name]
    assert os.path.exists(data_csv), f"missing data csv: {data_csv}"
    loaders = {"AITA": load_aita_clean, "OEQ": load_oeq, "PAS": load_pas}
    data = loaders[args.env_name](data_csv)

    if args.post_id is not None:
        matches = data[data["id"] == args.post_id]
        assert len(matches) == 1, (
            f"--post_id {args.post_id!r} matched {len(matches)} rows in "
            f"{args.env_name} data; expected exactly 1"
        )
        row = matches.iloc[0]
    else:
        row = data.sample(n=1, random_state=args.seed).iloc[0]
    post_text = row["post_text"]

    if args.env_name == "AITA":
        truth = "YTA" if int(row["is_asshole"]) == 1 else "NTA"
        human_response = None
        print(f"=== AITA post id={row['id']}  truth={truth} ===")
    elif args.env_name == "OEQ":
        truth = None
        human_response = row["human_response"]
        print(f"=== OEQ post id={row['id']} ===")
    else:  # PAS — single problematic sentence
        truth = None
        human_response = None
        print(f"=== PAS post id={row['id']} ===")
    print(post_text)
    print()

    client = make_client(args.model)
    if (args.env_name in ("OEQ", "PAS")
            and model_family(args.judge_model) != model_family(args.model)):
        judge_client = make_client(args.judge_model)
    else:
        judge_client = client

    result = run_mitigation(
        client, args.model, post_text, args.max_tokens, args.env_name
    )

    candidates = result["candidate_stakeholders"]
    kept = result["stakeholders"]
    dropped = [s for s in candidates if s not in kept]
    added = [s for s in kept if s not in candidates]

    print("=== STEP 1: identify stakeholders ===")
    print("--- raw LLM response ---")
    print(result["raw_stakeholders"])
    print(f"--- parsed candidates ({len(candidates)}) ---")
    for s in candidates:
        print(f"  - {s}")
    print()

    print("=== STEP 1.5: filter stakeholders ===")
    print("--- raw LLM response ---")
    print(result["raw_filtered_stakeholders"])
    print(f"--- kept ({len(kept)}) ---")
    for s in kept:
        print(f"  + {s}")
    print(f"--- dropped ({len(dropped)}) ---")
    for s in dropped:
        print(f"  - {s}")
    if added:
        # The filter prompt asks for a subset, but the LLM is free to
        # reword or introduce names; surface anything that wasn't a
        # verbatim candidate so the prune is fully auditable.
        print(f"--- in kept but not a verbatim candidate ({len(added)}) ---")
        for s in added:
            print(f"  ? {s}")
    print()

    print("=== STEP 2: per-stakeholder concerns + supports ===")
    n = len(result["concerns_entries"])
    for i, e in enumerate(result["concerns_entries"], 1):
        print(f"--- stakeholder {i}/{n}: {e['stakeholder']} ---")
        print(f"raw response: {e['raw']!r}")
        print(f"parsed concerns ({len(e['concerns'])}):")
        for c in e["concerns"]:
            print(f"  - {c}")
        print(f"parsed supports ({len(e['supports'])}):")
        for s in e["supports"]:
            print(f"  - {s}")
    print()

    print("=== STEP 3: judge synthesis ===")
    print("--- raw judge response ---")
    print(result["judge_raw"])
    print("--- parsed reasoning ---")
    print(result["judge_reasoning"])
    if args.env_name == "AITA":
        verdict = result["judge_output"]
        match = verdict == truth
        print(f"--- parsed verdict: {verdict}  (truth={truth}, "
              f"match={'yes' if match else 'NO'}) ---")
    else:
        print("--- parsed advice response ---")
        print(result["judge_output"])

    in_total = result["input_tokens"]
    out_total = result["output_tokens"]

    if args.env_name in ("OEQ", "PAS"):
        if args.env_name == "OEQ":
            print()
            print("=== HUMAN RESPONSE (from OSF, for comparison) ===")
            print(human_response)

        print()
        print(f"=== STEP 4: action-endorsement judge ({args.judge_model}) ===")
        label_model, j_in1, j_out1 = judge_endorsement(
            judge_client, args.judge_model, post_text, result["judge_output"]
        )
        judge_in_total = j_in1
        judge_out_total = j_out1
        if args.env_name == "OEQ":
            label_human, j_in2, j_out2 = judge_endorsement(
                judge_client, args.judge_model, post_text, human_response
            )
            judge_in_total += j_in2
            judge_out_total += j_out2
        label_descriptions = {
            0: "challenges the user's actions",
            1: "explicitly affirms the user's actions",
            2: "implicitly affirms the user's actions",
            3: "neutral / unrelated to the user's actions",
        }
        print(f"--- model advice label: {label_model} "
              f"({label_descriptions.get(label_model, 'unparseable')}) ---")
        if args.env_name == "OEQ":
            print(f"--- human advice label: {label_human} "
                  f"({label_descriptions.get(label_human, 'unparseable')}) ---")

    gen_in_price, gen_out_price = model_pricing(args.model)
    cost = (in_total / 1e6) * gen_in_price + (out_total / 1e6) * gen_out_price
    print()
    print("=== cost ===")
    print(f"  gen_input_tokens:    {in_total}  (priced at --model {args.model})")
    print(f"  gen_output_tokens:   {out_total}")
    if args.env_name in ("OEQ", "PAS"):
        judge_in_price, judge_out_price = model_pricing(args.judge_model)
        judge_cost = ((judge_in_total / 1e6) * judge_in_price
                      + (judge_out_total / 1e6) * judge_out_price)
        cost += judge_cost
        print(f"  judge_input_tokens:  {judge_in_total}  (priced at --judge_model {args.judge_model})")
        print(f"  judge_output_tokens: {judge_out_total}")
    print(f"  cost_usd:            ${cost:.4f}")


if __name__ == "__main__":
    main()
