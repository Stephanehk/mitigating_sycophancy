"""Score IFEval responses with the OFFICIAL programmatic checkers.

IFEval is verifiable instruction-following: each prompt carries a list of
instruction ids (e.g. 'length_constraints:number_words') with kwargs, and each
is checked PROGRAMMATICALLY (not by an LLM). We reuse Google's reference
implementation verbatim (vendored at <repo_root>/instruction_following_eval/,
from google-research/google-research) so the numbers match the published metric:

  - test_instruction_following_strict : checks the raw response.
  - test_instruction_following_loose  : checks several lenient transforms of the
                                        response (strip markdown, drop first/last
                                        line, etc.) and passes if ANY satisfies.

Reported metrics (the four standard IFEval numbers + their mean):
  - prompt-level   strict / loose : fraction of prompts where ALL instructions pass.
  - instruction-level strict / loose : fraction of individual instructions that pass.
Wald 95% CIs are added on each (binomial over prompts / over instructions).

Input contract:
  - --dataset (google/IFEval): supplies key, prompt, instruction_id_list, kwargs.
    The HF dataset pads every kwargs dict with all possible keys set to None;
    we DROP the None entries before build_description, matching the official
    input_data.jsonl (which omits absent keys).
  - --responses: one or more jsonl files (shards are concatenated) with objects
    {"prompt": <original prompt>, "response": <model output>}. Responses are
    joined to instructions by the EXACT original prompt string (Google's join
    key), so eval_ifeval.py records the bare prompt, not prompt+suffix.

Output: a summary json and a per-prompt csv under --out-dir, plus a printed
report. By default every dataset prompt must have a response (asserted);
--allow-partial scores only the covered prompts and reports the coverage.
"""

import argparse
import csv
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # training_mitigations/
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _REPO_ROOT)            # vendored 'instruction_following_eval'
sys.path.insert(0, os.path.join(_HERE, "eval"))

from instruction_following_eval import evaluation_lib  # noqa: E402
from ifeval_data import DEFAULT_DATA, load_records  # noqa: E402

DEFAULT_OUT_DIR = os.path.join(_HERE, "sycoharm_runs", "ifeval", "scored")


def load_inputs(dataset, split):
    """Build evaluation_lib.InputExample list from the IFEval records, dropping
    the None-padded kwargs entries so build_description sees only real
    arguments (matching Google's official input_data.jsonl)."""
    inputs = []
    for ex in load_records(dataset, split):
        kwargs = [{k: v for k, v in d.items() if v is not None} for d in ex["kwargs"]]
        assert len(kwargs) == len(ex["instruction_id_list"]), \
            f"key {ex['key']}: kwargs/instruction length mismatch"
        inputs.append(evaluation_lib.InputExample(
            key=ex["key"], instruction_id_list=ex["instruction_id_list"],
            prompt=ex["prompt"], kwargs=kwargs))
    return inputs


def load_responses(paths):
    """Concatenate response jsonl shards into {prompt: response}. Asserts no
    conflicting duplicate prompt maps to two different responses."""
    prompt_to_response = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                p, r = o["prompt"], o["response"]
                if p in prompt_to_response:
                    assert prompt_to_response[p] == r, \
                        f"conflicting responses for the same prompt in {path}"
                prompt_to_response[p] = r
    return prompt_to_response


def wald_ci(k, n):
    """Mean and 95% Wald half-width for k successes out of n Bernoulli trials."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    return p, 1.96 * math.sqrt(p * (1.0 - p) / n)


def aggregate(outputs):
    """Prompt-level and instruction-level success counts over a list of
    OutputExample (one scoring mode: strict or loose)."""
    prompt_ok = sum(1 for o in outputs if o.follow_all_instructions)
    inst_ok = sum(sum(1 for x in o.follow_instruction_list if x) for o in outputs)
    inst_tot = sum(len(o.follow_instruction_list) for o in outputs)
    return prompt_ok, len(outputs), inst_ok, inst_tot


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--responses", nargs="+", required=True,
                   help="one or more IFEval response jsonl files (shards concatenated)")
    p.add_argument("--tag", required=True, help="output stem under --out-dir")
    p.add_argument("--dataset", default=DEFAULT_DATA,
                   help="local IFEval jsonl (default) or an HF dataset name")
    p.add_argument("--split", default="train")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--allow-partial", action="store_true",
                   help="score only prompts that have a response (default: require all)")
    args = p.parse_args()

    inputs = load_inputs(args.dataset, args.split)
    prompt_to_response = load_responses(args.responses)

    covered = [inp for inp in inputs if inp.prompt in prompt_to_response]
    missing = len(inputs) - len(covered)
    if not args.allow_partial:
        assert missing == 0, \
            f"{missing}/{len(inputs)} dataset prompts have no response; pass --allow-partial to score the rest"
    inputs = covered

    strict = [evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
              for inp in inputs]
    loose = [evaluation_lib.test_instruction_following_loose(inp, prompt_to_response)
             for inp in inputs]

    sp_ok, sp_n, si_ok, si_n = aggregate(strict)
    lp_ok, lp_n, li_ok, li_n = aggregate(loose)

    res = {}
    for name, (k, n) in {
        "prompt_strict": (sp_ok, sp_n), "prompt_loose": (lp_ok, lp_n),
        "inst_strict": (si_ok, si_n), "inst_loose": (li_ok, li_n),
    }.items():
        m, ci = wald_ci(k, n)
        res[name] = {"acc": m, "ci95": ci, "k": k, "n": n}
    res["ifeval_avg"] = sum(res[m]["acc"] for m in
                            ("prompt_strict", "prompt_loose", "inst_strict", "inst_loose")) / 4.0

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {"tag": args.tag, "dataset": args.dataset, "n_prompts": len(inputs),
               "n_missing": missing, **res}
    sum_path = os.path.join(args.out_dir, f"summary_{args.tag}.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Per-prompt csv: key, prompt, #instructions, strict_all, loose_all.
    csv_path = os.path.join(args.out_dir, f"perprompt_{args.tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "n_instructions", "strict_all", "loose_all", "prompt"])
        for inp, s, l in zip(inputs, strict, loose):
            w.writerow([inp.key, len(inp.instruction_id_list),
                        int(s.follow_all_instructions), int(l.follow_all_instructions),
                        inp.prompt[:200]])

    # Report.
    print(f"\n===== IFEval [{args.tag}]  n={len(inputs)}"
          + (f"  (missing {missing})" if missing else "") + " =====")
    for label, key in [("prompt-level  strict", "prompt_strict"),
                       ("prompt-level  loose ", "prompt_loose"),
                       ("instruction   strict", "inst_strict"),
                       ("instruction   loose ", "inst_loose")]:
        r = res[key]
        print(f"  {label} = {r['acc']:.3f} [{r['acc']-r['ci95']:.3f}, {r['acc']+r['ci95']:.3f}]"
              f"  ({r['k']}/{r['n']})")
    print(f"  IFEval average        = {res['ifeval_avg']:.3f}")
    print(f"  -> {sum_path}")


if __name__ == "__main__":
    main()
