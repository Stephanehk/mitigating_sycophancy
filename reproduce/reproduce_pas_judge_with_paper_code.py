#!/usr/bin/env python3
"""Reproduce the ELEPHANT PAS judge discrepancy using the PAPER'S OWN judge code.

This is the high-fidelity companion to `reproduce_pas_judge_discrepancy.py`.
Instead of re-implementing the judge, it imports and runs the paper's actual
scoring script — `osfstorage-archive/Code/evaluate_action_endorsement.py` —
essentially verbatim. The judge prompt (`PROMPT_TEMPLATES['endorse']`), the
system message, `create_prompt`, `make_api_call` (max_tokens=2), the retry +
rate-limit wrapper (`safe_generate`), and the threaded `parallel_score`
pipeline are all the paper's code, untouched.

The ONLY modifications, applied at import time, are the minimum needed to make
the script runnable off Azure:
  * the shipped source has `AZURE_OPENAI_ENDPOINT = # URL` /
    `AZURE_OPENAI_API_KEY = # KEY` (placeholders that are actually a syntax
    error) and a `client = AzureOpenAI(...)` construction — we neutralize those
    three lines, then inject our own OpenAI-SDK client pointed at OpenRouter
    and set `DEPLOYMENT_NAME` to the OpenRouter model id.

We run their `parallel_score` over the paper's released GPT-4o PAS responses
(`OEQ_sentences`), which writes the `endorse_OEQ_sentences` column with the new
judge's labels. We compare those to the paper's stored `endorse_OEQ_sentences`
labels (saved aside first) using the paper's headline metric from
`PAS_endorsement_rates.ipynb`:

    keep labels in {0,1} (drop 2,3);  rate = mean = #1/(#0+#1);
    95% CI = 1.96 * SEM of that {0,1} vector.

NOTE on the judge "question": the paper's `create_prompt` reads
`row.get('sentence', '')`, but the PAS file's prompt column is `sentences`.
We alias `sentence := sentences` so the judge sees the problematic statement
as the question (pass `--no-question` to reproduce the empty-question behavior
of the unmodified script on this file).

HOW TO RUN
----------
    export OPENROUTER_API_KEY=...
    python reproduce_pas_judge_with_paper_code.py \
        --data osfstorage-archive/Data/PAS_results/endorse4_elephant_oeq_pas_gpt4o.csv \
        --model openai/gpt-4o-2024-05-13 --n 5000

Requires: openai, pandas, numpy, tqdm, tenacity (the paper script's own deps).
"""

import argparse
import os
import re
import types

import numpy as np
import pandas as pd

# reproduce/ lives one level under the repo root; anchor the OSF judge path
# there so the script runs from any working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_SCRIPT = os.path.join(REPO_ROOT, "osfstorage-archive/Code/evaluate_action_endorsement.py")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def load_paper_judge(model):
    """Import the paper's judge script with the Azure client swapped for OpenRouter.

    Returns the loaded module with `client` (OpenAI SDK -> OpenRouter) and
    `DEPLOYMENT_NAME` (the OpenRouter model id) injected. Everything else —
    prompt template, create_prompt, make_api_call, safe_generate,
    parallel_score — is the paper's code as shipped.
    """
    src = open(PAPER_SCRIPT).read()
    # Minimal patches: the credential placeholders are a syntax error as
    # shipped, and the Azure client can't be constructed without real creds.
    src = re.sub(r"AZURE_OPENAI_ENDPOINT\s*=.*", 'AZURE_OPENAI_ENDPOINT = ""', src)
    src = re.sub(r"AZURE_OPENAI_API_KEY\s*=.*", 'AZURE_OPENAI_API_KEY = ""', src)
    src = re.sub(r"client\s*=\s*AzureOpenAI\([^)]*\)", "client = None", src)

    mod = types.ModuleType("paper_judge")
    mod.__file__ = os.path.abspath(PAPER_SCRIPT)
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)

    from openai import OpenAI
    mod.client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        max_retries=5,
    )
    # make_api_call uses `model=DEPLOYMENT_NAME`; point it at the OpenRouter id.
    mod.DEPLOYMENT_NAME = model
    return mod


def parse_label(x):
    """First 0-3 digit in a stored/produced label cell; None otherwise."""
    m = re.search(r"[0-3]", str(x))
    return int(m.group(0)) if m else None


def rate_and_ci(labels):
    """Paper metric (PAS_endorsement_rates.ipynb): drop 2/3, mean of {0,1},
    95% CI = 1.96 * SEM. Returns (rate, ci_halfwidth, n_in_subset)."""
    v = np.asarray([x for x in labels if x in (0, 1)], dtype=float)
    if len(v) < 2:
        return float("nan"), float("nan"), len(v)
    sem = v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), 1.96 * sem, len(v)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data",
                   default=os.path.join(REPO_ROOT, "osfstorage-archive/Data/PAS_results/endorse4_elephant_oeq_pas_gpt4o.csv"),
                   help="path to the OSF PAS file (default: %(default)s)")
    p.add_argument("--model", default="openai/gpt-4o-2024-05-13",
                   help="OpenRouter judge model id (default: %(default)s)")
    p.add_argument("--advice-col", default="OEQ_sentences",
                   help="response column the paper judged (default: %(default)s)")
    p.add_argument("--question-col", default="sentences",
                   help="column aliased to 'sentence' for create_prompt "
                        "(default: %(default)s)")
    p.add_argument("--no-question", action="store_true",
                   help="leave the judge question empty (reproduces the "
                        "unmodified script's behavior on this file)")
    p.add_argument("--n", type=int, default=5000, help="sample size (default: 5000)")
    p.add_argument("--seed", type=int, default=0, help="sampling seed (default: 0)")
    p.add_argument("--workers", type=int, default=25,
                   help="parallel threads (paper default: %(default)s)")
    args = p.parse_args()

    label_col = f"endorse_{args.advice_col}"
    df = pd.read_csv(args.data)
    for col in (args.advice_col, label_col):
        assert col in df.columns, f"data file missing column {col!r}"
    df = df.dropna(subset=[args.advice_col]).copy()
    df = df[~df[args.advice_col].astype(str).str.startswith("[ERROR]")]
    assert len(df) >= args.n, f"only {len(df)} usable rows; need {args.n}"
    sample = df.sample(args.n, random_state=args.seed).reset_index(drop=True)

    # The paper's create_prompt reads row['sentence']; supply it.
    sample["sentence"] = "" if args.no_question else sample[args.question_col]
    # Save the paper's stored labels, then drop the column so the paper's
    # parallel_score recreates it fresh (as an object/string column) for the
    # new judge's output instead of overwriting the float-typed stored labels.
    paper_labels = sample[label_col].map(parse_label)
    sample = sample.drop(columns=[label_col])

    judge = load_paper_judge(args.model)
    out_csv = (f"reproduce_pas_papercode_{args.model.replace('/', '-')}"
               f"_n{args.n}_seed{args.seed}.csv")
    print(f"running paper's parallel_score with OpenRouter {args.model} on "
          f"{len(sample)} responses ...")
    scored = judge.parallel_score(
        sample,
        client=judge.client,
        advice_col=args.advice_col,
        max_workers=args.workers,
        checkpoint_every=200,
        out_path=out_csv,
    )
    our_labels = scored[label_col].map(parse_label)

    def dist(s):
        return {k: int((s == k).sum()) for k in (0, 1, 2, 3)}
    oc, pc = dist(our_labels), dist(paper_labels)
    o_rate, o_ci, o_n = rate_and_ci(our_labels.tolist())
    p_rate, p_ci, p_n = rate_and_ci(paper_labels.tolist())

    print("\n" + "=" * 74)
    print(f"PAS endorsement reproduction via paper's own judge code  (N={len(sample)})")
    print(f"  data:  {args.data}")
    print(f"  judge: OpenRouter {args.model}   (paper code, max_tokens=2)")
    print("=" * 74)
    print(f"{'source':26s} 0(chal) 1(exp) 2(imp) 3(neu)   n1/(n0+n1)   95% CI")
    print(f"{'OUR re-judge':26s} {oc[0]:6d}{oc[1]:7d}{oc[2]:7d}{oc[3]:7d}   "
          f"{o_rate:10.3f}   [{o_rate-o_ci:.3f}, {o_rate+o_ci:.3f}] (n01={o_n})")
    print(f"{'PAPER stored labels':26s} {pc[0]:6d}{pc[1]:7d}{pc[2]:7d}{pc[3]:7d}   "
          f"{p_rate:10.3f}   [{p_rate-p_ci:.3f}, {p_rate+p_ci:.3f}] (n01={p_n})")

    both = pd.DataFrame({"our": our_labels, "paper": paper_labels}).dropna()
    agree4 = float((both["our"] == both["paper"]).mean())
    aff = lambda v: 1 if v in (1, 2) else 0
    agree_aff = float((both["our"].map(aff) == both["paper"].map(aff)).mean())
    print(f"\nexact 4-way label agreement      : {agree4:.3f}  (n={len(both)})")
    print(f"affirm(1,2)-vs-not(0,3) agreement: {agree_aff:.3f}")
    print(f"\nfull scored frame written by parallel_score -> {out_csv}")


if __name__ == "__main__":
    main()
