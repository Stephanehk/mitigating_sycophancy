#!/usr/bin/env python3
"""Re-judge the paper's stored PAS responses with the paper's OWN Azure judge.

WHAT THIS DOES
--------------
This is the Azure-OpenAI counterpart to `reproduce_pas_judge_with_paper_code.py`.
It imports and runs the paper's actual scoring script —
`osfstorage-archive/Code/evaluate_action_endorsement.py` — essentially verbatim,
including its `AzureOpenAI` client (`api_version='2024-02-15-preview'`,
deployment `gpt-4o`), prompt template, `create_prompt`, `make_api_call`
(max_tokens=2), the retry + rate-limit wrapper, and the threaded
`parallel_score` pipeline. This is the same Azure path the paper used.

The ONLY edits, applied at import time, are the minimum needed to make the
shipped source compile and authenticate: the released file has
`AZURE_OPENAI_ENDPOINT = # URL` and `AZURE_OPENAI_API_KEY = # KEY` (blank
placeholders that are a syntax error). We rewrite those two lines to read the
real values from the environment; the `client = AzureOpenAI(...)` construction
and `api_version` are left exactly as shipped.

We run the paper's `parallel_score` over two of the paper's released PAS files
from `osfstorage-archive/Data/PAS_results/`:
  * GPT-4o:   advice column `OEQ_sentences`, stored labels `endorse_OEQ_sentences`
  * DeepSeek: advice column `response`,      stored labels `endorse_response`
For each, we sample N rows, re-judge, and compare our labels to the stored
labels on the SAME rows using the paper's headline metric from
`PAS_endorsement_rates.ipynb`:

    keep labels in {0,1} (drop 2,3);  rate = mean = #1/(#0+#1);
    95% CI = 1.96 * SEM of that {0,1} vector.

NOTE on the judge "question": the paper's `create_prompt` reads
`row.get('sentence', '')`, but the PAS files' prompt column is `sentences`.
We alias `sentence := sentences` so the judge sees the problematic statement as
the question (pass `--no-question` to reproduce the empty-question behavior of
the unmodified script run straight off these files).

HOW TO RUN  (NOT run automatically -- credentials required)
-----------------------------------------------------------
    export AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
    export AZURE_OPENAI_API_KEY=...
    python reproduce_pas_judge_azure.py --n 5000

    # judge only one dataset, or change the Azure deployment name:
    python reproduce_pas_judge_azure.py --datasets gpt4o --deployment gpt-4o --n 5000

Requires: openai (with Azure support), pandas, numpy, tqdm, tenacity.
"""

import argparse
import os
import re
import types

import numpy as np
import pandas as pd

# reproduce/ lives one level under the repo root; anchor OSF paths there so
# the script runs from any working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_SCRIPT = os.path.join(REPO_ROOT, "osfstorage-archive/Code/evaluate_action_endorsement.py")
PAS_DIR = os.path.join(REPO_ROOT, "osfstorage-archive/Data/PAS_results")

# Each entry: (csv path, advice/response column the paper judged).
# The stored-label column is always f"endorse_{advice_col}".
DATASETS = {
    "gpt4o": (os.path.join(PAS_DIR, "endorse4_elephant_oeq_pas_gpt4o.csv"), "OEQ_sentences"),
    "deepseek": (os.path.join(PAS_DIR, "endorse4_elephant_oeq_pas_deepseek.csv"), "response"),
}
QUESTION_COL = "sentences"  # the problematic first-person statement


def load_paper_judge(deployment):
    """Import the paper's judge script with Azure credentials filled from env.

    The shipped source compiles its own `AzureOpenAI` client at module level
    using `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_API_KEY`, which ship blank
    (a syntax error). We rewrite only those two assignment lines to read from
    the environment, then exec the module so its own `client = AzureOpenAI(...)`
    (api_version='2024-02-15-preview') runs unchanged. `DEPLOYMENT_NAME` is set
    to the requested Azure deployment after exec (read at call time by
    `make_api_call`). Everything else is the paper's code as shipped.
    """
    src = open(PAPER_SCRIPT).read()
    src = re.sub(r"AZURE_OPENAI_ENDPOINT\s*=.*",
                 'AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]', src)
    src = re.sub(r"AZURE_OPENAI_API_KEY\s*=.*",
                 'AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]', src)

    assert os.environ.get("AZURE_OPENAI_ENDPOINT"), "set AZURE_OPENAI_ENDPOINT"
    assert os.environ.get("AZURE_OPENAI_API_KEY"), "set AZURE_OPENAI_API_KEY"

    mod = types.ModuleType("paper_judge_azure")
    mod.__file__ = os.path.abspath(PAPER_SCRIPT)
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)

    mod.DEPLOYMENT_NAME = deployment
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


def run_dataset(judge, name, data_path, advice_col, args):
    """Sample, re-judge with the paper's parallel_score, and report one dataset.

    Returns nothing; prints the comparison table for this dataset and writes the
    full scored frame to a per-dataset CSV.
    """
    label_col = f"endorse_{advice_col}"
    assert os.path.exists(data_path), f"missing data file: {data_path}"
    df = pd.read_csv(data_path)
    for col in (QUESTION_COL, advice_col, label_col):
        assert col in df.columns, f"{data_path} missing column {col!r}"

    df = df.dropna(subset=[advice_col]).copy()
    df = df[~df[advice_col].astype(str).str.startswith("[ERROR]")]
    assert len(df) >= args.n, f"{name}: only {len(df)} usable rows; need {args.n}"
    sample = df.sample(args.n, random_state=args.seed).reset_index(drop=True)

    # The paper's create_prompt reads row['sentence']; supply it.
    sample["sentence"] = "" if args.no_question else sample[QUESTION_COL]
    # Save the stored labels, then drop the column so parallel_score recreates
    # it fresh (object dtype) for the new judge rather than overwriting floats.
    paper_labels = sample[label_col].map(parse_label)
    sample = sample.drop(columns=[label_col])

    out_csv = f"reproduce_pas_azure_{name}_{args.deployment}_n{args.n}_seed{args.seed}.csv"
    print(f"\n[{name}] running paper's parallel_score on Azure deployment "
          f"{args.deployment!r} over {len(sample)} responses -> {out_csv}")
    scored = judge.parallel_score(
        sample,
        client=judge.client,
        advice_col=advice_col,
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

    print("=" * 78)
    print(f"PAS endorsement via paper's Azure judge  [{name}]  (N={len(sample)})")
    print(f"  data:  {data_path}")
    print(f"  judge: Azure deployment {args.deployment} "
          f"(api_version=2024-02-15-preview, paper code, max_tokens=2)")
    print("=" * 78)
    print(f"{'source':26s} 0(chal) 1(exp) 2(imp) 3(neu)   n1/(n0+n1)   95% CI")
    print(f"{'OUR re-judge':26s} {oc[0]:6d}{oc[1]:7d}{oc[2]:7d}{oc[3]:7d}   "
          f"{o_rate:10.3f}   [{o_rate-o_ci:.3f}, {o_rate+o_ci:.3f}] (n01={o_n})")
    print(f"{'PAPER stored labels':26s} {pc[0]:6d}{pc[1]:7d}{pc[2]:7d}{pc[3]:7d}   "
          f"{p_rate:10.3f}   [{p_rate-p_ci:.3f}, {p_rate+p_ci:.3f}] (n01={p_n})")

    both = pd.DataFrame({"our": our_labels, "paper": paper_labels}).dropna()
    agree4 = float((both["our"] == both["paper"]).mean())
    aff = lambda v: 1 if v in (1, 2) else 0
    agree_aff = float((both["our"].map(aff) == both["paper"].map(aff)).mean())
    print(f"exact 4-way label agreement      : {agree4:.3f}  (n={len(both)})")
    print(f"affirm(1,2)-vs-not(0,3) agreement: {agree_aff:.3f}")
    print(f"full scored frame -> {out_csv}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["gpt4o", "deepseek"],
                   choices=sorted(DATASETS), help="which PAS files to judge")
    p.add_argument("--deployment", default="gpt-4o",
                   help="Azure deployment name (default: %(default)s, as shipped)")
    p.add_argument("--no-question", action="store_true",
                   help="leave the judge question empty (reproduces the "
                        "unmodified script's behavior on these files)")
    p.add_argument("--n", type=int, default=5000, help="sample size per dataset (default: 5000)")
    p.add_argument("--seed", type=int, default=0, help="sampling seed (default: 0)")
    p.add_argument("--workers", type=int, default=25,
                   help="parallel threads (paper default: %(default)s)")
    args = p.parse_args()

    judge = load_paper_judge(args.deployment)
    for name in args.datasets:
        data_path, advice_col = DATASETS[name]
        run_dataset(judge, name, data_path, advice_col, args)


if __name__ == "__main__":
    main()
