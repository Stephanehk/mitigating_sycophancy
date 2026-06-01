"""Sanity check: re-judge the paper's own PAS responses with a chosen judge.

Takes a sample of the paper's released GPT-4o PAS responses (the `OEQ_sentences`
column in the OSF PAS file, whose published judge labels live in
`endorse_OEQ_sentences`), re-judges each (question=`sentences`,
advice=`OEQ_sentences`) pair with `--judge_model`, and compares our labels
against the paper's stored labels on the SAME rows.

Purpose: isolate judge-snapshot drift from generation drift. If a pinned 2024
gpt-4o snapshot reproduces the paper's labels here, the headline-rate gap we
saw on freshly generated responses is attributable to generation, not judging.

The (question, advice) pairing and judge prompt are identical to the main
pipeline's PAS judging (`explore_mitigations.judge_endorsement`), so this is
apples-to-apples with our own runs.

Run from the repo root:

    python mitigations/sanity_rejudge_paper_pas.py --judge_model openai/gpt-4o-2024-08-06
"""

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# explore_mitigations lives in the sibling mitigations/ package.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mitigations"))

from explore_mitigations import (
    judge_endorsement, make_client, model_pricing, model_slug,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Paper's released GPT-4o PAS responses + their published judge labels.
PAPER_CSV = os.path.join(
    REPO_ROOT,
    "osfstorage-archive/Data/PAS_results/endorse4_elephant_oeq_pas_gpt4o.csv",
)
QUESTION_COL = "sentences"            # the problematic first-person statement
RESPONSE_COL = "OEQ_sentences"        # the model response the paper judged
PAPER_LABEL_COL = "endorse_OEQ_sentences"  # the paper's stored judge label


def first_digit(x):
    """Parse the first 0-3 digit from a stored label cell; None if absent."""
    m = re.search(r"[0-3]", str(x))
    return int(m.group(0)) if m else None


def label_dist(series):
    """Return (counts dict for 0-3, n_nan, n1/(n0+n1))."""
    c = {k: int((series == k).sum()) for k in (0, 1, 2, 3)}
    n_nan = int(series.isna().sum())
    denom = c[0] + c[1]
    rate = c[1] / denom if denom else float("nan")
    return c, n_nan, rate


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--judge_model", default="openai/gpt-4o-2024-08-06",
                   help="judge model id to evaluate (default: %(default)s)")
    p.add_argument("--n", type=int, default=100, help="sample size (default: 100)")
    p.add_argument("--seed", type=int, default=0,
                   help="sampling seed; keep fixed to compare across judges "
                        "(default: 0)")
    p.add_argument("--workers", type=int, default=10,
                   help="parallel judge calls (default: 10)")
    args = p.parse_args()

    assert os.path.exists(PAPER_CSV), f"missing paper PAS csv: {PAPER_CSV}"
    df = pd.read_csv(PAPER_CSV)
    for col in (QUESTION_COL, RESPONSE_COL, PAPER_LABEL_COL, "problematic_to_affirm"):
        assert col in df.columns, f"paper csv missing column {col!r}"

    df = df.dropna(subset=[QUESTION_COL, RESPONSE_COL]).copy()
    df = df[~df[RESPONSE_COL].astype(str).str.startswith("[ERROR]")]
    df = df[df["problematic_to_affirm"] == 1]
    assert len(df) >= args.n, f"only {len(df)} eligible rows; need {args.n}"
    sample = df.sample(args.n, random_state=args.seed).reset_index(drop=True)
    sample["paper_label"] = sample[PAPER_LABEL_COL].map(first_digit)

    # Output + checkpoint paths are scoped by (judge, n, seed) so different
    # sample sizes don't clobber each other and a run is resumable.
    slug = model_slug(args.judge_model)
    stem = f"sanity_paper_pas_rejudge_{slug}_n{args.n}_seed{args.seed}"
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    out = os.path.join(results_dir, stem + ".csv")
    ckpt = os.path.join(results_dir, stem + ".checkpoint.jsonl")

    # Resume: each checkpoint line is {"i", "label", "in", "out"} for one
    # completed sample position. Positions are stable given fixed (n, seed).
    done = {}
    if os.path.exists(ckpt):
        with open(ckpt) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done[rec["i"]] = rec
        print(f"resuming: {len(done)} of {len(sample)} already judged in {ckpt}")

    client = make_client(args.judge_model)
    lock = threading.Lock()
    todo = [i for i in range(len(sample)) if i not in done]

    def work(i):
        r = sample.iloc[i]
        label, j_in, j_out = judge_endorsement(
            client, args.judge_model, r[QUESTION_COL], r[RESPONSE_COL],
        )
        return {"i": i, "label": label, "in": j_in, "out": j_out}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, i) for i in todo]
        n_completed = 0
        for f in as_completed(futs):
            rec = f.result()
            done[rec["i"]] = rec
            with lock:
                with open(ckpt, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            n_completed += 1
            if n_completed % 250 == 0:
                print(f"  judged {n_completed}/{len(todo)} this run")

    sample["our_label"] = [done[i]["label"] for i in range(len(sample))]
    in_tok = sum(done[i]["in"] for i in range(len(sample)))
    out_tok = sum(done[i]["out"] for i in range(len(sample)))

    oc, onan, orate = label_dist(sample["our_label"])
    pc, pnan, prate = label_dist(sample["paper_label"])

    def ci95(series):
        """Paper-style 95% CI half-width: 1.96 * SEM over the {0,1} vector."""
        import scipy.stats as st
        v = series[series.isin([0, 1])].astype(float).values
        return 1.96 * st.sem(v) if len(v) > 1 else float("nan")

    o_ci = ci95(sample["our_label"])
    p_ci = ci95(sample["paper_label"])
    print(f"\nN judged = {len(sample)}  judge = {args.judge_model}  "
          f"(paper GPT-4o PAS responses)")
    print(f"{'':26s} 0(chal) 1(exp) 2(imp) 3(neu) nan   n1/(n0+n1)  95% CI")
    print(f"OUR judge ({slug[:14]:14s}): "
          f"{oc[0]:5d} {oc[1]:6d} {oc[2]:6d} {oc[3]:6d} {onan:4d}   {orate:.3f}    "
          f"[{orate-o_ci:.3f}, {orate+o_ci:.3f}]")
    print(f"PAPER stored labels        : "
          f"{pc[0]:5d} {pc[1]:6d} {pc[2]:6d} {pc[3]:6d} {pnan:4d}   {prate:.3f}    "
          f"[{prate-p_ci:.3f}, {prate+p_ci:.3f}]")

    both = sample.dropna(subset=["our_label", "paper_label"])
    agree4 = (both["our_label"] == both["paper_label"]).mean()
    to_aff = lambda v: 1 if v in (1, 2) else 0
    agree_aff = (both["our_label"].map(to_aff)
                 == both["paper_label"].map(to_aff)).mean()
    print(f"\nexact 4-way label agreement     : {agree4:.3f} on {len(both)} rows")
    print(f"affirm(1,2)-vs-not(0,3) agreement: {agree_aff:.3f}")

    p_in, p_out = model_pricing(args.judge_model)
    cost = in_tok / 1e6 * p_in + out_tok / 1e6 * p_out
    print(f"\njudge tokens: in={in_tok:,} out={out_tok:,}  "
          f"cost @ {args.judge_model} = ${cost:.4f}")

    sample[[QUESTION_COL, RESPONSE_COL, "paper_label", "our_label"]].to_csv(
        out, index=False)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
