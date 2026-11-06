"""MMLU capability-loss table for the rebuttal: accuracy of Qwen3-8B base and
the four "Our Model (3+1)" adapters, each tested against base with an exact
paired McNemar test on per-question correctness (same protocol as the IFEval
table: binomial test on the discordant pairs).

Reads the output of run_mmlu_qwen3_preempt.sbatch / next_mmlu_qwen3.sbatch:
    <root>/<name>/<model_sanitized>/samples_mmlu_<subject>_<ts>.jsonl
(one line per question, lm-eval 0.4 --log_samples format: `doc_id` + `acc`).

    python training_mitigations/eval/mmlu_mcnemar.py --root <runs>/mmlu_qwen3-8b

Prints micro accuracy (over all 14,042 questions; lm-eval's `mmlu` group
number is also micro-averaged), the lm-eval reported group accuracy as a
cross-check, b = questions base got right and the model got wrong, c = the
reverse, and the two-sided exact p. Holm-adjusted p across the four models is
included because four tests share one base.
"""

import argparse
import glob
import json
import os
import re
import sys

from scipy.stats import binomtest

NAMES = ["base", "oeq", "pas", "aita", "flip"]
N_EXPECTED = 14042


def load_correct(model_dir):
    """{(subject, doc_id): 0/1} from the newest samples file per subject."""
    files = glob.glob(os.path.join(model_dir, "*", "samples_mmlu_*.jsonl"))
    newest = {}
    for f in files:
        m = re.match(r"samples_(mmlu_.+?)_(\d{4}-\d{2}-\d{2}T.+)\.jsonl$", os.path.basename(f))
        assert m, f"unrecognised samples filename {f}"
        subj, ts = m.groups()
        if subj not in newest or ts > newest[subj][0]:
            newest[subj] = (ts, f)
    out = {}
    for subj, (_, f) in newest.items():
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                out[(subj, int(r["doc_id"]))] = int(round(float(r["acc"])))
    return out


def reported_acc(model_dir):
    res = sorted(glob.glob(os.path.join(model_dir, "*", "results_*.json")))
    if not res:
        return None
    return json.load(open(res[-1]))["results"].get("mmlu", {}).get("acc,none")


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dir holding base/ oeq/ pas/ aita/ flip/")
    args = ap.parse_args()

    data = {}
    for n in NAMES:
        d = os.path.join(args.root, n)
        if not os.path.isdir(d):
            print(f"[mmlu] {n}: missing ({d})", file=sys.stderr)
            continue
        data[n] = load_correct(d)
        if len(data[n]) != N_EXPECTED:
            print(f"[mmlu] WARNING {n}: {len(data[n])} questions, expected {N_EXPECTED}",
                  file=sys.stderr)
    assert "base" in data, "base results are required for the paired test"
    base = data["base"]

    rows, ps = [], []
    for n in NAMES:
        if n not in data:
            continue
        cur = data[n]
        acc = sum(cur.values()) / len(cur)
        rep = reported_acc(os.path.join(args.root, n))
        if n == "base":
            rows.append((n, len(cur), acc, rep, None, None, None))
            continue
        keys = base.keys() & cur.keys()
        assert len(keys) == len(base), f"{n}: question sets differ from base"
        b = sum(1 for k in keys if base[k] == 1 and cur[k] == 0)
        c = sum(1 for k in keys if base[k] == 0 and cur[k] == 1)
        p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        rows.append((n, len(cur), acc, rep, b, c, p))
        ps.append(p)

    adj = iter(holm(ps)) if ps else iter([])
    base_acc = rows[0][2]
    print(f"{'model':<6} {'n':>6} {'acc':>7} {'d(base)':>8} {'lm-eval':>8} "
          f"{'b(lost)':>8} {'c(won)':>7} {'p':>8} {'p_holm':>7}")
    for n, cnt, acc, rep, b, c, p in rows:
        rep_s = f"{rep:.4f}" if rep is not None else "-"
        if n == "base":
            print(f"{n:<6} {cnt:>6} {acc:>7.4f} {'-':>8} {rep_s:>8} {'-':>8} {'-':>7} {'-':>8} {'-':>7}")
        else:
            print(f"{n:<6} {cnt:>6} {acc:>7.4f} {acc - base_acc:>+8.4f} {rep_s:>8} "
                  f"{b:>8} {c:>7} {p:>8.3g} {next(adj):>7.3g}")


if __name__ == "__main__":
    main()
