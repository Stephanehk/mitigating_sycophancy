"""Summarize Skywork RM scores: mean reward per method per dataset, plus
cross-RM rank agreement.

The two RMs have unrelated output scales, so their raw means are NEVER pooled
or averaged across RMs — they are reported side by side and compared only by
the RANK they induce over methods within a dataset.

`trunc_at_cap` is the fraction of responses at the generation token cap (512 for
OEQ/PAS, 1536 for AITA). It is reported next to mean reward because reward
models penalize mid-sentence cutoffs, so any arm ordering that tracks this
column is suspect.
"""
import csv, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_reward_models import ENVS, RUNS, conditions

csv.field_size_limit(10**9)
RMS = ["llama", "qwen"]
CAP = {"OEQ": 512, "PAS": 512, "AITA": 1536}

SHORT = {
    "base": "base", "base-becritical": "becritical", "base-explicit": "explicit",
    "base-sharma": "sharma", "base-dontbesyco": "dontbesyco",
    "base-perspshiftassess": "perspshiftassess",
    "base-besycophantic": "besycophantic (CTRL)",
}
def short(label):
    if label in SHORT:
        return SHORT[label]
    for k, v in [("P6_perspshift", "P6_rpo0p5 (DPO)"), ("neutral_mixed", "neutral_mixed (DPO)"),
                 ("neutral", "neutral (DPO)"), ("sft_lora", "becritical_sft (SFT)")]:
        if k in label:
            return v
    return label


def load(rundir, tag, rm):
    f = f"{RUNS}/{rundir}/eval/rm_{rm}_{tag}.csv"
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        return {r["id"]: (float(r["score"]), int(r["n_resp_tok"]),
                          int(r["rm_input_truncated"])) for r in csv.DictReader(fh)}


def spearman(a, b):
    def rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):  # average ties
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    den = (sum((x - ma) ** 2 for x in ra) * sum((x - mb) ** 2 for x in rb)) ** 0.5
    return num / den if den else float("nan")


def main():
    for env in ["OEQ", "PAS", "AITA"]:
        conds = conditions(env)
        rows, means = [], {rm: [] for rm in RMS}
        for label, rundir, tag in conds:
            rec = {"method": short(label)}
            ok = True
            for rm in RMS:
                d = load(rundir, tag, rm)
                if d is None:
                    ok = False; break
                sc = [v[0] for v in d.values()]
                rec[f"{rm}_mean"] = statistics.mean(sc)
                rec[f"{rm}_sem"] = (statistics.stdev(sc) / len(sc) ** 0.5) if len(sc) > 1 else 0.0
                rec["n"] = len(sc)
                rec["trunc"] = sum(1 for v in d.values() if v[1] >= CAP[env] - 1) / len(d)
                rec["rm_trunc"] = sum(v[2] for v in d.values())
            if not ok:
                continue
            for rm in RMS:
                means[rm].append(rec[f"{rm}_mean"])
            rows.append(rec)

        if not rows:
            print(f"\n### {env}: no RM scores yet\n"); continue

        print(f"\n### {env}  (n={rows[0]['n']} held-out items/arm, cap={CAP[env]} tok)")
        print(f"{'method':22s} {'llama_mean':>16s} {'qwen_mean':>16s} {'trunc@cap':>10s}")
        for r in sorted(rows, key=lambda x: -x["llama_mean"]):
            print(f"{r['method']:22s} {r['llama_mean']:+8.3f} ±{r['llama_sem']:.3f} "
                  f"{r['qwen_mean']:+8.3f} ±{r['qwen_sem']:.3f} {100*r['trunc']:9.1f}%")
        rho = spearman(means["llama"], means["qwen"])
        print(f"cross-RM rank agreement (Spearman over {len(rows)} methods): rho = {rho:+.3f}")
        nrt = sum(r["rm_trunc"] for r in rows)
        if nrt:
            print(f"WARNING: {nrt} responses hit the RM's own max_length")


if __name__ == "__main__":
    main()
