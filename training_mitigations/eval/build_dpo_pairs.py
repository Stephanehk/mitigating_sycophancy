"""Stage 3: turn labelled candidates into DPO preference pairs.

Pairing rule (within-prompt, which is the whole point):
    rejected = a candidate labelled 1 or 2 for this prompt
    chosen   = a candidate labelled 0 for the SAME prompt, drawn uniformly at
               random (seeded)
one pair per rejected candidate, capped at --max-pairs-per-prompt.

Pairing within a prompt is what makes this worth doing as DPO rather than SFT:
the DPO gradient acts on the DIFFERENCE between chosen and rejected for a shared
prompt, so anything the two responses have in common cancels and the update
targets the thing that actually differs -- the voice. Absolute-likelihood
training on the survivors alone would instead import every correlate of
"survived the filter", including blandness.

Both sides must be COMPLETE (finished=True). A truncated response teaches
mid-sentence stopping, and a systematic truncation asymmetry between chosen and
rejected is a length signal DPO will happily learn instead of the voice signal.
For the same reason the manifest reports the chosen/rejected length balance --
check it before believing any result.

Prompts contribute nothing when either pool is empty (no failures to learn from,
or no clean response to pair against). Both counts are recorded rather than
silently dropped: they say how much of the train set the round actually saw.

Output:
    <out-dir>/pairs/pairs.jsonl    {"prompt","chosen","rejected"} per line
    <out-dir>/pairs/manifest.json  every count above + length balance

Usage:
    python eval/build_dpo_pairs.py --out-dir /projects/m000254/runs/<new-arm>
"""

import argparse
import csv
import json
import os
import random
import statistics

csv.field_size_limit(10**9)


def load_candidates(path):
    """-> {(id, sample_idx): row}, last write wins (stage 1 rewrites on resume)."""
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[(str(r["id"]), int(r["sample_idx"]))] = r
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-pairs-per-prompt", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--require-finished", type=int, default=1,
                    help="1 (default): both sides must have hit EOS")
    args = ap.parse_args()

    pdir = os.path.join(args.out_dir, "pairs")
    cands = load_candidates(os.path.join(pdir, "candidates.jsonl"))
    labels = {}
    with open(os.path.join(pdir, "labels.csv")) as f:
        for r in csv.DictReader(f):
            labels[(str(r["id"]), int(r["sample_idx"]))] = int(r["label"])

    # Group by prompt.
    by_prompt = {}
    for key, row in cands.items():
        lab = labels.get(key)
        if lab is None:
            continue
        by_prompt.setdefault(key[0], {"sentence": row["sentence"],
                                      "chosen": [], "rejected": [],
                                      "n_seen": 0, "n_trunc": 0})
        g = by_prompt[key[0]]
        g["n_seen"] += 1
        if args.require_finished and not row.get("finished", False):
            g["n_trunc"] += 1
            continue
        if lab == 0:
            g["chosen"].append(row["response"])
        elif lab in (1, 2):
            g["rejected"].append(row["response"])
        # -1 (unparsed) is dropped on purpose: never guess a label.

    rng = random.Random(args.seed)
    pairs = []
    n_no_rejected = n_no_chosen = n_capped = 0
    clen, rlen = [], []

    for pid in sorted(by_prompt):
        g = by_prompt[pid]
        if not g["rejected"]:
            n_no_rejected += 1
            continue
        if not g["chosen"]:
            n_no_chosen += 1
            continue
        rejected = g["rejected"]
        if len(rejected) > args.max_pairs_per_prompt:
            rejected = rng.sample(rejected, args.max_pairs_per_prompt)
            n_capped += 1
        for rej in rejected:
            ch = rng.choice(g["chosen"])
            pairs.append({"prompt": g["sentence"], "chosen": ch, "rejected": rej})
            clen.append(len(ch))
            rlen.append(len(rej))

    rng.shuffle(pairs)
    out_path = os.path.join(pdir, "pairs.jsonl")
    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    lab_counts = {}
    for v in labels.values():
        lab_counts[v] = lab_counts.get(v, 0) + 1

    def stats(xs):
        if not xs:
            return {"n": 0}
        return {"n": len(xs), "mean_chars": round(statistics.mean(xs), 1),
                "median_chars": statistics.median(xs)}

    manifest = {
        "n_candidates": len(cands),
        "n_labelled": len(labels),
        "label_counts": {str(k): v for k, v in sorted(lab_counts.items())},
        "n_prompts_seen": len(by_prompt),
        "n_prompts_contributing": len(by_prompt) - n_no_rejected - n_no_chosen,
        "n_prompts_dropped_no_failure": n_no_rejected,
        "n_prompts_dropped_no_clean": n_no_chosen,
        "n_prompts_hit_pair_cap": n_capped,
        "n_truncated_dropped": sum(g["n_trunc"] for g in by_prompt.values()),
        "max_pairs_per_prompt": args.max_pairs_per_prompt,
        "require_finished": bool(args.require_finished),
        "seed": args.seed,
        "n_pairs": len(pairs),
        # Length balance: DPO's best-known failure mode is learning "be shorter"
        # when rejected responses are systematically longer. If these diverge,
        # any downstream result is suspect.
        "chosen_length": stats(clen),
        "rejected_length": stats(rlen),
    }
    with open(os.path.join(pdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    print(f"[pairs] wrote {len(pairs)} pairs -> {out_path}")
    if pairs and clen and rlen:
        delta = statistics.mean(rlen) - statistics.mean(clen)
        rel = delta / max(statistics.mean(clen), 1)
        if abs(rel) > 0.15:
            print(f"[pairs] WARNING: rejected responses average {rel:+.1%} "
                  f"vs chosen in length. DPO can satisfy this objective by "
                  f"changing length alone — check the held-out length shift.")
    assert pairs, ("0 pairs built — nothing to train on. Check the label "
                   "distribution above.")


if __name__ == "__main__":
    main()
