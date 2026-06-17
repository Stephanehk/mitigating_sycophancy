"""Split a samples_scored CSV into blind labeling batch files.

Each batch file lists items as plain markdown with ONLY a global item id,
the env, the prompt, and the response — no veto/judge/my_label columns — so
a labeler (human or subagent) judges sycophancy without anchoring.

Item id = "<csv basename without extension>:<row index>" so labels can be
merged back unambiguously.

Usage:
    python make_blind_batches.py <samples_scored.csv> --batch-size 50
Writes <csv dir>/blind_<csv stem>_batch<k>.md
"""

import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    stem = os.path.splitext(os.path.basename(args.csv))[0]
    out_dir = os.path.dirname(os.path.abspath(args.csv))
    paths = []
    for k, start in enumerate(range(0, len(df), args.batch_size)):
        chunk = df.iloc[start:start + args.batch_size]
        path = os.path.join(out_dir, f"blind_{stem}_batch{k}.md")
        with open(path, "w") as f:
            for i, r in chunk.iterrows():
                f.write(f"\n===== ITEM {stem}:{i} (env={r['env']}) =====\n\n"
                        f"PROMPT:\n{r['sentence']}\n\n"
                        f"RESPONSE:\n{r['response']}\n")
        paths.append(path)
        print(f"wrote {path} ({len(chunk)} items)")
    return paths


if __name__ == "__main__":
    main()
