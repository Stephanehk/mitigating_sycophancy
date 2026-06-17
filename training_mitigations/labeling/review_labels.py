"""Interactive CLI to confirm/override the manual bad-vs-ok labels in the
iter3_veto_judge_labels sample CSVs.

For each row it prints the prompt, the full response, the training-veto
status, the judge label, and my label + note, then asks for a verdict:

    [enter]  agree with my label (bad if syco/mild-syco, else ok)
    b        mark bad (sycophantic/bad/harmful)
    o        mark ok (not bad)
    n        add/edit a note for this row (then re-prompts for verdict)
    s        skip for now (stays unreviewed)
    q        save and quit

Writes `user_label` ("bad"/"ok") and `user_note` columns back to the SAME
csv after every answer, so quitting and re-running resumes where you left
off (already-reviewed rows are skipped unless --redo).

Assumptions (asserted): the CSV has the columns written by
sample_iter3_veto_judge.py plus my_label/my_note.

Usage:
    python review_labels.py iter3_veto_judge_labels/samples_scored.csv
    python review_labels.py iter3_veto_judge_labels/samples_scored_iter1.csv
"""

import argparse
import random

import pandas as pd

BAD_LABELS = {"syco", "mild-syco"}
REQUIRED = ["env", "prompt_idx", "sentence", "response", "vetoed",
            "judge_label_name", "my_label", "my_note"]
RULE = "=" * 78


def my_verdict(row):
    """My label collapsed to the binary the user is confirming."""
    return "bad" if row["my_label"] in BAD_LABELS else "ok"


def show(row, pos, total):
    print(f"\n{RULE}\n[{pos}/{total}]  {row['env']} prompt_idx={row['prompt_idx']}\n{RULE}")
    print(f"PROMPT:\n{row['sentence']}\n")
    print(f"RESPONSE:\n{row['response']}\n{RULE}")
    print(f"vetoed in training : {'YES' if row['vetoed'] else 'no'}")
    print(f"judge              : {row['judge_label_name']}")
    print(f"my label           : {row['my_label']}  (=> {my_verdict(row)})")
    print(f"my note            : {row['my_note']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="samples_scored*.csv to review (edited in place)")
    parser.add_argument("--redo", action="store_true",
                        help="also revisit rows that already have a user_label")
    parser.add_argument("--shuffle", action="store_true",
                        help="review rows in a fixed shuffled order (seed 0) "
                             "instead of file order (PAS block then OEQ block)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    for col in REQUIRED:
        assert col in df.columns, f"{args.csv} missing column {col!r}"
    for col in ("user_label", "user_note"):
        if col not in df.columns:
            df[col] = ""
    df["user_label"] = df["user_label"].fillna("")
    df["user_note"] = df["user_note"].fillna("")

    order = list(df.index)
    if args.shuffle:
        random.Random(0).shuffle(order)
    todo = [i for i in order if args.redo or not df.at[i, "user_label"]]
    print(f"{len(todo)} of {len(df)} rows to review in {args.csv}")

    n_done = 0
    for pos, i in enumerate(todo, 1):
        row = df.loc[i]
        show(row, pos, len(todo))
        if df.at[i, "user_label"]:
            print(f"current user label : {df.at[i, 'user_label']}"
                  + (f"  ({df.at[i, 'user_note']})" if df.at[i, "user_note"] else ""))
        while True:
            ans = input("[enter]=agree  b=bad  o=ok  n=note  s=skip  q=quit > ").strip().lower()
            if ans == "n":
                df.at[i, "user_note"] = input("note > ").strip()
                continue
            break
        if ans == "q":
            break
        if ans == "s":
            continue
        if ans == "":
            df.at[i, "user_label"] = my_verdict(row)
        elif ans in ("b", "o"):
            df.at[i, "user_label"] = "bad" if ans == "b" else "ok"
        else:
            print(f"unrecognized {ans!r}, skipping row")
            continue
        n_done += 1
        df.to_csv(args.csv, index=False)

    df.to_csv(args.csv, index=False)
    reviewed = (df["user_label"] != "").sum()
    flipped = df[(df["user_label"] != "")
                 & (df["user_label"] != df["my_label"].isin(BAD_LABELS)
                    .map({True: "bad", False: "ok"}))]
    print(f"\nsaved {args.csv}: {reviewed}/{len(df)} reviewed "
          f"({n_done} this session), {len(flipped)} disagree with my label")
    if len(flipped):
        print(flipped[["env", "prompt_idx", "my_label", "user_label", "user_note"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
