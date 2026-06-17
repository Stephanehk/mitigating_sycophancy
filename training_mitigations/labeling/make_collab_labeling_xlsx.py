"""Build the collaborator labeling workbook: 20 responses sampled across the
six gpt-5.4-mini sycoharm-judged sets, exported to Excel WITHOUT the LLM
labels (blind labeling), plus a local answer key for recovering them.

Sampling: STRATIFIED BY JUDGE LABEL — 5 rows per sycoharm level (0-3) = 20
rows, seed 7, restricted to rows with a valid judge label (asserted).
Within each label stratum the picks are spread round-robin across the six
datasets (heldout_pas, heldout_oeq, pas_deepseek, pas_gpt4o, oeq_deepseek,
oeq_gpt4o) so no dataset dominates a stratum. The underlying judge-label
distribution is heavily skewed (~76% neutral, ~4% critical), so a uniform
sample would contain almost no non-neutral rows and per-class agreement
with the human labels could not be assessed; the label-balanced sample
measures agreement on every level (reweight by the population label
frequencies if a population-level agreement number is needed). Sheet row
order is shuffled so neither the source sets nor the label balance are
apparent.

Recovery: each row gets id = sha256("<dataset>:<source_row>:<response>")[:12].
The id appears in both the workbook and the answer key, and is recomputable
from the response text itself, so labels are recoverable even if the
collaborator reorders rows.

Outputs in training_mitigations/stephane_xiaoxuan_labeling/:
  - to_label.xlsx   : sheet "labeling" (id, question, response, your_label,
                      notes) + sheet "instructions" (the judge rubric with
                      the 0-3 scale). Send THIS file only.
  - answer_key.csv  : id -> dataset, source_row, sycoharm label + score.
                      Do NOT send.
"""

import hashlib
import os
import random
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, "eval"))  # judge_sycophancy now lives in eval/
from judge_sycophancy import PROMPT_TEMPLATE, LEVEL_NAMES, METRIC
RUNS = os.path.join(_HERE, "sycoharm_runs")
OUT_DIR = os.path.join(_HERE, "stephane_xiaoxuan_labeling")
SEED = 7
DATASETS = ["heldout_pas", "heldout_oeq", "pas_deepseek", "pas_gpt4o",
            "oeq_deepseek", "oeq_gpt4o"]
# Rows per sycoharm level (0-3). Balanced so per-class human/judge
# agreement is measurable on all four levels; totals 20.
LABEL_PLAN = {0: 5, 1: 5, 2: 5, 3: 5}


def row_id(dataset, source_row, response):
    return hashlib.sha256(
        f"{dataset}:{source_row}:{response}".encode()).hexdigest()[:12]


def load_scored_rows(dataset):
    """Load one scored set and return its valid-judge-label rows as dicts
    in the answer-key/sheet schema (id, dataset, source_row, question,
    response, label fields). `source_row` is the row's index in the
    scored CSV, matching the previous batches' convention.
    """
    df = pd.read_csv(os.path.join(RUNS, f"scored_{dataset}.csv"))
    label_col = f"{METRIC}_response"
    score_col = f"{METRIC}_score_response"
    assert label_col in df.columns, f"{dataset}: not yet scored/renamed"
    valid = df[df[score_col].notna()]
    rows = []
    for i, r in valid.iterrows():
        label = int(round(r[score_col] * 3))
        assert label in LEVEL_NAMES, f"{dataset} row {i}: bad label {label}"
        rows.append({
            "id": row_id(dataset, i, r["response"]),
            "dataset": dataset, "source_row": i,
            "question": r["sentence"], "response": r["response"],
            "sycoharm_label": label,
            "sycoharm_label_name": LEVEL_NAMES[label],
            "sycoharm_score": r[score_col],
        })
    return rows


def sample_label_stratum(stratum_rows, k, rng):
    """Pick `k` rows from one label's pool, spreading the picks across
    datasets: datasets are visited round-robin in rng-shuffled order,
    taking one rng-chosen row from each dataset that still has any left,
    until k are picked. With pools larger than k this caps any dataset's
    contribution near ceil(k / num_datasets_with_rows); with scarce pools
    it degrades gracefully (a dataset with no rows for this label is
    simply skipped). Asserts the pool is big enough.
    """
    assert len(stratum_rows) >= k, (
        f"only {len(stratum_rows)} rows available for a stratum of {k}"
    )
    by_ds = {}
    for r in stratum_rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    for pool in by_ds.values():
        rng.shuffle(pool)
    order = sorted(by_ds)
    rng.shuffle(order)
    picked = []
    while len(picked) < k:
        for ds in order:
            if len(picked) >= k:
                break
            if by_ds[ds]:
                picked.append(by_ds[ds].pop())
    return picked


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(SEED)

    all_rows = []
    for dataset in DATASETS:
        all_rows.extend(load_scored_rows(dataset))

    rows = []
    for label in sorted(LABEL_PLAN):
        stratum = [r for r in all_rows if r["sycoharm_label"] == label]
        rows.extend(sample_label_stratum(stratum, LABEL_PLAN[label], rng))

    assert len(rows) == sum(LABEL_PLAN.values())
    assert len({r["id"] for r in rows}) == len(rows), "hash collision"
    for label, k in LABEL_PLAN.items():
        got = sum(r["sycoharm_label"] == label for r in rows)
        assert got == k, f"label {label}: wanted {k}, got {got}"
    rng.shuffle(rows)

    key = pd.DataFrame(rows)[["id", "dataset", "source_row", "sycoharm_label",
                              "sycoharm_label_name", "sycoharm_score"]]
    key_path = os.path.join(OUT_DIR, "answer_key.csv")
    key.to_csv(key_path, index=False)

    sheet = pd.DataFrame(rows)[["id", "question", "response"]]
    sheet["your_label"] = ""
    sheet["notes"] = ""
    # The rubric the collaborator labels by: the judge prompt minus the
    # per-row Question/Response tail.
    rubric = PROMPT_TEMPLATE.split("Question: {question}")[0].strip()
    instructions = pd.DataFrame({"instructions": [
        "Please read each row's question (the user's post) and response "
        "(advice given to that user) on the `labeling` sheet, and put a "
        "single number 0-3 in `your_label` following the rubric below. "
        "Use `notes` for anything you want to flag. Please don't reorder "
        "or edit the id column.",
        "",
        *rubric.splitlines(),
    ]})

    xlsx_path = os.path.join(OUT_DIR, "to_label.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xl:
        sheet.to_excel(xl, sheet_name="labeling", index=False)
        instructions.to_excel(xl, sheet_name="instructions", index=False,
                              header=False)
    print(f"wrote {xlsx_path} ({len(sheet)} rows, labels withheld)")
    print(f"wrote {key_path} (answer key -- do not send)")
    print(key.to_string(index=False))


if __name__ == "__main__":
    main()
