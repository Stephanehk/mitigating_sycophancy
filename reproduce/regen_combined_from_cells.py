#!/usr/bin/env python3
"""Rebuild the combined AITA provider-consistency summary + predictions CSVs
from the 6 per-(model, provider) cell CSVs already on disk.

Why this exists: `aita_provider_consistency.py` was run as two concurrent lanes
(one per provider) that both write the SAME combined paths
(`aita_consistency_summary_<tag>.csv` / `aita_consistency_predictions_<tag>.csv`).
Running them in parallel makes the lane that finishes last clobber the other's
combined output, so those two files ended up holding only one provider's rows.
The per-cell CSVs (`aita_consistency_<model>_<provider>_<tag>.csv`) are written
per combo and are unaffected, so this script reassembles the correct combined
files from them — no API calls, no re-running.

It reuses the harness metric (`compute_fnr_fpr`) and the script's own
`wald_ci` / `report_model` so the regenerated summary schema and numbers are
byte-for-byte what a single clean full run would have produced. It asserts every
expected cell is present and has exactly the expected row count before writing,
so a missing or short cell fails loud instead of silently shrinking the matrix.
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from aita_provider_consistency import (
    MODELS,
    compute_fnr_fpr,
    report_model,
)

EXPECTED_N = 2000
TAG = f"n{EXPECTED_N}_seed0_temp1.0"
PROVIDERS = ["azure", "openrouter"]
CELL_COLS = ["id", "is_asshole", "raw", "pred", "served_model"]


def load_cell(model_key, provider):
    """Load one per-(model, provider) cell CSV and validate it is complete.

    Assumptions checked: file exists, has exactly EXPECTED_N rows, carries the
    columns `compute_fnr_fpr` consumes, and every row has a non-null verdict
    (so the cell really ran to completion rather than being a partial
    checkpoint). Any violation raises immediately.
    """
    slug = model_key.replace("/", "-")
    path = os.path.join(HERE, f"aita_consistency_{slug}_{provider}_{TAG}.csv")
    assert os.path.exists(path), f"missing cell CSV: {path}"
    df = pd.read_csv(path)
    assert len(df) == EXPECTED_N, (
        f"{path}: {len(df)} rows, expected {EXPECTED_N}")
    for col in CELL_COLS:
        assert col in df.columns, f"{path}: missing column {col!r}"
    assert df["pred"].notna().all(), (
        f"{path}: has {int(df['pred'].isna().sum())} unfilled (NaN) verdict "
        f"rows; cell is not complete")
    return df


def main():
    per_model = {}
    long_frames = []
    for model_key in MODELS:
        per_provider = {}
        for prov in PROVIDERS:
            res = load_cell(model_key, prov)
            metrics = compute_fnr_fpr(res)
            # compute_fnr_fpr is the single source of truth for n_total etc.;
            # assert its denominator matches the file so a schema drift in the
            # metric can't silently disagree with the raw cell.
            assert metrics["n_total"] == EXPECTED_N, (
                f"{model_key}/{prov}: compute_fnr_fpr n_total="
                f"{metrics['n_total']} != {EXPECTED_N}")
            per_provider[prov] = (res, metrics)
            tall = res.copy()
            tall.insert(0, "model", model_key)
            tall.insert(1, "provider", prov)
            long_frames.append(tall)
        per_model[model_key] = per_provider

    # report_model prints the per-model FPR/FNR table + Azure-vs-OpenRouter
    # agreement and returns the flat summary rows — identical to the live run.
    all_rows = []
    for model_key in MODELS:
        all_rows.extend(report_model(model_key, per_model[model_key]))

    summary = pd.DataFrame(all_rows)
    summary_path = os.path.join(HERE, f"aita_consistency_summary_{TAG}.csv")
    summary.to_csv(summary_path, index=False)

    predictions = pd.concat(long_frames, ignore_index=True)
    long_path = os.path.join(HERE, f"aita_consistency_predictions_{TAG}.csv")
    predictions.to_csv(long_path, index=False)

    # Final cross-checks on the combined predictions: full matrix coverage and
    # exact per-cell counts, so the written file can't quietly under-cover.
    counts = predictions.groupby(["model", "provider"]).size()
    print("\ncombined predictions: rows per (model, provider)")
    print(counts.to_string())
    assert len(predictions) == EXPECTED_N * len(MODELS) * len(PROVIDERS), (
        f"combined predictions has {len(predictions)} rows, expected "
        f"{EXPECTED_N * len(MODELS) * len(PROVIDERS)}")
    assert (counts == EXPECTED_N).all(), "a (model, provider) cell != EXPECTED_N"
    assert len(summary) == len(MODELS) * len(PROVIDERS), (
        f"summary has {len(summary)} rows, expected "
        f"{len(MODELS) * len(PROVIDERS)}")

    print(f"\nregenerated summary     -> {summary_path} ({len(summary)} rows)")
    print(f"regenerated predictions -> {long_path} ({len(predictions)} rows)")


if __name__ == "__main__":
    main()
