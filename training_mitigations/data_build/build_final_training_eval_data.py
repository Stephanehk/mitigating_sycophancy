"""Build the FINAL, canonical train / held-out splits for all four envs.

Motivation
----------
The previous 1000-prompt held-out sets (OEQ/PAS `heldout_*_final1000_ids.json`,
AITA/FLIP `runs/baseline_*/test_ids.json`) were used to TUNE the mitigation
method, so they are contaminated for a final, unbiased evaluation and must be
discarded. This script freezes a fresh set of splits under
`final_training_eval_data/` and writes explicit id files so the split is an
auditable artifact rather than an implicit seed derivation.

Per-env contract
----------------
- OEQ, PAS: pool is several k prompts. Sample a fresh 1000 test + 1000 train,
  UNIFORM, from the pool with the OLD held-out ids EXCLUDED. This guarantees the
  new held-out never appeared in the tuning eval. (OEQ is tight: 3010 pool -
  1000 old held-out = 2010 available for 2000 new prompts.) The new set may
  still overlap the OLD *training* pool — that is fine: reusing an old train
  prompt as a new test item is not method-design contamination (design
  decisions were made on the EVAL set, which we exclude), and the final models
  train only on the new train split.
- AITA: pool is ~80k. Sample a fresh 1000 test + 1000 train, BALANCED 500/500 on
  `is_asshole`, excluding BOTH the old AITA held-out AND all 1591 FLIP ids. FLIP
  originals are AITA posts, so an AITA training prompt whose flipped twin sits in
  the FLIP eval would leak — hence the FLIP exclusion.
- AITA-NTA-FLIP: pool is a fixed, released 1591-pair artifact (cannot be grown
  in-repo). A fresh disjoint 1000+1000 is impossible, so per the agreed decision
  we resample 1000 test + 500 train (pairs) UNIFORM from the FULL 1591 with NO
  exclusion and fix it as canonical. This does NOT de-contaminate FLIP: ~94% of
  the pairs were touched during tuning (see the printed overlap). FLIP is treated
  as a small fixed consistency benchmark, not a genuinely-unseen held-out.

Split mechanics mirror `core/train_data.split_train_test`: test is sampled
first, then train is sampled from the disjoint complement with `seed + 1000`, so
the two samplers never collide. `FINAL_SEED` is distinct from the old seed (0) to
signal these are the frozen final splits.

Outputs (under `final_training_eval_data/`)
-------------------------------------------
  <env>_test_ids.json, <env>_train_ids.json  — ordered id lists (the artifact)
  <env>_test.csv,       <env>_train.csv       — materialized rows (post text +
                                                 is_asshole for AITA + flipped
                                                 post for FLIP)
  MANIFEST.json                               — seed, sizes, exclusions, overlaps

Run:
  /next/u/stephhk/miniconda3/envs/llm/bin/python \
    training_mitigations/data_build/build_final_training_eval_data.py
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "mitigations"))

from core.config import (  # noqa: E402
    OEQ_DATA_CSV, PAS_DATA_CSV, AITA_DATA_CSV, AITA_NTA_FLIP_DATA_CSV,
)
from core.data import (  # noqa: E402
    load_oeq, load_pas, load_aita_clean, load_aita_nta_flip,
    sample_uniform, sample_balanced,
)

FINAL_SEED = 20260710
OUT_DIR = os.path.join(_REPO, "final_training_eval_data")

# Old, contaminated held-out id files to EXCLUDE from the fresh draw (None = do
# not exclude, i.e. resample from the full pool).
OLD_HELDOUT = {
    "OEQ": os.path.join(
        _REPO, "training_mitigations/sycoharm_runs/heldout_oeq_final1000_ids.json"),
    "PAS": os.path.join(
        _REPO, "training_mitigations/sycoharm_runs/heldout_pas_final1000_ids.json"),
    "AITA": os.path.join(
        _REPO, "training_mitigations/runs/baseline_AITA/test_ids.json"),
    "AITA-NTA-FLIP": None,
}
# Reference to the old FLIP held-out purely for reporting the overlap caveat.
OLD_FLIP_HELDOUT = os.path.join(
    _REPO, "training_mitigations/runs/baseline_AITA-NTA-FLIP/test_ids.json")

ENVS = {
    "OEQ": dict(loader=load_oeq, csv=OEQ_DATA_CSV, sampler="uniform",
                test_n=1000, train_n=1000, stem="oeq"),
    "PAS": dict(loader=load_pas, csv=PAS_DATA_CSV, sampler="uniform",
                test_n=1000, train_n=1000, stem="pas"),
    "AITA": dict(loader=load_aita_clean, csv=AITA_DATA_CSV, sampler="balanced",
                 test_n=1000, train_n=1000, stem="aita"),
    "AITA-NTA-FLIP": dict(loader=load_aita_nta_flip, csv=AITA_NTA_FLIP_DATA_CSV,
                          sampler="uniform", test_n=1000, train_n=500,
                          stem="flip"),
}


def load_id_list(path):
    """Load a JSON list of ids as a set of strings. Fails loud if absent."""
    assert os.path.exists(path), f"missing id file: {path}"
    with open(path) as f:
        ids = json.load(f)
    assert isinstance(ids, list) and ids, f"{path}: expected a non-empty list"
    return set(str(i) for i in ids)


def draw_split(env, cfg, exclude_ids):
    """Return (train_df, test_df) for one env.

    Drops `exclude_ids` from the pool, then samples test then a disjoint train
    (test seed = FINAL_SEED, train seed = FINAL_SEED + 1000), mirroring
    `split_train_test`. Uniform for OEQ/PAS/FLIP, balanced-on-is_asshole for
    AITA. Asserts the pool is large enough and the two splits are disjoint.
    """
    data = cfg["loader"](os.path.join(_REPO, cfg["csv"]))
    data["id"] = data["id"].astype(str)
    n_full = len(data)
    if exclude_ids:
        pool = data[~data["id"].isin(exclude_ids)].reset_index(drop=True)
    else:
        pool = data.reset_index(drop=True)
    need = cfg["test_n"] + cfg["train_n"]
    assert len(pool) >= need, (
        f"{env}: pool after exclusion has {len(pool)} rows, need {need}")

    if cfg["sampler"] == "balanced":
        test_df = sample_balanced(pool, cfg["test_n"], FINAL_SEED)
        rest = pool[~pool["id"].isin(set(test_df["id"]))].reset_index(drop=True)
        train_df = sample_balanced(rest, cfg["train_n"], FINAL_SEED + 1000)
    else:
        test_df = sample_uniform(pool, cfg["test_n"], FINAL_SEED)
        rest = pool[~pool["id"].isin(set(test_df["id"]))].reset_index(drop=True)
        train_df = sample_uniform(rest, cfg["train_n"], FINAL_SEED + 1000)

    assert set(test_df["id"]).isdisjoint(set(train_df["id"])), \
        f"{env}: train/test id overlap"
    if exclude_ids:
        assert set(test_df["id"]).isdisjoint(exclude_ids), \
            f"{env}: new test overlaps the excluded old held-out"
        assert set(train_df["id"]).isdisjoint(exclude_ids), \
            f"{env}: new train overlaps the excluded old held-out"
    return train_df, test_df, n_full, len(pool)


def write_split(stem, split_name, df):
    """Write `<stem>_<split>_ids.json` and `<stem>_<split>.csv`. Returns ids."""
    ids = df["id"].tolist()
    with open(os.path.join(OUT_DIR, f"{stem}_{split_name}_ids.json"), "w") as f:
        json.dump(ids, f, indent=2)
    df.to_csv(os.path.join(OUT_DIR, f"{stem}_{split_name}.csv"), index=False)
    return ids


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    old_flip = load_id_list(OLD_FLIP_HELDOUT)
    manifest = {"final_seed": FINAL_SEED, "envs": {}}

    for env, cfg in ENVS.items():
        exclude = set()
        if OLD_HELDOUT[env] is not None:
            exclude |= load_id_list(OLD_HELDOUT[env])
        if env == "AITA":
            # FLIP originals are AITA posts: exclude every FLIP id so an AITA
            # training prompt can't leak into the FLIP consistency eval.
            flip_all = load_aita_nta_flip(
                os.path.join(_REPO, AITA_NTA_FLIP_DATA_CSV))["id"].astype(str)
            exclude |= set(flip_all)

        train_df, test_df, n_full, n_pool = draw_split(env, cfg, exclude)
        train_ids = write_split(cfg["stem"], "train", train_df)
        test_ids = write_split(cfg["stem"], "test", test_df)

        entry = dict(pool_total=n_full, pool_after_exclusion=n_pool,
                     n_excluded=len(exclude), sampler=cfg["sampler"],
                     test_n=len(test_ids), train_n=len(train_ids),
                     old_heldout_excluded=OLD_HELDOUT[env] is not None)
        if env == "AITA-NTA-FLIP":
            # Report how much of the "fresh" FLIP set the tuning eval already saw.
            new_ids = set(test_ids) | set(train_ids)
            entry["overlap_new_vs_old_flip_heldout"] = len(new_ids & old_flip)
            entry["contamination_note"] = (
                "Resampled from the full 1591-pair set with NO exclusion; "
                "overlaps the tuning eval. Treat FLIP as a fixed consistency "
                "benchmark, not an unseen held-out.")
        if env == "AITA":
            entry["balanced_on"] = "is_asshole (500/500 per split)"
        manifest["envs"][env] = entry
        print(f"[{env}] pool={n_full} -> after_excl={n_pool} "
              f"(excluded {len(exclude)}) | test={len(test_ids)} "
              f"train={len(train_ids)}"
              + (f" | overlap_vs_old_flip={entry.get('overlap_new_vs_old_flip_heldout')}"
                 if env == "AITA-NTA-FLIP" else ""))

    with open(os.path.join(OUT_DIR, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {OUT_DIR}/ (MANIFEST.json + per-env ids/csv)")


if __name__ == "__main__":
    main()
