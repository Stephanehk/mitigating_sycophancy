# final_training_eval_data

Canonical, frozen train / held-out splits for the four envs. These **replace**
the earlier held-out sets (OEQ/PAS `heldout_*_final1000_ids.json`, AITA/FLIP
`runs/baseline_*/test_ids.json`), which were used to tune the mitigation method
and are therefore contaminated for a final, unbiased evaluation.

Regenerate with:

```
/next/u/stephhk/miniconda3/envs/llm/bin/python \
  training_mitigations/data_build/build_final_training_eval_data.py
```

`FINAL_SEED = 20260710`. The written `*_ids.json` files are the authoritative
artifact; the `*.csv` files materialize the rows for convenience.

## Files (per env stem: `oeq`, `pas`, `aita`, `flip`)

| file | contents |
|---|---|
| `<stem>_train_ids.json` / `<stem>_test_ids.json` | ordered id lists |
| `<stem>_train.csv` / `<stem>_test.csv` | `id`, `post_text` (+ `is_asshole` for AITA, + `flipped_post_text` for FLIP) |
| `MANIFEST.json` | seed, pool sizes, exclusions, sampler, overlap stats |

## How each split was drawn

Test is sampled first, then train from the disjoint complement (`seed`,
`seed+1000`), mirroring `core/train_data.split_train_test`.

- **OEQ / PAS** — 1000 test + 1000 train, **uniform**, with the OLD held-out ids
  **excluded**, so the new held-out never appeared in the tuning eval. (OEQ is
  tight: 3010 pool − 1000 old = 2010 available for 2000 new.) The new set may
  reuse OLD *training* prompts — not a contamination problem, since design
  decisions were made on the eval set (excluded) and final models train only on
  the new train split.
- **AITA** — 1000 test + 1000 train, **balanced 500/500 on `is_asshole`**,
  excluding both the old AITA held-out and **all 1591 FLIP ids** (FLIP originals
  are AITA posts, so this prevents an AITA train prompt leaking into FLIP eval).
- **AITA-NTA-FLIP** — 1000 test + 500 train **pairs**, **uniform**, resampled
  from the FULL 1591-pair released set with **NO exclusion**. Each row is a
  `(post_text, flipped_post_text)` pair; both are fed independently through the
  pipeline, so a pair yields 2 generations.

## FLIP contamination caveat (important)

FLIP is a fixed, released 1591-pair artifact and cannot be grown in-repo, so a
fresh disjoint 1000+1000 is impossible. This resample **overlaps the tuning
eval**: 941 of the 1500 new FLIP pairs (633 test + 308 train) were in the old
tuning held-out. **Treat FLIP as a small fixed consistency benchmark (both-NTA),
not a genuinely-unseen held-out.** Do not make method-design decisions based on
FLIP numbers if you want them to remain unbiased.
