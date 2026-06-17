"""Build the 500-row input CSVs (columns: sentence, response) for the
gpt-5.4-mini sycoharm judge experiments.

Held-out group (one CSV per env x adapter tag, from the n1000 stakeholder-DPO
run dirs' eval shards endorse_<tag>_temp1.0_seed0_shard{0,1}of2.csv):

  heldout_pas / heldout_oeq                       : tag `base` (bare Qwen3-8B)
  heldout_pas_becritical / heldout_oeq_becritical : tag `base-becritical`
      (base + appended "Be critical." instruction)
  heldout_pas_trained / heldout_oeq_trained       : tag `final` (Qwen3-8B
      after stakeholder-DPO training, aka stakeholders-postraining-v1)

Paper group (skipped with --skip-paper; needs osfstorage-archive/Data):

  pas_deepseek / pas_gpt4o  : 500 sampled (seed 0) from the paper's
      PAS_results endorse4 files; question col `sentences`, advice cols
      `response` (deepseek) and `OEQ_sentences` (gpt4o, the paper's own
      naming for the gpt-4o response column).
  oeq_deepseek / oeq_gpt4o  : 500 sampled (seed 0) from the paper's
      OEQ_results 3k files; question col `prompt`; the gpt4o frame has no
      prompt column so prompts are aligned ROW-WISE from the deepseek frame
      (both 3027 rows, asserted).

Sampling matches the cost estimate exactly (df.sample(500, random_state=0)).
Outputs land in training_mitigations/sycoharm_runs/inputs/.
"""

import argparse
import glob
import os

import pandas as pd

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_HERE)
OSF = os.path.join(_REPO_ROOT, "osfstorage-archive", "Data")
OUT = os.path.join(_HERE, "sycoharm_runs", "inputs")


def save(name, q, a, expect=500):
    df = pd.DataFrame({"sentence": q.values, "response": a.values})
    assert len(df) == expect, f"{name}: expected {expect} rows, got {len(df)}"
    assert df["sentence"].notna().all() and df["response"].notna().all()
    path = os.path.join(OUT, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"wrote {path} ({len(df)} rows)")


# eval shard adapter tag -> output-name suffix appended to heldout_<env>
HELDOUT_TAGS = {"base": "", "base-becritical": "_becritical",
                "base-dontbesyco": "_dontbesyco", "final": "_trained"}


def build_from_run(run_basename, out_name, tag="final", expect=500):
    """Build a single heldout input CSV from an explicit run-dir basename
    (e.g. 'dpo_lora_pas_qwen3-8b_n1000_mixed') by concatenating its two
    eval shards for `tag`. Reuses save() (asserts `expect` rows / notna).
    Use this for run dirs that don't match the canonical n1000 naming that
    build_heldout assumes (e.g. the n1000_mixed / n1000_selfpref variants).
    `expect` is the merged row count (250+250=500 for the PAS held-out set,
    500+500=1000 for the OEQ final1000 set).

    Prefers the responses_<tag> CSV written by a generation-only
    (eval_endorsement_heldout.py --no-judge) run; falls back to the
    endorse_<tag> CSV from a judged run. Both carry sentence + response."""
    eval_dir = os.path.join(_HERE, "runs", run_basename, "eval")
    shards = sorted(glob.glob(os.path.join(
        eval_dir, f"responses_{tag}_temp1.0_seed0_shard*of2.csv")))
    if not shards:
        shards = sorted(glob.glob(os.path.join(
            eval_dir, f"endorse_{tag}_temp1.0_seed0_shard*of2.csv")))
    assert len(shards) == 2, f"{run_basename}/{tag}: expected 2 shards, got {shards}"
    df = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)
    # The two shards must be disjoint halves of the held-out set: a duplicated
    # id means a shard was misgenerated or the wrong pair was globbed.
    assert "id" in df.columns, f"{run_basename}/{tag}: shards lack an 'id' column"
    assert not df["id"].duplicated().any(), \
        f"{run_basename}/{tag}: duplicate ids across shards ({df['id'].duplicated().sum()})"
    save(out_name, df["sentence"], df["response"], expect=expect)


def build_heldout(tags, envs=("pas", "oeq")):
    """Build heldout_<env><suffix>.csv for each env x tag by concatenating the
    two 250-row eval shards of the n1000 run dirs. Asserts exactly 2 shards
    per (env, tag) and 500 total rows (via save)."""
    for env in envs:
        for tag in tags:
            build_from_run(f"dpo_lora_{env}_qwen3-8b_n1000",
                           f"heldout_{env}{HELDOUT_TAGS[tag]}", tag)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout-tags", nargs="*", default=["base"],
                        choices=sorted(HELDOUT_TAGS),
                        help="which eval adapter tags to build heldout inputs for")
    parser.add_argument("--envs", nargs="*", default=["pas", "oeq"],
                        choices=("pas", "oeq"),
                        help="which envs to build heldout inputs for")
    parser.add_argument("--skip-paper", action="store_true",
                        help="skip the paper deepseek/gpt4o samples")
    parser.add_argument("--variant", action="append", default=[],
                        help="build a heldout input from an explicit run-dir "
                             "basename, as "
                             "'<run_basename>:<out_name>[:<tag>[:<expect>]]' "
                             "(tag defaults to 'final', expect to 500). "
                             "Repeatable. When any --variant is given, ONLY the "
                             "variants are built (the default heldout + paper "
                             "builds are skipped).")
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)

    if args.variant:
        for spec in args.variant:
            parts = spec.split(":")
            assert len(parts) in (2, 3, 4), (
                f"bad --variant {spec!r}; want "
                "'<run_basename>:<out_name>[:<tag>[:<expect>]]'"
            )
            run_basename, out_name = parts[0], parts[1]
            tag = parts[2] if len(parts) >= 3 else "final"
            expect = int(parts[3]) if len(parts) == 4 else 500
            build_from_run(run_basename, out_name, tag, expect)
        return

    build_heldout(args.heldout_tags, envs=args.envs)

    if args.skip_paper:
        return

    d = pd.read_csv(os.path.join(OSF, "PAS_results",
                                 "endorse4_elephant_oeq_pas_deepseek.csv"))
    s = d.sample(500, random_state=0)
    save("pas_deepseek", s["sentences"], s["response"])

    g = pd.read_csv(os.path.join(OSF, "PAS_results",
                                 "endorse4_elephant_oeq_pas_gpt4o.csv"))
    s = g.sample(500, random_state=0)
    save("pas_gpt4o", s["sentences"], s["OEQ_sentences"])

    d3 = pd.read_csv(os.path.join(OSF, "OEQ_results",
                                  "endorse4_elephant_oeq3k_deepseek.csv"))
    s = d3.sample(500, random_state=0)
    save("oeq_deepseek", s["prompt"], s["deepseek"])

    g3 = pd.read_csv(os.path.join(OSF, "OEQ_results",
                                  "endorse4_elephant_oeq3k_gpt4o.csv"))
    assert len(g3) == len(d3), "gpt4o/deepseek OEQ frames misaligned"
    merged = pd.DataFrame({"prompt": d3["prompt"], "resp": g3["GPT-4o"]})
    s = merged.sample(500, random_state=0)
    save("oeq_gpt4o", s["prompt"], s["resp"])


if __name__ == "__main__":
    main()
