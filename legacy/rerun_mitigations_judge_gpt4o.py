"""Re-judge the OEQ and PAS mitigations sweep outputs with GPT-4o.

Why: our mitigations sweep (3 generation models x {OEQ, PAS}) was scored by a
GPT-5.4 action-endorsement judge (the `label` column, see the `_gpt-5.4`
filename suffix). We want a second, independent set of labels from GPT-4o --
the judge the ELEPHANT paper actually used (DIFF-D in `explore_mitigations.py`)
-- so endorsement rates can be reported against the paper's judge.

What this does NOT do: it does not re-generate any model responses. The
generated responses (and their generation token counts) are read verbatim from
the existing result CSVs; only the judge call is re-run. For each retained row
we look up the post text by `id` from the OSF source corpus, call
`explore_mitigations.judge_endorsement` with the new judge on the
(post_text, response) pair, and write a new CSV whose schema matches what
`explore_mitigations.main()` would have produced -- same columns as the input,
with `label`, `judge_input_tokens`, `judge_output_tokens` (and the
`input_tokens` / `output_tokens` totals) refreshed from the new judge.

Scope: by default only the four methods we report on are re-judged --
`baseline` ("no prompt"), `be_critical`, `no_syco_critical`
("be critical, don't be sycophantic"), and `stakeholder_concerns_v2`
("prompt_stakeholders_v2"). Override with --conditions to re-judge others.

Output filenames mirror the inputs with the judge slug swapped, e.g.
`explore_mitigations_results_OEQ_gpt-5.4_gpt-5.4.csv` ->
`explore_mitigations_results_OEQ_gpt-5.4_gpt-4o.csv`. A per-row checkpoint
sidecar (`.checkpoint.jsonl`) makes the run resumable: re-running the same
command picks up only the (id, condition) pairs not already judged.

Run from the repo root:

    python mitigations/rerun_mitigations_judge_gpt4o.py --workers 10
"""

import argparse
import concurrent.futures
import os
import sys
import threading

import pandas as pd
from tqdm import tqdm

# Make `explore_mitigations` (and, transitively, `open_router_utils`)
# importable regardless of where this is invoked from -- same bootstrap the
# main script uses.
# Moved to legacy/; explore_mitigations lives in ../mitigations/, and the repo
# root is added so the relative OSF-corpus paths resolve regardless of cwd.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mitigations"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import explore_mitigations
from explore_mitigations import (
    append_checkpoint,
    checkpoint_path_for,
    judge_endorsement,
    load_checkpoint,
    load_oeq,
    load_pas,
    make_client,
    model_pricing,
    model_slug,
)

# Repo root, so the relative OSF-corpus paths in `explore_mitigations` resolve
# no matter the caller's cwd.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The mitigation result CSVs this script re-judges live under mitigations/.
RESULTS_DIR = os.path.join(REPO_ROOT, "mitigations", "results")

# The new action-endorsement judge. Routed through OpenRouter (DIFF-D: GPT-4o
# is the judge the ELEPHANT paper used).
NEW_JUDGE_MODEL = "openai/gpt-4o"

# The four methods we report on. Names are the `condition` values registered in
# `explore_mitigations.get_conditions`; the parenthetical is the paper-facing
# label requested for this re-run.
DEFAULT_CONDITIONS = (
    "baseline",                 # "no prompt"
    "be_critical",              # "be critical"
    "no_syco_critical",         # "be critical, don't be sycophantic"
    "stakeholder_concerns_v2",  # "prompt_stakeholders_v2"
)

# CSV column order produced by `explore_mitigations.main()` for OEQ/PAS. We
# emit exactly these so the GPT-4o output is a drop-in for the same downstream
# readers (plot_explore_mitigations.py etc.).
OUTPUT_COLUMNS = [
    "id", "condition", "response", "label",
    "gen_input_tokens", "gen_output_tokens",
    "judge_input_tokens", "judge_output_tokens",
    "input_tokens", "output_tokens",
]


def build_post_map(env_name):
    """Return {id: post_text} for the OEQ or PAS source corpus.

    Delegates to `explore_mitigations.load_oeq` / `load_pas` so the id scheme
    (md5 of the raw prompt) stays defined in exactly one place. Reimplementing
    the hashing here would risk drifting from the ids stored in the result
    files. Assumes the corpus is a superset of every id in the result file;
    `rejudge_file` asserts that.
    """
    assert env_name in ("OEQ", "PAS"), f"unsupported env {env_name!r}"
    if env_name == "OEQ":
        df = load_oeq(os.path.join(REPO_ROOT, explore_mitigations.OEQ_DATA_CSV))
    else:
        df = load_pas(os.path.join(REPO_ROOT, explore_mitigations.PAS_DATA_CSV))
    post_map = dict(zip(df["id"], df["post_text"]))
    assert len(post_map) == len(df), f"{env_name} id collision in source corpus"
    return post_map


def env_of(in_path):
    """Return 'OEQ' or 'PAS' for a mitigations result filename.

    The filename scheme is
    `explore_mitigations_results_<env>_<gen>_<judge>.csv`. We only support the
    open-ended endorsement envs here (AITA uses YTA/NTA parsing, not the
    endorsement judge; OEQ_2 uses the three-axis sycophancy judge), so anything
    else fails loud rather than being silently mis-handled.
    """
    name = os.path.basename(in_path)
    assert name.startswith("explore_mitigations_results_"), \
        f"{name} is not a mitigations result file"
    # Reject OEQ_2 explicitly: it shares the 'OEQ' prefix but has a different
    # schema and judge, and is out of scope for endorsement re-judging.
    assert "_OEQ_2_" not in name, f"{name}: OEQ_2 is not supported"
    tail = name[len("explore_mitigations_results_"):]
    if tail.startswith("OEQ_"):
        return "OEQ"
    if tail.startswith("PAS_"):
        return "PAS"
    raise AssertionError(f"{name}: env must be OEQ or PAS for endorsement re-judging")


def canonical_out_path(in_path, new_judge_slug):
    """Path `explore_mitigations.main()` would have produced under the new judge.

    Swaps the input filename's trailing judge slug for `new_judge_slug`,
    keeping the env and generation slugs unchanged. Model slugs never contain
    underscores (see `model_slug`), so the judge slug is exactly the segment
    after the final underscore. Returns an absolute path under RESULTS_DIR.
    """
    name = os.path.basename(in_path)
    assert name.endswith(".csv"), f"{name} is not a .csv"
    stem = name[:-len(".csv")]
    head, _old_judge = stem.rsplit("_", 1)
    out_name = f"{head}_{new_judge_slug}.csv"
    assert out_name != name, (
        f"output path equals input ({name}); the file is already judged by "
        f"{new_judge_slug}. Refusing to overwrite."
    )
    return os.path.join(RESULTS_DIR, out_name)


def rejudge_file(in_path, conditions, post_map, client, judge_model, workers):
    """Re-judge the requested conditions in `in_path`; write a new-judge CSV.

    Reads the input result CSV, keeps only rows whose `condition` is in
    `conditions`, and re-runs the endorsement judge on each row's stored
    `response` against its post text (looked up by `id` in `post_map`). The
    generation-side columns are copied through unchanged; `label`,
    `judge_input_tokens`, `judge_output_tokens`, and the `input_tokens` /
    `output_tokens` totals are replaced with the new judge's results.

    Resumable: completed (id, condition) records are appended to a
    `.checkpoint.jsonl` sidecar as they finish, and a re-run skips them.
    Output is written sorted by (condition, id) to match `main()`.

    Returns (judge_in_tokens, judge_out_tokens) summed over rows judged THIS
    run (checkpoint-resumed rows are not re-counted).
    """
    df = pd.read_csv(in_path)
    for col in OUTPUT_COLUMNS:
        assert col in df.columns, f"{in_path} missing column {col!r}"

    df = df[df["condition"].isin(conditions)].copy()
    assert len(df) > 0, (
        f"{in_path}: none of the requested conditions {sorted(conditions)} "
        f"are present"
    )

    # Every retained id must exist in the source corpus -- otherwise we'd be
    # judging a response against the wrong (or missing) post. Fail loud.
    missing = sorted(set(df["id"]) - set(post_map))
    assert not missing, (
        f"{in_path}: {len(missing)} id(s) not in source corpus; "
        f"first few: {missing[:5]}"
    )

    out_path = canonical_out_path(in_path, model_slug(judge_model))
    ckpt_path = checkpoint_path_for(out_path)
    done = load_checkpoint(ckpt_path)
    if done:
        print(f"  resuming: {len(done)} (id, condition) already judged in {ckpt_path}")

    todo = [row for _, row in df.iterrows()
            if (row["id"], row["condition"]) not in done]

    lock = threading.Lock()
    run_judge_in = 0
    run_judge_out = 0

    def judge_one(row):
        post_text = post_map[row["id"]]
        label, j_in, j_out = judge_endorsement(
            client, judge_model, post_text, row["response"],
        )
        gen_in = int(row["gen_input_tokens"])
        gen_out = int(row["gen_output_tokens"])
        return {
            "id": row["id"],
            "condition": row["condition"],
            "response": row["response"],
            "label": label,
            "gen_input_tokens": gen_in,
            "gen_output_tokens": gen_out,
            "judge_input_tokens": j_in,
            "judge_output_tokens": j_out,
            "input_tokens": gen_in + j_in,
            "output_tokens": gen_out + j_out,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(judge_one, row) for row in todo]
        for f in tqdm(concurrent.futures.as_completed(futures),
                      total=len(futures), desc=os.path.basename(out_path)):
            rec = f.result()
            done[(rec["id"], rec["condition"])] = rec
            append_checkpoint(ckpt_path, rec, lock)
            run_judge_in += rec["judge_input_tokens"]
            run_judge_out += rec["judge_output_tokens"]

    out_df = pd.DataFrame(list(done.values()))
    out_df["condition"] = pd.Categorical(
        out_df["condition"], categories=list(conditions), ordered=True
    )
    out_df = out_df.sort_values(["condition", "id"]).reset_index(drop=True)
    out_df = out_df[OUTPUT_COLUMNS]
    out_df.to_csv(out_path, index=False)
    print(f"  wrote {len(out_df)} rows to {out_path}")
    return run_judge_in, run_judge_out


def discover_files(only, judge_slug):
    """Return the OEQ/PAS mitigations result CSVs to re-judge.

    If `only` is non-empty, those paths are used verbatim (asserted to exist).
    Otherwise every `explore_mitigations_results_{OEQ,PAS}_*.csv` sweep result
    under RESULTS_DIR is included, excluding OEQ_2 and the trace/checkpoint
    sidecars. Files already judged by `judge_slug` (i.e. this script's own
    outputs from a prior run) are excluded so a resumed run doesn't try to
    re-judge them onto themselves. Sorted for deterministic order.
    """
    if only:
        for p in only:
            assert os.path.exists(p), f"--only file not found: {p}"
        return sorted(only)
    out = []
    for name in os.listdir(RESULTS_DIR):
        if not name.endswith(".csv"):
            continue
        if not name.startswith("explore_mitigations_results_"):
            continue
        if "_OEQ_2_" in name:
            continue
        # Skip our own prior outputs: the judge slug is the last underscore
        # segment of the stem, and model slugs never contain underscores.
        if name[:-len(".csv")].rsplit("_", 1)[-1] == judge_slug:
            continue
        tail = name[len("explore_mitigations_results_"):]
        if tail.startswith("OEQ_") or tail.startswith("PAS_"):
            out.append(os.path.join(RESULTS_DIR, name))
    assert out, f"no OEQ/PAS mitigations result CSVs found under {RESULTS_DIR}"
    return sorted(out)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--judge_model", default=NEW_JUDGE_MODEL,
        help="OpenRouter judge model id (default: %(default)s)",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=list(DEFAULT_CONDITIONS),
        help="condition names to re-judge (default: the four reported methods "
             "%(default)s)",
    )
    parser.add_argument(
        "--workers", type=int, default=explore_mitigations.DEFAULT_WORKERS,
        help="parallel judge calls per file (default: %(default)s)",
    )
    parser.add_argument(
        "--only", nargs="*", default=[],
        help="specific mitigations result CSVs to re-judge (default: every "
             "OEQ/PAS sweep result under results/)",
    )
    args = parser.parse_args()

    files = discover_files(args.only, model_slug(args.judge_model))
    print(f"re-judging {len(files)} file(s) with judge {args.judge_model}; "
          f"conditions: {args.conditions}")

    client = make_client(args.judge_model)
    # Build the post maps once and reuse across files of the same env.
    post_maps = {}
    j_in_price, j_out_price = model_pricing(args.judge_model)

    total_in = 0
    total_out = 0
    for in_path in files:
        env_name = env_of(in_path)
        if env_name not in post_maps:
            post_maps[env_name] = build_post_map(env_name)
        print(f"\n{os.path.basename(in_path)}  (env={env_name})")
        j_in, j_out = rejudge_file(
            in_path, args.conditions, post_maps[env_name],
            client, args.judge_model, args.workers,
        )
        total_in += j_in
        total_out += j_out

    cost = (total_in / 1e6) * j_in_price + (total_out / 1e6) * j_out_price
    print(f"\n=== done. judge tokens this run: in={total_in} out={total_out} "
          f"(priced at {args.judge_model}: ${cost:.4f}) ===")


if __name__ == "__main__":
    main()
