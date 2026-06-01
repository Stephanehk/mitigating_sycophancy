"""Score OEQ / PAS responses on the Lend an Ear empathy sub-components.

For each (id, condition) row in
    mitigations/results/explore_mitigations_results_<env>_<family>.csv
this script issues 3 judge calls (one per sub-component in
LEND_AN_EAR_SUBCOMPONENTS) and writes one output row per (id, condition):
    validating_emotions, demonstrating_understanding, dismissing_emotions
Each value is the first 1-5 digit emitted by the judge for that
sub-component. Rows whose judge output didn't contain a 1-5 digit are
stored as None and reported via per-sub-component _nan counts; the plotter
drops them from per-condition means.

Conversation passed to the judge:
    Seeker: <post_text from the env source data>
    Supporter: <model response from the results CSV>

For OEQ the post_text comes from osfstorage-archive/Data/OEQ_results/
endorse4_elephant_oeq3k_human.csv (sentence column). For PAS it comes from
osfstorage-archive/Data/PAS_results/endorse4_elephant_oeq_pas_claude.csv
(sentences column). Joining on id (the md5-hash id `oeq_*`/`pas_*` is
shared between the explore_mitigations results CSV and the env source).

Outputs:
  mitigations/results/empathy_<env>_<family>_judge-<judge-slug>.csv
Checkpointing follows the same pattern as explore_mitigations.py — one
JSON record per completed (id, condition) is appended to
<out>.checkpoint.jsonl; pass --resume to skip rows present in the
checkpoint.

CLI:
  --env OEQ|PAS                env name (one per invocation)
  --family <gen>_<judge>       maps to the input results CSV
  --judge-model openai/gpt-5.4-mini  judge model (default: gpt-5.4-mini —
                                      the high-volume per-row 3-call cost
                                      makes the mini model the default here
                                      even though explore_mitigations.py
                                      defaults to the full gpt-5.4)
  --workers N                  ThreadPoolExecutor size (default 12)
  --max-rows N                 cap rows for a smoke test (default: all)
  --out PATH                   override output CSV path
  --resume                     resume from <out>.checkpoint.jsonl
"""

import argparse
import concurrent.futures
import os
import re
import sys
import threading
import warnings

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
# judge_empathy.py now lives in eval_empathy/ but reuses the LLM client, the
# judge-prompt builder, and the OEQ/PAS loaders that live in mitigations/.
# Put that dir on the path so the imports below resolve regardless of cwd.
MITIGATIONS_DIR = os.path.join(REPO_ROOT, "mitigations")
# empathy_judge_prompt lives alongside this file in eval_empathy/.
sys.path.insert(0, HERE)
sys.path.insert(0, MITIGATIONS_DIR)

from explore_mitigations import (
    make_client, model_slug, call_llm,
    load_oeq, load_pas,
    OEQ_DATA_CSV, PAS_DATA_CSV,
    checkpoint_path_for, load_checkpoint, append_checkpoint,
)
from empathy_judge_prompt import build_prompt, LEND_AN_EAR_SUBCOMPONENTS


SUB_KEYS = list(LEND_AN_EAR_SUBCOMPONENTS.keys())
# The OEQ/PAS pipeline in main() reads/writes the explore_mitigations results
# that live alongside the mitigations code, so RESULTS_DIR stays anchored
# there rather than following this file into eval_empathy/.
RESULTS_DIR = os.path.join(MITIGATIONS_DIR, "results")
SUPPORTED_ENVS = ("OEQ", "PAS")

# Restrict the empathy judge to the four conditions reported in the paper's
# main empathy comparison (plot labels in plot_explore_mitigations.py):
#   baseline             -> "no prompt"
#   be_critical          -> "be critical"
#   no_syco_critical     -> "(a) + (b)"
#   stakeholder_concerns -> "ours"
# All other conditions present in the explore_mitigations results CSV are
# dropped before judging.
CONDITIONS_TO_JUDGE = (
    "baseline",
    "be_critical",
    "no_syco_critical",
    "stakeholder_concerns",
)


def load_post_text(env):
    """Return dict[id -> post_text] from the env's source data file.

    For OEQ/PAS the explore_mitigations loaders produce a stable hashed id
    matching the one written into the results CSV, so a plain dict lookup
    suffices to pair each model response with the original seeker prompt.
    Asserts the env is supported and that the loader gave us a non-empty,
    unique-id frame.
    """
    assert env in SUPPORTED_ENVS, f"unsupported env {env!r}"
    # Both loaders read from paths relative to the repo root.
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        if env == "OEQ":
            df = load_oeq(OEQ_DATA_CSV)
        else:
            df = load_pas(PAS_DATA_CSV)
    finally:
        os.chdir(cwd)
    assert len(df) > 0, f"{env}: env source file produced no rows"
    assert df["id"].is_unique, f"{env}: env source ids are not unique"
    return dict(zip(df["id"], df["post_text"]))


def load_results(env, family):
    """Read explore_mitigations_results_<env>_<family>.csv, return rows with
    [id, condition, response]. Drops rows whose `response` is NaN (the
    generation step may emit empty rows on parse failure; the empathy judge
    has nothing to score there).
    """
    path = os.path.join(
        RESULTS_DIR, f"explore_mitigations_results_{env}_{family}.csv",
    )
    assert os.path.exists(path), f"missing results CSV: {path}"
    df = pd.read_csv(path)
    for col in ("id", "condition", "response"):
        assert col in df.columns, f"{path}: missing column {col!r}"
    n_before = len(df)
    df = df.dropna(subset=["response"]).copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"[{env}/{family}] dropped {n_dropped} rows with NaN response")
    present = set(df["condition"].unique())
    missing = [c for c in CONDITIONS_TO_JUDGE if c not in present]
    assert not missing, (
        f"{path}: required conditions absent from results CSV: {missing}. "
        f"Present conditions: {sorted(present)}"
    )
    n_before_cond = len(df)
    df = df[df["condition"].isin(CONDITIONS_TO_JUDGE)].copy()
    print(
        f"[{env}/{family}] kept {len(df)}/{n_before_cond} rows after filtering "
        f"to conditions {list(CONDITIONS_TO_JUDGE)}"
    )
    return df[["id", "condition", "response"]]


def format_conversation(post_text, response):
    """Build the Seeker/Supporter transcript expected by the judge prompt."""
    return f"Seeker: {post_text}\nSupporter: {response}"


def parse_score(text):
    """Return the first 1-5 digit in `text` as int, or None if absent.

    The judge prompt ends with "Respond with ONLY the number.", so the
    expected output is a single digit; we still scan for the first 1-5
    digit to tolerate stray whitespace or a leading rationale char from
    less-instructable judges.
    """
    m = re.search(r"[1-5]", text or "")
    return int(m.group(0)) if m else None


def judge_one_sub(client, model, conversation, sub_key):
    """One judge call for one sub-component.

    Returns (score_or_none, input_tokens, output_tokens, raw_text).
    """
    sub = LEND_AN_EAR_SUBCOMPONENTS[sub_key]
    prompt = build_prompt(conversation, sub)
    text, in_tok, out_tok = call_llm(client, model, prompt, max_tokens=8)
    score = parse_score(text)
    if score is None:
        warnings.warn(
            f"empathy[{sub_key}] judge produced no 1-5 digit; raw={text!r}"
        )
    return score, in_tok, out_tok, text


def run_one(client, model, env, pid, condition, post_text, response,
            print_responses=False):
    """Score one (id, condition) row across all three sub-components.

    When `print_responses` is True, prints the seeker/supporter transcript
    plus each sub-component's parsed score and raw judge output. Intended
    for small `--max-rows` debugging runs; with multiple workers the
    prints from concurrent tasks will interleave.
    """
    conversation = format_conversation(post_text, response)
    rec = {"id": pid, "condition": condition, "env": env}
    in_sum = 0
    out_sum = 0
    if print_responses:
        print(f"\n=== {env} / {pid} / condition={condition} ===")
        print(f"--- seeker (post_text) ---\n{post_text}")
        print(f"--- supporter (model response) ---\n{response}")
    for sub_key in SUB_KEYS:
        score, in_tok, out_tok, raw = judge_one_sub(
            client, model, conversation, sub_key,
        )
        rec[sub_key] = score
        in_sum += in_tok
        out_sum += out_tok
        if print_responses:
            print(f"--- judge[{sub_key}] --- parsed={score}  raw={raw!r}")
    rec["judge_input_tokens"] = in_sum
    rec["judge_output_tokens"] = out_sum
    return rec


def default_out_path(env, family, judge_model):
    return os.path.join(
        RESULTS_DIR,
        f"empathy_{env}_{family}_judge-{model_slug(judge_model)}.csv",
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--env", required=True, choices=SUPPORTED_ENVS,
                   help="env whose results CSV to judge")
    p.add_argument("--family", required=True,
                   help="<gen-slug>_<judge-slug> segment matching the "
                        "explore_mitigations_results_<env>_<family>.csv to read")
    p.add_argument("--judge-model", default="openai/gpt-5.4-mini",
                   help="judge model id (default: openai/gpt-5.4-mini; the "
                        "empathy judge issues 3 calls per (id, condition), "
                        "so the mini model is the default to keep cost in "
                        "check — pass --judge-model openai/gpt-5.4 to match "
                        "the explore_mitigations judge instead)")
    p.add_argument("--workers", type=int, default=12,
                   help="ThreadPoolExecutor size (default: 12)")
    p.add_argument("--max-rows", type=int, default=None,
                   help="cap input rows for a smoke test (default: all)")
    p.add_argument("--out", default=None,
                   help="override the output CSV path")
    p.add_argument("--resume", action="store_true",
                   help="load any existing <out>.checkpoint.jsonl and skip "
                        "(id, condition) pairs already completed there")
    p.add_argument("--print-responses", action="store_true",
                   help="for each task, print the seeker post, the model "
                        "response, and each sub-component's parsed score "
                        "and raw judge output. Use with --max-rows 1 "
                        "--workers 1 for a clean single-task trace.")
    args = p.parse_args()

    out_path = args.out or default_out_path(
        args.env, args.family, args.judge_model,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"env={args.env} family={args.family} judge={args.judge_model}")
    print(f"out={out_path}")

    post_text_by_id = load_post_text(args.env)
    results = load_results(args.env, args.family)
    if args.max_rows is not None:
        results = results.head(args.max_rows)
        print(f"--max-rows: capped input to {len(results)} rows")

    missing = set(results["id"]) - set(post_text_by_id)
    assert not missing, (
        f"{len(missing)} ids in results CSV have no paired post_text in the "
        f"env source file (e.g. {sorted(missing)[:3]})"
    )

    tasks = []
    for _, row in results.iterrows():
        tasks.append((
            row["id"], row["condition"],
            post_text_by_id[row["id"]], row["response"],
        ))
    print(f"total tasks (id × condition): {len(tasks)}")

    ckpt_path = checkpoint_path_for(out_path)
    if args.resume:
        checkpoint = load_checkpoint(ckpt_path)
        print(f"--resume: loaded {len(checkpoint)} rows from {ckpt_path}")
    else:
        checkpoint = {}
        if os.path.exists(ckpt_path):
            print(f"overwriting existing checkpoint {ckpt_path}")
            os.remove(ckpt_path)
    remaining = [t for t in tasks if (t[0], t[1]) not in checkpoint]
    print(f"remaining after checkpoint: {len(remaining)}")

    client = make_client(args.judge_model)
    ckpt_lock = threading.Lock()
    results_acc = list(checkpoint.values())
    errors = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(
                run_one, client, args.judge_model, args.env,
                pid, cond, post, resp, args.print_responses,
            )
            for (pid, cond, post, resp) in remaining
        ]
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                rec = f.result()
            except Exception as e:
                errors.append(e)
                continue
            results_acc.append(rec)
            append_checkpoint(ckpt_path, rec, ckpt_lock)
            if i % 25 == 0 or i == len(futures):
                print(f"  done {i}/{len(futures)}")

    if errors:
        n_done_this_run = len(results_acc) - len(checkpoint)
        print(f"\n{len(errors)} task(s) raised this run. "
              f"Checkpoint preserves {n_done_this_run} completed this run "
              f"(plus {len(checkpoint)} from prior runs). Re-run with --resume.")
        raise errors[0]

    df = pd.DataFrame(results_acc)
    df = df.sort_values(["condition", "id"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(df)} rows)")
    for sub in SUB_KEYS:
        n_total = len(df)
        n_nan = int(df[sub].isna().sum())
        mean = df[sub].dropna().mean()
        print(f"  {sub:30s} mean={mean:.2f}  ({n_nan}/{n_total} unparsed)")


if __name__ == "__main__":
    main()
