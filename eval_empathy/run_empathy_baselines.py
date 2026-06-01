"""Run three response-generation baselines on the EPITOME emotional-reactions
prompts and score each response with the Lend an Ear empathy judge.

Pipeline (for a uniform random sample of N seeker posts):
  1. Sample N distinct `seeker_post` strings from
     data/emotional-reactions-reddit.csv (uniform, fixed --seed).
  2. For each post, generate a supporter response under three methods:
       no_prompt            -> the bare post (explore_mitigations "baseline")
       be_critical          -> "Be critical." prepended (explore_mitigations
                               "be_critical")
       stakeholder_concerns_v2 -> the multi-stakeholder concerns-and-judge
                               method (our_mitigation_v2, run as the OEQ
                               open-ended condition)
     Generation reuses the canonical condition values from
     explore_mitigations.get_conditions("OEQ") and the same prefix-assembly /
     callable-dispatch logic as explore_mitigations.run_one, so a method here
     produces byte-identical prompts to the main harness.
  3. Score each (post, response) with judge_empathy.run_one — three judge
     calls per response (validating_emotions, demonstrating_understanding,
     dismissing_emotions), each a 1-5 Likert score.

Each (post_id, method) is one task. Tasks run in a ThreadPoolExecutor and
each completed record is appended to <out>.checkpoint.jsonl immediately, so a
crash is recoverable with --resume (same checkpoint contract as
explore_mitigations.py / judge_empathy.py).

Outputs (under eval_empathy/results/ by default):
  - <out>.csv          one row per (post_id, method) with the three empathy
                       sub-scores + token counts. Column schema
                       (id, condition, validating_emotions,
                       demonstrating_understanding, dismissing_emotions, ...)
                       matches judge_empathy.py's output so plot_empathy-style
                       code and plot_empathy_baselines.py can read it.
  - <out>.responses.jsonl  one record per (post_id, method) carrying the full
                       seeker_post, generated response, and the v2 stakeholder
                       trace (when present) for auditing.
  - <out>.checkpoint.jsonl  per-task checkpoint (full record incl. response).

Generation model is --model (default claude-sonnet-4-6, matching
explore_mitigations); empathy judge is --judge-model (default
openai/gpt-5.4-mini, matching judge_empathy). Both authenticate from the same
env vars make_client uses (ANTHROPIC_API_KEY / OPENROUTER_API_KEY).

CLI:
  --data PATH        emotional-reactions CSV (default: data/emotional-reactions-reddit.csv)
  --n N              number of distinct posts to sample (default: 100)
  --seed S           sampling seed (default: 0)
  --model M          generation model (default: claude-sonnet-4-6)
  --judge-model M    empathy judge model (default: openai/gpt-5.4-mini)
  --max-tokens N     generation max_tokens (default: 2048)
  --workers N        thread pool size (default: 10)
  --out PATH         output CSV path (default: results/empathy_baselines_<gen>_<judge>.csv)
  --resume           resume from <out>.checkpoint.jsonl
"""

import argparse
import concurrent.futures
import json
import os
import sys
import threading

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
MITIGATIONS_DIR = os.path.join(REPO_ROOT, "mitigations")
sys.path.insert(0, HERE)
sys.path.insert(0, MITIGATIONS_DIR)

from explore_mitigations import (
    make_client, model_slug, call_llm,
    get_conditions, PROMPT_TEMPLATE_OEQ,
    checkpoint_path_for, load_checkpoint, append_checkpoint,
)
import judge_empathy as je

# Env tag carried on every record. The empathy judge is env-agnostic (it only
# needs a Seeker/Supporter transcript), but get_conditions / the v2 method
# dispatch on env_name, and "OEQ" is the open-ended advice path whose prompt
# assembly (bare post, no YTA/NTA template) matches an emotional-support turn.
GEN_ENV = "OEQ"
RECORD_ENV = "emotional_reactions"
DEFAULT_DATA = os.path.join(REPO_ROOT, "data", "emotional-reactions-reddit.csv")

# The three methods to compare, named by their canonical key in
# get_conditions("OEQ"). Using the canonical keys keeps generation
# byte-identical to the main harness AND makes the stored `condition` column
# match plot_explore_mitigations.CONDITION_COLORS / display_name, which render
# "baseline" as "no prompt", "be_critical" as "be critical.", and
# "stakeholder_concerns_v2" as "prompt-stakeholders v2".
METHODS = ["baseline", "be_critical", "stakeholder_concerns_v2"]

SUB_KEYS = je.SUB_KEYS
RESULTS_DIR = os.path.join(HERE, "results")


def sample_posts(data_path, n, seed):
    """Return a list of (post_id, seeker_post) for n distinct seeker posts.

    Assumptions / validations:
      - The CSV has a `seeker_post` column (the EPITOME schema) and parses as
        a normal CSV (a stray HTML download will lack the column and fail
        here rather than silently producing garbage).
      - Posts are deduplicated on the exact `seeker_post` string before
        sampling, so the sample is n *distinct* prompts, not n rows (the raw
        file has duplicate posts across response rows).
      - n must not exceed the number of distinct non-empty posts.
    post_id is a stable `er_<row-index-in-dedup-frame>` so reruns with the
    same --seed reproduce the same ids.
    """
    assert os.path.exists(data_path), f"missing data CSV: {data_path}"
    df = pd.read_csv(data_path)
    assert "seeker_post" in df.columns, (
        f"{data_path}: no 'seeker_post' column (found {list(df.columns)}). "
        "Is this the EPITOME emotional-reactions CSV?"
    )
    posts = df["seeker_post"].dropna().astype(str).str.strip()
    posts = posts[posts != ""]
    distinct = posts.drop_duplicates().reset_index(drop=True)
    assert len(distinct) >= n, (
        f"requested n={n} but only {len(distinct)} distinct non-empty "
        f"seeker posts available"
    )
    sampled = distinct.sample(n=n, random_state=seed).sort_index()
    return [(f"er_{idx}", text) for idx, text in sampled.items()]


def generate_response(gen_client, gen_model, post_text, condition,
                      max_tokens, env_name):
    """Generate one supporter response under `condition`.

    Mirrors the open-ended branch of explore_mitigations.run_one so prompts
    are identical to the main harness:
      - callable condition (the v2 method): invoked as
        fn(client, model, post_text, max_tokens, env_name) ->
        (text, in_tok, out_tok, trace).
      - string / (prefix, suffix) condition: prompt is
        prefix + PROMPT_TEMPLATE_OEQ.format(post=post_text + suffix), one
        call_llm. trace is None.
    Returns (text, input_tokens, output_tokens, trace).
    """
    if callable(condition):
        return condition(gen_client, gen_model, post_text, max_tokens, env_name)
    if isinstance(condition, tuple):
        prefix, suffix = condition
    else:
        prefix, suffix = condition, ""
    prompt = prefix + PROMPT_TEMPLATE_OEQ.format(post=post_text + suffix)
    text, in_tok, out_tok = call_llm(gen_client, gen_model, prompt, max_tokens)
    return text, in_tok, out_tok, None


def run_trial(gen_client, gen_model, judge_client, judge_model,
              post_id, seeker_post, method_name, condition, max_tokens):
    """Generate a response for one (post, method) and score its empathy.

    Returns a single JSON-able record carrying the empathy sub-scores
    (from judge_empathy.run_one), generation + judge token counts, the full
    response text, and the method's trace (v2 stakeholder data, else None).
    """
    response, gen_in, gen_out, trace = generate_response(
        gen_client, gen_model, seeker_post, condition, max_tokens, GEN_ENV,
    )
    assert response is not None and str(response).strip(), (
        f"empty response for {post_id}/{method_name}"
    )
    rec = je.run_one(
        judge_client, judge_model, RECORD_ENV,
        post_id, method_name, seeker_post, response,
    )
    # je.run_one already set id, condition, env, the three sub-keys, and
    # judge_input/output_tokens. Add generation side + audit fields.
    rec["seeker_post"] = seeker_post
    rec["response"] = response
    rec["gen_input_tokens"] = gen_in
    rec["gen_output_tokens"] = gen_out
    rec["trace"] = trace
    return rec


def default_out_path(gen_model, judge_model):
    return os.path.join(
        RESULTS_DIR,
        f"empathy_baselines_{model_slug(gen_model)}_{model_slug(judge_model)}.csv",
    )


def write_outputs(records, out_path):
    """Write the scores CSV and the responses JSONL from completed records.

    The CSV holds the plotting columns (id, condition, the three sub-scores,
    token counts) so it is a drop-in for judge_empathy.py's schema. The
    JSONL holds the bulky audit fields (seeker_post, response, trace).
    """
    df = pd.DataFrame(records).sort_values(["condition", "id"]).reset_index(drop=True)
    csv_cols = [
        "id", "condition", "env",
        *SUB_KEYS,
        "gen_input_tokens", "gen_output_tokens",
        "judge_input_tokens", "judge_output_tokens",
    ]
    df[csv_cols].to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(df)} rows)")

    base, _ = os.path.splitext(out_path)
    resp_path = base + ".responses.jsonl"
    with open(resp_path, "w") as f:
        for rec in records:
            f.write(json.dumps({
                "id": rec["id"],
                "condition": rec["condition"],
                "seeker_post": rec["seeker_post"],
                "response": rec["response"],
                "trace": rec.get("trace"),
            }) + "\n")
    print(f"wrote {resp_path}")
    return df


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data", default=DEFAULT_DATA,
                   help="emotional-reactions CSV (default: "
                        "data/emotional-reactions-reddit.csv)")
    p.add_argument("--n", type=int, default=100,
                   help="number of distinct posts to sample (default: 100)")
    p.add_argument("--seed", type=int, default=0,
                   help="sampling seed (default: 0)")
    p.add_argument("--model", default="claude-sonnet-4-6",
                   help="generation model (default: claude-sonnet-4-6)")
    p.add_argument("--judge-model", default="openai/gpt-5.4-mini",
                   help="empathy judge model (default: openai/gpt-5.4-mini)")
    p.add_argument("--max-tokens", type=int, default=2048,
                   help="generation max_tokens (default: 2048)")
    p.add_argument("--workers", type=int, default=10,
                   help="ThreadPoolExecutor size (default: 10)")
    p.add_argument("--out", default=None, help="override output CSV path")
    p.add_argument("--resume", action="store_true",
                   help="resume from <out>.checkpoint.jsonl")
    args = p.parse_args()

    out_path = args.out or default_out_path(args.model, args.judge_model)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"data={args.data}  n={args.n}  seed={args.seed}")
    print(f"gen_model={args.model}  judge_model={args.judge_model}")
    print(f"out={out_path}")

    posts = sample_posts(args.data, args.n, args.seed)
    print(f"sampled {len(posts)} distinct posts")

    conditions = get_conditions(GEN_ENV)
    for method_name in METHODS:
        assert method_name in conditions, (
            f"method {method_name!r} not in get_conditions({GEN_ENV!r})"
        )

    # One task per (post, method).
    tasks = []
    for post_id, seeker_post in posts:
        for method_name in METHODS:
            tasks.append((post_id, seeker_post, method_name, conditions[method_name]))
    print(f"total tasks (post x method): {len(tasks)}")

    ckpt_path = checkpoint_path_for(out_path)
    if args.resume:
        checkpoint = load_checkpoint(ckpt_path)
        print(f"--resume: loaded {len(checkpoint)} rows from {ckpt_path}")
    else:
        checkpoint = {}
        if os.path.exists(ckpt_path):
            print(f"overwriting existing checkpoint {ckpt_path}")
            os.remove(ckpt_path)
    remaining = [t for t in tasks if (t[0], t[2]) not in checkpoint]
    print(f"remaining after checkpoint: {len(remaining)}")

    gen_client = make_client(args.model)
    judge_client = make_client(args.judge_model)
    ckpt_lock = threading.Lock()
    results_acc = list(checkpoint.values())
    errors = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(
                run_trial, gen_client, args.model, judge_client,
                args.judge_model, post_id, seeker_post, method_name, cond,
                args.max_tokens,
            )
            for (post_id, seeker_post, method_name, cond) in remaining
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
        n_done = len(results_acc) - len(checkpoint)
        print(f"\n{len(errors)} task(s) raised this run. Checkpoint preserves "
              f"{n_done} completed this run (plus {len(checkpoint)} prior). "
              f"Re-run with --resume.")
        raise errors[0]

    df = write_outputs(results_acc, out_path)

    # Per-method summary so a smoke run shows the numbers without plotting.
    print("\n=== per-method mean empathy sub-scores ===")
    for method_name in METHODS:
        sub = df[df["condition"] == method_name]
        line = [f"{method_name:24s} n={len(sub):3d}"]
        for k in SUB_KEYS:
            vals = sub[k].dropna()
            n_nan = int(sub[k].isna().sum())
            mean = vals.mean() if len(vals) else float("nan")
            line.append(f"{k}={mean:.2f}({n_nan}nan)")
        print("  " + "  ".join(line))


if __name__ == "__main__":
    main()
