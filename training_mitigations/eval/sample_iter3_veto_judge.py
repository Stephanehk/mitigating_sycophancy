"""Sample representative round-3 (iter 3) training responses from the
completed PAS / OEQ n1000 DPO runs and judge them with the paper's Azure
gpt-4o endorse judge, for manual comparison of:

  1. veto      - did the simulated-stakeholder sim veto this response during
                 training? (read from responses_iter_3.jsonl, saved by trainer)
  2. judge     - ELEPHANT 4-way endorse label (0 challenges / 1 explicit
                 affirm / 2 implicit affirm / 3 neutral), scored NOW via the
                 reused judge (this is not saved during training)
  3. my_label  - left blank; filled in by hand afterwards.

Sampling (seed 42, per env):
  - Pool = all (prompt, response) slots in responses_iter_3.jsonl
    (1000 prompts x 10 samples). Veto status of a slot = membership of the
    response TEXT in that prompt's `vetoed` list (matches trainer semantics:
    duplicate texts share one status).
  - Draw N_PER_ENV slots stratified by veto status so the sample matches the
    overall round-3 veto rate (each stratum clamped to >= 1 so both classes
    appear), at most one response per prompt for prompt diversity.

Assumptions (asserted): run dirs + jsonl exist, 10 responses per prompt,
vetoed is a subset of responses, Azure creds in env, judge returns a valid
0-3 label for every sampled row.

Outputs (in training_mitigations/iter3_veto_judge_labels/):
  - samples_scored.csv : one row per sampled response with veto + judge label
                         and an empty my_label column
  - samples.md         : full prompt/response text per sample, for hand review
"""

import argparse
import importlib.util
import json
import os
import random

import pandas as pd

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_HERE)

N_PER_ENV = 20
SEED = 42
RUNS = {
    "PAS": os.path.join(_HERE, "runs", "dpo_lora_pas_qwen3-8b_n1000"),
    "OEQ": os.path.join(_HERE, "runs", "dpo_lora_oeq_qwen3-8b_n1000"),
}
OUT_DIR = os.path.join(_HERE, "iter3_veto_judge_labels")
JUDGE_LABEL_NAMES = {0: "challenges", 1: "explicit-affirm",
                     2: "implicit-affirm", 3: "neutral"}


def load_judge_module():
    """Import the paper's Azure endorse judge (mirrors
    eval_endorsement_heldout.load_judge_module; duplicated to avoid that
    module's torch/llama imports for this CPU-only script)."""
    path = os.path.join(_REPO_ROOT, "osfstorage-archive", "Code",
                        "evaluate_action_endorsement_stephane.py")
    assert os.path.exists(path), f"judge script not found: {path}"
    spec = importlib.util.spec_from_file_location("endorse_judge", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_iter_slots(run_dir, it):
    """Read responses_iter_<it>.jsonl -> list of slot dicts
    (prompt_idx, prompt, response, vetoed). One slot per sampled response."""
    path = os.path.join(run_dir, f"responses_iter_{it}.jsonl")
    assert os.path.exists(path), f"missing {path}"
    slots = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            responses, vetoed = rec["responses"], rec["vetoed"]
            assert len(responses) == 10, (
                f"expected 10 responses, got {len(responses)} "
                f"(prompt_idx={rec['prompt_idx']})")
            vetoed_set = set(vetoed)
            assert vetoed_set <= set(responses), (
                f"vetoed not a subset of responses (prompt_idx={rec['prompt_idx']})")
            for resp in responses:
                slots.append({"prompt_idx": rec["prompt_idx"],
                              "prompt": rec["prompt"],
                              "response": resp,
                              "vetoed": resp in vetoed_set})
    return slots


def sample_representative(slots, n, rng):
    """Stratified sample of n slots matching the pool's veto rate (each
    stratum >= 1), at most one response per prompt. Returns list of slots."""
    veto_rate = sum(s["vetoed"] for s in slots) / len(slots)
    n_veto = min(max(round(n * veto_rate), 1), n - 1)
    quotas = {True: n_veto, False: n - n_veto}
    shuffled = slots[:]
    rng.shuffle(shuffled)
    picked, used_prompts = [], set()
    for s in shuffled:
        if quotas[s["vetoed"]] > 0 and s["prompt_idx"] not in used_prompts:
            picked.append(s)
            used_prompts.add(s["prompt_idx"])
            quotas[s["vetoed"]] -= 1
    assert quotas == {True: 0, False: 0}, f"unfilled quotas: {quotas}"
    return picked, veto_rate


def matched_sample(prev_csv, it, rng):
    """For each (env, prompt_idx) in a prior sample CSV, draw one random
    iteration-<it> response for that SAME prompt and record its veto status.
    Preserves the prior row order so outputs align side by side."""
    prev = pd.read_csv(prev_csv)
    by_env = {}
    for env, run_dir in RUNS.items():
        path = os.path.join(run_dir, f"responses_iter_{it}.jsonl")
        assert os.path.exists(path), f"missing {path}"
        with open(path) as f:
            by_env[env] = {r["prompt_idx"]: r for r in map(json.loads, f)}
    rows = []
    for _, p in prev.iterrows():
        rec = by_env[p["env"]][p["prompt_idx"]]
        assert rec["prompt"] == p["sentence"], (
            f"prompt mismatch at {p['env']} idx {p['prompt_idx']}")
        resp = rng.choice(rec["responses"])
        rows.append({"env": p["env"], "prompt_idx": p["prompt_idx"],
                     "sentence": rec["prompt"], "response": resp,
                     "vetoed": resp in set(rec["vetoed"])})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=3,
                        help="which training iteration's responses to sample")
    parser.add_argument("--match-sample", default=None,
                        help="prior samples_scored CSV; reuse its exact "
                             "(env, prompt_idx) set instead of fresh sampling")
    parser.add_argument("--n", type=int, default=N_PER_ENV,
                        help="responses to sample per env (default 20)")
    parser.add_argument("--out-dir", default=OUT_DIR,
                        help="output directory (default the original "
                             "iter3_veto_judge_labels dir)")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(SEED)
    # Original dir keeps its legacy naming (iter-3 files untagged); any other
    # out dir always tags files with the iteration for clarity.
    if out_dir == OUT_DIR:
        tag = f"_iter{args.iter}" if args.iter != 3 else ""
    else:
        tag = f"_iter{args.iter}"

    if args.match_sample:
        df = matched_sample(args.match_sample, args.iter, rng)
        for env in RUNS:
            n_veto = int(df[df["env"] == env]["vetoed"].sum())
            print(f"[{env}] matched prompts, iter {args.iter}: "
                  f"{n_veto} vetoed / {args.n - n_veto} non-vetoed")
    else:
        rows = []
        for env, run_dir in RUNS.items():
            slots = load_iter_slots(run_dir, args.iter)
            picked, veto_rate = sample_representative(slots, args.n, rng)
            n_veto = sum(s["vetoed"] for s in picked)
            print(f"[{env}] pool={len(slots)} slots, iter-{args.iter} veto rate "
                  f"{veto_rate:.1%} -> sampled {n_veto} vetoed / "
                  f"{args.n - n_veto} non-vetoed")
            for s in picked:
                rows.append({"env": env, "prompt_idx": s["prompt_idx"],
                             "sentence": s["prompt"], "response": s["response"],
                             "vetoed": s["vetoed"]})
        df = pd.DataFrame(rows)

    # Judge with the reused Azure scorer: reads sentence + response, writes
    # endorse_response (raw digit string) per row.
    jm = load_judge_module()
    jm.client = jm.build_client(False)
    scored_path = os.path.join(out_dir, f"samples_scored{tag}.csv")
    scored = jm.parallel_score(df, client=jm.client, advice_col="response",
                               max_workers=4, checkpoint_every=args.n,
                               out_path=scored_path)

    # Azure's RAI content filter rejects some prompts outright (e.g. sexual
    # content at medium severity); the scorer records those cells as "ERROR".
    # Keep the sampled row (the sample stays representative) and mark the
    # judge column explicitly unscored instead of crashing — but only tolerate
    # a couple, since widespread parse failures would mean a broken judge.
    labels = scored["endorse_response"].map(jm.first_label)
    n_unscored = int(labels.isna().sum())
    max_unscored = max(2, round(0.03 * len(scored)))
    assert n_unscored <= max_unscored, (
        f"{n_unscored}/{len(scored)} judge labels unparsable")
    if n_unscored:
        print(f"[judge] {n_unscored} row(s) hit the Azure content filter; "
              f"marked content-filtered")
    scored["judge_label"] = labels.astype("Int64")
    scored["judge_label_name"] = labels.map(JUDGE_LABEL_NAMES).fillna(
        "content-filtered")
    scored["my_label"] = ""
    scored.to_csv(scored_path, index=False)
    print(f"wrote {scored_path}")

    md_path = os.path.join(out_dir, f"samples{tag}.md")
    sampling_note = (f"same (env, prompt_idx) set as {args.match_sample}, one "
                     f"random iter-{args.iter} response per prompt"
                     if args.match_sample else
                     f"{args.n} per env, stratified to iter-{args.iter} veto rate")
    with open(md_path, "w") as f:
        f.write(f"# Round-{args.iter} training responses: veto vs judge vs manual label\n\n"
                f"Runs: {', '.join(os.path.basename(d) for d in RUNS.values())}; "
                f"seed {SEED}; {sampling_note}.\n")
        for i, r in scored.iterrows():
            f.write(f"\n---\n\n## {r['env']} #{i} (prompt_idx={r['prompt_idx']})\n\n"
                    f"**Vetoed in training:** {'YES' if r['vetoed'] else 'no'}  \n"
                    f"**Judge:** {r['judge_label']} ({r['judge_label_name']})  \n"
                    f"**My label:** _(pending)_\n\n"
                    f"**Prompt:** {r['sentence']}\n\n"
                    f"**Response:**\n\n{r['response']}\n")
    print(f"wrote {md_path}")

    for env in RUNS:
        sub = scored[scored["env"] == env]
        print(f"[{env}] judge label counts:",
              sub["judge_label_name"].value_counts().to_dict())


if __name__ == "__main__":
    main()
