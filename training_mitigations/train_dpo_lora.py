"""Fine-tune a local HuggingFace model via DPO + LoRA against preference
pairs derived from the `SimulatedStakeholders` veto signal, on a single
dataset env (OEQ or PAS).

This is the shared training entry point. The
`run_train_<env>.sbatch` wrappers invoke it with env-specific defaults.

What it does:
  1. Load the dataset for `--env` from disk.
  2. Hold out `--test-n` prompts as the eval split (using the harness's
     `sample_uniform` with `--seed`). Their ids are written to
     `<output_dir>/test_ids.json` so a downstream eval script can
     reproduce the same split.
  3. From the disjoint complement, sample `--train-n` prompts as the DPO
     training prompts. Train ids → `<output_dir>/train_ids.json`.
  4. Load TWO copies of the model (default `Qwen/Qwen3-8B`):
       - `policy`  on `--device`     (trainable; LoRA-adapted)
       - `sim_base` on `--sim-device` (frozen; never updated)
     The policy is what gets sampled from each iteration and what DPO
     updates. The sim_base drives every LLM call inside the
     stakeholder pipeline (identify / filter / concerns / veto) so
     the veto judge's behavior does not drift with the policy.
  5. Construct `SimulatedStakeholders` wired to sim_base;
     `use_weak_model=True` routes the identify/filter/concerns calls
     through `weak_pipeline.py`.
  6. Construct `Trainer(sim_stake, policy, tokenizer, n_samples,
     temperature, max_new_tokens)` and call `trainer.train(train_prompts,
     output_dir, num_iterations=K, ...)`, which:
       - wraps the policy with a LoRA adapter once;
       - for each of K iterations: samples N candidate responses per
         prompt from the current policy, vetoes via the (frozen) sim,
         cross-pairs vetoed × non-vetoed into a `datasets.Dataset`,
         logs the count of all-vetoed prompts, runs one DPO epoch on
         the iteration's fresh dataset;
       - all-vetoed prompts contribute 0 pairs that iteration; the
         next iteration re-samples them from the now-updated policy
         and may produce non-vetoed responses that contribute pairs;
       - saves the final LoRA adapter to `output_dir`.

Per-env user prompt format used for sampling AND for the DPO prompt
column (so DPO sees the same surface the responses were sampled
against): bare post_text for OEQ and PAS (matches the harness's
`PROMPT_TEMPLATE_OEQ = "{post}"`).

The script does not call any external API. Both model copies run
locally on the configured GPUs.
"""

import argparse
import json
import os
import re
import sys

# Make `mitigations/` (the harness) and the local `training_mitigations/`
# importable so `our_mitigation*`, `core.*`, `weak_pipeline`,
# `simulated_stakeholders`, and `trainer` all resolve.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "mitigations"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import our_mitigation
import our_mitigation_v2
import weak_pipeline

from explore_mitigations import (
    OEQ_DATA_CSV, PAS_DATA_CSV,
    DEFAULT_MAX_TOKENS, DEFAULT_SEED,
    load_oeq, load_pas,
    sample_uniform,
)

from llama_local import DEFAULT_HF_MODEL_ID, load_llama, make_call_llm
from mem_probe import start_mem_sampler
from simulated_stakeholders import SimulatedStakeholders
from trainer import Trainer


# Same tolerant stakeholder parser used in `test_v2_llama.py`. The harness
# default `parse_stakeholders` asserts strictly fewer than `MAX_STAKEHOLDERS
# =10` items; Qwen3-8B occasionally emits exactly 10 (or theme-shaped
# strings) on OEQ-style prompts, which would abort the entire train run.
# Truncate at a higher local cap instead.
_LOCAL_MAX_STAKEHOLDERS = 15


def _tolerant_parse_stakeholders(text, min_count=None, max_count=None):
    """Drop the strict-cap assert from `our_mitigation.parse_stakeholders`
    so a single out-of-spec model emission doesn't kill the whole train
    run. Truncates at `_LOCAL_MAX_STAKEHOLDERS`. Behavior is otherwise
    identical: strip bullet/number prefixes and whitespace, keep non-
    empty lines.
    """
    items = []
    for raw in (text or "").splitlines():
        s = raw.strip()
        s = re.sub(r"^(?:[\-\*•]|\d+[\.\)])\s*", "", s).strip()
        s = s.strip("*_`").strip()
        if s:
            items.append(s)
    if len(items) > _LOCAL_MAX_STAKEHOLDERS:
        items = items[:_LOCAL_MAX_STAKEHOLDERS]
    return items


def _resolve_csv(env_name):
    """Return the abs path to the env's CSV, anchored to the repo root so
    the script runs identically from any cwd.
    """
    rel = {
        "OEQ":  OEQ_DATA_CSV,
        "PAS":  PAS_DATA_CSV,
    }[env_name]
    path = rel if os.path.isabs(rel) else os.path.join(_REPO_ROOT, rel)
    assert os.path.exists(path), f"missing {env_name} csv: {path}"
    return path


def _load_data(env_name, csv_path):
    """Dispatch to the per-env loader. Each returns a DataFrame with at
    least `id` and `post_text`.
    """
    if env_name == "OEQ":
        return load_oeq(csv_path)
    if env_name == "PAS":
        return load_pas(csv_path)
    raise AssertionError(f"unknown env_name: {env_name!r}")


def _split_train_test(env_name, data, train_n, test_n, seed):
    """Pick `test_n` test rows and a disjoint `train_n` train rows from
    `data`.

    OEQ / PAS use `sample_uniform`. Disjointness is enforced by dropping
    the test ids from the pool before sampling train. The train seed is
    `seed + 1000` so the two samplers don't collide.

    Returns (train_df, test_df).
    """
    test_df = sample_uniform(data, test_n, seed)
    pool = data[~data["id"].isin(set(test_df["id"]))].reset_index(drop=True)
    train_df = sample_uniform(pool, train_n, seed + 1000)
    assert set(test_df["id"]).isdisjoint(set(train_df["id"])), (
        "train/test id overlap — split logic is broken"
    )
    return train_df, test_df


def _format_prompt(env_name, post_text):
    """Return the user message the model under training will see at both
    sampling time and DPO time.
    """
    if env_name in ("OEQ", "PAS"):
        return post_text
    raise AssertionError(f"unknown env_name: {env_name!r}")


def _write_ids(path, df):
    """Dump the `id` column of `df` to `path` as a JSON list. Used for
    the test_ids.json / train_ids.json split records.
    """
    with open(path, "w") as f:
        json.dump([str(x) for x in df["id"].tolist()], f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--env", required=True, choices=("OEQ", "PAS"),
        help="dataset env to train on",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="directory the LoRA adapter and split-id files are written to",
    )
    parser.add_argument(
        "--model-id", default=DEFAULT_HF_MODEL_ID,
        help=f"HF model id of the policy under training (default: {DEFAULT_HF_MODEL_ID})",
    )
    parser.add_argument(
        "--cache-dir", default=os.environ.get("HF_HOME"),
        help="HF cache dir; defaults to $HF_HOME if set, else HF's default",
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="torch device for the trainable policy model (default: cuda:0)",
    )
    parser.add_argument(
        "--sim-device", default="cuda:1",
        help="torch device for the frozen sim_base model (default: cuda:1). "
             "Set equal to --device to share one GPU.",
    )
    parser.add_argument(
        "--num-iterations", type=int, default=3,
        help="number of outer training iterations. Each iteration "
             "re-samples all training prompts from the current policy, "
             "rebuilds the (chosen, rejected) preference dataset, and "
             "runs one DPO epoch on it. Prompts whose responses are all "
             "vetoed in iter K contribute 0 pairs that iter; iter K+1 "
             "re-samples them with the updated policy. Default: 3.",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"split / sampler seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--train-n", type=int, required=True,
        help="number of training prompts",
    )
    parser.add_argument(
        "--test-n", type=int, required=True,
        help="number of held-out test prompts",
    )
    parser.add_argument(
        "--n-samples", type=int, default=10,
        help="N — candidate responses sampled per training prompt (default: 10)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"max new tokens for stakeholder / veto LLM calls (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--sample-max-tokens", type=int, default=512,
        help="max new tokens per sampled candidate response (default: 512)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="sampling temperature for candidate generation (default: 1.0)",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.95,
        help="nucleus sampling top-p for candidate generation (default: 0.95)",
    )
    parser.add_argument(
        "--beta", type=float, default=0.1,
        help="DPO beta (default: 0.1)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-5,
        help="DPO learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--num-train-epochs", type=float, default=1.0,
        help="DPO epochs over the preference dataset (default: 1.0)",
    )
    parser.add_argument(
        "--per-device-batch-size", type=int, default=1,
        help="DPO per-device batch size (default: 1)",
    )
    parser.add_argument(
        "--grad-accum", type=int, default=8,
        help="DPO gradient accumulation steps (default: 8)",
    )
    parser.add_argument(
        "--max-length", type=int, default=3072,
        help="DPO total prompt+completion sequence length cap (default: 3072). "
             "trl>=1.x has no separate prompt cap; sequences over this are "
             "truncated per DPOConfig.truncation_mode (keep_start).",
    )
    parser.add_argument(
        "--lora-r", type=int, default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=32,
        help="LoRA alpha (default: 32)",
    )
    parser.add_argument(
        "--lora-dropout", type=float, default=0.05,
        help="LoRA dropout (default: 0.05)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Dataset + train/test split.
    csv_path = _resolve_csv(args.env)
    print(f"[data] env={args.env}  csv={csv_path}")
    data = _load_data(args.env, csv_path)
    print(f"[data] pool size: {len(data)} rows")
    train_df, test_df = _split_train_test(
        args.env, data, args.train_n, args.test_n, args.seed,
    )
    print(f"[data] held out test_n={len(test_df)}  train_n={len(train_df)}  seed={args.seed}")

    test_ids_path = os.path.join(args.output_dir, "test_ids.json")
    train_ids_path = os.path.join(args.output_dir, "train_ids.json")
    _write_ids(test_ids_path, test_df)
    _write_ids(train_ids_path, train_df)
    print(f"[data] wrote {test_ids_path}")
    print(f"[data] wrote {train_ids_path}")

    train_prompts = [_format_prompt(args.env, t) for t in train_df["post_text"].tolist()]

    # 2. Two model copies:
    #    - policy: trainable, on args.device. Sampled from each
    #      iteration and updated by DPO.
    #    - sim_base: frozen, on args.sim_device. Drives every LLM call
    #      in the stakeholder pipeline (identify, filter, concerns,
    #      veto) so the judge's behavior never drifts with the policy.
    print(f"[load] policy   {args.model_id} -> {args.device}  cache_dir={args.cache_dir}")
    model, tokenizer = load_llama(
        model_id=args.model_id,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    print(f"[load] sim_base {args.model_id} -> {args.sim_device}  (frozen)")
    sim_model, sim_tokenizer = load_llama(
        model_id=args.model_id,
        device=args.sim_device,
        cache_dir=args.cache_dir,
    )
    # Freeze sim_base — it's only used for inference inside the pipeline.
    # Disabling grad on parameters is belt-and-suspenders; eval() already
    # suppresses dropout, and we never call backward on sim outputs.
    for p in sim_model.parameters():
        p.requires_grad = False
    sim_model.eval()

    # Memory diagnostics: a background thread logs host RSS + per-GPU
    # allocator counters to mem.csv every 2s, line-buffered so the curve
    # survives the SIGKILL an OOM kill sends. Off the training path, so
    # it does not slow generate. Started after both models are resident
    # so the baseline row reflects loaded weights.
    mem_csv = os.path.join(args.output_dir, "mem.csv")
    start_mem_sampler(mem_csv)
    print(f"[mem] sampling host+GPU memory -> {mem_csv}")

    # 3. call_llm closure for the stakeholder + veto pipeline. Greedy
    # because reproducibility of the veto signal across train iterations
    # is more useful than diversity here; only the candidate-response
    # sampler inside Trainer needs stochasticity (and it has its own
    # do_sample=True path).
    sim_call_llm = make_call_llm(sim_model, sim_tokenizer, do_sample=False)

    # Rewire the harness's module-level `call_llm` bindings to the
    # frozen sim_base. Three modules hold independent imports of
    # `call_llm` (`our_mitigation`, `our_mitigation_v2`,
    # `weak_pipeline`); the SimulatedStakeholders constructor below
    # also does this, but doing it here too is harmless and makes the
    # script readable on its own. CRITICAL: the binding points at
    # sim_base, NOT the policy — the sim must never see the updated
    # policy or the veto signal would drift as training progresses.
    our_mitigation.call_llm = sim_call_llm
    our_mitigation_v2.call_llm = sim_call_llm
    weak_pipeline.call_llm = sim_call_llm

    # Tolerant stakeholder parser — patch all three modules that hold
    # an independent binding of `parse_stakeholders`.
    our_mitigation.parse_stakeholders = _tolerant_parse_stakeholders
    our_mitigation_v2.parse_stakeholders = _tolerant_parse_stakeholders
    weak_pipeline.parse_stakeholders = _tolerant_parse_stakeholders

    # 4. Veto judge — wired to sim_base. Default sim_stake routing:
    # identify + filter + concerns + veto all go through sim_call_llm
    # (the frozen sim). We don't pass concerns_call_llm / veto_call_llm
    # — the single-backend default does what we want.
    sim = SimulatedStakeholders(
        call_llm=sim_call_llm,
        model_id=args.model_id,
        client=None,
        stakeholder_max_tokens=args.max_tokens,
        use_weak_model=True,
    )

    # 5. Trainer + train. Trainer.model is the policy (trainable);
    # Trainer's `_sample_responses` calls policy.generate() directly,
    # so policy updates are picked up in the next iteration's sampling.
    trainer = Trainer(
        sim_stake=sim,
        model=model,
        tokenizer=tokenizer,
        n_samples=args.n_samples,
        max_new_tokens=args.sample_max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print(f"[train] starting iterative DPO+LoRA  iters={args.num_iterations}  output_dir={args.output_dir}")
    dpo_trainer = trainer.train(
        prompts=train_prompts,
        output_dir=args.output_dir,
        num_iterations=args.num_iterations,
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    print(f"[done] adapter saved to {args.output_dir}")
    # log_history is a list of per-step dicts; show the last entry so
    # the user can eyeball the final loss without reading the full HF
    # trainer log.
    if dpo_trainer.state.log_history:
        print(f"[done] last log entry: {dpo_trainer.state.log_history[-1]}")


if __name__ == "__main__":
    main()
