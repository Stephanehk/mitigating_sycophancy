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
  4. Stand up TWO instances of the model (default `Qwen/Qwen3-8B`):
       - `policy` on `--device`: a trainable HF model (LoRA-adapted).
       - the sim judge on `--sim-device`: frozen; never updated. Served
         by a vLLM server subprocess by default (`--sim-backend vllm`),
         whose continuous batching lets the per-(stakeholder, response)
         veto calls run concurrently (`--veto-concurrency`) instead of
         one batch-1 HF generate at a time; `--sim-backend hf` keeps
         the original in-process frozen HF copy as a fallback.
     The policy is what gets sampled from each iteration and what DPO
     updates. The sim judge drives every LLM call inside the
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
       - snapshots the adapter to `output_dir/adapter_iter_<k>` after
         each iteration so every iteration is independently loadable
         for evaluation;
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
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, "mitigations"))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "core"))

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
from simulated_stakeholders import SimulatedStakeholders, VETO_PROMPTS
from trainer import Trainer
from vllm_sim import (launch_vllm_server, make_vllm_call_llm, wait_for_vllm_server,
                      make_vllm_policy_sampler, load_lora_adapter, unload_lora_adapter)


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
        help="GPU for the frozen sim judge (default: cuda:1). With "
             "--sim-backend hf this is a torch device the sim_base model is "
             "loaded onto (set equal to --device to share one GPU). With "
             "--sim-backend vllm the GPU is given exclusively to the vLLM "
             "server subprocess and must NOT be shared with --device.",
    )
    parser.add_argument(
        "--sim-backend", default="vllm", choices=("vllm", "hf"),
        help="how the frozen sim judge is served (default: vllm). 'vllm' "
             "launches a vLLM OpenAI-compatible server subprocess on "
             "--sim-device and fans veto calls out concurrently "
             "(--veto-concurrency); 'hf' is the original in-process "
             "HF-transformers path (sequential, batch-1 — slow; kept as a "
             "fallback).",
    )
    parser.add_argument(
        "--veto-concurrency", type=int, default=16,
        help="max in-flight (response, stakeholder) veto calls when "
             "--sim-backend vllm (default: 16). Ignored (forced to 1) for "
             "--sim-backend hf, which cannot batch concurrent calls.",
    )
    parser.add_argument(
        "--vllm-gpu-mem-util", type=float, default=0.90,
        help="vLLM gpu_memory_utilization on the sim GPU (default: 0.90)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=64,
        help="prompts processed per chunk in the per-stage batched build "
             "loop (default: 64). Each chunk runs stages identify/sample/veto/"
             "preference breadth-first across its prompts.",
    )
    parser.add_argument(
        "--stage-concurrency", type=int, default=128,
        help="max in-flight LLM calls per stage in the batched build loop "
             "(default: 128) — the width of the fan-out the vLLM server "
             "continuous-batches. This is the knob that matters for the "
             "batched pipeline; --veto-concurrency only affects the legacy "
             "per-prompt veto path.",
    )
    parser.add_argument(
        "--veto-prompt", default="baseline", choices=tuple(VETO_PROMPTS),
        help="which stakeholder veto prompt to use (default: baseline). "
             "'P4' = proposal 1 (reframe to 'would you be okay with') + "
             "proposal 2 (expanded examples). See simulated_stakeholders.py.",
    )
    parser.add_argument(
        "--policy-on-vllm", action="store_true",
        help="sample candidate responses + greedy preference labels on the "
             "shared vLLM engine (base + current LoRA) instead of HF "
             "model.generate. Requires --sim-backend vllm. The engine is "
             "launched with --enable-lora and the policy adapter is hot-loaded "
             "each iteration. Validated to HF logprob parity (build_vllm_parity_check.py).",
    )
    parser.add_argument(
        "--resume-adapter", default=None,
        help="path to a saved adapter (e.g. <output_dir>/adapter_iter_1) to "
             "resume training FROM; combined with --start-iteration. Reuses the "
             "existing <output_dir>/train_ids.json split instead of resampling.",
    )
    parser.add_argument(
        "--start-iteration", type=int, default=0,
        help="0-based outer iteration to start at when resuming (e.g. 1 means "
             "iteration 1 is already done; begin at the 2nd iteration). "
             "Default 0 (fresh run).",
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
        "--mixed-sampling", action="store_true",
        help="variation 1: split the N candidates per prompt across sources "
             "instead of all from the bare post — --n-source-model from the "
             "bare post, --n-source-critical from the post + 'Be critical.', "
             "and the remaining N budget round-robin across the prompt's "
             "identified stakeholders (each voiced with its concerns). The "
             "DPO prompt column stays the bare post.",
    )
    parser.add_argument(
        "--n-source-model", type=int, default=2,
        help="mixed sampling: candidates drawn from the bare post (default: 2)",
    )
    parser.add_argument(
        "--n-source-critical", type=int, default=2,
        help="mixed sampling: candidates drawn from the post + 'Be critical.' "
             "(default: 2). The stakeholder budget is n_samples minus these two.",
    )
    parser.add_argument(
        "--self-preference", action="store_true",
        help="variation 2: for every unordered pair of NON-vetoed responses, "
             "ask the policy (the model under training) to pick the better "
             "advice (post in context) and add the resulting preference pair "
             "to the DPO dataset, on top of the veto-based pairs.",
    )
    parser.add_argument(
        "--preference-max-tokens", type=int, default=128,
        help="max new tokens per self-preference label call (default: 128)",
    )
    parser.add_argument(
        "--preference-max-pairs", type=int, default=0,
        help="per-prompt cap on self-preference pairs (0 = no cap, i.e. all "
             "C(k,2) pairs over k non-vetoed responses). Default: 0.",
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
        "--precompute-ref-batch-size", type=int, default=8,
        help="batch size for trl's reference log-prob precompute pass "
             "(default: 8). The pass is no-grad inference, so this can be "
             "much larger than --per-device-batch-size; at batch 1 it took "
             "2h+ per iteration on the n1000 runs.",
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

    assert not (args.policy_on_vllm and args.sim_backend != "vllm"), (
        "--policy-on-vllm requires --sim-backend vllm (the policy is sampled on "
        "the same vLLM engine that serves the sim)"
    )

    # 1. Dataset + train/test split.
    csv_path = _resolve_csv(args.env)
    print(f"[data] env={args.env}  csv={csv_path}")
    data = _load_data(args.env, csv_path)
    print(f"[data] pool size: {len(data)} rows")
    train_ids_path = os.path.join(args.output_dir, "train_ids.json")
    if args.resume_adapter is not None:
        # Resume: reuse the EXACT training split the original run saved, so the
        # continued iterations train on the same prompts (not a fresh sample).
        assert os.path.exists(train_ids_path), (
            f"--resume-adapter set but {train_ids_path} is missing; cannot reuse "
            f"the original split"
        )
        with open(train_ids_path) as f:
            saved_ids = json.load(f)
        by_id = {str(i): t for i, t in zip(data["id"].tolist(), data["post_text"].tolist())}
        missing = [i for i in saved_ids if i not in by_id]
        assert not missing, f"{len(missing)} saved train ids not in data (e.g. {missing[:3]})"
        train_prompts = [_format_prompt(args.env, by_id[i]) for i in saved_ids]
        print(f"[data] RESUME: reusing {len(train_prompts)} train prompts from {train_ids_path}")
    else:
        train_df, test_df = _split_train_test(
            args.env, data, args.train_n, args.test_n, args.seed,
        )
        print(f"[data] held out test_n={len(test_df)}  train_n={len(train_df)}  seed={args.seed}")
        test_ids_path = os.path.join(args.output_dir, "test_ids.json")
        _write_ids(test_ids_path, test_df)
        _write_ids(train_ids_path, train_df)
        print(f"[data] wrote {test_ids_path}")
        print(f"[data] wrote {train_ids_path}")
        train_prompts = [_format_prompt(args.env, t) for t in train_df["post_text"].tolist()]

    # 2. Two model instances:
    #    - policy: trainable HF model, on args.device. Sampled from each
    #      iteration and updated by DPO.
    #    - sim judge: frozen, on args.sim_device. Drives every LLM call
    #      in the stakeholder pipeline (identify, filter, concerns,
    #      veto) so the judge's behavior never drifts with the policy.
    #      Served either by a vLLM server subprocess (default — its
    #      continuous batching lets veto calls run concurrently) or by
    #      an in-process HF copy (--sim-backend hf fallback).
    # The vLLM server is launched BEFORE the policy load so its multi-
    # minute startup overlaps with loading the policy weights; we only
    # block on its health check afterwards.
    sim_proc = None
    if args.sim_backend == "vllm":
        assert args.sim_device != args.device, (
            "--sim-backend vllm gives the whole sim GPU to the server "
            "subprocess; --sim-device must differ from --device"
        )
        vllm_log = os.path.join(args.output_dir, "vllm_server.log")
        print(f"[load] sim judge: launching vLLM server {args.model_id} -> {args.sim_device}  log={vllm_log}")
        # NB: the server resolves the model through the default HF hub
        # cache ($HF_HOME/hub), not args.cache_dir — see launch_vllm_server.
        sim_proc, sim_base_url = launch_vllm_server(
            args.model_id,
            args.sim_device,
            gpu_memory_utilization=args.vllm_gpu_mem_util,
            log_path=vllm_log,
            enable_lora=args.policy_on_vllm,
            max_lora_rank=args.lora_r,
        )

    print(f"[load] policy   {args.model_id} -> {args.device}  cache_dir={args.cache_dir}")
    model, tokenizer = load_llama(
        model_id=args.model_id,
        device=args.device,
        cache_dir=args.cache_dir,
    )

    # 3. call_llm for the stakeholder + veto pipeline. Greedy on both
    # backends (temperature=0 for vLLM, do_sample=False for HF) because
    # reproducibility of the veto signal across train iterations is more
    # useful than diversity here; only the candidate-response sampler
    # inside Trainer needs stochasticity (and it has its own
    # do_sample=True path).
    if args.sim_backend == "vllm":
        wait_for_vllm_server(sim_proc, sim_base_url, log_path=os.path.join(args.output_dir, "vllm_server.log"))
        print(f"[load] sim judge: vLLM server healthy at {sim_base_url}")
        sim_call_llm = make_vllm_call_llm(sim_base_url, args.model_id)
        veto_concurrency = args.veto_concurrency
        if args.policy_on_vllm:
            # Same engine serves the policy: model="policy" -> base + current
            # LoRA (hot-loaded each iteration); the sim keeps using model=base.
            policy_sampler = make_vllm_policy_sampler(sim_base_url, args.model_id)

            def sync_policy_to_vllm(adapter_dir):
                unload_lora_adapter(sim_base_url, "policy")
                load_lora_adapter(sim_base_url, "policy", os.path.abspath(adapter_dir))

            print("[setup] policy sampling on shared vLLM engine (base + 'policy' LoRA)")
        else:
            policy_sampler = None
            sync_policy_to_vllm = None
    else:
        print(f"[load] sim_base {args.model_id} -> {args.sim_device}  (frozen)")
        sim_model, sim_tokenizer = load_llama(
            model_id=args.model_id,
            device=args.sim_device,
            cache_dir=args.cache_dir,
        )
        # Freeze sim_base — it's only used for inference inside the
        # pipeline. Disabling grad on parameters is belt-and-suspenders;
        # eval() already suppresses dropout, and we never call backward
        # on sim outputs.
        for p in sim_model.parameters():
            p.requires_grad = False
        sim_model.eval()
        sim_call_llm = make_call_llm(sim_model, sim_tokenizer, do_sample=False)
        # A single in-process HF model serves requests one at a time;
        # concurrent calls would interleave without batching.
        veto_concurrency = 1
        # HF sim backend has no vLLM engine to sample the policy on; use the
        # in-process HF generation path. (--policy-on-vllm is asserted off here.)
        policy_sampler = None
        sync_policy_to_vllm = None

    # Memory diagnostics: a background thread logs host RSS + per-GPU
    # allocator counters to mem.csv every 2s, line-buffered so the curve
    # survives the SIGKILL an OOM kill sends. Off the training path, so
    # it does not slow generate. Started after the models are resident
    # so the baseline row reflects loaded weights. (With the vLLM
    # backend, the sim GPU's usage lives in the server subprocess and is
    # invisible to this process's allocator counters.)
    mem_csv = os.path.join(args.output_dir, "mem.csv")
    start_mem_sampler(mem_csv)
    print(f"[mem] sampling host+GPU memory -> {mem_csv}")

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
        veto_concurrency=veto_concurrency,
        veto_prompt=VETO_PROMPTS[args.veto_prompt],
    )
    print(f"[setup] veto prompt: {args.veto_prompt}")

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
        mixed_sampling=args.mixed_sampling,
        n_source_model=args.n_source_model,
        n_source_critical=args.n_source_critical,
        self_preference=args.self_preference,
        preference_max_tokens=args.preference_max_tokens,
        preference_max_pairs=args.preference_max_pairs,
        policy_sampler=policy_sampler,
        sync_policy_to_vllm=sync_policy_to_vllm,
        chunk_size=args.chunk_size,
        stage_concurrency=args.stage_concurrency,
    )
    print(
        f"[train] mixed_sampling={args.mixed_sampling} "
        f"(model={args.n_source_model} critical={args.n_source_critical} "
        f"stakeholder={args.n_samples - args.n_source_model - args.n_source_critical})  "
        f"self_preference={args.self_preference} "
        f"(max_pairs={args.preference_max_pairs or 'unlimited'})"
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
        precompute_ref_batch_size=args.precompute_ref_batch_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        start_iteration=args.start_iteration,
        resume_adapter=args.resume_adapter,
    )

    print(f"[done] adapter saved to {args.output_dir}")
    # log_history is a list of per-step dicts; show the last entry so
    # the user can eyeball the final loss without reading the full HF
    # trainer log. dpo_trainer is None if every iteration produced 0 pairs
    # (nothing to train on) — guard so that path doesn't crash.
    if dpo_trainer is not None and dpo_trainer.state.log_history:
        print(f"[done] last log entry: {dpo_trainer.state.log_history[-1]}")


if __name__ == "__main__":
    main()
