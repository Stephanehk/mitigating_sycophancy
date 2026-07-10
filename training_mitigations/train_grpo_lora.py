"""GRPO + LoRA neutrality baseline: fine-tune a local HuggingFace policy with
GRPO whose reward is the v5 gpt-5-mini endorse judge — +1 if the policy's
response is labeled 3 (NEUTRAL toward the user's action), 0 otherwise.

This is the on-policy RL counterpart to the off-policy DPO neutrality baseline
(`train_dpo_lora.py --judge neutrality-v5`). Both optimize the SAME target — the
v5 label-3 rate — and share `core/neutrality_judge.py` (so the train-time
criterion is identical to the v5 eval judge) and `core/train_data.py` (so a GRPO
run and a DPO run on the same env/seed/split train and evaluate on identical
prompts).

Pipeline:
  1. Load the `--env` dataset; hold out `--test-n` prompts (ids -> test_ids.json)
     and sample a disjoint `--train-n` (ids -> train_ids.json), via the shared
     split helpers — the SAME split contract the DPO baseline uses.
  2. Build a CONVERSATIONAL train dataset: each row is
        {"prompt": [{"role": "user", "content": post}], "post": post}
     The `prompt` column is what the policy generates from; TRL applies the
     chat template internally with `chat_template_kwargs={"enable_thinking":
     False}` (the Qwen3 quirk) + add_generation_prompt — so generation matches
     the surface `eval_endorsement_heldout.py` evaluates on, with no double-BOS.
     The bare `post` column rides along for the reward function (the v5 judge
     scores (post, completion), the (question, response) pair it expects).
  3. GRPO with `num_generations` (G) completions per prompt, group-normalized
     advantages, KL-to-reference (`--beta`). The reward is +1 for v5 label 3
     else 0 (see core/neutrality_judge.make_neutrality_reward). Requires
     OPENROUTER_API_KEY.
  4. Wrap the policy with a LoRA adapter (peft_config); with PEFT, TRL uses the
     adapter-disabled base as the KL reference, so no second 8B copy is loaded.
  5. Save the final LoRA adapter to `--output-dir` (loadable by the eval script).

Reward hacking caveat: "neutral" is the easy-to-game direction (refusals /
evasive non-answers also score label 3). The KL term (`--beta`) constrains drift
from the base policy; a quality judge + a held-out judge (run at eval time) are
the intended Goodhart detectors. See core/neutrality_judge.py.

GPU: this script runs single-process; pin it to one GPU with CUDA_VISIBLE_DEVICES
in the launcher. With `--use-vllm` TRL colocates a vLLM engine on the same GPU
for generation (faster, more memory); the default HF generation path is slower
but simpler.
"""

import argparse
import os
import sys

# Make `mitigations/` (the harness) and the local `training_mitigations/` +
# `core/` importable so `train_data`, `neutrality_judge`, `llama_local`, and the
# harness modules all resolve — same path setup as train_dpo_lora.py.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, "mitigations"))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "core"))

from explore_mitigations import DEFAULT_SEED

from train_data import (resolve_csv, load_env_data, split_train_test,
                        build_train_prompts, write_ids)
from neutrality_judge import make_neutrality_reward, make_openrouter_client
from llama_local import DEFAULT_HF_MODEL_ID, load_llama


def _build_dataset(prompts):
    """Build the conversational GRPO dataset from bare-post prompt strings.

    Each row: {"prompt": [{"role": "user", "content": post}], "post": post}.
    `prompt` (conversational) is what TRL templates + generates from; `post`
    (bare) is passed through to the reward function. Imports `datasets` lazily.
    """
    from datasets import Dataset
    rows = [{"prompt": [{"role": "user", "content": p}], "post": p}
            for p in prompts]
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env", required=True,
                        choices=("OEQ", "PAS", "AITA", "AITA-NTA-FLIP"),
                        help="dataset env to train on")
    parser.add_argument("--output-dir", required=True,
                        help="directory the LoRA adapter + split-id files go to")
    parser.add_argument("--model-id", default=DEFAULT_HF_MODEL_ID,
                        help=f"HF model id of the policy (default: {DEFAULT_HF_MODEL_ID})")
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME"),
                        help="HF cache dir; defaults to $HF_HOME if set")
    parser.add_argument("--device", default="cuda:0",
                        help="torch device the policy is loaded on (default: "
                             "cuda:0). Pin the process to one GPU with "
                             "CUDA_VISIBLE_DEVICES in the launcher.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"split / sampler / RL seed (default: {DEFAULT_SEED})")
    parser.add_argument("--train-n", type=int, required=True,
                        help="number of training prompts (IGNORED under --use-final-data)")
    parser.add_argument("--test-n", type=int, required=True,
                        help="number of held-out test prompts (IGNORED under --use-final-data)")
    parser.add_argument("--use-final-data", action="store_true",
                        help="train/eval on the frozen final_training_eval_data/ "
                             "split for --env instead of the seed-derived sample; "
                             "--train-n/--test-n/--seed are ignored.")

    # GRPO knobs.
    parser.add_argument("--num-generations", type=int, default=8,
                        help="G — completions sampled per prompt per step "
                             "(default: 8). per-device-batch-size * grad-accum "
                             "must be divisible by this (TRL requirement).")
    parser.add_argument("--num-train-epochs", type=float, default=1.0,
                        help="epochs over the train prompts (default: 1.0). "
                             "Ignored if --max-steps > 0.")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="optimizer steps cap (default: -1 = use epochs)")
    parser.add_argument("--per-device-batch-size", type=int, default=2,
                        help="per-device train batch size in completions "
                             "(default: 2). With grad-accum 4 and G=8 the "
                             "product 8 is divisible by G.")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="gradient accumulation steps (default: 4)")
    parser.add_argument("--learning-rate", type=float, default=1e-6,
                        help="GRPO learning rate (default: 1e-6)")
    parser.add_argument("--beta", type=float, default=0.04,
                        help="KL coefficient to the reference policy (default: "
                             "0.04). Higher = stronger anchor against drift "
                             "toward degenerate-neutral reward hacking.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="generation temperature (default: 1.0)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="generation nucleus top-p (default: 1.0)")
    parser.add_argument("--max-completion-length", type=int, default=512,
                        help="max new tokens per generated completion "
                             "(default: 512)")
    parser.add_argument("--use-vllm", action="store_true",
                        help="generate completions with a colocated vLLM engine "
                             "on the same GPU (faster, more memory) instead of "
                             "HF generate (default off).")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                        action="store_false",
                        help="disable gradient checkpointing (on by default to "
                             "fit the 8B policy + grad on a 48 GB card).")
    parser.add_argument("--logging-steps", type=int, default=1)

    # LoRA.
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="all-linear",
                        help="LoRA target modules (default: all-linear)")

    # v5 neutrality judge (reward) — same knobs as the DPO baseline's judge.
    parser.add_argument("--judge-model", default="openai/gpt-5-mini",
                        help="reward judge model id (default: openai/gpt-5-mini)")
    parser.add_argument("--judge-reasoning", default="minimal",
                        help="OpenRouter reasoning effort for the judge "
                             "(default: minimal); 'none' to omit")
    parser.add_argument("--judge-max-tokens", type=int, default=4000,
                        help="judge output token budget incl. reasoning "
                             "(default: 4000)")
    parser.add_argument("--judge-temp", type=float, default=0.0,
                        help="judge temperature (default: 0.0 deterministic; "
                             "negative omits temperature)")
    parser.add_argument("--judge-concurrency", type=int, default=16,
                        help="max in-flight OpenRouter judge calls (default: 16)")
    parser.add_argument("--judge-price-in", type=float, default=0.25,
                        help="judge input price $/1M tokens (default: 0.25); "
                             "used only for the spend cap")
    parser.add_argument("--judge-price-out", type=float, default=2.00,
                        help="judge output price $/1M tokens (default: 2.00)")
    parser.add_argument("--judge-max-cost", type=float, default=None,
                        help="hard cap (USD) on cumulative reward-judge spend "
                             "across the run (default: none). On breach the run "
                             "aborts (fail-fast).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # TRL requires (per_device_batch_size * grad_accum) to be divisible by G
    # (single process); fail fast with a clear message before any heavy load.
    global_batch = args.per_device_batch_size * args.grad_accum
    assert global_batch % args.num_generations == 0, (
        f"per-device-batch-size * grad-accum ({global_batch}) must be divisible "
        f"by --num-generations ({args.num_generations})"
    )
    assert args.num_generations >= 2, "GRPO needs >= 2 generations per prompt"

    # 1. Dataset + train/test split (shared contract with the DPO baseline).
    csv_path = resolve_csv(args.env)
    print(f"[data] env={args.env}  csv={csv_path}")
    data = load_env_data(args.env, csv_path)
    print(f"[data] pool size: {len(data)} rows")
    train_df, test_df = split_train_test(
        args.env, data, args.train_n, args.test_n, args.seed,
        use_final_data=args.use_final_data)
    if args.use_final_data:
        print(f"[data] FINAL split for {args.env}: test_n={len(test_df)}  "
              f"train_n={len(train_df)} (from final_training_eval_data/; "
              f"--train-n/--test-n/--seed ignored)")
    else:
        print(f"[data] held out test_n={len(test_df)}  train_n={len(train_df)}  seed={args.seed}")
    write_ids(os.path.join(args.output_dir, "test_ids.json"), test_df)
    write_ids(os.path.join(args.output_dir, "train_ids.json"), train_df)
    train_prompts = build_train_prompts(args.env, train_df)
    train_dataset = _build_dataset(train_prompts)
    print(f"[data] built GRPO dataset: {len(train_dataset)} prompts")

    # 2. Policy model + tokenizer. load_llama matches the rest of the repo
    # (dtype, pad token). GRPO needs LEFT padding for batched generation.
    print(f"[load] policy {args.model_id} -> {args.device}  cache_dir={args.cache_dir}")
    model, tokenizer = load_llama(
        model_id=args.model_id, device=args.device, cache_dir=args.cache_dir)
    tokenizer.padding_side = "left"

    # 3. Reward function: +1 for v5 label 3, else 0 (over OpenRouter).
    judge_temperature = None if args.judge_temp < 0 else args.judge_temp
    judge_reasoning = None if args.judge_reasoning == "none" else args.judge_reasoning
    reward_fn = make_neutrality_reward(
        client=make_openrouter_client(),
        model=args.judge_model,
        reasoning=judge_reasoning,
        max_tokens=args.judge_max_tokens,
        temperature=judge_temperature,
        concurrency=args.judge_concurrency,
        price_in=args.judge_price_in,
        price_out=args.judge_price_out,
        max_cost=args.judge_max_cost,
    )
    print(f"[setup] reward: v5 neutrality ({args.judge_model}, "
          f"reasoning={args.judge_reasoning})  +1 iff label==3 else 0")

    # 4. GRPO config + LoRA. With peft_config, TRL uses the adapter-disabled
    # base as the KL reference (no separate ref copy).
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    target_modules = args.lora_target_modules
    if "," in target_modules:
        target_modules = [m.strip() for m in target_modules.split(",") if m.strip()]
    peft_config = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        seed=int(args.seed),
        num_generations=int(args.num_generations),
        per_device_train_batch_size=int(args.per_device_batch_size),
        gradient_accumulation_steps=int(args.grad_accum),
        num_train_epochs=float(args.num_train_epochs),
        max_steps=int(args.max_steps),
        learning_rate=float(args.learning_rate),
        beta=float(args.beta),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_completion_length=int(args.max_completion_length),
        use_vllm=bool(args.use_vllm),
        gradient_checkpointing=bool(args.gradient_checkpointing),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        logging_steps=int(args.logging_steps),
        save_strategy="no",
        report_to="none",
        # Qwen3 chat-template quirk: suppress the thinking block so generation
        # matches the eval surface. Passed straight to apply_chat_template.
        chat_template_kwargs={"enable_thinking": False},
        log_completions=True,
    )

    print(f"[train] GRPO  G={args.num_generations}  global_batch={global_batch}  "
          f"epochs={args.num_train_epochs}  max_steps={args.max_steps}  "
          f"beta={args.beta}  use_vllm={args.use_vllm}  output_dir={args.output_dir}")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()

    # 5. Persist the final LoRA adapter (loadable by eval_endorsement_heldout.py).
    trainer.save_model(args.output_dir)
    print(f"[done] adapter saved to {args.output_dir}")
    if trainer.state.log_history:
        print(f"[done] last log entry: {trainer.state.log_history[-1]}")


if __name__ == "__main__":
    main()
