"""SFT + LoRA baseline: distill the Qwen3-8B + "Be critical." behavior into the
bare-prompt policy.

This is the supervised-fine-tuning counterpart of the DPO mitigations, built to
be directly comparable to them. The teacher targets are responses sampled from
the UNTRAINED base Qwen3-8B with "\n\nBe critical." appended to the prompt
(generated separately by eval_endorsement_heldout.py --base --be-critical over
the SAME train-id split the DPO runs used). Here we SFT the base model so that,
given the BARE prompt, it reproduces those critical responses — i.e. context
distillation of the "be critical" steer into the unconditioned policy.

Comparability with the DPO runs (deliberately identical):
  - base Qwen3-8B in bfloat16, UNQUANTIZED (not QLoRA);
  - the exact DPO LoraConfig: r=16, alpha=32, dropout=0.05,
    target_modules="all-linear", bias="none", task_type="CAUSAL_LM";
  - the same chat-template surface (llama_local.build_chat_input semantics:
    add_generation_prompt=True, enable_thinking=False, no double-BOS), with EOS
    appended to the target exactly as TRL DPO does;
  - effective batch 8 (per-device 1 x grad-accum 8), lr 5e-5, max_length 3072,
    1 epoch, gradient checkpointing (use_reentrant=False).
Only the OBJECTIVE differs: plain next-token NLL on the teacher response
(masked to the response tokens only), no preference margin, no reference model.

Sharding: the 8B bf16 weights (~16.4 GB) do not fit one 16 GB a4000, so the base
is split across the visible GPUs with device_map="auto" (naive model-parallel).
HF Trainer detects the multi-device hf_device_map and runs model-parallel (no
DDP). On a 2x a4000 pair this trains at ~one-GPU throughput, which is fine at the
~1000-example scale.

Inputs : teacher (sentence, response) CSV shards under --teacher-glob (written by
         eval_endorsement_heldout.py --no-judge --base --be-critical). `sentence`
         is the BARE prompt; `response` is the critical teacher target.
Outputs: the LoRA adapter saved to --output-dir (adapter_config.json +
         adapter_model.safetensors, loadable by eval_endorsement_heldout.py), and
         a `.complete` sentinel.

Run (see slurm_scripts/run_train_<env>_becritical_sft.sbatch for the wrapper):
    python train_sft_lora.py --output-dir runs/sft_lora_oeq_qwen3-8b_n1000_becritical
"""

import argparse
import glob
import os
import sys

import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "core"))

from llama_local import DEFAULT_HF_MODEL_ID, load_llama
from train_data import write_complete_sentinel


def load_teacher_pairs(teacher_glob, expect):
    """Concatenate the teacher CSV shards into a list of (prompt, response).

    Assumptions: every shard has `sentence` (bare prompt) + `response` columns;
    responses are non-empty; the total row count equals `expect` (the train-id
    set size) so a missing/short shard fails loudly rather than silently
    training on a subset.
    """
    paths = sorted(glob.glob(teacher_glob))
    assert paths, f"no teacher CSV shards matched {teacher_glob!r}"
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    assert "sentence" in df.columns and "response" in df.columns, (
        f"teacher CSVs must have sentence+response columns; got {list(df.columns)}")
    df = df[["sentence", "response"]].dropna()
    assert len(df) == expect, (
        f"expected {expect} teacher pairs across {len(paths)} shards, got "
        f"{len(df)} (missing/short shard?)")
    return list(df.itertuples(index=False, name=None))


def build_example(tokenizer, prompt, response, max_length):
    """Tokenize one (prompt, response) into input_ids + response-only labels.

    The prompt side reproduces build_chat_input's surface exactly
    (add_generation_prompt=True, enable_thinking=False, no double-BOS); the
    response tokens (+ EOS, matching TRL DPO) are the supervised targets and the
    prompt tokens are masked to -100. If the full sequence exceeds max_length,
    the PROMPT is left-truncated (the response + its labels are always kept
    intact), mirroring TRL's left-truncation of the prompt.
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False)
    prompt_ids = tokenizer(text, add_special_tokens=False).input_ids
    response_ids = tokenizer(response, add_special_tokens=False).input_ids
    response_ids = response_ids + [tokenizer.eos_token_id]

    # Keep the whole response; left-truncate the prompt if we overflow.
    if len(prompt_ids) + len(response_ids) > max_length:
        keep = max_length - len(response_ids)
        assert keep > 0, (
            f"response ({len(response_ids)} tok) alone exceeds max_length "
            f"{max_length}; raise --max-length or shorten teacher targets")
        prompt_ids = prompt_ids[-keep:]

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    return {"input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids)}


class PadCollator:
    """Right-pad a batch of variable-length examples. Pads input_ids with the
    tokenizer pad id, attention_mask with 0, and labels with -100 (ignored by
    the CE loss), so padding never contributes to the objective."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        maxlen = max(len(e["input_ids"]) for e in batch)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for e in batch:
            pad = maxlen - len(e["input_ids"])
            out["input_ids"].append(e["input_ids"] + [self.pad_token_id] * pad)
            out["attention_mask"].append(e["attention_mask"] + [0] * pad)
            out["labels"].append(e["labels"] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


def main():
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True,
                   help="dir the LoRA adapter is saved to; teacher CSVs default "
                        "to <output-dir>/eval/")
    p.add_argument("--teacher-glob", default=None,
                   help="glob for the teacher (sentence,response) CSV shards "
                        "(default: <output-dir>/eval/responses_*becritical*shard*.csv)")
    p.add_argument("--expect", type=int, default=1000,
                   help="assert this many teacher pairs (the train-id set size)")
    p.add_argument("--model-id", default=DEFAULT_HF_MODEL_ID)
    p.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    p.add_argument("--device-map", default="auto",
                   help="device_map for the base model (default: auto — shard "
                        "across the visible a4000s). Set to '' for single-GPU.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=3072)
    p.add_argument("--limit", type=int, default=None,
                   help="smoke test: train on only the first N teacher pairs")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    teacher_glob = args.teacher_glob or os.path.join(
        args.output_dir, "eval", "responses_*becritical*shard*.csv")
    pairs = load_teacher_pairs(teacher_glob, args.expect)
    if args.limit is not None:
        pairs = pairs[:args.limit]
    print(f"[sft] teacher pairs: {len(pairs)} (glob={teacher_glob})")

    device_map = args.device_map or None
    model, tokenizer = load_llama(
        args.model_id, dtype=torch.bfloat16, device_map=device_map,
        cache_dir=args.cache_dir)
    print(f"[sft] base loaded  device_map={device_map}  "
          f"hf_device_map={getattr(model, 'hf_device_map', None)}")

    # Tokenize on CPU into a Dataset.
    examples = [build_example(tokenizer, q, r, args.max_length) for q, r in pairs]
    lens = [len(e["input_ids"]) for e in examples]
    print(f"[sft] seq len: min={min(lens)} max={max(lens)} mean={sum(lens)//len(lens)}")
    ds = Dataset.from_list(examples)

    # Gradient checkpointing on a frozen base needs the input embeddings to
    # require grad, else HF retains the full activation stack (OOM). Same fix the
    # DPO trainer applies (core/trainer.py).
    model.enable_input_require_grads()
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules="all-linear", bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "_hf_trainer"),
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="linear",
        warmup_ratio=0.0,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model, args=targs, train_dataset=ds,
        data_collator=PadCollator(tokenizer.pad_token_id))
    print(f"[sft] starting SFT  examples={len(ds)}  "
          f"eff_batch={args.per_device_batch_size * args.grad_accum}  "
          f"lr={args.learning_rate}  epochs={args.num_train_epochs}")
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    write_complete_sentinel(args.output_dir)
    print(f"[sft] done -> adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
