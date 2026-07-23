"""One round of OFFLINE, vanilla DPO on a precomputed pair file, continuing an
already-trained LoRA adapter.

Stage 4 of the filtered-pair round. Stages 1-3 (eval/sample_train_candidates.py,
eval/classify_failure_mode.py, eval/build_dpo_pairs.py) produce
`<out-dir>/pairs/pairs.jsonl`; this trains on it.

Why a separate entrypoint rather than a flag on train_dpo_lora.py: that script's
pairs can ONLY come from `Trainer._build_iteration` (core/trainer.py:1487),
which unconditionally re-samples every prompt and runs the veto panel -- the
expensive part -- and its `main()` also stands up the vLLM sim server and the
simulated-stakeholder panel before training. None of that is wanted here: the
pairs already exist on disk. This deliberately does NOT refactor core/trainer.py,
which produced the frozen final results; the ~50 lines below are a faithful copy
of its training block (trainer.py:1505-1572) with the iteration loop, the
checkpoint/resume machinery, and the RPO/distill term removed.

Differences from the P6 recipe, both intentional:
  - DPO by default. With no --rpo-alpha there is no SFT term: the SFT term
    anchors chosen log-probs in absolute terms, which is exactly the drift a
    second-stage round usually should not have; the point is the contrast.
    PASS --rpo-alpha to reproduce the P6 recipe's RPO/distill objective
    (loss = L_DPO + alpha * CE(chosen)), implemented exactly as core/trainer.py
    :1524-1540 does it — trl multi-loss loss_type=["sigmoid","sft"] with
    loss_weights=[1.0, alpha]. NOTE: trl's built-in `sft` term pools every
    `chosen` in the micro-batch, which is why the P6 pipeline wraps it in
    SelfDistillDPOTrainer to mask the term to on-policy samples. That mask is
    unnecessary here and only here: this trainer's pairs come from a single
    on-policy source (sample_train_candidates.py draws every candidate from the
    adapter being continued), so the mask would be all-ones, and with an
    all-ones mask the masked term is numerically identical to trl's built-in
    one (core/self_distill_trainer.py:24-26). Do not pass --rpo-alpha here for
    pairs built from mixed sources.
  - save_strategy stays "no" (dpo_save_steps is not plumbed at all), so the
    mid-epoch checkpoint-resume path that silently zeroes LoRA weights is
    UNREACHABLE rather than merely unused. See _next_dpo_body_final.sh:25-34.

pi_ref is the adapter-DISABLED base model -- trl's behaviour for a PeftModel
with ref_model=None (core/trainer.py:1254-1258). That is what every P6 iteration
already used, so this round is methodologically a further iteration on the same
adapter with a different pair-construction rule. Note the consequence: the KL
term pulls toward the BASE model, not toward P6.

The adapter is loaded with `is_trainable=True` and saved to a NEW --out-dir.
The input adapter is never written to.

Usage:
    python train_dpo_offline_lora.py \
        --adapter /projects/m000254/runs/dpo_lora_pas_..._final \
        --out-dir /projects/m000254/runs/dpo_lora_pas_..._final_voicedpo
"""

import argparse
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "core"))

from llama_local import DEFAULT_HF_MODEL_ID, load_llama
from train_data import write_complete_sentinel


def retemplate_prompt(tokenizer, prompt):
    """Render `prompt` through the chat template exactly as sampling did.

    Verbatim copy of Trainer._retemplate_prompt (core/trainer.py:1187-1206).
    The pair file stores BARE prompts, but the completions were generated
    against the chat-templated prefix; handing DPO the raw prompt would score
    log-probs for "raw text -> response" while the policy was shaped under
    "<chat-template> -> response", scrambling the gradient.
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True,
                    help="trained LoRA adapter to CONTINUE (read-only)")
    ap.add_argument("--out-dir", required=True,
                    help="new run dir; pairs read from <out-dir>/pairs/pairs.jsonl")
    ap.add_argument("--pairs-file", default=None,
                    help="override the pair file path")
    ap.add_argument("--model-id", default=DEFAULT_HF_MODEL_ID)
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--rpo-alpha", type=float, default=None,
                    help="if set (>0), add an SFT/NLL term on `chosen`: "
                         "loss = L_DPO + rpo_alpha * CE(chosen), via trl "
                         "multi-loss. Reproduces the P6 RPO/distill recipe. "
                         "Only valid when the pairs come from a single "
                         "on-policy source — see the module docstring.")
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--num-train-epochs", type=float, default=1.0)
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--precompute-ref-batch-size", type=int, default=8)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N pairs")
    args = ap.parse_args()

    from datasets import Dataset
    from peft import PeftModel
    from trl import DPOConfig, DPOTrainer

    assert os.path.exists(os.path.join(args.adapter, "adapter_config.json")), (
        f"not a PEFT adapter dir: {args.adapter}")
    assert os.path.normpath(args.adapter) != os.path.normpath(args.out_dir), (
        "--out-dir must differ from --adapter; this must never overwrite the "
        "trained adapter it continues")

    pairs_file = args.pairs_file or os.path.join(args.out_dir, "pairs", "pairs.jsonl")
    pairs = []
    with open(pairs_file) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    if args.limit:
        pairs = pairs[:args.limit]
    assert pairs, f"no pairs in {pairs_file}"
    print(f"[offline-dpo] {len(pairs)} pairs from {pairs_file}")

    torch.manual_seed(args.seed)
    model, tokenizer = load_llama(model_id=args.model_id, device=args.device,
                                  cache_dir=args.cache_dir,
                                  device_map=args.device_map)

    # Gradient checkpointing only saves memory on a LoRA model if the input
    # embeddings are forced to require grad (the base is frozen). Must run on
    # the raw base BEFORE PEFT wrapping; the hook survives it.
    # See core/trainer.py:1330-1342.
    model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    print(f"[offline-dpo] continuing adapter {args.adapter}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable > 0, (
        "0 trainable parameters — the adapter did not load as trainable")
    print(f"[offline-dpo] trainable params: {trainable:,}")

    dataset = Dataset.from_list([
        dict(p, prompt=retemplate_prompt(tokenizer, p["prompt"])) for p in pairs
    ])

    use_bf16 = next(model.parameters()).dtype == torch.bfloat16
    # RPO / "distill": trl sums loss_type[i] weighted by loss_weights[i], so
    # ["sigmoid","sft"] with [1.0, alpha] is L_DPO + alpha * CE(chosen). Passing
    # a single-element list when alpha is unset would change trl's code path, so
    # leave loss_type untouched in that case.
    rpo_kwargs = {}
    if args.rpo_alpha is not None:
        assert args.rpo_alpha > 0, (
            f"--rpo-alpha must be > 0 when set (got {args.rpo_alpha})")
        rpo_kwargs = {"loss_type": ["sigmoid", "sft"],
                      "loss_weights": [1.0, float(args.rpo_alpha)]}
        print(f"[dpo-offline] RPO/distill: loss = L_DPO + {args.rpo_alpha} * CE(chosen)")

    dpo_args = DPOConfig(
        output_dir=os.path.join(args.out_dir, "dpo"),
        beta=float(args.beta),
        **rpo_kwargs,
        learning_rate=float(args.learning_rate),
        num_train_epochs=float(args.num_train_epochs),
        per_device_train_batch_size=int(args.per_device_batch_size),
        gradient_accumulation_steps=int(args.grad_accum),
        max_length=int(args.max_length),
        # Ref log-probs in a separate cached pre-pass: the in-step ref forward
        # materialises a full-vocab logits tensor (~6 GB on Qwen3-8B's 152k
        # vocab) and OOMs the 48 GB card on long sequences.
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=int(args.precompute_ref_batch_size),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=use_bf16,
        remove_unused_columns=False,
        logging_steps=int(args.logging_steps),
        report_to="none",
        # Never "steps": keeps the LoRA-zeroing mid-epoch resume path unreachable.
        save_strategy="no",
        seed=int(args.seed),
    )

    model.train()
    trainer = DPOTrainer(
        model=model,
        ref_model=None,          # PeftModel + None => adapter-disabled base as pi_ref
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=None,        # already wrapped; passing this would nest
    )
    # HF wraps in nn.DataParallel when device_count > 1, replicating the policy
    # onto the second GPU and gathering large-vocab logits back — ~30 GB of
    # overhead that OOMs a 48 GB card. See core/trainer.py:1548.
    trainer.args._n_gpu = 1

    trainer.train()

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    write_complete_sentinel(args.out_dir)
    print(f"[offline-dpo] saved adapter -> {args.out_dir}")

    hist = [h for h in trainer.state.log_history if "loss" in h]
    if hist:
        print(f"[offline-dpo] first loss={hist[0]['loss']:.4f}  "
              f"last loss={hist[-1]['loss']:.4f}  ({len(hist)} logged steps)")


if __name__ == "__main__":
    main()
