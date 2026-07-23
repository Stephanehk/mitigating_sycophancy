"""Create a FRESH, untrained LoRA adapter on a base model and save it to disk.

Why this exists: train_dpo_offline_lora.py is built to CONTINUE an existing
adapter (`--adapter` is required and the LoRA geometry is read from it). To train
a brand-new model (e.g. Qwen/Qwen3-32B) offline on a pair file we first need an
adapter to hand it. This writes a zero-initialised adapter whose geometry matches
the P6 runs' adapter_config.json exactly, so that continuing it is identical to
training a fresh LoRA from the base:

  init_lora_weights=True  =>  lora_B is zero at init  =>  delta W = 0 at step 0,
  i.e. the policy starts equal to the base model. No prior training is baked in.

The saved adapter is tiny (only the LoRA A/B tensors); the base is not copied.

Assumptions:
  - The base model_id is available in the HF cache (compute nodes are air-gapped;
    pre-download on a login node first).
  - The target module names below exist in the base (they are the standard
    Qwen3 attention+MLP projections; get_peft_model asserts if any are absent).

Inputs (CLI): --model-id, --out-dir, --cache-dir, --device.
Output: <out-dir>/adapter_config.json + adapter_model.safetensors.
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "core"))

from llama_local import DEFAULT_HF_MODEL_ID, load_llama

# Geometry copied verbatim from the P6 rpo0p5_final adapter_config.json so a
# 32B fresh LoRA is dimension-for-dimension the same recipe as the 8B P6 runs.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default=DEFAULT_HF_MODEL_ID)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model

    assert not os.path.exists(os.path.join(args.out_dir, "adapter_config.json")), (
        f"adapter already exists at {args.out_dir}; refusing to overwrite")

    model, tokenizer = load_llama(model_id=args.model_id, device=args.device,
                                  cache_dir=args.cache_dir)

    cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
        init_lora_weights=True,   # zero-init B => starts equal to base
        use_rslora=False,
        use_dora=False,
    )
    peft_model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    assert trainable > 0, "0 trainable params after get_peft_model"
    print(f"[make-fresh-lora] {args.model_id}: trainable params {trainable:,}")

    os.makedirs(args.out_dir, exist_ok=True)
    peft_model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    assert os.path.exists(os.path.join(args.out_dir, "adapter_model.safetensors")), (
        "save_pretrained did not write adapter_model.safetensors")
    print(f"[make-fresh-lora] saved fresh adapter -> {args.out_dir}")


if __name__ == "__main__":
    main()
