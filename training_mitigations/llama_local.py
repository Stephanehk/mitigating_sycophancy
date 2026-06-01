"""Local HuggingFace-model wrapper exposing a `call_llm`-compatible
function so the single-turn mitigation pipeline in
`mitigations/our_mitigation*.py` can be driven against a chat model
loaded with HuggingFace transformers instead of going through Anthropic
/ OpenRouter.

Default checkpoint is `Qwen/Qwen3-8B` (ungated). The caller is expected
to have $HF_TOKEN set if loading a gated model; the loader does not
attempt interactive login.

Public surface:
  - `load_llama(model_id, dtype, device, cache_dir)`:
        eager load of (model, tokenizer). bf16 on CUDA by default; one
        a6000-48G holds an 8B comfortably. Name kept as `load_llama`
        for backward compatibility with the original probe; works for
        any AutoModelForCausalLM-compatible checkpoint.
  - `make_call_llm(model, tokenizer, do_sample, temperature)`:
        returns a closure with the exact signature the harness uses,
        `call_llm(client, model_name, prompt, max_tokens, system=None)`,
        ignoring `client` and `model_name` (we only have one local model)
        and returning `(text, in_tok, out_tok)` where `in_tok` is the
        templated prompt's token length and `out_tok` is the number of
        newly generated tokens.

Greedy decoding is the default (do_sample=False) so step-by-step runs
across the 4-call v2 pipeline are reproducible. Pass `do_sample=True` and
a temperature if you want sampled outputs.

Qwen3 quirk: these models default to "thinking" mode and emit
<think>...</think> blocks before the actual reply, which inflates token
usage and buries the structured output the v2 pipeline parsers expect.
`build_chat_input` always passes `enable_thinking=False` to the chat
template; the kwarg is silently ignored by tokenizers whose templates
don't reference it (Llama-3, Qwen2.5, etc.), so this is safe to pass
unconditionally.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_HF_MODEL_ID = "Qwen/Qwen3-8B"


def load_llama(
    model_id=DEFAULT_HF_MODEL_ID,
    dtype=None,
    device="cuda",
    cache_dir=None,
):
    """Load `model_id` and its tokenizer onto `device`.

    Assumptions:
      - `device` exists. We assert torch.cuda.is_available() when device
        starts with 'cuda' so a missing GPU fails loud at load time rather
        than later inside generate().
      - The user is authenticated for gated repos. HF picks up $HF_TOKEN
        automatically; if missing for a gated id, model load will raise.
      - One process, one GPU. We do not use device_map="auto" because the
        8B fits in a single a6000-48G in bf16 (~16 GB) and a single device
        avoids cross-GPU traffic that we don't need for inference.

    `dtype` defaults to bfloat16 on CUDA, float32 on CPU.
    `cache_dir` is passed through to HF; if None the env var $HF_HOME / the
    default ~/.cache/huggingface is used.
    """
    if device.startswith("cuda"):
        assert torch.cuda.is_available(), (
            f"requested device={device!r} but torch.cuda.is_available() is False"
        )
    if dtype is None:
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )
    model.to(device)
    model.eval()

    # Greedy decoding doesn't need a pad token, but generate() warns if one
    # is missing; Llama-3 tokenizers ship without a pad token, so reuse EOS.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def build_chat_input(tokenizer, prompt, system, device):
    """Build the templated input_ids for a single user turn (plus optional
    system turn) using the model's chat template.

    Assumptions:
      - The tokenizer has a chat template registered (Llama-3.1 instruct
        does). We don't fall back to a hand-rolled template because Llama-3
        chat formatting is non-trivial and getting it wrong silently
        degrades instruction-following.
    Returns a 1xN LongTensor on `device`.
    """
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    # Two-step (render-then-tokenize) is the only call shape that behaves
    # identically across the 4.x → 5.x transformers boundary: 4.x's
    # `apply_chat_template` returned a list[int] by default, 5.x's
    # returns the rendered string; explicitly rendering then tokenizing
    # sidesteps both. `add_special_tokens=False` because the chat
    # template already inserts BOS / role markers — letting the
    # tokenizer add them again doubles the BOS and degrades quality.
    # `enable_thinking=False` is honored by Qwen3's chat template and
    # ignored by templates that don't reference it (Llama-3, Qwen2.5,
    # Mistral, etc.) — `apply_chat_template` forwards unknown kwargs as
    # Jinja variables, and unset variables are simply unused.
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return enc.input_ids.to(device)


def make_call_llm(model, tokenizer, do_sample=False, temperature=1.0):
    """Return a function with the signature
        call_llm(client, model_name, prompt, max_tokens, system=None)
    that runs `prompt` (plus optional `system`) through the loaded HF model
    and returns `(text, input_tokens, output_tokens)`.

    `client` and `model_name` are accepted for signature compatibility with
    the harness's provider-agnostic dispatcher and intentionally ignored —
    there is one local model. Token counts are exact: `input_tokens` is
    the length of the chat-templated prompt; `output_tokens` is the count
    of newly generated tokens (EOS-stop included if emitted before the
    `max_tokens` cap).

    `do_sample=False` (greedy) is the default so a 4-call v2 pipeline is
    reproducible across invocations.
    """
    device = next(model.parameters()).device

    def call_llm(client, model_name, prompt, max_tokens, system=None):
        del client, model_name  # local model; provider args ignored.
        input_ids = build_chat_input(tokenizer, prompt, system, device)
        in_tok = input_ids.shape[1]

        gen_kwargs = dict(
            max_new_tokens=int(max_tokens),
            do_sample=bool(do_sample),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)

        with torch.no_grad():
            output_ids = model.generate(input_ids, **gen_kwargs)

        # `output_ids` is the concatenation of input_ids and the generated
        # tokens. Slice off the prompt to recover only the new tokens for
        # decoding and for the output-token count.
        new_ids = output_ids[0, in_tok:]
        out_tok = int(new_ids.shape[0])
        text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return text, int(in_tok), out_tok

    return call_llm
