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


def _is_multimodal_wrapper(model_id, cache_dir=None):
    """True for checkpoints whose top-level architecture is a multimodal
    `*ForConditionalGeneration` wrapper, which is NOT registered under
    `AutoModelForCausalLM` and must load via `AutoModelForMultimodalLM`.

    Covers Gemma-4 (`Gemma4UnifiedForConditionalGeneration`, model_type
    `gemma4_unified`) and Mistral-3 (`Mistral3ForConditionalGeneration`,
    model_type `mistral3`); verified 2026-07-28 that AutoModelForMultimodalLM
    resolves both to exactly those concrete classes.

    Assumption: the decision comes from the CHECKPOINT'S OWN CONFIG, not its
    name. A name substring silently sends an unrecognised new family down the
    causal-LM path, where it fails deep inside from_pretrained with an
    unrelated-looking error. Text-only families (Qwen3ForCausalLM,
    LlamaForCausalLM, ...) are unaffected — their architectures do not end in
    ForConditionalGeneration.

    Reading the config needs it present in the HF cache, which is already true
    wherever the weights are about to be loaded from.
    """
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
    architectures = getattr(config, "architectures", None) or []
    assert architectures, (
        f"{model_id} config declares no `architectures`; cannot tell whether it "
        f"needs the multimodal or the causal-LM auto class"
    )
    return any(a.endswith("ForConditionalGeneration") for a in architectures)


def assert_chat_template_round_trips(tokenizer, model_id):
    """Fail loud unless `tokenize(render(x)) == apply_chat_template(tokenize=True)`.

    `build_chat_input` below renders the chat template to a STRING and
    re-tokenizes it, and `Trainer._retemplate_prompt` does the same because TRL's
    DPO dataset API is string-based. Both silently assume the round trip is
    lossless. It is for Qwen3-8B/32B and Gemma-4-12B (verified 2026-07-28), but
    NOT for tokenizers whose control tokens are one-way: Ministral-3-14B's
    mistral-common backend renders `[INST]` as text that never encodes back to
    the control token, so BOS disappears and turn markers become literal
    characters — a model trained and sampled outside its own instruction format,
    with nothing in the logs to show for it.

    Cheap enough to run once at load time, which is where the cost of a wrong
    answer is lowest.
    """
    for prompt in ("hello world", "My friend ghosted me. AITA for being upset?"):
        messages = [{"role": "user", "content": prompt}]
        direct = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            enable_thinking=False)
        direct = list(direct["input_ids"]) if hasattr(direct, "keys") else list(direct)
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False)
        round_trip = list(tokenizer(text, add_special_tokens=False).input_ids)
        assert direct == round_trip, (
            f"{model_id}: chat template does not survive render-then-retokenize "
            f"({len(direct)} vs {len(round_trip)} tokens). build_chat_input and "
            f"Trainer._retemplate_prompt both depend on it; this tokenizer "
            f"({type(tokenizer).__name__}) would feed the model malformed prompts."
        )


def load_llama(
    model_id=DEFAULT_HF_MODEL_ID,
    dtype=None,
    device="cuda",
    cache_dir=None,
    device_map=None,
    max_memory=None,
):
    """Load `model_id` and its tokenizer onto `device`.

    Assumptions:
      - `device` exists. We assert torch.cuda.is_available() when device
        starts with 'cuda' so a missing GPU fails loud at load time rather
        than later inside generate().
      - The user is authenticated for gated repos. HF picks up $HF_TOKEN
        automatically; if missing for a gated id, model load will raise.
      - By default (device_map=None): one process, one GPU — the 8B fits in
        a single a6000-48G in bf16 (~16 GB) and a single device avoids the
        cross-GPU traffic we don't need for inference. Pass `device_map`
        (e.g. "auto") to shard the weights across multiple GPUs instead;
        accelerate then dispatches the forward pass and we do NOT call
        `.to(device)` (the model is already placed). This is the path for
        running the 8B on >1 small GPU (e.g. 2x16 GB a4000) when one card
        can't hold the bf16 weights.

    `dtype` defaults to bfloat16 on CUDA, float32 on CPU.
    `cache_dir` is passed through to HF; if None the env var $HF_HOME / the
    default ~/.cache/huggingface is used.
    `max_memory` (dict like {0: "40GiB", 1: "0GiB", ...}) is forwarded to
    from_pretrained alongside `device_map`; it caps per-device placement so
    you can restrict an "auto"/"balanced" shard to a chosen subset of GPUs
    (set the excluded ones to "0GiB"). Ignored when device_map is None.
    """
    if device.startswith("cuda") or device_map is not None:
        assert torch.cuda.is_available(), (
            f"requested device={device!r} / device_map={device_map!r} but "
            f"torch.cuda.is_available() is False"
        )
    if dtype is None:
        on_cuda = device.startswith("cuda") or device_map is not None
        dtype = torch.bfloat16 if on_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    assert_chat_template_round_trips(tokenizer, model_id)
    if _is_multimodal_wrapper(model_id, cache_dir):
        # Multimodal wrappers (gemma4_unified, mistral3) load via
        # AutoModelForMultimodalLM, not
        # AutoModelForCausalLM. We train TEXT-ONLY: the tokenizer above already
        # carries the -it chat template, and a text-only forward (input_ids, no
        # pixel_values/audio) never enters the vision/audio input projections
        # (verified: they receive zero grad on a text-only backward), so the
        # causal-LM training path is behaviorally unchanged. Import lazily inside
        # this branch so the Qwen env — whose older transformers lacks
        # AutoModelForMultimodalLM — is unaffected. Uses `dtype=` (the modern
        # kwarg) rather than the deprecated `torch_dtype=` alias.
        from transformers import AutoModelForMultimodalLM
        model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            dtype=dtype,
            cache_dir=cache_dir,
            device_map=device_map,
            max_memory=max_memory if device_map is not None else None,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            device_map=device_map,
            max_memory=max_memory if device_map is not None else None,
        )
    # When device_map is set, accelerate has already placed the shards;
    # calling .to(device) would try to move the whole model onto one GPU and
    # defeat the point (and OOM). Only place explicitly in the single-device
    # path.
    if device_map is None:
        model.to(device)
    model.eval()

    # Greedy decoding doesn't need a pad token, but generate() warns if one
    # is missing; Llama-3 tokenizers ship without a pad token, so reuse EOS.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Restore the checkpoint's own stop-token set. transformers ALIGNS
    # generation_config.eos_token_id to the tokenizer's single eos at load time
    # ("Updated tokens: {'eos_token_id': ...}"), which for GLM-4 throws away the
    # two ids that actually end an assistant turn. Re-read it from disk so
    # eos_ids_for_generate sees the real list rather than the collapsed one.
    if os.environ.get("GEN_EOS_FROM_GENERATION_CONFIG") == "1":
        from transformers import GenerationConfig
        disk = GenerationConfig.from_pretrained(model_id, cache_dir=cache_dir)
        assert disk.eos_token_id is not None, (
            f"{model_id}: GEN_EOS_FROM_GENERATION_CONFIG=1 but its "
            f"generation_config declares no eos_token_id")
        model.generation_config.eos_token_id = disk.eos_token_id
        print(f"[llama_local] stop ids restored from generation_config: "
              f"{disk.eos_token_id} (tokenizer alone would give "
              f"{tokenizer.eos_token_id})")

    return model, tokenizer


def eos_ids_for_generate(model, tokenizer):
    """Stop-token id(s) to pass to `model.generate()`.

    Default is `tokenizer.eos_token_id` -- what every existing arm used, and
    CORRECT for Qwen3 and phi-4, whose tokenizer eos (<|im_end|>) IS the token
    that ends an assistant turn.

    GEN_EOS_FROM_GENERATION_CONFIG=1 uses the checkpoint's own list instead.
    GLM-4 needs it: its tokenizer eos is <|endoftext|> (151329) but its chat
    template ends an assistant turn with <|user|> (151336) --
    "<|assistant|>\\nhello<|user|>\\n..." with no <|endoftext|> anywhere -- so a
    stop set of {151329} never fires. Measured on GLM-4-9B (job 402399): every
    generation ran to the full 512-token cap and kept writing past the end of the
    answer, versus 195/242 tokens and a clean stop with the full list.

    Opt-in rather than always-on so pre-existing arms still generate
    byte-identically: Qwen's and phi-4's generation_config ALSO list
    <|endoftext|>, which would introduce a new stop token for them.
    """
    if os.environ.get("GEN_EOS_FROM_GENERATION_CONFIG") != "1":
        return tokenizer.eos_token_id
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    assert eos is not None, (
        "GEN_EOS_FROM_GENERATION_CONFIG=1 but the model exposes no "
        "generation_config.eos_token_id; load via load_llama so it is restored")
    return eos


def build_chat_input(tokenizer, prompt, system, device):
    """Build the templated input_ids for one user turn (plus optional system
    turn), or for a full pre-built multi-turn conversation, using the model's
    chat template.

    `prompt` is either a bare string (single user turn -- the original
    contract) or a list of {"role","content"} dicts (e.g. k-shot exemplar
    turns followed by the real target turn); `system`, if given, is
    prepended as a system turn in both cases.

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
    if isinstance(prompt, str):
        messages.append({"role": "user", "content": prompt})
    else:
        messages.extend(prompt)
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
            eos_token_id=eos_ids_for_generate(model, tokenizer),
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
