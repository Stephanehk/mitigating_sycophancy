"""OpenRouter-backed frozen sim: serves the stakeholder pipeline's LLM calls
from an API model (e.g. openai/gpt-5-mini) instead of a local Qwen3-8B.

`make_openrouter_call_llm` returns a closure with the harness signature

    call_llm(client, model, prompt, max_tokens, system=None)
        -> (text, in_tokens, out_tokens)

so it is a drop-in for `vllm_sim.make_vllm_call_llm` / `llama_local.make_call_llm`
at the ONE place `train_dpo_lora.py` builds `sim_call_llm`. That single swap
reroutes every frozen call — identify / filter / concerns and the whole veto
panel — because the pipeline modules resolve `call_llm` from their module-level
binding, which `SimulatedStakeholders.__init__` rewrites to this closure.

WHY NOT `open_router_utils.call_openrouter`
-------------------------------------------
It already matches the signature, but it deliberately (a) drops `max_tokens`
(`call.py`: `del max_tokens`) and (b) sends no `temperature`. Both matter here:

  - temperature: the frozen sim is greedy on BOTH local backends
    (`vllm_sim` temperature=0.0, `llama_local` do_sample=False), and three
    places depend on that determinism — `trainer._record_contribution` rebuilds
    pairs from disk without re-vetoing ("valid because the sim is frozen"), the
    batched pipeline claims equivalence to the per-prompt loop "for a
    deterministic judge", and `simulated_stakeholders`' sequential/concurrent
    veto paths claim the same veto set "for a deterministic (greedy) judge".
    Decoding at a provider default temperature would silently invalidate all
    three, so temperature is pinned here.
  - max_tokens: dropping it removes the harness's 128-token veto ceiling, and
    the cost of an unbounded veto reply is paid ~10^5 times per run.

So this module talks to OpenRouter through the OpenAI SDK (OpenRouter is
OpenAI-compatible), mirroring `neutrality_judge.make_openrouter_client` — the
same client shape the v5 judge has used in this pipeline for a while.

REASONING MODELS AND `max_tokens`
---------------------------------
For a reasoning model, `max_tokens` bounds reasoning + visible content, and the
reasoning comes first. The harness asks for 128 tokens on a veto call, which a
reasoning model can burn entirely on hidden reasoning, returning EMPTY content.
`max_tokens_floor` (default 4000, matching the v5 judge's `--judge-max-tokens`)
raises any smaller request to that floor. This is a cap, not a target — billing
is on tokens actually generated — so the floor costs nothing on short replies
and only prevents truncation.

ASSUMPTIONS
  - OPENROUTER_API_KEY is set (asserted at construction, not per call).
  - The model returns OpenAI-shaped `choices[0].message.content` + `usage`.
    Empty content is an ASSERT, not a silent "": an empty stakeholder list or
    empty veto verdict would corrupt the preference dataset rather than fail.
  - Callers bound their own fan-out; see `--sim-concurrency` (128-way
    `--stage-concurrency` is tuned for a local vLLM server and will rate-limit
    an API).
"""

import os
import sys

from openai import OpenAI

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from cost import CostTracker  # noqa: E402,F401  (re-exported for callers)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Matches neutrality_judge's --judge-max-tokens default; see module docstring.
DEFAULT_MAX_TOKENS_FLOOR = 4000


def make_openrouter_call_llm(model_id, temperature=0.0, reasoning="minimal",
                             max_tokens_floor=DEFAULT_MAX_TOKENS_FLOOR,
                             max_retries=12, cost_tracker=None):
    """Return a harness-signature `call_llm` routed to an OpenRouter model.

    Assumptions / inputs:
      model_id: OpenRouter model id (e.g. "openai/gpt-5-mini"). The returned
        closure IGNORES the `client` and `model` arguments the harness passes
        (one configured backend per run), exactly as the vLLM/HF closures do.
      temperature: pinned decoding temperature; 0.0 (default) keeps the frozen
        sim deterministic — see the module docstring on why that is
        load-bearing. Pass None to omit the field (provider default) only if you
        accept a non-reproducible veto signal.
      reasoning: OpenRouter reasoning effort, forwarded as
        `extra_body={"reasoning": {"effort": ...}}` like the v5 judge. None
        omits it (for non-reasoning models, which reject the field).
      max_tokens_floor: raise any requested max_tokens to at least this, so a
        reasoning model's hidden tokens cannot crowd out the visible answer.
      max_retries: OpenAI SDK retry budget (429 / 5xx). 12 matches
        `neutrality_judge.make_openrouter_client`.
      cost_tracker: optional `CostTracker`; when set, every call's usage is
        added, and a breach of its cap raises out of the call (fail-fast).

    Returns: `call_llm(client, model, prompt, max_tokens, system=None)`
      -> (text, in_tokens, out_tokens). Thread-safe (the OpenAI client is), so
      it is safe under `concurrency.concurrent_map`.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    assert api_key, (
        "OPENROUTER_API_KEY is required for --sim-backend openrouter; export "
        "it before launching (the sim's stakeholder + veto calls all go "
        "through it)"
    )
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key,
                    max_retries=max_retries)

    def call_llm(client_unused, model_name, prompt, max_tokens, system=None):
        del client_unused, model_name  # one configured backend; provider args ignored.
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max(int(max_tokens), int(max_tokens_floor)),
        }
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if reasoning is not None:
            kwargs["extra_body"] = {"reasoning": {"effort": reasoning}}
        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        content = choice.message.content
        assert content and content.strip(), (
            f"OpenRouter sim returned empty content (model={model_id!r}, "
            f"finish_reason={choice.finish_reason!r}). If finish_reason is "
            f"'length', the reasoning budget crowded out the answer — raise "
            f"--sim-max-tokens-floor. Prompt: {prompt[:200]!r}"
        )
        usage = resp.usage
        assert usage is not None, (
            f"OpenRouter sim response missing usage block (model={model_id!r}); "
            f"cost tracking would silently under-count"
        )
        in_tok = int(usage.prompt_tokens)
        out_tok = int(usage.completion_tokens)
        if cost_tracker is not None:
            cost_tracker.add(in_tok, out_tok)
        return content.strip(), in_tok, out_tok

    return call_llm
