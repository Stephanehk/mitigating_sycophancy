"""Single-call wrapper for OpenRouter's chat-completions endpoint.

The public entry point `call_openrouter` mirrors the signature of
`explore_mitigations.call_llm` so the mitigations harness can swap it
in without rewriting call sites:

  call_openrouter(client, model, prompt, max_tokens, system=None,
                  prefill=None)
    -> (text, input_tokens, output_tokens)

`model` is an OpenRouter model id (e.g. "openai/gpt-4o",
"google/gemini-2.5-pro"). `system`, if provided, is prepended as a
system-role message — OpenRouter's chat-completions endpoint is
OpenAI-compatible, so the role conventions match. `prefill`, if
provided, is appended as a final assistant-role message so the chat
template the upstream model sees has `prefill` as the start of the
assistant turn (used by the WAIT condition to inject "Wait a minute, ").
True prefill semantics — the response continuing from `prefill` —
depend on the upstream provider; some (Anthropic) honor it natively,
others (OpenAI) treat it as conversation history. Either way, the
returned text is `prefill + continuation` so judges see the full
assistant utterance.

Retries: HTTP 429 responses (rate-limit) and the upstream-cutoff
failure mode (HTTP 200 with null `finish_reason` and all-zero `usage`,
which OpenRouter returns when an upstream provider drops a streaming
response mid-flight) are retried via `tenacity`. All other HTTP errors
fail fast so the caller sees the underlying provider error instead of
having it silently swallowed. The retry budget is wall-time-bounded
(90s) so a deterministically-broken prompt does not loop forever.
"""

import json

import requests
import tenacity

from .client import OPENROUTER_BASE_URL


class OpenRouterEmptyResponseError(Exception):
    """OpenRouter returned 200 OK with no usable content and no
    terminal `finish_reason`.

    Two observed variants, both transient:
      - Gemini reasoning loop: `total_tokens=0`, provider abandoned
        the request before billing anything.
      - GPT-5 family: `prompt_tokens>0`, `completion_tokens=0` —
        provider accepted (and billed) the prompt but emitted zero
        output tokens.
    The common signature is `finish_reason is None` (and `content` is
    empty / null). Retried by the tenacity wrapper, capped at 3
    attempts (initial + 2 retries) so a deterministically-broken
    prompt does not loop indefinitely. The parsed response body is
    preserved on `.response_data` for post-mortem inspection if all
    attempts fail.
    """

    def __init__(self, message, response_data):
        super().__init__(message)
        self.response_data = response_data


def _is_retryable(exc):
    """True if `exc` is a transient failure worth retrying.

    Retries:
      - HTTP 429 from OpenRouter (per-minute rate-limit or upstream
        provider rate-limit; both clear quickly).
      - `OpenRouterEmptyResponseError` (upstream-cutoff signature; see
        the class docstring).
    Other HTTPError statuses (4xx auth/validation, 5xx server) and any
    other exception are not retried — they indicate a real problem
    that backoff cannot fix.
    """
    if isinstance(exc, OpenRouterEmptyResponseError):
        return True
    if isinstance(exc, requests.HTTPError):
        r = exc.response
        return r is not None and r.status_code == 429
    return False


_retry = tenacity.retry(
    retry=tenacity.retry_if_exception(_is_retryable),
    wait=tenacity.wait_random_exponential(multiplier=2, max=30),
    # Cap at 3 attempts (1 initial + 2 retries). The upstream-cutoff
    # failure modes are either truly transient (one retry usually
    # rescues) or deterministic on this prompt (no number of retries
    # will fix it) — there is little value in spending more than 2
    # retries either way.
    stop=tenacity.stop_after_attempt(3),
    reraise=True,
)


@_retry
def call_openrouter(client, model, prompt, max_tokens, system=None,
                    prefill=None):
    """Send `prompt` to OpenRouter as a single user turn.

    Assumptions:
      - `client` is an `OpenRouterClient` from
        `open_router_utils.make_openrouter_client()`.
      - `model` is a valid OpenRouter model id; OpenRouter validates
        and returns a 4xx if not.
      - The endpoint responds with the OpenAI-compatible shape:
        `choices[0].message.content` plus a `usage` block.

    Inputs:
      client:     OpenRouterClient.
      model:      OpenRouter model id (e.g. "openai/gpt-4o").
      prompt:     user-turn text.
      max_tokens: accepted for signature compatibility with
                  `explore_mitigations.call_llm` but intentionally NOT
                  forwarded to OpenRouter. Reasoning models (e.g.
                  `google/gemini-3.1-pro-preview`) burn most of the
                  output budget on hidden chain-of-thought before any
                  visible `content` appears; capping at the harness
                  default of 2048 left `content` empty. Omitting the
                  field lets OpenRouter / the upstream provider use the
                  model's own max-output cap.
      system:     optional system prompt; sent as a system-role
                  message ahead of the user turn.
      prefill:    optional assistant-prefill string. When set, an
                  assistant-role message with this content is appended
                  after the user turn so the chat template the upstream
                  model sees has `prefill` as the beginning of the
                  assistant turn. Whether the model truly continues
                  from it (Anthropic) or treats it as a completed prior
                  turn (most other providers) varies upstream; the
                  returned text is `prefill + content` either way so
                  judges and the rest of the harness see the full
                  assistant utterance.

    Returns:
      (text, input_tokens, output_tokens). `text` is stripped of
      surrounding whitespace and, when `prefill` is set, has `prefill`
      prepended so the returned string is the full assistant turn the
      upstream model produced (prefill + continuation). Token counts
      are read from `usage.prompt_tokens` / `usage.completion_tokens`.

    Asserts on empty content, missing choices, or a missing usage
    block — silent empties / missing usage would propagate junk into
    the harness metrics and the cost report.
    """
    del max_tokens  # see docstring — intentionally not forwarded
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    if prefill is not None:
        messages.append({"role": "assistant", "content": prefill})
    payload = {
        "model": model,
        "messages": messages,
    }
    # No request-side timeout: reasoning models (e.g. Gemini 3 Pro)
    # routinely take many minutes per call, especially on the
    # stakeholder-concerns prompt. A shorter timeout was masking real
    # responses as connection failures.
    response = requests.post(
        OPENROUTER_BASE_URL + "/chat/completions",
        headers=client.headers(),
        data=json.dumps(payload),
        timeout=None,
    )
    response.raise_for_status()
    data = response.json()
    assert "choices" in data and data["choices"], (
        "OpenRouter response missing choices; "
        "model=" + repr(model) + " raw=" + repr(data)
    )
    choice = data["choices"][0]
    message = choice.get("message") or {}
    text = (message.get("content") or "").strip()
    usage = data.get("usage") or {}
    if not text:
        finish_reason = choice.get("finish_reason")
        native_finish_reason = choice.get("native_finish_reason")
        # Upstream-cutoff signature: empty `content` AND no terminal
        # `finish_reason`. The response was aborted before completing
        # rather than finishing cleanly with no output. Two observed
        # variants share this signature — see
        # `OpenRouterEmptyResponseError`. Both are transient; retry.
        # The retry is gated by the tenacity wrapper above; this
        # branch just raises the typed exception its predicate matches.
        if finish_reason is None and native_finish_reason is None:
            raise OpenRouterEmptyResponseError(
                "OpenRouter upstream-cutoff signature "
                "(model=" + repr(model) + ", finish_reason=None, "
                "native_finish_reason=None); retrying. "
                "usage=" + repr(usage),
                response_data=data,
            )
        # Otherwise: the response really did complete but produced no
        # visible content. Usually a reasoning model that spent its
        # entire output budget on hidden chain-of-thought, or a
        # content-filter trip. Not retryable — fail fast with full
        # diagnostic detail so the user can decide whether to switch
        # model / adjust the reasoning knob / rewrite the prompt.
        raise AssertionError(
            "empty content from OpenRouter "
            "(model=" + repr(model) + ", "
            "finish_reason=" + repr(finish_reason) + ", "
            "native_finish_reason=" + repr(native_finish_reason) + ")"
            "\n--- message ---\n" + json.dumps(message, indent=2) +
            "\n--- usage ---\n" + json.dumps(data.get("usage"), indent=2)
        )
    assert usage, (
        "OpenRouter response missing usage block; "
        "model=" + repr(model) + " raw=" + repr(data)
    )
    in_tok = int(usage.get("prompt_tokens", 0))
    out_tok = int(usage.get("completion_tokens", 0))
    if prefill is not None:
        text = prefill + text
    return text, in_tok, out_tok
