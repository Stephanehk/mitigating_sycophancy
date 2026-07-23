"""Direct Google Gemini API backend for the frozen sim.

Serves the stakeholder pipeline's LLM calls (identify / filter / concerns and the
whole veto panel) from a Gemini model instead of a local Qwen3-8B, using
$GEMINI_API_KEY via `google-genai`. It is a drop-in for
`vllm_sim.make_vllm_call_llm` / `llama_local.make_call_llm` /
`openrouter_sim.make_openrouter_call_llm` at the ONE place train_dpo_lora.py
builds `sim_call_llm`.

WHY NOT OpenRouter (`openrouter_sim.py`)
----------------------------------------
Gemini traffic goes to Google directly: it avoids the middleman's markup and,
more importantly, its provider routing — the frozen sim's determinism is
load-bearing (`trainer._record_contribution` rebuilds pairs from disk without
re-vetoing "because the sim is frozen"), and a router that can silently move you
between backends undermines that. OpenRouter remains the path for non-Gemini API
models (e.g. the v5 gpt-5-mini endorse judge).

DETERMINISM
  temperature is pinned to 0.0 and thinking is disabled (thinking_budget=0), for
  the same reason vllm_sim pins temperature=0.0 and llama_local uses
  do_sample=False. Note this is best-effort: an API model is not guaranteed
  reproducible across model updates the way a pinned local checkpoint is.

THINKING / COST
  Google bills thinking tokens at the standard OUTPUT rate. thinking_budget=0
  disables them (verified: a 1-token reply bills 1 output token). Leaving
  thinking on multiplies the output bill several-fold for calls whose visible
  answer is one word (VETO/ACCEPT), which is ~70% of the run's volume.

RETRIES
  Transient failures (429 rate-limit, 5xx, timeouts) retry with exponential
  backoff + jitter, capped at `backoff_max` seconds (default 300 = 5 min), for
  `max_retries` attempts (default 5). try/except here is deliberate and
  narrowly scoped to transport-level faults — the repo's fail-fast rule still
  governs everything else. A "successful" reply with EMPTY content but a
  NON-block finish_reason (observed: None, and MAX_TOKENS) is also treated as
  transient and retried on the same schedule: it raises no exception, so the
  transport-fault path never sees it, yet it is an intermittent glitch that
  clears on a re-request rather than a deterministic refusal. After the last
  retry it still raises loudly (fail-fast preserved).

SAFETY BLOCKS ARE NOT RETRYABLE
  Gemini's core child-safety filter returns finish_reason=PROHIBITED_CONTENT with
  EMPTY content. Measured: deterministic (3/3 identical retries), fires on BOTH
  3.1-flash-lite and 3-flash-preview, and is NOT relaxable via safety_settings
  (all four HARM_CATEGORY_* thresholds at BLOCK_NONE change nothing). It fired on
  1/30 audited prompts — the filter step of an AITA post enumerating minors.
  Retrying it would burn the whole backoff schedule to fail anyway, so it raises
  `GeminiBlockedError` immediately. The CALLER must decide the policy; see
  `blocked_action`. Dropping blocked prompts is NOT neutral — it removes
  child-related scenarios systematically, which biases the training set.

ASSUMPTIONS
  - GEMINI_API_KEY is set (asserted at construction, not per call).
  - The response exposes .text and .usage_metadata; empty text with a non-block
    finish_reason is RETRIED (see RETRIES) and, if still empty after the last
    attempt, an ASSERT — never a silent "" (an empty stakeholder list or veto
    verdict would corrupt the preference dataset rather than fail).
  - Callers bound their own fan-out; see --sim-concurrency.
"""

import os
import random
import sys
import time

from google import genai
from google.genai import types

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from cost import CostTracker  # noqa: E402,F401  (re-exported for callers)
from sim_errors import SimRefusedError

DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_MAX = 300.0   # seconds; the specified 5-minute ceiling
# base**(attempt+1) capped at BACKOFF_MAX gives per-attempt ceilings
#   4, 16, 64, 256, 300   (5 attempts)
# so the schedule actually REACHES the 5-minute ceiling on the last retry.
# base=2 would top out at 32s and make BACKOFF_MAX decorative. Waits are
# full-jittered (uniform over [0, ceiling]), so expected waits are half these.
DEFAULT_BACKOFF_BASE = 4.0

# finish_reason values that mean "the platform refused", not "transient fault".
_BLOCK_REASONS = ("PROHIBITED_CONTENT", "SAFETY", "BLOCKLIST", "SPII",
                  "IMAGE_SAFETY", "RECITATION")


class GeminiBlockedError(SimRefusedError):
    """Gemini refused to answer (finish_reason in _BLOCK_REASONS).

    A `SimRefusedError`, so `simulated_stakeholders` can apply its per-stage skip
    policy without importing this backend. Retrying is pointless (the refusal is
    deterministic for a given prompt), and silently treating it as an ordinary
    empty answer would corrupt the preference dataset -- hence a typed error
    rather than "".
    """


class GeminiEmptyError(AssertionError):
    """Gemini returned an empty body with a non-block finish_reason on the FINAL
    retry (see the raise site for the full diagnosis).

    Subclasses AssertionError so existing fail-fast behaviour is unchanged for
    every caller: the core sim steps (identify / filter / concerns / veto) still
    die loudly, because an empty answer there would silently corrupt the veto and
    therefore the preference dataset.

    It is a distinct TYPE only so that AUXILIARY per-candidate classifiers can
    opt into failing open instead. Observed 2026-07-20: the assistant-role veto
    on OEQ-length inputs exhausted all 5 retries and killed a 1h45m run at
    iteration 2. That classifier only decides whether one candidate becomes a
    rejected exemplar, so degrading to "unparseable, not flagged" (counted, and
    surfaced in the cumulative WARN) is strictly better than losing the run.
    """


def _is_transient(exc):
    """True for faults worth retrying: rate limits, server errors, timeouts.

    Matches on the exception's text rather than SDK-specific classes so it keeps
    working across google-genai versions (which have moved these types around).
    A block is never transient — it is raised as GeminiBlockedError before this
    is consulted.
    """
    s = f"{type(exc).__name__}: {exc}".lower()
    markers = ("429", "resource_exhausted", "resourceexhausted", "rate limit",
               "quota", "500", "502", "503", "504", "internal", "unavailable",
               "deadline", "timeout", "timed out", "connection", "temporarily")
    return any(m in s for m in markers)


def _sleep_for(attempt, backoff_base, backoff_max):
    """Exponential backoff with full jitter, capped at backoff_max.

    Jitter matters because the sim fans out --sim-concurrency calls at once; a
    deterministic schedule would re-synchronize every retry into the same spike
    that caused the 429.
    """
    ceiling = min(backoff_base ** (attempt + 1), backoff_max)
    return random.uniform(0.0, ceiling)


def make_gemini_call_llm(model_id,
                         temperature=0.0,
                         thinking_budget=0,
                         max_retries=DEFAULT_MAX_RETRIES,
                         backoff_max=DEFAULT_BACKOFF_MAX,
                         backoff_base=DEFAULT_BACKOFF_BASE,
                         max_tokens_floor=0,
                         cost_tracker=None,
                         on_blocked=None):
    """Return a harness-signature `call_llm` routed to a Gemini model.

    Inputs:
      model_id: Gemini model id, e.g. "gemini-3.1-flash-lite" (NOT the
        OpenRouter-style "google/..." id).
      temperature: pinned decoding temperature; 0.0 keeps the frozen sim
        deterministic. None omits the field (provider default) — only if you
        accept a non-reproducible veto signal.
      thinking_budget: 0 disables thinking (the cost lever). None leaves the
        model default, which bills thinking at the output rate.
      max_retries / backoff_max / backoff_base: transient-fault retry schedule.
        Defaults: 5 attempts, exponential from 2s, capped at 300s (5 min).
      max_tokens_floor: raise any requested max_tokens to at least this. Only
        needed if thinking is re-enabled (hidden tokens can crowd out the answer).
      cost_tracker: optional `CostTracker`; every call's usage is added, and a
        cap breach raises out of the call (fail-fast).
      on_blocked: optional callable(prompt, finish_reason) invoked on every
        refusal before GeminiBlockedError is raised — for counting/logging, so a
        silently-shrinking dataset is impossible.

    On a refusal this ALWAYS raises GeminiBlockedError (a `SimRefusedError`).
    It never returns "": `simulated_stakeholders` applies a per-stage skip policy
    that needs to tell a refusal apart from an ordinary empty/unparsable reply,
    and collapsing the two would discard the wrong thing (e.g. exclude a response
    when only one voter's vote should have been dropped).

    Returns: call_llm(client, model, prompt, max_tokens, system=None)
      -> (text, in_tokens, out_tokens). Thread-safe, so it is safe under
      `concurrency.concurrent_map`.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    assert api_key, (
        "GEMINI_API_KEY is required for --sim-backend gemini; export it before "
        "launching (the sim's stakeholder + veto calls all go through it)"
    )
    client = genai.Client(api_key=api_key)

    def _config(max_tokens, system):
        kw = dict(max_output_tokens=max(int(max_tokens), int(max_tokens_floor)))
        if temperature is not None:
            kw["temperature"] = float(temperature)
        if system is not None:
            kw["system_instruction"] = system
        if thinking_budget is not None:
            kw["thinking_config"] = types.ThinkingConfig(
                thinking_budget=int(thinking_budget))
        return types.GenerateContentConfig(**kw)

    def call_llm(client_unused, model_name, prompt, max_tokens, system=None):
        del client_unused, model_name  # one configured backend; provider args ignored.
        cfg = _config(max_tokens, system)
        last = None
        for attempt in range(int(max_retries)):
            try:
                resp = client.models.generate_content(
                    model=model_id, contents=prompt, config=cfg)
            except Exception as exc:                      # noqa: BLE001
                # Transport-level fault. Retry per the user-specified schedule;
                # anything else propagates immediately (fail-fast).
                if not _is_transient(exc) or attempt == int(max_retries) - 1:
                    raise
                last = exc
                delay = _sleep_for(attempt, backoff_base, backoff_max)
                print(f"[gemini] transient {type(exc).__name__} "
                      f"(attempt {attempt + 1}/{max_retries}); retrying in "
                      f"{delay:.1f}s", flush=True)
                time.sleep(delay)
                continue

            finish = resp.candidates[0].finish_reason if resp.candidates else None
            fname = getattr(finish, "name", str(finish))
            if fname in _BLOCK_REASONS:
                if on_blocked is not None:
                    on_blocked(prompt, fname)
                raise GeminiBlockedError(
                    f"Gemini refused (finish_reason={fname}, model={model_id!r}). "
                    f"This is deterministic and NOT relaxable via safety_settings, "
                    f"so it is never retried. simulated_stakeholders catches this "
                    f"and SKIPS the affected step. Prompt: {prompt[:200]!r}")

            text = resp.text
            if not (text and text.strip()):
                # Empty body with a NON-block finish reason (observed: None, and
                # MAX_TOKENS when a thinking budget crowds out the answer). This
                # is neither a deterministic refusal (those are _BLOCK_REASONS,
                # raised above) nor a raised transport fault (caught above), so it
                # would otherwise fall straight through to a hard crash. Measured
                # to be an intermittent glitch that clears on a re-request, so
                # treat it like any other transient: back off and retry within the
                # same schedule. Fail-fast is preserved — after the last attempt
                # it still raises loudly, keeping the MAX_TOKENS guidance.
                last = RuntimeError(f"empty content (finish_reason={fname!r})")
                if attempt == int(max_retries) - 1:
                    raise GeminiEmptyError(
                        f"Gemini sim returned empty content (model={model_id!r}, "
                        f"finish_reason={fname!r}) on the final of {max_retries} "
                        f"retries. If it is MAX_TOKENS the thinking budget crowded "
                        f"out the answer — raise --sim-max-tokens-floor or lower "
                        f"--sim-thinking-budget. Prompt: {prompt[:200]!r}")
                delay = _sleep_for(attempt, backoff_base, backoff_max)
                print(f"[gemini] empty content (finish_reason={fname}) "
                      f"(attempt {attempt + 1}/{max_retries}); retrying in "
                      f"{delay:.1f}s", flush=True)
                time.sleep(delay)
                continue

            usage = resp.usage_metadata
            assert usage is not None, (
                f"Gemini response missing usage_metadata (model={model_id!r}); "
                f"cost tracking would silently under-count")
            in_tok = int(usage.prompt_token_count or 0)
            out_tok = int((usage.candidates_token_count or 0)
                          + (getattr(usage, "thoughts_token_count", 0) or 0))
            if cost_tracker is not None:
                cost_tracker.add(in_tok, out_tok)
            return text.strip(), in_tok, out_tok

        raise RuntimeError(
            f"gemini sim exhausted {max_retries} retries; last error: {last!r}")

    return call_llm
