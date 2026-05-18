"""OpenRouter routing helpers for the sycophancy mitigations harness.

Provides drop-in replacements for the provider-specific helpers in
`mitigations/explore_mitigations.py` so any non-Claude model can be
called through OpenRouter (https://openrouter.ai/) using the OpenAI-
compatible `/api/v1/chat/completions` endpoint described in the
OpenRouter quickstart.

Public API:
  make_openrouter_client()
    Build the client used for every non-Claude call. Reads
    `OPENROUTER_API_KEY` from the environment.
  call_openrouter(client, model, prompt, max_tokens, system=None)
    Single chat-completion call. Same shape as
    `explore_mitigations.call_llm` so it slots directly into the
    existing dispatch.
  OpenRouterClient
    Thin client object returned by `make_openrouter_client`. Exported
    for type-introspection or test stubbing — callers do not need to
    construct it directly.
"""

from .client import OpenRouterClient, make_openrouter_client
from .call import call_openrouter, OpenRouterEmptyResponseError

__all__ = [
    "OpenRouterClient",
    "make_openrouter_client",
    "call_openrouter",
    "OpenRouterEmptyResponseError",
]
