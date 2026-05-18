"""Tiny client object for the OpenRouter chat-completions endpoint.

Holds the API key and the optional app-attribution headers OpenRouter
uses for its rankings page (HTTP-Referer, X-Title). No persistent HTTP
session — each call in `call.py` issues its own `requests.post`, which
keeps the harness's threaded worker pool unambiguously safe.

Authentication:
  OPENROUTER_API_KEY    required.
  OPENROUTER_HTTP_REFERER, OPENROUTER_X_TITLE
                        optional; forwarded verbatim if set.
"""

import os


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    """Carrier for OpenRouter auth + attribution headers.

    Assumptions:
      - `api_key` is a non-empty string. The constructor asserts this so
        a misconfigured client cannot silently issue unauthenticated
        requests.
    Inputs:
      api_key:       the OpenRouter API key.
      http_referer:  optional value for the OpenRouter HTTP-Referer
                     header (used for rankings).
      x_title:       optional value for the OpenRouter X-Title header.

    The client itself has no methods other than `headers()`, which
    builds the per-request header dict. Threaded callers can share a
    single instance freely because nothing on it is mutated after
    construction.
    """

    def __init__(self, api_key, http_referer=None, x_title=None):
        assert api_key, "OpenRouterClient requires a non-empty api_key"
        self.api_key = api_key
        self.http_referer = http_referer
        self.x_title = x_title

    def headers(self):
        """Build the per-request header dict.

        Always includes Authorization and Content-Type. HTTP-Referer and
        X-Title are added only when the corresponding env vars were set
        at client-construction time — OpenRouter accepts requests
        without them, the headers just opt the run out of the rankings
        attribution.
        """
        h = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
        }
        if self.http_referer:
            h["HTTP-Referer"] = self.http_referer
        if self.x_title:
            h["X-Title"] = self.x_title
        return h


def make_openrouter_client():
    """Build the OpenRouter client used for every non-Claude call.

    Assumptions:
      - `OPENROUTER_API_KEY` is exported in the environment. Asserted —
        a missing key fails the run at startup, rather than letting
        every worker thread spawn a 401.
    Inputs: none (reads env vars).
    Returns: a configured `OpenRouterClient`.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    assert api_key, (
        "OPENROUTER_API_KEY env var is required to route non-Claude "
        "models through OpenRouter"
    )
    return OpenRouterClient(
        api_key=api_key,
        http_referer=os.environ.get("OPENROUTER_HTTP_REFERER"),
        x_title=os.environ.get("OPENROUTER_X_TITLE"),
    )
