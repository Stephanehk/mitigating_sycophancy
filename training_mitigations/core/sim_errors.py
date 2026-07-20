"""Backend-neutral error types for the frozen sim.

Lives in its own module so `simulated_stakeholders` can handle a provider
refusal WITHOUT importing a backend (importing `gemini_sim` would make
`google.genai` a hard dependency of the local Qwen path, which has no business
needing it). Backends raise these; the pipeline decides the policy.
"""


class SimRefusedError(RuntimeError):
    """The sim backend refused to answer, and retrying will not help.

    Distinct from a transport fault (retryable) and from a malformed reply
    (which the pipeline's own parsers catch). Raised today by `gemini_sim` when
    Gemini returns finish_reason=PROHIBITED_CONTENT / SAFETY / BLOCKLIST with
    empty content: measured deterministic across retries, on both
    3.1-flash-lite and 3-flash-preview, and NOT relaxable via safety_settings.

    Local backends (vLLM / HF Qwen) never raise this, so every `except
    SimRefusedError` in the pipeline is inert for the existing arms.
    """
