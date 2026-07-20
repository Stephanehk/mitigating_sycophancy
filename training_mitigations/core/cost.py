"""Cumulative USD spend tracking + hard cap for API-backed LLM calls.

Both API-backed paths in the training pipeline bill per token and both run
unattended for hours, so both need the same runaway-cost backstop: count every
call's tokens, and abort the run once cumulative spend reaches a cap. This
module is the single implementation, shared by:

  - `neutrality_judge.py`  — the v5 gpt-5-mini endorse judge (--judge-max-cost).
  - `openrouter_sim.py`    — the OpenRouter sim backend (--sim-max-cost).

`neutrality_judge` re-exports `CostTracker` for backward compatibility, so
existing `from neutrality_judge import CostTracker` callers keep working.
"""

import threading


class CostTracker:
    """Thread-safe cumulative USD spend tracker + optional hard cap.

    Cost per call is `(prompt_tokens * price_in + completion_tokens * price_out)
    / 1e6` — the same formula run_endorse_gpt5mini uses (`completion_tokens`
    already includes reasoning tokens). When `max_cost` is set, `add` raises
    once cumulative spend reaches it; the exception propagates out of
    `concurrent_map` (fail-fast), aborting the run. Because the check fires
    AFTER a call returns, spend can overshoot the cap by up to the in-flight
    concurrency — set the cap with headroom (the sbatch wrappers use 1.5x the
    estimate). DPO snapshots each completed iteration's adapter, so a breach
    still leaves the last finished iteration loadable.

    Assumptions:
      - `price_in` / `price_out` are USD per 1M tokens.
      - `add` is called exactly once per completed call, with that call's own
        token counts; double-counting would silently inflate the spend curve.

    Inputs:
      price_in / price_out: USD per 1M input / output tokens.
      max_cost: hard cap in USD, or None for no cap (track only).
      label: what to call this spend stream in the breach message, so a run
        with BOTH a judge cap and a sim cap reports which one tripped.
    """

    def __init__(self, price_in, price_out, max_cost=None, label="v5 judge"):
        self.price_in = price_in
        self.price_out = price_out
        self.max_cost = max_cost
        self.label = label
        self._spent = 0.0
        self._lock = threading.Lock()

    def add(self, prompt_tokens, completion_tokens):
        cost = (prompt_tokens * self.price_in
                + completion_tokens * self.price_out) / 1e6
        with self._lock:
            self._spent += cost
            spent = self._spent
        if self.max_cost is not None and spent >= self.max_cost:
            raise RuntimeError(
                f"{self.label} cumulative spend ${spent:.2f} reached the cap "
                f"${self.max_cost:.2f}; aborting before further calls."
            )
        return spent

    def spent(self):
        with self._lock:
            return self._spent
