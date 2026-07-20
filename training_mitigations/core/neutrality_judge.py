"""Neutrality judging signal — the v5 gpt-5-mini endorse judge repurposed as a
TRAINING signal that targets NEUTRAL responses (v5 label 3).

The v5 endorse judge (run_endorse_gpt5mini.py) classifies a response toward the
user's action as 0=challenge, 1=explicit-affirm, 2=implicit-affirm, 3=neutral.
The "neutral" baselines want a policy whose responses the v5 judge labels 3, so
"good" == label 3 and "bad" == any of {0, 1, 2} (and any unparseable label —
see below). This module is the single place that maps a (question, response)
pair to that good/bad decision, so both baselines share one criterion that is
IDENTICAL to the v5 eval judge:

  - DPO baseline (train_dpo_lora.py --judge neutrality-v5): `NeutralityVetoJudge`
    mirrors the subset of the `SimulatedStakeholders` public surface that
    `Trainer` calls (prime_stakeholder_cache / veto_responses /
    veto_responses_batch). A response is VETOED iff the v5 judge does NOT label
    it 3; vetoed -> DPO `rejected`, non-vetoed (neutral) -> DPO `chosen`.
  - GRPO baseline (train_grpo_lora.py): `make_neutrality_reward` builds a TRL
    reward function returning +1 for v5 label 3 and 0 otherwise.

The judge prompt + per-call client contract are reused VERBATIM from
run_endorse_gpt5mini.py (PROMPTS['v5'], judge_one, first_label) so the label a
response gets at TRAIN time is the same label it would get at EVAL time. Reusing
the same judge for training and evaluation is a deliberate, documented choice:
the baselines' explicit goal is to maximize the v5 label-3 rate, so training
against the same judge is the strongest form of the baseline. (The downside —
Goodharting the judge — is exactly why the eval pipeline also reports a quality
judge and should cross-check with a held-out judge; that is out of scope here.)

INVALID LABELS. The judge is asked for a single 0-3 digit (reasoning=minimal,
temperature=0). `first_label` returns None when it cannot parse one. An
unparseable label is NOT a confident "neutral", so for a "produce neutral"
objective the conservative, fail-safe choice is to treat None as non-neutral
(vetoed in DPO / reward 0 in GRPO) rather than reward uncertainty. The count of
invalids is surfaced (printed) so a judge that starts failing to emit digits is
visible rather than silently degrading the signal.
"""

import os
import sys
import threading

from openai import OpenAI

# Reuse the v5 endorse judge prompt + per-call helpers VERBATIM (single source
# of truth for the judge surface). run_endorse_gpt5mini lives under eval/.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "eval"))
from run_endorse_gpt5mini import PROMPTS, judge_one, first_label  # noqa: E402

# core/ is on sys.path for the train entry points; concurrency lives there.
sys.path.insert(0, _HERE)
from concurrency import concurrent_map  # noqa: E402
# CostTracker moved to cost.py so the OpenRouter sim backend can share it.
# Imported (not just referenced) so `neutrality_judge.CostTracker` keeps
# resolving for existing callers.
from cost import CostTracker  # noqa: E402,F401

# The neutral label in the v5 judge's 0-3 scheme.
NEUTRAL_LABEL = 3


def make_openrouter_client():
    """Construct the OpenRouter-backed OpenAI client the v5 judge calls go
    through. Mirrors run_endorse_gpt5mini.main's client (same base_url +
    retries). Requires OPENROUTER_API_KEY in the environment.
    """
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        max_retries=12,
    )


def judge_labels(client, model, pairs, reasoning, max_tokens, temperature,
                 concurrency, cost_tracker=None):
    """Label every (question, response) pair with the v5 endorse judge.

    Inputs:
      client:      an OpenRouter OpenAI client (see make_openrouter_client).
      model:       judge model id (e.g. 'openai/gpt-5-mini').
      pairs:       list of (question, response) string tuples.
      reasoning:   OpenRouter reasoning effort passed to judge_one (e.g.
                   'minimal'); None to omit.
      max_tokens:  judge output budget (incl. reasoning tokens).
      temperature: judge sampling temperature; None to omit (provider default).
                   Pin to 0.0 for deterministic labels.
      concurrency: max in-flight judge calls (bounds OpenRouter fan-out).
      cost_tracker: optional CostTracker; when set, every call's token usage is
                   added to it (and a hard cap, if any, is enforced — see
                   CostTracker).

    Returns a list of labels aligned with `pairs`; each is an int in 0..3 or
    None when the judge's reply did not contain a parseable 0-3 digit. Fans the
    calls out with `concurrent_map` (order-preserving, fail-fast).
    """
    template = PROMPTS["v5"]

    def _one(pair):
        question, response = pair
        content, ptok, ctok, _rtok, _served = judge_one(
            client, model, template, question, response, max_tokens, reasoning,
            temperature)
        if cost_tracker is not None:
            cost_tracker.add(ptok, ctok)
        return first_label(content)

    return concurrent_map(_one, pairs, concurrency)


class NeutralityVetoJudge:
    """v5-neutrality veto judge with the subset of the `SimulatedStakeholders`
    surface that `Trainer` calls, so it is a drop-in `sim_stake` for the DPO
    trainer. A response is vetoed iff the v5 judge does NOT label it 3 (neutral).

    Unlike `SimulatedStakeholders` there is no stakeholder "world model": the
    veto is a single per-response judge call. `prime_stakeholder_cache` is a
    no-op and `_stakeholder_data_for` raises — mixed sampling (which sources
    candidates from per-stakeholder voices) is therefore unsupported with this
    judge, and the caller asserts `--mixed-sampling` is off.

    Concurrency for the judge calls is bounded by this object's own
    `concurrency` (set from --judge-concurrency), NOT by the `concurrency`
    argument `Trainer` threads through (which is tuned for the local vLLM
    server's fan-out, far larger than OpenRouter should see at once).
    """

    def __init__(self, client, model, reasoning="minimal", max_tokens=4000,
                 temperature=0.0, concurrency=16, price_in=0.25, price_out=2.00,
                 max_cost=None):
        self.client = client
        self.model = model
        self.reasoning = reasoning
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.concurrency = concurrency
        # Cumulative spend tracker (+ optional hard cap). Shared across all
        # veto_responses_batch calls, so the cap bounds the WHOLE run's judge
        # spend, not one iteration's.
        self.cost = CostTracker(price_in, price_out, max_cost)
        self._invalid = 0
        self._invalid_lock = threading.Lock()

    def prime_stakeholder_cache(self, prompts, concurrency):
        """No-op. The neutrality judge has no per-prompt stakeholder pipeline to
        prime; present only so `Trainer`'s Stage A call is a harmless no-op.
        """
        return

    def _stakeholder_data_for(self, prompt):
        raise NotImplementedError(
            "NeutralityVetoJudge has no stakeholder data; --mixed-sampling is "
            "unsupported for the neutrality-v5 baseline (assert it off)."
        )

    def veto_responses(self, prompt, responses):
        """Single-prompt convenience wrapper over `veto_responses_batch`, so the
        per-prompt and batched paths share one implementation.
        """
        return self.veto_responses_batch([(prompt, responses)], self.concurrency)[0]

    def veto_responses_batch(self, items, concurrency):
        """For each (prompt, responses) item, return the vetoed subset — every
        response the v5 judge does NOT label 3 — in original response order.

        The `concurrency` argument (passed by `Trainer`) is IGNORED in favor of
        `self.concurrency` so the OpenRouter fan-out stays bounded by the
        judge's own knob regardless of the vLLM-tuned stage concurrency. Every
        (item, response) judge call across the whole batch is fanned out at once
        so a single round-trip covers the chunk.
        """
        index = []
        pairs = []
        for item_idx, (prompt, responses) in enumerate(items):
            for pos, response in enumerate(responses):
                index.append((item_idx, pos))
                pairs.append((prompt, response))

        labels = judge_labels(
            self.client, self.model, pairs, self.reasoning, self.max_tokens,
            self.temperature, self.concurrency, cost_tracker=self.cost)
        print(f"[neutrality-judge] judged {len(pairs)} responses; "
              f"cumulative spend=${self.cost.spent():.3f}"
              + (f"/${self.cost.max_cost:.2f}" if self.cost.max_cost else ""))

        n_invalid = sum(1 for label in labels if label is None)
        if n_invalid:
            with self._invalid_lock:
                self._invalid += n_invalid
            print(f"[neutrality-judge] {n_invalid}/{len(labels)} responses got "
                  f"no parseable v5 label this batch -> treated as non-neutral "
                  f"(vetoed); cumulative invalid={self._invalid}")

        out = [[] for _ in items]
        for (item_idx, pos), label in zip(index, labels):
            # Veto iff NOT confidently neutral. None (unparseable) is not a
            # confident neutral, so it is vetoed (see module docstring).
            if label != NEUTRAL_LABEL:
                out[item_idx].append(items[item_idx][1][pos])
        return out


def _completion_text(completion):
    """Normalize a TRL completion to its text. For a conversational dataset TRL
    hands the reward function each completion as a message list
    (`[{"role": "assistant", "content": ...}]`); for a plain-text dataset it is
    already a string. Returns the assistant text either way; asserts on any
    other shape so a silent mis-judge can't slip through.
    """
    if isinstance(completion, str):
        return completion
    if (isinstance(completion, list) and completion
            and isinstance(completion[-1], dict) and "content" in completion[-1]):
        return completion[-1]["content"]
    raise AssertionError(f"unexpected completion shape: {completion!r}")


def make_neutrality_reward(client, model, reasoning="minimal", max_tokens=4000,
                           temperature=0.0, concurrency=16, price_in=0.25,
                           price_out=2.00, max_cost=None):
    """Build a TRL GRPO reward function: +1 if the v5 judge labels the response
    3 (neutral), else 0.

    TRL calls the reward function as
        reward_func(prompts=..., completions=..., **dataset_columns)
    where every extra dataset column is repeated to align with `completions`.
    The GRPO dataset (see train_grpo_lora.py) carries a `post` column holding
    the BARE question text (the `prompt` column is the conversational message
    the policy generates from), so the judge scores (post, completion_text) —
    exactly the (question, response) pair the v5 eval judge scores. Completions
    are normalized via `_completion_text` (conversational message list or plain
    string). An unparseable label yields reward 0 (non-neutral; see module
    docstring).

    A `CostTracker` (price_in/price_out/max_cost) is closed over so the cap
    bounds the WHOLE run's reward-judge spend. Returns a callable with
    `__name__ == 'neutrality_v5_reward'` (TRL uses the name for per-reward
    logging).
    """
    cost = CostTracker(price_in, price_out, max_cost)

    def neutrality_v5_reward(completions, post, **kwargs):
        texts = [_completion_text(c) for c in completions]
        pairs = list(zip(post, texts))
        labels = judge_labels(
            client, model, pairs, reasoning, max_tokens, temperature,
            concurrency, cost_tracker=cost)
        print(f"[neutrality-reward] judged {len(pairs)} completions; "
              f"cumulative spend=${cost.spent():.3f}"
              + (f"/${cost.max_cost:.2f}" if cost.max_cost else ""))
        n_invalid = sum(1 for label in labels if label is None)
        if n_invalid:
            print(f"[neutrality-reward] {n_invalid}/{len(labels)} completions "
                  f"got no parseable v5 label -> reward 0")
        return [1.0 if label == NEUTRAL_LABEL else 0.0 for label in labels]

    neutrality_v5_reward.__name__ = "neutrality_v5_reward"
    return neutrality_v5_reward
