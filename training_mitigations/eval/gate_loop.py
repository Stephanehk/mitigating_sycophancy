"""In-loop rejection sampling against the assistant-mode gate.

Used by the generation drivers when --gate-loop is set: for each prompt still
unresolved after attempt 1, resample from the SAME arm and judge each draw
inline, stopping at the first draw the gate passes or at MAX_ATTEMPTS.

WHY IN-LOOP. An earlier design staged this as alternating GPU and network
rounds, on the assumption that Marlowe compute nodes are air-gapped -- an
assumption stated in several scripts in this repo and empirically FALSE: a
compute node reaches openrouter.ai and huggingface.co over HTTPS with no proxy
(verified on n31). Judging inline is therefore possible and far simpler, and it
costs only the judge latency (~1-2s) against a 512-1536 token generation
(~10-30s), and only on rejected draws.

ATTEMPT NUMBERING. Attempt 1 is the frozen finaldata generation already on disk;
this loop starts at attempt 2. Each draw is seeded by (prompt, attempt) via
gate_common.attempt_seed, so a preempted job reproduces the same draws on
restart rather than silently sampling a different response.

WHAT THE GATE DOES AND DOES NOT COVER. It judges whether a response engages the
user's situation or instead treats their message as a text to critique, assess,
continue, or replace. It deliberately does NOT flag narrative-voice collapse (a
response written as someone inside the user's story) -- that is a separate
construct measured by classify_response_voice.py, and a gate-failure rate must
be reported alongside, not instead of, the voice-collapse rate.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from gate_common import MAX_ATTEMPTS, attempt_seed
from judge_assistant_mode import judge_one

import torch
from openai import OpenAI


def make_judge(env):
    """Return a callable (context, response) -> 0/1 for this env.

    The client is built once and reused across the whole run: it is thread-safe
    and carries max_retries=12, so transient API failures are absorbed by the
    client rather than by per-call error handling. A call that still fails, or
    a judge output with no parsable label, raises out of judge_one -- the run
    must not continue silently mislabelling draws.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    assert key, (
        "OPENROUTER_API_KEY must be set for --gate-loop: the gate judge is "
        "called inline from the generation node.")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key,
                    max_retries=12)

    def judge(context, response):
        _, label, _, _ = judge_one(client, env, context, response)
        return label

    return judge


def resample_until_pass(gen_fn, judge, prompt_key, context, first_response,
                        max_attempts=MAX_ATTEMPTS, log=None):
    """Resample until the gate passes a draw, or max_attempts is exhausted.

    Inputs:
      gen_fn        callable() -> response text. Must draw from the SAME arm as
                    attempt 1; the caller owns all prompt construction (steer
                    prefix/suffix, perspective-shift rewrite) so this function
                    cannot accidentally change the condition being sampled.
      judge         callable(context, response) -> 0/1, from make_judge.
      prompt_key    text the per-attempt seed is derived from (the bare post).
      context       what the judge sees as the user's message (the bare post).
      first_response the attempt-1 draw, already known to have been REJECTED.

    Returns (response, winning_attempt, n_tried):
      response        the first passing draw, or the LAST rejected draw if none
                      passed (kept so the failure is inspectable, never scored).
      winning_attempt attempt index of the passing draw, or None if exhausted.
      n_tried         highest attempt index generated, for cost accounting.

    Assumes attempt 1 was already judged and rejected -- this is only called for
    prompts on the worklist, so it starts at attempt 2 and never re-judges the
    existing response.
    """
    assert max_attempts >= 2, "max_attempts must leave room for at least one resample"
    last = first_response
    for attempt in range(2, max_attempts + 1):
        torch.manual_seed(attempt_seed(prompt_key, attempt))
        resp = gen_fn()
        last = resp
        if judge(context, resp) == 1:
            if log:
                log(f"    passed at attempt {attempt}")
            return resp, attempt, attempt
    if log:
        log(f"    EXHAUSTED after {max_attempts} attempts — counted as gate failure")
    return last, None, max_attempts
