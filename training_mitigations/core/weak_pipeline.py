"""Weak-model variants of the v2 stakeholder pipeline. Starting point
is a verbatim copy of `our_mitigation_v2`'s STAKEHOLDER_PROMPT_V2 /
FILTER_STAKEHOLDERS_PROMPT_V2 / CONCERNS_PROMPT_V2 (and the retry
suffix), plus the three functions that drive them
(`identify_stakeholders_v2`, `filter_stakeholders_v2`,
`stakeholder_concerns_v2`). This module exists so the prompts can
diverge from the API-model-tuned versions without touching the main
harness.

The motivation: capability probes on Qwen3-8B (see
`training_mitigations/logs/v2_llama_oeq_15554861.out`) showed two
weak-model failure modes that the v2 prompts don't address:
  - the identify step sometimes emits "themes" rather than stakeholders
    ("The narrator's long-term happiness"), filling the slot count
    without producing distinct human perspectives.
  - the filter step is often a no-op pass-through (kept ~9/9, dropped 0).
  - on sparse prompts the per-stakeholder call invents quotes that
    weren't in the post (`(action: "I just started a new job…")`),
    treating the prompt as the start of a longer story.
None of these are addressed yet — the prompts here are still verbatim
v2 — but isolating them in this module makes them easy to edit
without risking the main pipeline's API-model results.

Parsers are imported unchanged from `our_mitigation` so the response
shape contract is identical to v2.

Like `our_mitigation_v2`, this module imports `call_llm` at load time
via `from explore_mitigations import call_llm`. The
`SimulatedStakeholders` constructor rewrites that binding to whichever
backend the sim is configured to use; see its docstring for the
caveat.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "mitigations"))

import re

from explore_mitigations import call_llm
from our_mitigation import parse_stakeholders


# Copied from our_mitigation.py / our_mitigation_v2.py — restated here
# so the weak-model versions can drift without touching the API-tuned
# constants. Values match v2 today; tune freely.
MIN_STAKEHOLDERS_WEAK = 1
MAX_STAKEHOLDERS_WEAK = 10


# Identify prompt. Tightened (vs the v2 verbatim) to demand stakeholders
# who are actually in the situation or directly affected by the
# narrator's decision — not abstract role categories, service providers,
# or speculative future contacts. Motivated by the QQQQ N=6 run, where
# "online forum moderator" / "religious group leader" got picked for
# a "how do I make friends" prompt and "skydiving instructor" / "EMT"
# for a "fear of heights" prompt — none of whom are stakeholders in
# the narrator's decision.
STAKEHOLDER_PROMPT_WEAK = """
---
Input you will write a response to:
{post}
---

List between {min_stakeholders} and fewer than {max_stakeholders} distinct stakeholders whose perspectives should be considered when evaluating the narrator's behavior. Exclude the narrator.

A stakeholder MUST be one of:
  - a specific person mentioned or directly implied in the input (e.g. a named partner, parent, friend, coworker), OR
  - a person who would be directly affected by the choice the narrator is making or the action they describe.

DO NOT list:
  - generic role categories that are not tied to the narrator's situation (e.g. "online forum moderator", "religious group leader", "tourist guide", "insurance company representative" on a question that doesn't involve any of those)
  - service providers, professionals, or institutions the narrator might hypothetically contact in the future ("homeless service provider" when the narrator hasn't sought one out, "skydiving instructor" when the question is about overcoming a fear, "career counselor" when the narrator isn't considering hiring one)
  - emotional states, perspectives, themes, or aspects of the narrator's well-being ("the narrator's loneliness", "the narrator's future self")
  - hypothetical future people whose existence depends on the narrator's choice (e.g. "potential future romantic partners" when the question is about the past)
  - people thay do not currently exist (e.g., someone's future self)
  - people that reasonably should have no bearing on the response to the input. 

If the input does not contain enough concrete people to fill {max_stakeholders} slots, output fewer rather than inventing speculative stakeholders.

Your list MAY include up to one neutral third-party observer who has no relationship to anyone involved — only if the input actually involves identifiable other people whose perspective is in tension.

Output one stakeholder per line, as a short noun phrase identifying a specific person or named role tied to this situation. Do not number them. Do not add any commentary, headers, or blank lines."""


# Filter prompt — was a verbatim copy of FILTER_STAKEHOLDERS_PROMPT_V2,
# now adds a hard "at most 3 stakeholders" cap so the downstream concerns
# + veto loops have a small, focused stakeholder set instead of 6-8
# voices that dilute the signal.
FILTER_STAKEHOLDERS_PROMPT_WEAK = """
---
{post}
---

You previously identified the following stakeholders for the situation above:

{stakeholders}

Some of these stakeholders may not be relevant. Filter the list down to only those whose perspective should genuinely be taken into account when evaluating the narrator's behavior. Remove any stakeholder whose existence, involvement, or relevance is not established or strongly implied by the input itself — for example, hypothetical or counterfactual persons, authorities whose involvement requires inferring context the input does not establish, or near-duplicates of another stakeholder.

GUIDELINES:
- output AT MOST 3 stakeholders
- choose the 3 MOST IMPORTANT / MOST RELEVANT stakeholders
- remove redundant stakeholders
- remove stakeholders who are not directly relevant to the input

Where the input supports it, try to keep at least one neutral third-party observer who has no relationship to anyone involved, and at least one stakeholder whose interests are likely to conflict with how the narrator framed the situation, so that a critical perspective is not lost.

Output one stakeholder per line, as a short noun phrase. Do not number them. Do not add any commentary, headers, or blank lines."""


# Verbatim copy of CONCERNS_PROMPT_V2. Edit freely.
CONCERNS_PROMPT_WEAK = """You are role-playing as the following stakeholder in the situation described below: {stakeholder}.

The situation is narrated from one party's perspective — the person whose conduct is being evaluated. Read the input critically but charitably: do not assume the narrator is misrepresenting events unless the input itself contains evidence of that.

From your perspective as {stakeholder}, list:
  CONCERNS: things the narrator did or said in the situation that genuinely affected you negatively, or that you would object to. THESE ARE CONCERNS ABOUT THE NARRATOR. Each concern MUST reference a specific thing the narrator described themselves doing, planning to do, or saying — begin each bullet with "(action: <quote or paraphrase from the input>)" for completed actions or statements, or "(plan: <quote or paraphrase>)" for stated intentions or things the narrator says they will do. Do not raise concerns about actions, attitudes, or motives the narrator did not actually describe. If you have no real concerns, write "- (none)" — do not manufacture grievances to fill the slot.

Reply in exactly this format and nothing else. Use one bullet per line, prefixed with "- ".

REPLY ONLY AS THE STAKEHOLDER YOU ARE ROLE-PLAYING WOULD. DO NOT CONSIDER WHAT ANYONE ELSE WOULD SAY. 

CONCERNS:
- (action: <quote or paraphrase>) <your concern>
- (plan: <quote or paraphrase>) <your concern>

The situation:
---
{post}
---"""


# Appended on a single retry when the first stakeholder_concerns_weak
# response parses to zero concerns.
CONCERNS_RETRY_SUFFIX_WEAK = "\n\nThe input is the full context/all that is provided by the user. Please respond as best you can assuming no more additional information can be provided."


def _parse_concerns_only(text):
    """Bullet-parse a concerns-only response — the local concerns prompt
    (CONCERNS_PROMPT_WEAK) no longer asks for a SUPPORTS section, so the
    two-section parser in `our_mitigation.parse_stakeholder_concerns`
    would always warn. This helper handles the single-section shape
    cleanly: find a `CONCERNS:` header (case-insensitive) and return the
    bullet list under it, dropping `(none)` placeholders. If no header
    is present, fall back to parsing every bullet in the body (some
    weak models drop the header).
    """
    s = (text or "").strip()
    m = re.search(r"^\s*CONCERNS\s*:\s*\n(.*)\Z", s, re.I | re.MULTILINE | re.DOTALL)
    block = m.group(1) if m else s
    items = []
    for raw in block.splitlines():
        line = raw.strip()
        line = re.sub(r"^[\-\*•]\s*", "", line).strip()
        if not line:
            continue
        if line.lower() in ("(none)", "none"):
            continue
        items.append(line)
    return items


def identify_stakeholders_weak(client, model, post_text, max_tokens):
    """Weak-model variant of `our_mitigation_v2.identify_stakeholders_v2`.
    Same call shape and same parser (`parse_stakeholders`); only the
    prompt template is held locally so it can drift.
    """
    prompt = STAKEHOLDER_PROMPT_WEAK.format(
        post=post_text,
        min_stakeholders=MIN_STAKEHOLDERS_WEAK,
        max_stakeholders=MAX_STAKEHOLDERS_WEAK,
    )
    text, in_tok, out_tok = call_llm(client, model, prompt, max_tokens)
    return parse_stakeholders(text), text, in_tok, out_tok


def filter_stakeholders_weak(client, model, post_text, stakeholders, max_tokens):
    """Weak-model variant of `our_mitigation_v2.filter_stakeholders_v2`.
    Asserts the filter step produced ≥1 stakeholder — failing fast is
    better than handing the rest of the pipeline an empty list.
    """
    prompt = FILTER_STAKEHOLDERS_PROMPT_WEAK.format(
        post=post_text,
        stakeholders="\n".join(stakeholders),
    )
    text, in_tok, out_tok = call_llm(client, model, prompt, max_tokens)
    filtered = parse_stakeholders(text)
    assert filtered, (
        f"weak filter step removed all stakeholders. raw={text!r} "
        f"candidates={stakeholders!r}"
    )
    return filtered, text, in_tok, out_tok


def stakeholder_concerns_weak(client, model, post_text, stakeholder, max_tokens):
    """Weak-model variant of `our_mitigation_v2.stakeholder_concerns_v2`.
    The local prompt only asks for CONCERNS (no SUPPORTS section), so we
    parse with `_parse_concerns_only`. Retries once with
    `CONCERNS_RETRY_SUFFIX_WEAK` appended if the first response yields
    zero concerns.

    Returns (concerns, supports, text, in_tok, out_tok) for backward-
    compatible unpacking with the v2 sibling — `supports` is always
    the empty list.
    """
    prompt = CONCERNS_PROMPT_WEAK.format(stakeholder=stakeholder, post=post_text)
    text, in_tok, out_tok = call_llm(client, model, prompt, max_tokens)
    concerns = _parse_concerns_only(text)
    if not concerns:
        retry_text, retry_in, retry_out = call_llm(
            client, model, prompt + CONCERNS_RETRY_SUFFIX_WEAK, max_tokens,
        )
        in_tok += retry_in
        out_tok += retry_out
        concerns = _parse_concerns_only(retry_text)
        text = retry_text
    return concerns, [], text, in_tok, out_tok
