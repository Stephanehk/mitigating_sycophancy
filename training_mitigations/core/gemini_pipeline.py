"""Gemini-tuned variants of the stakeholder pipeline prompts.

Starting point is a near-verbatim copy of `weak_pipeline`'s
STAKEHOLDER_PROMPT_WEAK / FILTER_STAKEHOLDERS_PROMPT_WEAK /
CONCERNS_PROMPT_WEAK. This module exists so the Gemini prompts can diverge
from the Qwen3-8B-tuned ones WITHOUT touching the Qwen path — every existing
arm keeps `weak_pipeline` byte-for-byte unchanged.

WHY THIS MODULE EXISTS
----------------------
A 16-prompt comparison (Gemini 3.1 Flash-Lite vs the persisted real Qwen3-8B
stakeholder outputs in `runs/stakeholder_examples/`, same prompts, same
pipeline) found:

  - mean identify-list size:            Qwen 9.5   vs  Gemini 2.4
  - filter violated the subset rule:    Qwen 6/16  vs  Gemini 16/16
  - kept a generic/"self" voice:        Qwen 6/16  vs  Gemini 13/16
  - shared kept-stakeholders (of 3):    0.06 mean -- near-disjoint panels

The causal story is NOT "Gemini disobeys". STAKEHOLDER_PROMPT_WEAK renders as
"List between 1 and fewer than 10 ..." and explicitly says "output fewer rather
than inventing speculative stakeholders", so Gemini's terse list OBEYS it; it is
Qwen at 9.5 that fills slots with the speculative service-providers the same
prompt forbids. Gemini is then punished by the FILTER step, which demands "the 3
MOST IMPORTANT" and invites "at least one neutral third-party observer" -- with
only ~2.4 candidates in hand, fabrication is the only way to comply. Hence the
100% subset-violation rate is arithmetic, not misbehavior.

So the fix lives mostly in the filter:
  1. identify: ONE added paragraph asking for completeness *within the existing
     rules* (no anti-speculation rule is loosened), so the filter has genuine
     candidates to select from.
  2. filter: an explicit verbatim-subset constraint, permission to return fewer
     than 3, and the observer clause rewritten from "try to keep at least one"
     (which reads as *add one* when none exists) to a strictly conditional keep.
  3. concerns: deliberately UNCHANGED -- Gemini already honored the
     (action:)/(plan:) contract and spoke in the stakeholder's voice. The one bad
     output (the narrator lecturing themselves) came from the filter admitting
     "The narrator", which (2) fixes upstream.

KNOWN CONSEQUENCE (not a bug): stopping the fabrication SHRINKS the panels. On
terse PAS posts there often are not 3 genuine stakeholders, so Gemini will field
1-2 voices where Qwen fields 3. Fewer voices => fewer veto opportunities =>
fewer preference pairs. Watch `contributing_prompts` in iter_stats.

Parsers and the concerns retry suffix are imported unchanged from
`weak_pipeline` / `our_mitigation` so the response-shape contract is identical.

Like `weak_pipeline`, this module imports `call_llm` at load time; the
`SimulatedStakeholders` constructor rewrites that binding to whichever backend
the sim is configured to use (see its docstring for the caveat).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "mitigations"))

from explore_mitigations import call_llm
from our_mitigation import parse_stakeholders
from weak_pipeline import CONCERNS_RETRY_SUFFIX_WEAK, _parse_concerns_only


# Same bounds as the weak pipeline; restated so they can drift independently.
MIN_STAKEHOLDERS_GEMINI = 1
MAX_STAKEHOLDERS_GEMINI = 10


# Identify prompt. Near-verbatim copy of STAKEHOLDER_PROMPT_WEAK; the ONLY
# change is the "be COMPLETE" paragraph, which pushes for coverage without
# loosening any of the DO-NOT rules above it. Motivated by Gemini's 2.4-item
# lists starving the filter (see module docstring).
STAKEHOLDER_PROMPT_GEMINI = """
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

Within the rules above, be COMPLETE: list every distinct person who qualifies, including those referred to only by role or in passing — not only the one or two most obvious. Listing each qualifying person separately is preferred over collapsing them into a single group.

Your list MAY include up to one neutral third-party observer who has no relationship to anyone involved — only if the input actually involves identifiable other people whose perspective is in tension.

Output one stakeholder per line, as a short noun phrase identifying a specific person or named role tied to this situation. Do not number them. Do not add any commentary, headers, or blank lines.
"""


# Filter prompt. Near-verbatim copy of FILTER_STAKEHOLDERS_PROMPT_WEAK with the
# three changes that this module exists for: (a) an explicit verbatim-subset
# constraint, (b) permission to output fewer than 3 rather than padding, and
# (c) the neutral-observer clause made strictly conditional on the observer
# ALREADY being present. The "at least one conflicting stakeholder" half is
# preserved so the critical perspective is not lost.
FILTER_STAKEHOLDERS_PROMPT_GEMINI = """
---
{post}
---

You previously identified the following stakeholders for the situation above:

{stakeholders}

Some of these stakeholders may not be relevant. Filter the list down to only those whose perspective should genuinely be taken into account when evaluating the narrator's behavior. Remove any stakeholder whose existence, involvement, or relevance is not established or strongly implied by the input itself — for example, hypothetical or counterfactual persons, authorities whose involvement requires inferring context the input does not establish, or near-duplicates of another stakeholder.

This step is a SELECTION, not a rewrite. Every stakeholder you output MUST appear VERBATIM in the list above. Do not add, invent, rename, reword, generalize, or substitute a stakeholder, and never output the narrator themselves. If you think an important perspective is missing from the list, leave it out anyway.

GUIDELINES:
- output AT MOST 3 stakeholders
- choose the 3 MOST IMPORTANT / MOST RELEVANT stakeholders
- output FEWER than 3 when fewer than 3 are genuinely relevant; do not pad the list to reach 3
- remove redundant stakeholders
- remove stakeholders who are not directly relevant to the input

If — and only if — the list above ALREADY contains a neutral third-party observer who has no relationship to anyone involved, you may keep it. Do not add one if it is not already in the list. Where the list supports it, keep at least one stakeholder whose interests are likely to conflict with how the narrator framed the situation, so that a critical perspective is not lost.

Output one stakeholder per line, as a short noun phrase. Do not number them. Do not add any commentary, headers, or blank lines.
"""


# VERBATIM copy of CONCERNS_PROMPT_WEAK — deliberately identical (Gemini already
# honored this contract; see module docstring). Held as its OWN constant rather
# than aliased to weak_pipeline's so that editing the Gemini concerns prompt
# cannot silently change the Qwen path.
CONCERNS_PROMPT_GEMINI = """You are role-playing as the following stakeholder in the situation described below: {stakeholder}.

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


def identify_stakeholders_gemini(client, model, post_text, max_tokens):
    """Gemini variant of `weak_pipeline.identify_stakeholders_weak`. Same call
    shape and same parser (`parse_stakeholders`); only the prompt differs.

    Assumptions: `parse_stakeholders` asserts len < MAX_STAKEHOLDERS_GEMINI, so
    a runaway list still fails loud (the failure mode Qwen hits at exactly 10).
    Returns (stakeholders, raw_text, in_tok, out_tok).
    """
    prompt = STAKEHOLDER_PROMPT_GEMINI.format(
        post=post_text,
        min_stakeholders=MIN_STAKEHOLDERS_GEMINI,
        max_stakeholders=MAX_STAKEHOLDERS_GEMINI,
    )
    text, in_tok, out_tok = call_llm(client, model, prompt, max_tokens)
    return parse_stakeholders(text), text, in_tok, out_tok


def filter_stakeholders_gemini(client, model, post_text, stakeholders, max_tokens):
    """Gemini variant of `weak_pipeline.filter_stakeholders_weak`.

    Asserts the filter produced >= 1 stakeholder (same contract as the weak
    sibling — an empty list would hand the rest of the pipeline nothing).
    The verbatim-subset rule is stated in the PROMPT but deliberately NOT
    asserted here: the weak/Qwen path violates it 6/16 of the time, so
    enforcing it in code would change the Qwen contract too and would turn a
    measurable prompt-adherence property into a crash. Subset violations are
    measured by the comparison harness instead.

    Returns (filtered, raw_text, in_tok, out_tok).
    """
    prompt = FILTER_STAKEHOLDERS_PROMPT_GEMINI.format(
        post=post_text,
        stakeholders="\n".join(stakeholders),
    )
    text, in_tok, out_tok = call_llm(client, model, prompt, max_tokens)
    filtered = parse_stakeholders(text)
    assert filtered, (
        f"gemini filter step removed all stakeholders. raw={text!r} "
        f"candidates={stakeholders!r}"
    )
    return filtered, text, in_tok, out_tok


def stakeholder_concerns_gemini(client, model, post_text, stakeholder, max_tokens):
    """Gemini variant of `weak_pipeline.stakeholder_concerns_weak`. The prompt
    only asks for CONCERNS (no SUPPORTS section), so it parses with the shared
    `_parse_concerns_only`. Retries once with `CONCERNS_RETRY_SUFFIX_WEAK`
    appended if the first response yields zero concerns.

    Returns (concerns, supports, text, in_tok, out_tok) for backward-compatible
    unpacking with its siblings — `supports` is always the empty list.
    """
    prompt = CONCERNS_PROMPT_GEMINI.format(stakeholder=stakeholder, post=post_text)
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
