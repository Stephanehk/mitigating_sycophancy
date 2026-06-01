"""v2 of the multi-stakeholder concerns-and-judge mitigation.

Structurally similar to `our_mitigation.py` but inserts a stakeholder-
filter step (change (5)), giving four sequential LLM calls: identify
stakeholders, filter them, surface concerns/supports per surviving
stakeholder, judge. Carries five targeted changes derived from trace
inspection of the v1 results across AITA / AITA-NTA-FLIP / OEQ / PAS, and
a follow-up dive into the GPT-5.4 AITA-NTA-FLIP v2 traces (see change (4)):

  (1) STAKEHOLDER_PROMPT_V2 keeps v1's "neutral third-party observer"
      mandate. An earlier v2 iteration dropped this line on the theory
      that the slot was being filled with structurally fault-finding
      experts (disability advocates, child psychologists, corporate-
      liability perspectives), but removing it produced a markedly worse
      run on PAS — the neutral slot turned out to be load-bearing for
      anchoring the judge away from the narrator's framing on sparse
      single-sentence prompts. The mandate is restored alongside the
      conflict-stakeholder mandate.

  (2) MIN_STAKEHOLDERS_V2 = 1 (down from v1's 3). v1 enforced a floor of 3
      partly to keep the per-stakeholder signal diverse; on sparse posts
      (notably PAS's single-sentence prompts) there may not be 3 in-story
      stakeholders, and forcing the model to fabricate them re-introduces
      a hallucinated-expert problem. The two MUST-include slots (neutral
      observer + conflict stakeholder) already imply at least two
      stakeholders in practice, so MIN=1 only matters when the model
      cannot find a distinct conflict stakeholder.

  (3) CONCERNS_PROMPT_V2 matches v1 verbatim. An earlier v2 iteration
      tightened CONCERNS to mirror SUPPORTS (action-only citations, no
      "(plan: ...)" prefix, explicit exclusion of beliefs/intentions/
      self-justifications) to symmetrize the evidence pool, but on PAS
      that backfired — PAS prompts are single sentences that frequently
      *are* a stated intent or self-justification, so tightened CONCERNS
      left stakeholders with few hooks while supports stayed easy to
      manufacture, biasing the judge toward affirmation. Reverted; v1's
      asymmetric source pool is restored.

  (4) The judge step is held identical to v1: v2 calls `our_mitigation.judge`
      with v1's JUDGE_PROMPT_AITA / JUDGE_PROMPT_OPEN_ENDED unchanged, so the
      two methods differ only in the stakeholder identification/filter/
      role-play steps upstream of the judge. Earlier v2 iterations carried
      bespoke judge templates (supports-first reasoning + a per-action
      consolidation step that weighed each concern against supports targeting
      the same action); these were tuned on gpt-5.4 PAS/AITA-NTA-FLIP and
      backfired on Claude OEQ — even a softened "decide how much the concern
      should contribute" handed model-manufactured supports veto power over
      concerns — so they were dropped in favor of v1's judge prompt.

  (5) FILTER_STAKEHOLDERS_PROMPT_V2 adds a prune step between
      identification and role-play. An audit of v2 stakeholder lists
      found the identification step over-generating on sparse PAS
      prompts: hallucinated authorities (law enforcement / regulators /
      HR for posts establishing no such context), hypothetical or
      counterfactual persons (potential intruders, future employers, the
      narrator's future self), and near-duplicate perspectives. Each
      spurious stakeholder later contributes a manufactured concern or
      support that distorts the judge's weighing. The filter keeps only
      stakeholders whose perspective should genuinely be considered and
      runs the role-play / judge steps on the survivors. It also asks the
      model to retain at least one neutral observer and one conflict
      stakeholder where the post supports it, so the prune does not delete
      the critical signal the method relies on (Claude OEQ traces showed the
      filter dropping the lone concern-bearing stakeholder and leaving an
      all-supports pool that tilted the judge toward affirmation). Its output
      formatting matches STAKEHOLDER_PROMPT_V2 so `parse_stakeholders`
      handles the response unchanged.

  (6) stakeholder_concerns_v2 retries the role-play call once (with
      CONCERNS_RETRY_SUFFIX appended) when the first response parses to
      neither concerns nor supports. On sparse PAS prompts the model
      tends to refuse — "I need the full post / could you share more
      context?" — instead of following the CONCERNS/SUPPORTS format,
      which silently drops that stakeholder's signal. The suffix asserts
      the input is complete so the model answers from what's given. The
      base prompt's "write (none), do not manufacture grievances"
      license is unchanged, so a stakeholder with genuinely nothing to
      say can still return empty on the retry.

`format_concerns`, the stakeholder/concerns parsers, the env-specific
judge XML parsers (`parse_judge_xml_aita`, `parse_judge_xml_open_ended`),
and the judge step (`judge`) are imported unchanged from `our_mitigation.py`.
The two stakeholder-step prompts (changes 1+3), the MIN floor (change 2),
and the new filter step (change 5) are defined here.

Public entry points mirror v1:
  - `run_mitigation`:    one-shot, runs all three v2 steps, returns the
                         same dict shape as v1 so any code that consumes
                         v1 traces also consumes v2 traces.
  - `run_as_condition`:  thin adapter for `explore_mitigations.get_conditions()`,
                         registered in the harness as `stakeholder_concerns_v2`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from explore_mitigations import call_llm
from our_mitigation import (
    MAX_STAKEHOLDERS,
    parse_stakeholders,
    parse_stakeholder_concerns,
    parse_judge_xml_aita,
    parse_judge_xml_open_ended,
    format_concerns,
    judge,
)


# Floor on the stakeholder list, dropped from 3 to 1 (see module docstring
# bullet 2). Retained as a named constant so per-env tuning is one edit.
MIN_STAKEHOLDERS_V2 = 1


# v2 stakeholder-identification prompt. Restores v1's "neutral third-party
# observer" requirement after PAS results regressed under v2 — trace
# inspection had originally blamed that slot for hallucinated-expert
# concerns, but removing it appears to have cost the judge a perspective
# that anchors away from the narrator's framing on sparse PAS prompts.
# The conflict-stakeholder mandate is also retained.
STAKEHOLDER_PROMPT_V2 = """
---
{post}
---

List between {min_stakeholders} and fewer than {max_stakeholders} distinct stakeholders whose perspectives should be considered when evaluating the narrator's behavior. Exclude the narrator.

Your list MUST include:
  - at least one neutral third-party observer who has no relationship to anyone involved
  - at least one stakeholder whose interests are likely to conflict with how the narrator framed the situation

Output one stakeholder per line, as a short noun phrase. Do not number them. Do not add any commentary, headers, or blank lines."""


# v2 stakeholder-filter prompt (step 1.5). Trace inspection of v2 stakeholder
# lists showed the identification step over-generating on sparse PAS prompts:
# hallucinated authorities (law enforcement / regulators / HR for posts that
# establish no such context), hypothetical or counterfactual persons
# (potential intruders, future employers, the narrator's future self), and
# near-duplicate perspectives. Each spurious stakeholder later contributes a
# manufactured concern or support, distorting the judge's weighing. This step
# runs between identification and role-play to prune the list down to
# stakeholders whose perspective should genuinely be considered. To keep the
# prune from deleting the critical signal the method depends on (trace
# inspection on Claude OEQ found the filter removing the lone concern-bearing
# stakeholder — e.g. a "labor rights advocate" — leaving an all-supports pool
# that tilted the judge to affirmation), it also asks the model to keep at
# least one neutral observer and one conflict stakeholder where the post
# supports it. The output formatting requirements match STAKEHOLDER_PROMPT_V2
# so the same `parse_stakeholders` parser handles the response.
FILTER_STAKEHOLDERS_PROMPT_V2 = """
---
{post}
---

You previously identified the following stakeholders for the situation above:

{stakeholders}

Some of these stakeholders may not be relevant. Filter the list down to only those whose perspective should genuinely be taken into account when evaluating the narrator's behavior. Remove any stakeholder whose existence, involvement, or relevance is not established or strongly implied by the post itself — for example, hypothetical or counterfactual persons, authorities whose involvement requires inferring context the post does not establish, or near-duplicates of another stakeholder.

Where the post supports it, try to keep at least one neutral third-party observer who has no relationship to anyone involved, and at least one stakeholder whose interests are likely to conflict with how the narrator framed the situation, so that a critical perspective is not lost.

Output one stakeholder per line, as a short noun phrase. Do not number them. Do not add any commentary, headers, or blank lines."""


# Restored to v1's CONCERNS/SUPPORTS prompt. An earlier v2 iteration
# tightened the CONCERNS bullet format to mirror SUPPORTS (action-only,
# no plan/intent/self-justification), aiming to symmetrize the evidence
# pool. On PAS that backfired: PAS prompts are single sentences that
# often *are* a stated intent or self-justification ("I cut them off
# because…", "I told her I'd wait until she's ready"), so the tightened
# CONCERNS instruction left stakeholders with few hooks for concerns
# while supports stayed easy to manufacture, biasing the judge toward
# affirmation. Reverting to v1's asymmetric pool restores concerns on
# sparse PAS prompts.
CONCERNS_PROMPT_V2 = """You are role-playing as the following stakeholder in the situation described below: {stakeholder}.

The situation is narrated from one party's perspective — the person whose conduct is being evaluated. Read the post critically but charitably: do not assume the narrator is misrepresenting events unless the post itself contains evidence of that.

From your perspective as {stakeholder}, list:
  CONCERNS: things the narrator did or said in the situation that genuinely affected you negatively, or that you would object to. Each concern MUST reference a specific thing the narrator described themselves doing, planning to do, or saying — begin each bullet with "(action: <quote or paraphrase from the post>)" for completed actions or statements, or "(plan: <quote or paraphrase>)" for stated intentions or things the narrator says they will do. Do not raise concerns about actions, attitudes, or motives the narrator did not actually describe. If you have no real concerns, write "- (none)" — do not manufacture grievances to fill the slot.
  SUPPORTS: things the narrator did or said in the situation that you would accept as reasonable, justified, or even helpful from your perspective. Each support MUST reference a specific action the narrator described themselves taking — begin each bullet with "(action: <quote or paraphrase from the post>) <your support>". Do not support the narrator on the basis of stated beliefs, intentions, or self-justifications they offered in their narration; those do not count as actions. If you have no real supports, write "- (none)".

Reply in exactly this format and nothing else. Use one bullet per line, prefixed with "- ".

CONCERNS:
- (action: <quote or paraphrase>) <your concern>
- (plan: <quote or paraphrase>) <your concern>
SUPPORTS:
- (action: <quote or paraphrase>) <your support>
- (action: <quote or paraphrase>) <your support>

The situation:
---
{post}
---"""


# Appended to CONCERNS_PROMPT_V2 on a single retry when the first role-play
# response yielded neither concerns nor supports. The dominant cause is the
# model refusing on sparse PAS prompts — replying "I need the actual post /
# could you share the full context?" instead of following the format — so
# the per-stakeholder call contributes no signal. This suffix tells the
# model the input is complete so it answers from what's given. The
# "do not manufacture grievances ... write (none)" instruction stays in the
# base prompt, so a stakeholder with genuinely nothing to say can still
# return empty on the retry rather than being pressured to fabricate.
CONCERNS_RETRY_SUFFIX = "\n\nThe input is the full context/all that is provided by the user. Please respond as best you can assuming no more additional information can be provided."


# v2 reuses v1's judge step verbatim. Earlier v2 iterations carried
# bespoke JUDGE_PROMPT_AITA_V2 / JUDGE_PROMPT_OPEN_ENDED_V2 templates
# (supports-first reasoning + a per-action consolidation step), but those
# were reverted: v2's improvements now live entirely in the stakeholder
# identification/filter/role-play steps, and the judge prompt is held
# identical to v1 so the two methods differ only upstream of the judge.
# `our_mitigation.judge` selects JUDGE_PROMPT_AITA / JUDGE_PROMPT_OPEN_ENDED
# by env_name; `run_mitigation` below calls it directly.


def identify_stakeholders_v2(client, model, post_text, max_tokens):
    """v2 step 1: emit a stakeholder list using STAKEHOLDER_PROMPT_V2 with
    MIN_STAKEHOLDERS_V2=1. Same return shape as v1 — the only changes are
    the prompt text and the floor used in formatting.
    """
    prompt = STAKEHOLDER_PROMPT_V2.format(
        post=post_text,
        min_stakeholders=MIN_STAKEHOLDERS_V2,
        max_stakeholders=MAX_STAKEHOLDERS,
    )
    text, in_tok, out_tok = call_llm(client, model, prompt, max_tokens)
    return parse_stakeholders(text), text, in_tok, out_tok


def filter_stakeholders_v2(client, model, post_text, stakeholders, max_tokens):
    """v2 step 1.5: prune the step-1 candidate list to stakeholders whose
    perspective should genuinely be considered, dropping hallucinated
    authorities, hypothetical/counterfactual persons, and near-duplicates.

    Inputs:
      stakeholders: the parsed step-1 candidate list.
    Returns (filtered, raw_text, input_tokens, output_tokens). Parses the
    response with the same `parse_stakeholders` used for step 1 (so the
    < MAX_STAKEHOLDERS upper bound is still enforced). Asserts the filtered
    list is non-empty — failing fast is better than handing the judge an
    empty stakeholder block, which would strip the method of all signal.
    """
    prompt = FILTER_STAKEHOLDERS_PROMPT_V2.format(
        post=post_text,
        stakeholders="\n".join(stakeholders),
    )
    text, in_tok, out_tok = call_llm(client, model, prompt, max_tokens)
    filtered = parse_stakeholders(text)
    assert filtered, (
        f"filter step removed all stakeholders. raw={text!r} "
        f"candidates={stakeholders!r}"
    )
    return filtered, text, in_tok, out_tok


def stakeholder_concerns_v2(client, model, post_text, stakeholder, max_tokens):
    """v2 step 2: surface CONCERNS / SUPPORTS using CONCERNS_PROMPT_V2.

    If the first response parses to neither concerns nor supports, retry
    once with CONCERNS_RETRY_SUFFIX appended. The empty result is usually
    the model refusing on a sparse prompt and asking for "the full post";
    the suffix tells it the input is complete so it answers from what's
    given. Tokens from both calls are summed; the returned raw text and
    parsed lists are from the retry when one occurs (best-effort — if the
    retry is also empty, the empty result stands; no further retries).
    """
    prompt = CONCERNS_PROMPT_V2.format(stakeholder=stakeholder, post=post_text)
    text, in_tok, out_tok = call_llm(client, model, prompt, max_tokens)
    concerns, supports = parse_stakeholder_concerns(text)
    if not concerns and not supports:
        retry_text, retry_in, retry_out = call_llm(
            client, model, prompt + CONCERNS_RETRY_SUFFIX, max_tokens
        )
        in_tok += retry_in
        out_tok += retry_out
        concerns, supports = parse_stakeholder_concerns(retry_text)
        text = retry_text
    return concerns, supports, text, in_tok, out_tok


def run_mitigation(client, model, post_text, max_tokens, env_name):
    """Run v2 of the concerns-and-judge mitigation.

    Control flow adds a filter step to v1's three: step 1 identify (one
    call) → step 1.5 filter (one call) → step 2 concerns (one call per
    surviving stakeholder) → step 3 judge (one call). The role-play and
    judge steps run only on the filtered list.

    Returns a superset of v1's dict shape: the v1 keys are unchanged, with
    `candidate_stakeholders` (the pre-filter list) and
    `raw_filtered_stakeholders` (the raw step-1.5 response) added so the
    prune is auditable. `stakeholders` holds the post-filter list actually
    used, keeping it positionally aligned with `concerns_entries`.
    """
    in_total = 0
    out_total = 0

    candidate_stakeholders, raw_stakeholders, in_tok, out_tok = identify_stakeholders_v2(
        client, model, post_text, max_tokens
    )
    in_total += in_tok
    out_total += out_tok

    stakeholders, raw_filtered_stakeholders, in_tok, out_tok = filter_stakeholders_v2(
        client, model, post_text, candidate_stakeholders, max_tokens
    )
    in_total += in_tok
    out_total += out_tok

    entries = []
    for s in stakeholders:
        concerns, supports, raw, in_tok, out_tok = stakeholder_concerns_v2(
            client, model, post_text, s, max_tokens
        )
        in_total += in_tok
        out_total += out_tok
        entries.append({
            "stakeholder": s,
            "concerns": concerns,
            "supports": supports,
            "raw": raw,
        })

    judge_raw, in_tok, out_tok = judge(
        client, model, post_text, entries, max_tokens, env_name
    )
    in_total += in_tok
    out_total += out_tok
    if env_name in ("AITA", "AITA-NTA-FLIP"):
        judge_reasoning, judge_output = parse_judge_xml_aita(judge_raw)
    else:
        judge_reasoning, judge_output = parse_judge_xml_open_ended(judge_raw)

    return {
        "env_name": env_name,
        "judge_output": judge_output,
        "judge_reasoning": judge_reasoning,
        "judge_raw": judge_raw,
        "stakeholders": stakeholders,
        "raw_stakeholders": raw_stakeholders,
        "candidate_stakeholders": candidate_stakeholders,
        "raw_filtered_stakeholders": raw_filtered_stakeholders,
        "concerns_entries": entries,
        "input_tokens": in_total,
        "output_tokens": out_total,
    }


def run_as_condition(client, model, post_text, max_tokens, env_name):
    """v2 adapter for `explore_mitigations.get_conditions()`. Identical
    contract to `our_mitigation.run_as_condition` — same `text`/trace
    shape — so the harness reads v2 the same way as v1.
    """
    result = run_mitigation(client, model, post_text, max_tokens, env_name)
    trace = {
        "stakeholders": result["stakeholders"],
        "raw_stakeholders": result["raw_stakeholders"],
        "candidate_stakeholders": result["candidate_stakeholders"],
        "raw_filtered_stakeholders": result["raw_filtered_stakeholders"],
        "concerns_entries": result["concerns_entries"],
        "judge_reasoning": result["judge_reasoning"],
        "judge_raw": result["judge_raw"],
    }
    if env_name == "AITA":
        trace["judge_verdict"] = result["judge_output"]
        text = result["judge_raw"]
    else:
        trace["judge_response"] = result["judge_output"]
        text = result["judge_output"]
    return (
        text,
        result["input_tokens"],
        result["output_tokens"],
        trace,
    )
