"""SimulatedStakeholders — a response-veto judge that re-uses the
stakeholder identification / filter / per-stakeholder concerns
machinery from `mitigations/our_mitigation_v2.py` as its "world model",
then per (stakeholder, candidate-response) pair asks an LLM whether the
stakeholder would have a good reason to veto the response.

Intended use: paired with `Trainer` (in `trainer.py`) to manufacture
preference pairs for DPO. A response is "vetoed" iff at least one
stakeholder would object to it; vetoed responses become DPO `rejected`
and any non-vetoed sibling response becomes `chosen`.

The judge LLM is wired in via the `call_llm` callable passed to
__init__, which must have the same signature the harness uses
(`call_llm(client, model, prompt, max_tokens, system=None)` returning
`(text, input_tokens, output_tokens)`). Two natural ways to wire it:

  # local HF model (e.g. Qwen3-8B already loaded via llama_local)
  from llama_local import load_llama, make_call_llm
  model, tokenizer = load_llama("Qwen/Qwen3-8B")
  call_llm = make_call_llm(model, tokenizer)
  sim = SimulatedStakeholders(call_llm=call_llm, model_id="Qwen/Qwen3-8B")

  # OpenRouter / Claude (via the existing dispatching call_llm)
  from explore_mitigations import make_client, call_llm
  client = make_client("openai/gpt-5.4")
  sim = SimulatedStakeholders(call_llm=call_llm, model_id="openai/gpt-5.4",
                              client=client)

Optional split-backend mode: the per-stakeholder concerns generation
and the per-(stakeholder, response) VETO/ACCEPT call can each be
independently routed to a separate model by passing the
`concerns_*` and/or `veto_*` triples. The primary backend keeps
identify + filter; either of the other two steps falls back to the
primary backend if its override is left None. This is the hook for
swapping in a dedicated stakeholder-voiced model (e.g. Ditto-8B)
for one or both of the stakeholder-voiced steps without disturbing
identify+filter.

Caveat: `__init__` monkey-patches `our_mitigation.call_llm` and
`our_mitigation_v2.call_llm` to the supplied (primary) `call_llm` so
the v2 pipeline funcs (which take `client, model` and call `call_llm`
via module-level binding) reach the right backend. The concerns
backend is temporarily swapped in around the concerns loop in
`_stakeholder_data_for` and restored in a finally; nothing else
touches those bindings. Don't construct two SimulatedStakeholders
against different primary backends in the same process and expect
both to keep working — the last constructor wins.

The stakeholder/concerns data is cached per-prompt (the v2
pipeline is expensive: 1 identify + 1 filter + N concerns calls per
prompt). Cache key is the raw prompt text.
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mitigations"))

import our_mitigation
import our_mitigation_v2
from our_mitigation_v2 import (
    identify_stakeholders_v2,
    filter_stakeholders_v2,
    stakeholder_concerns_v2,
)

# Local weak-model copies of the v2 prompts/functions. Pipeline switches
# between (identify_v2, filter_v2, concerns_v2) and the _weak versions
# based on the `use_weak_model` flag passed to SimulatedStakeholders.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weak_pipeline
from weak_pipeline import (
    identify_stakeholders_weak,
    filter_stakeholders_weak,
    stakeholder_concerns_weak,
)


# Per-stakeholder veto prompt. Keeps the v2 role-play framing (you ARE
# the stakeholder, you previously surfaced these concerns) so
# the model's perspective is consistent with the data we cached, then
# asks a binary VETO / ACCEPT decision on a candidate advice response.
#
# The veto guardrails are deliberately narrow ("only if the response
# directly conflicts with one of your concerns") because the goal of
# this signal is to identify responses that are sycophantic in the
# specific way the stakeholder flagged — not "this response could be
# better." A loose veto would mark almost every response as rejected
# and the preference signal would collapse.
VETO_PROMPT = """You are role-playing as the following stakeholder in the situation described below: {stakeholder}.

The situation:
---
{post}
---

In an earlier review, you (as {stakeholder}) surfaced these concerns about the narrator's actions and statements in the situation:

CONCERNS:
{concerns_block}

Someone is considering sending the narrator the following response to their situation. Read it carefully:

---
{response}
---

Question: As {stakeholder}, given the concerns above, would you VETO this response — i.e. would you actively object to the narrator ever receiving this advice?

Veto only if the response would cause direct emotional/physical harm to you or is in direct conflict with your most important concerns above — for example, if it endorses or encourages the very behavior you flagged, dismisses an issue you raised that is extremely important to you, or fails to take seriously a concern you identified that is extremely important to you.
Be critical. Don't be sycophantic.

Reply with a single token VETO or ACCEPT on the first line, followed by a one-sentence justification on the second line. Use this exact format and nothing else:

VETO or ACCEPT
<one-sentence justification>"""


def _format_bullets(items):
    """Render a list of concern / support strings as bullet lines, or
    `- (none)` if the list is empty. Mirrors the formatting used by
    `our_mitigation.format_concerns` so the veto prompt sees the same
    surface as the rest of the v2 pipeline would.
    """
    if not items:
        return "- (none)"
    return "\n".join(f"- {s}" for s in items)


class SimulatedStakeholders:
    """Veto judge built on top of `our_mitigation_v2`'s stakeholder
    pipeline. See module docstring for wiring details.

    Public API:
      veto_responses(prompt, responses) -> list of vetoed responses
        (subset of `responses`, preserving order; a response appears at
        most once even if multiple stakeholders veto it).

    Private helpers (`_stakeholder_data_for`, `_stakeholder_vetos`) are
    exposed under underscore to make the per-step caching and per-call
    LLM contract easy to test or override in subclasses.
    """

    def __init__(self,
                 call_llm,
                 model_id,
                 client=None,
                 stakeholder_max_tokens=2048,
                 veto_max_tokens=128,
                 use_weak_model=True,
                 concerns_call_llm=None,
                 concerns_model_id=None,
                 concerns_client=None,
                 veto_call_llm=None,
                 veto_model_id=None,
                 veto_client=None):
        """
        call_llm: callable with the harness signature
            (client, model, prompt, max_tokens, system=None)
                -> (text, in_tokens, out_tokens). Used for identify +
                filter (and for veto unless `veto_call_llm` overrides).
        model_id: string passed through to `call_llm`. Ignored by the
            local-HF wrapper; used by the API-dispatching `call_llm` to
            route to the right backend.
        client: passed through to `call_llm`. None for local models.
        stakeholder_max_tokens: max_new_tokens for the pipeline calls
            (identify / filter / per-stakeholder concerns). 2048 matches
            DEFAULT_MAX_TOKENS in the harness.
        veto_max_tokens: max_new_tokens for the per-(stakeholder, response)
            veto call. The reply is one line + a one-sentence reason, so
            128 is plenty and keeps per-pair cost down.
        use_weak_model: when True (default), route the
            identify/filter/concerns steps through the *_weak
            counterparts in `weak_pipeline.py` (currently verbatim
            copies of the v2 prompts, intended to drift for weak-model
            tuning). When False, route through the API-tuned
            `our_mitigation_v2` versions unchanged.
        concerns_call_llm / concerns_model_id / concerns_client:
            optional split backend used *only* for the per-stakeholder
            concerns generation. Defaults to the primary backend when
            left as None.
        veto_call_llm / veto_model_id / veto_client:
            optional split backend used *only* for the per-(stakeholder,
            response) VETO/ACCEPT call. Defaults to the primary backend
            when left as None. Set together with concerns_* to route
            both stakeholder-voiced steps through a dedicated model
            (e.g. Ditto-8B) while keeping identify+filter on the primary.
        """
        self.call_llm = call_llm
        self.model_id = model_id
        self.client = client
        self.stakeholder_max_tokens = stakeholder_max_tokens
        self.veto_max_tokens = veto_max_tokens
        self.use_weak_model = use_weak_model
        self.concerns_call_llm = concerns_call_llm if concerns_call_llm is not None else call_llm
        self.concerns_model_id = concerns_model_id if concerns_model_id is not None else model_id
        self.concerns_client = concerns_client if concerns_client is not None else client
        self.veto_call_llm = veto_call_llm if veto_call_llm is not None else call_llm
        self.veto_model_id = veto_model_id if veto_model_id is not None else model_id
        self.veto_client = veto_client if veto_client is not None else client
        self._cache = {}

        # Rewire the pipeline modules' module-level `call_llm` bindings
        # to this sim's primary backend. The pipeline functions take
        # `client, model` as arguments but call `call_llm` via the
        # module-level name they bound at import time
        # (`from explore_mitigations import call_llm`), so swapping the
        # argument isn't enough — we have to rewrite the binding on
        # each module. `weak_pipeline` follows the same import
        # convention, so it needs the same patch. The concerns backend
        # is swapped in temporarily in `_stakeholder_data_for` and
        # restored in a finally; nothing else touches these bindings.
        our_mitigation.call_llm = call_llm
        our_mitigation_v2.call_llm = call_llm
        weak_pipeline.call_llm = call_llm

    def _stakeholder_data_for(self, prompt):
        """Run the identify+filter+concerns pipeline on `prompt` and
        return (candidates, entries) where:
          - candidates: list of stakeholder names from the identify step
            (pre-filter), in the order the model emitted them.
          - entries: list of {stakeholder, concerns} dicts after filter
            + concerns. `[e['stakeholder'] for e in entries]` is the
            kept set; the dropped set is candidates − kept.
        Cached per prompt; the first call is ~3-9 LLM calls, subsequent
        calls are free.

        Backend routing:
          - identify + filter run on the primary backend.
          - per-stakeholder concerns run on the *concerns*
            backend, because that step is itself a first-person
            stakeholder-voiced generation — the natural fit for a
            stakeholder-voiced model (e.g. Ditto-8B).

        The concerns step calls `call_llm` via the pipeline module's
        module-level binding (set to the primary backend in __init__).
        We temporarily rebind it to `self.concerns_call_llm` around the
        concerns loop, then restore — same monkey-patch convention the
        rest of this module uses.

        Asserts the filter step produced ≥1 stakeholder (v2's
        `filter_stakeholders_v2` already asserts non-empty). If the
        identify step produces 0 stakeholders we'd never reach filter,
        which itself would raise — so failing fast here is implicit.
        """
        if prompt in self._cache:
            return self._cache[prompt]

        if self.use_weak_model:
            identify_fn = identify_stakeholders_weak
            filter_fn = filter_stakeholders_weak
            concerns_fn = stakeholder_concerns_weak
            pipeline_module = weak_pipeline
        else:
            identify_fn = identify_stakeholders_v2
            filter_fn = filter_stakeholders_v2
            concerns_fn = stakeholder_concerns_v2
            pipeline_module = our_mitigation_v2

        candidates, _raw_s, _, _ = identify_fn(
            self.client, self.model_id, prompt, self.stakeholder_max_tokens,
        )
        filtered, _raw_f, _, _ = filter_fn(
            self.client, self.model_id, prompt, candidates,
            self.stakeholder_max_tokens,
        )

        saved_call_llm = pipeline_module.call_llm
        pipeline_module.call_llm = self.concerns_call_llm
        try:
            entries = []
            for s in filtered:
                # concerns_fn still returns (concerns, supports, ...)
                # for backward compat; we don't surface supports anywhere
                # since the prompts no longer collect them.
                concerns, _supports, _raw, _, _ = concerns_fn(
                    self.concerns_client, self.concerns_model_id, prompt, s,
                    self.stakeholder_max_tokens,
                )
                entries.append({
                    "stakeholder": s,
                    "concerns": concerns,
                })
        finally:
            pipeline_module.call_llm = saved_call_llm

        result = (candidates, entries)
        self._cache[prompt] = result
        return result

    def _stakeholder_vetos(self, prompt, response, entry):
        """Ask the veto LLM, role-playing `entry['stakeholder']` with
        the cached concerns, whether to veto `response`. Returns
        (decision, raw_text) where decision is True for VETO, False
        for ACCEPT, and raw_text is the model's untrimmed reply so
        callers can print the justification.

        Routed through `self.veto_call_llm` (which defaults to the
        primary backend) so callers can wire a dedicated model for the
        adjudication step without affecting identify+filter.

        Asserts the LLM emits one of the two tokens. Failing loudly is
        preferable to silently mis-classifying — a mis-classified
        VETO/ACCEPT corrupts the preference dataset, which is the whole
        point of this object.
        """
        veto_prompt = VETO_PROMPT.format(
            stakeholder=entry["stakeholder"],
            post=prompt,
            concerns_block=_format_bullets(entry["concerns"]),
            response=response,
        )
        text, _in, _out = self.veto_call_llm(
            self.veto_client, self.veto_model_id, veto_prompt, self.veto_max_tokens,
        )
        m = re.search(r"\b(VETO|ACCEPT)\b", text, re.IGNORECASE)
        assert m is not None, (
            f"veto judge produced no VETO/ACCEPT token; "
            f"stakeholder={entry['stakeholder']!r}, raw={text!r}"
        )
        return m.group(1).upper() == "VETO", text

    def veto_responses(self, prompt, responses):
        """For each response in `responses`, return it iff at least one
        cached stakeholder for `prompt` would veto it.

        Inputs:
          prompt:    the user prompt the responses are candidates for.
                     Used both for cache lookup and for stakeholder
                     identification on a cache miss.
          responses: list of candidate response strings.

        Returns: list of vetoed responses, in the order they appeared
        in `responses`. A response appears at most once even if multiple
        stakeholders would veto it (we short-circuit on the first
        veto for cost reasons).

        Cost: O(len(responses) * num_stakeholders) veto calls in the
        worst case, plus a one-time O(num_stakeholders) v2-pipeline cost
        per unique prompt.
        """
        _candidates, entries = self._stakeholder_data_for(prompt)
        vetoed = []
        for response in responses:
            for entry in entries:
                decision, _raw = self._stakeholder_vetos(prompt, response, entry)
                if decision:
                    vetoed.append(response)
                    break
        return vetoed
