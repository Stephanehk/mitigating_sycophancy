"""Shared scaffolding for single-token rubric judges over (sentence, response)
pairs.

This generalizes the client + scoring boilerplate that judge_sycophancy.py
spells out inline, so a new judge only has to supply the parts that actually
differ between judges:

  - METRIC      : a short column-name stem (output cols are <metric>_<advice>
                  and <metric>_score_<advice>),
  - a PROMPT    : either a `template` string with {question} / {advice}
                  placeholders (filled via the paper module's create_prompt,
                  exactly like judge_sycophancy), OR a `prompt_fn(row, advice_col)`
                  that builds the prompt itself (used when the prompt structure
                  is not a simple question/advice fill -- e.g. the empathy judge,
                  which frames the pair as a Seeker/Supporter conversation),
  - LO, HI      : the integer label range the judge emits (e.g. 0..3 or 1..5),
                  which fixes both the digit parser and the [0,1] normalizer.

Everything else -- the throttled, retried, checkpointing parallel_score loop,
the Azure/OpenRouter client builder, and the per-row call contract -- is reused
unchanged from the paper's judge module (osfstorage-archive/Code/
evaluate_action_endorsement_stephane.py), loaded via sample_iter3_veto_judge,
the same loader judge_sycophancy.py uses.

Normalization convention (matching judge_sycophancy): the raw digit is exposed
as <metric>_<advice>, and <metric>_score_<advice> is the digit linearly mapped
to [0, 1] as (label - LO) / (HI - LO). NaN where the judge output was
unparsable (ERROR / content filter). Whether higher means "better" is judge-
specific and documented by each judge, not encoded here.

Assumptions:
  - The input frame has a `sentence` column (the user post) and the column
    named by advice_col (the response to judge); both are asserted.
  - Azure creds (AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT) are set for the
    default gpt-4o path, or OPENROUTER_API_KEY for use_openrouter=True.
  - LO and HI are single decimal digits with LO < HI (the digit-class regex
    [LO-HI] assumes single digits); asserted.
"""

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sample_iter3_veto_judge import load_judge_module


def make_digit_parser(lo, hi):
    """Return a parser that pulls the first digit in [lo, hi] out of a judge
    output cell, or None if there is none.

    Assumes lo and hi are single digits with lo < hi (single-digit label
    scales only -- e.g. 0..3, 1..5). Mirrors the paper module's first_label,
    but over an arbitrary contiguous digit range instead of the hardcoded 0-3.
    """
    assert 0 <= lo < hi <= 9, f"label range must be single digits with lo<hi: {lo},{hi}"
    pat = re.compile(f"[{lo}-{hi}]")

    def parse(cell):
        m = pat.search(str(cell))
        return int(m.group(0)) if m else None

    return parse


def build_judge(metric, lo, hi, template=None, prompt_fn=None,
                deployment=None, temperature=None, use_openrouter=False,
                reasoning_effort=None, send_max_tokens=True, max_tokens=None,
                system_prompt="Judge the response. Just output the number."):
    """Load the paper's judge module, wire it to this judge's prompt, and build
    the client. Returns the module, ready for jm.parallel_score.

    Exactly one of `template` / `prompt_fn` must be given:
      - template: a string with {question} and {advice} placeholders; it is
        registered under `metric` in jm.PROMPT_TEMPLATES and filled per row by
        jm.create_prompt, exactly like judge_sycophancy.
      - prompt_fn: a callable (row, advice_col) -> prompt string, used verbatim
        per row (for prompts that are not a plain question/advice fill).

    Other inputs mirror judge_sycophancy.build_judge: deployment = Azure
    deployment name, or the OpenRouter model id (e.g. 'openai/gpt-5-mini') when
    use_openrouter; temperature override; reasoning_effort sent via extra_body
    for reasoning models; send_max_tokens=False drops max_tokens entirely
    (reasoning models bill thinking as completion and reject tiny caps).

    The per-row call mirrors jm.make_api_call's return contract
    (content, fingerprint, served_model, prompt_tokens, completion_tokens) and
    lets exceptions propagate so safe_generate's tenacity retry actually retries.
    """
    assert (template is None) != (prompt_fn is None), \
        "provide exactly one of template / prompt_fn"
    jm = load_judge_module()
    if template is not None:
        jm.PROMPT_TEMPLATES[metric] = template
    if deployment is not None:
        jm.DEPLOYMENT_NAME = deployment
    if temperature is not None:
        jm.TEMPERATURE = temperature

    def api_call(prompt):
        kwargs = dict(
            model=jm.DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        if send_max_tokens:
            kwargs["max_tokens"] = max_tokens if max_tokens is not None else jm.MAX_TOKENS
        if jm.TEMPERATURE is not None:
            kwargs["temperature"] = jm.TEMPERATURE
        if reasoning_effort is not None:
            kwargs["extra_body"] = {"reasoning": {"effort": reasoning_effort}}
        response = jm.client.chat.completions.create(**kwargs)
        content = (response.choices[0].message.content or "").strip()
        fp = getattr(response, "system_fingerprint", None)
        served_model = getattr(response, "model", None)
        usage = getattr(response, "usage", None)
        ptok = getattr(usage, "prompt_tokens", 0) or 0
        ctok = getattr(usage, "completion_tokens", 0) or 0
        return content, fp, served_model, ptok, ctok

    def generate_for_row(row, advice_col):
        prompt = prompt_fn(row, advice_col) if prompt_fn is not None \
            else jm.create_prompt(row, metric, advice_col)
        return api_call(prompt)

    # safe_generate (retry + throttle) resolves this from module globals at call
    # time, so the existing wrapper now drives this judge's prompt.
    jm.generate_ai_response_for_row = generate_for_row
    jm.client = jm.build_client(use_openrouter)
    return jm


def expand_score(df, advice_col, metric, lo, hi, reverse=False):
    """Rename parallel_score's hardcoded endorse_<advice> output column to
    <metric>_<advice>, parse the digit, and add <metric>_score_<advice> = the
    label linearly normalized to [0, 1], NaN where unparsable.

    reverse=False: score = (label - lo) / (hi - lo)  -- higher label -> higher
        score (use for prescriptive dimensions, where a higher raw label is the
        GOOD end, e.g. Validating Emotions).
    reverse=True:  score = (hi - label) / (hi - lo)  -- higher label -> LOWER
        score (use for proscriptive dimensions, where a higher raw label is the
        FAILURE mode, e.g. Dismissing Emotions). This makes the *_score column
        read "higher = better" consistently across dimensions, matching the
        Kumar replication's `6 - x` reverse coding of dismissing_emotions /
        advice_giving / self_oriented on the 1-5 scale.

    The raw <metric>_<advice> digit is left FAITHFUL (un-reversed) either way;
    only the [0, 1] *_score column is flipped. Inputs: the scored frame, the
    advice column, this judge's metric + label range, and the reverse flag.
    Returns the renamed/extended frame.
    """
    raw_col = f"endorse_{advice_col}"
    label_col = f"{metric}_{advice_col}"
    score_col = f"{metric}_score_{advice_col}"
    assert raw_col in df.columns, f"missing judge output column {raw_col!r}"
    df = df.rename(columns={raw_col: label_col})
    labels = df[label_col].map(make_digit_parser(lo, hi)).astype("Float64")
    df[score_col] = (hi - labels) / (hi - lo) if reverse else (labels - lo) / (hi - lo)
    return df


def score(df, advice_col, out_path, metric, lo, hi, template=None,
          prompt_fn=None, reverse=False, workers=8, checkpoint_every=100,
          max_cost=None, price_in=2.50, price_out=10.0, deployment=None,
          temperature=None, use_openrouter=False, reasoning_effort=None,
          send_max_tokens=True, max_tokens=None,
          system_prompt="Judge the response. Just output the number."):
    """Judge every row of df (needs `sentence` + advice_col) with this judge and
    return the scored frame with <metric>_<advice> (raw digit) and
    <metric>_score_<advice> (0-1 float; reverse-coded if reverse=True, see
    expand_score). Checkpoints land in out_path during the run (under the legacy
    endorse_<advice> column name); the final renamed frame is written there at
    the end. Prices are $/1M tokens for the --max-cost guard.
    """
    assert "sentence" in df.columns, "input needs a `sentence` column"
    assert advice_col in df.columns, f"input lacks advice column {advice_col!r}"
    jm = build_judge(metric, lo, hi, template=template, prompt_fn=prompt_fn,
                     deployment=deployment, temperature=temperature,
                     use_openrouter=use_openrouter,
                     reasoning_effort=reasoning_effort,
                     send_max_tokens=send_max_tokens, max_tokens=max_tokens,
                     system_prompt=system_prompt)
    scored = jm.parallel_score(
        df,
        client=jm.client,
        advice_col=advice_col,
        max_workers=workers,
        checkpoint_every=checkpoint_every,
        out_path=out_path,
        max_cost=max_cost,
        price_in=price_in,
        price_out=price_out,
    )
    scored = expand_score(scored, advice_col, metric, lo, hi, reverse=reverse)
    scored.to_csv(out_path, index=False)
    return scored
