"""Model routing and cost reporting: pricing lookup, filename slug, the
claude-vs-openrouter family switch, client construction, and total-cost
printing."""

from open_router_utils import make_openrouter_client

from .config import PRICING_PER_MTOK


def model_pricing(model):
    """Return (input_usd_per_mtok, output_usd_per_mtok) for `model`.

    Matches by prefix so dated model ids resolve to the family entry. Fails
    loudly if the model is unknown — pricing must be explicit, not guessed.
    """
    for prefix, prices in PRICING_PER_MTOK.items():
        if model.startswith(prefix):
            return prices
    raise AssertionError(f"no pricing entry for model {model!r}; add it to PRICING_PER_MTOK")


def model_slug(model):
    """Return the filename-safe tail of a model id.

    Drops any OpenRouter `provider/` prefix so e.g. `openai/gpt-4o`
    becomes `gpt-4o`. Bare ids (no slash, e.g. `claude-sonnet-4-6`) are
    returned unchanged. Used by the default `--out` construction in
    `main()` so the output path uniquely identifies the (env, gen-model,
    judge-model) combination — runs across different model choices no
    longer clobber each other's CSVs / checkpoints / traces files.
    """
    return model.rsplit("/", 1)[-1]


def model_family(model):
    """Return the routing family for a model id: 'claude' or 'openrouter'.

    Claude model ids (the `claude-*` family) are routed through the
    Anthropic SDK directly. Every other id is routed through OpenRouter
    via `open_router_utils.call_openrouter` — so the family is just a
    two-way switch, not a full provider taxonomy. The returned string is
    also used as a filename suffix so a Claude run and an OpenRouter run
    can coexist on disk; the OpenRouter run gets `_openrouter` and the
    caller can pass `--out` explicitly if a more specific name is wanted.
    """
    if model.startswith("claude-"):
        return "claude"
    return "openrouter"


def make_client(model):
    """Return the client matched to `model`'s routing family.

    The clients are lightweight; making two of them (one for the
    generation model, one for the judge model) is fine. Authentication
    is done from environment variables:
      claude     → ANTHROPIC_API_KEY  (Anthropic SDK)
      openrouter → OPENROUTER_API_KEY (via `open_router_utils`)
    """
    family = model_family(model)
    if family == "claude":
        import anthropic
        return anthropic.Anthropic()
    assert family == "openrouter"
    return make_openrouter_client()


def rows_cost(rows, gen_model, judge_model):
    """Return the USD cost of an iterable of result dicts.

    Generation tokens (`gen_input_tokens`/`gen_output_tokens`) are priced at
    `gen_model`; judge tokens (`judge_input_tokens`/`judge_output_tokens`) at
    `judge_model`. Old rows that predate the gen/judge token split (only
    `input_tokens`/`output_tokens`) are priced wholly at `gen_model`, matching
    `print_total_cost`.

    Assumptions: each row is a mapping with the token fields above; missing or
    None token values count as 0. Used both for the final cost report and for
    the in-run `--max-cost` cap, so the two never disagree on what a run costs.
    """
    gen_in_price, gen_out_price = model_pricing(gen_model)
    judge_in_price, judge_out_price = model_pricing(judge_model)
    total = 0.0
    for r in rows:
        if "gen_input_tokens" in r:
            total += (r.get("gen_input_tokens") or 0) / 1e6 * gen_in_price
            total += (r.get("gen_output_tokens") or 0) / 1e6 * gen_out_price
            total += (r.get("judge_input_tokens") or 0) / 1e6 * judge_in_price
            total += (r.get("judge_output_tokens") or 0) / 1e6 * judge_out_price
        else:
            total += (r.get("input_tokens") or 0) / 1e6 * gen_in_price
            total += (r.get("output_tokens") or 0) / 1e6 * gen_out_price
    return total


def print_total_cost(results_df, gen_model, judge_model):
    """Print the total token usage + cost across all rows in `results_df`.

    Split into generation tokens (priced at `gen_model`) and judge tokens
    (priced at `judge_model`). Old checkpoint rows that predate the
    gen/judge token breakdown only have `input_tokens` / `output_tokens`;
    those get priced wholly at `gen_model`.

    Safe to call on partial results — e.g. after a worker raised and the
    harness is about to re-raise — so the user can see how much was burned
    on a crashed run instead of being stuck guessing.
    """
    gen_in_price, gen_out_price = model_pricing(gen_model)
    judge_in_price, judge_out_price = model_pricing(judge_model)
    if len(results_df) == 0:
        g_in = g_out = j_in = j_out = 0
    elif "gen_input_tokens" in results_df.columns:
        g_in = int(results_df["gen_input_tokens"].fillna(0).sum())
        g_out = int(results_df["gen_output_tokens"].fillna(0).sum())
        j_in = int(results_df["judge_input_tokens"].fillna(0).sum())
        j_out = int(results_df["judge_output_tokens"].fillna(0).sum())
    else:
        g_in = int(results_df["input_tokens"].fillna(0).sum())
        g_out = int(results_df["output_tokens"].fillna(0).sum())
        j_in = j_out = 0
    total_cost = ((g_in / 1e6) * gen_in_price + (g_out / 1e6) * gen_out_price
                  + (j_in / 1e6) * judge_in_price + (j_out / 1e6) * judge_out_price)
    print(f"\n=== TOTAL across all conditions ===")
    print(f"  gen_input_tokens:    {g_in}  (priced at --model {gen_model})")
    print(f"  gen_output_tokens:   {g_out}")
    if j_in or j_out:
        print(f"  judge_input_tokens:  {j_in}  (priced at --judge_model {judge_model})")
        print(f"  judge_output_tokens: {j_out}")
    print(f"  total_cost_usd: ${total_cost:.4f}")
