"""Configuration constants for the mitigation harness: data-file paths,
default sample sizes / models / token caps, the HUMAN_REFERENCE sentinel,
and the per-model pricing table."""


AITA_DATA_CSV = "data/aita_clean.csv"
# Any of the per-model OSF OEQ files works — they all carry the same `sentence`
# (prompt) and `Human` (advice columnist / top Reddit comment) columns. The
# `_human` file additionally holds the GPT-4o judge labels of the Human
# responses in `endorse_Human`, useful for sanity-checking our re-judging
# results against the paper's published labels. We don't read `endorse_Human`
# in the main flow — we re-judge with our own Claude judge for consistency.
OEQ_DATA_CSV = "osfstorage-archive/Data/OEQ_results/endorse4_elephant_oeq3k_human.csv"
# Any of the per-model PAS files works for the prompt set (`sentences` column);
# they share the same 6,433 pre-filtered problematic sentences. We default to
# the Claude file by convention. The model `response` column is ignored — we
# generate our own responses and re-judge them with our own judge.
PAS_DATA_CSV = "osfstorage-archive/Data/PAS_results/endorse4_elephant_oeq_pas_claude.csv"
# AITA-NTA-FLIP: paired-post dataset where each row has an `original_post` and
# a `flipped_story` (the same scenario rewritten from the other party's POV).
# A YTA/NTA-stable method should flip its verdict alongside the POV flip; a
# method that says NTA for BOTH is sycophantic (always-nice), one that says
# YTA for BOTH is over-critical (always-harsh).
AITA_NTA_FLIP_DATA_CSV = "ELEPHANT_datasets/AITA-NTA-FLIP.csv"
# OEQ_2: ELEPHANT_datasets/OEQ.csv. Same prompt column as the OSF OEQ data
# (`prompt` here vs. `sentence` there), plus per-row precomputed binary human
# labels on three sycophancy axes (validation_human, indirectness_human,
# framing_human). The model response is judged on the same three axes with
# the prompts copied from `elephant/sycophancy_scorers.py` and the per-row
# score is "1 if judge label != human label else 0" (judge-vs-human
# disagreement, not action endorsement).
OEQ_2_DATA_CSV = "ELEPHANT_datasets/OEQ.csv"

# Default sample sizes per env. AITA: 50 (balanced 25/25). OEQ/PAS: 30
# (uniform); the `stakeholder_concerns` condition is ~5-9 calls per post plus a
# judge call, so this keeps explore runs cheap. AITA-NTA-FLIP: 100 (uniform);
# each post runs the condition TWICE (original + flipped) so per-post cost is
# 2× the AITA cost. OEQ_2: 100 (uniform); each post adds 3 judge calls on the
# response (one per sycophancy axis). Override with --n.
DEFAULT_N_AITA = 50
DEFAULT_N_OEQ = 30
DEFAULT_N_PAS = 30
DEFAULT_N_AITA_NTA_FLIP = 100
DEFAULT_N_OEQ_2 = 100
DEFAULT_SEED = 0
# DIFF-C: paper's Claude version is unspecified in the OSF archive.
# Generation model (the methods themselves) defaults to Sonnet — cheap enough
# for sweeping conditions. The action-endorsement judge defaults to GPT-5.4
# (same OpenAI family as the paper's GPT-4o judge) so a single fixed judge is
# used across all generation models — cross-model comparisons are not
# confounded by judge variation.
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_JUDGE_MODEL = "openai/gpt-5.4"
# Sized for the `repeat_question` condition, which makes the model echo the
# post (often hundreds of tokens) before producing the YTA/NTA verdict. The
# baseline / single-line-prefix conditions still produce ~1-token replies;
# output cost is per token actually generated, so over-allocating is free.
DEFAULT_MAX_TOKENS = 2048
DEFAULT_WORKERS = 10

# Sentinel value for the OEQ-only `human_reference` "condition": instead of
# calling the LLM, run_one looks up the human advice from the row.
HUMAN_REFERENCE = object()

# USD per 1M tokens, (input, output). Keyed by model-id prefix so versioned
# ids like `claude-sonnet-4-5-20250929` resolve to the same family. Order
# matters: longer / more specific prefixes are listed first so they match
# before a shorter prefix that would also be a valid `startswith`. For
# example, `gemini-2.5-flash-lite` is listed before `gemini-2.5-flash`.
PRICING_PER_MTOK = {
    "claude-opus-4": (15.0, 75.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-3-7-sonnet": (3.0, 15.0),
    "anthropic/claude-3.7-sonnet": (3.0, 15.0),
    "claude-haiku-4": (1.0, 5.0),
    "google/gemini-3.1-pro-preview": (2.00, 12.00),
    "openai/gpt-5.4-mini": (0.25, 2.00),
    "openai/gpt-5.4":(2.50, 15.0),
    "openai/gpt-4o": (2.50, 10.00),
    "deepseek/deepseek-v3.2": (0.252, 0.378),
    # DeepSeek V4 Pro (OpenRouter standard provider, looked up May 2026).
    "deepseek/deepseek-v4-pro": (0.435, 0.87),
    # Original DeepSeek-V3 (`deepseek/deepseek-chat`). Approximate OpenRouter
    # rate for cost reporting only — adjust if the routed provider differs.
    "deepseek/deepseek-chat": (0.28, 0.88),
    # Llama 4 Maverick (OpenRouter standard provider, looked up May 2026).
    "meta-llama/llama-4-maverick": (0.15, 0.60),
    # Llama 4 Scout (OpenRouter standard provider, looked up Jun 2026).
    "meta-llama/llama-4-scout": (0.08, 0.30),
    "meta-llama/llama-3.3-70b-instruct": (0.10, 0.32),
    "qwen/qwen3-235b-a22b-2507": (0.071, 0.10),
    "qwen/qwen3.6-plus": (0.325, 1.95),
    # Qwen2.5-7B-Instruct (OpenRouter standard provider, looked up Jun 2026).
    # Together's "-Turbo" variant maps here on OpenRouter.
    "qwen/qwen-2.5-7b-instruct": (0.04, 0.10),
}


# Anthropic SDK refuses non-streaming requests when its static time estimate
# `(60*60) * max_tokens / 128_000` exceeds 10 minutes (`_base_client.py:731`),
# which kicks in for `max_tokens > 21333`. 20000 sits comfortably under that
# threshold while giving ~10x headroom over the previous 2048 default, which
# was truncating long `our_mitigation.judge` XML responses mid-`<verdict>`.
_CLAUDE_MAX_TOKENS = 20000
