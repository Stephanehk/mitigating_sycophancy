# sycophancy

Research code for studying and mitigating sycophancy in LLM responses to
interpersonal-advice scenarios. Builds on the ELEPHANT social-sycophancy
benchmark (arxiv:2510.01395) and the SYCON-Bench multi-turn benchmark
(Hong et al. 2025, https://aclanthology.org/2025.findings-emnlp.121.pdf),
adds new prompt- and multi-call mitigations on top, and sweeps them
against a no-mitigation baseline across single-turn and multi-turn
settings.

## Layout

- `mitigations/` — the main harness. Single-turn sweeps live in
  `explore_mitigations.py` (envs: `PAS`, `AITA`, `OEQ`, `AITA-NTA-FLIP`,
  `OEQ_2`); multi-turn sweeps live in `explore_mitigations_multiturn.py`
  (envs: `ethical`, `fp`). The proposed stakeholder-concerns method is in
  `our_mitigation.py` / `our_mitigation_v2.py`. Empathy judging is in
  `judge_empathy.py`. Plot scripts (`plot_*.py`) read the CSVs in
  `mitigations/results/` and write into `mitigations/plots_*/`.
- `open_router_utils/` — thin OpenRouter chat-completion client used by
  the harness to route non-Claude models. Reads `OPENROUTER_API_KEY` from
  the environment. Claude models are routed through the Anthropic SDK
  directly (reads `ANTHROPIC_API_KEY`).
- `data/` — `aita_clean.csv` only. The harness loads AITA post text from
  here because the OSF archive does not ship AITA post text (see
  `DIFF-A` in `mitigations/explore_mitigations.py`).
- `osfstorage-archive/` — the OSF reproducibility bundle for the
  ELEPHANT paper. See "Required data" below for how to obtain it.
- `accommodation/`, `elephant/`, `ELEPHANT_datasets/`, `SYCON-Bench/` —
  third-party benchmark/dataset checkouts referenced by the harness for
  prompts and source CSVs. Not part of this repo; see "Datasets" below.

## Required data

Download the OSF archive from

  https://osf.io/smvw7/?view_only=ad71a7201c71477d921000c90c565da7

and extract it as `osfstorage-archive/` at the repo root. The harness
reads OEQ prompts from
`osfstorage-archive/Data/OEQ_results/endorse4_elephant_oeq3k_human.csv`
and PAS prompts from
`osfstorage-archive/Data/PAS_results/endorse4_elephant_oeq_pas_claude.csv`;
without the archive in place, OEQ and PAS runs will fail at CSV-read
time. AITA runs only need `data/aita_clean.csv`, which is checked in.

`osfstorage-archive/Code/CLAUDE.md` documents the three-study layout of
the OSF bundle (AITA / Study 2 / Study 3) and the judge script
`evaluate_action_endorsement.py` that produced the `endorse_*` columns
visible in the OSF CSVs.

## Environment

Use the project Python interpreter:

```
/next/u/stephhk/miniconda3/envs/llm/bin/python
```

The base environment lacks `matplotlib` and `anthropic`; use the `llm`
env above. API keys (`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`) must be
set in the shell before running anything that calls an LLM.

## Typical run

Single-turn sweep on AITA with Claude as the generator and GPT-5.4 as
the action-endorsement judge:

```
python mitigations/explore_mitigations.py \
  --env_name AITA \
  --model claude-sonnet-4-6 \
  --judge_model openai/gpt-5.4 \
  --n 50 --workers 10
```

Outputs land in `mitigations/results/` as
`explore_mitigations_results_<env>_<gen-slug>_<judge-slug>.csv` plus a
matching `.checkpoint.jsonl` (resumable) and `_traces.jsonl` (per-call
traces). Plot with `python mitigations/plot_explore_mitigations.py`.
