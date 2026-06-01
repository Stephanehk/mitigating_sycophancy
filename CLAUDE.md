# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository nature

This is a research codebase for **studying and mitigating sycophancy in LLM responses to interpersonal-advice scenarios**, built on the ELEPHANT social-sycophancy benchmark. It is a git repo, but not a packaged product: no build system or package manifest, and work is done ad hoc in Python / Jupyter — though there are runnable harness scripts and a couple of `test_*` files under `mitigations/`. Use the project interpreter `/next/u/stephhk/miniconda3/envs/llm/bin/python` (the base env lacks `matplotlib` / `anthropic`); set `ANTHROPIC_API_KEY` / `OPENROUTER_API_KEY` before running anything that calls an LLM.

## Repository layout

See `README.md` for the full description. In brief:

- `mitigations/` — the single-turn mitigation harness. The entry point `explore_mitigations.py` (envs `PAS`/`AITA`/`OEQ`/`AITA-NTA-FLIP`/`OEQ_2`) re-exports its shared library from the `mitigations/core/` package (`config`, `models`, `prompts`, `data`, `llm`, `checkpoint`, `metrics`, `run`), so existing `from explore_mitigations import ...` callers keep working — preserve those re-exports. The proposed stakeholder-concerns method is in `our_mitigation.py` / `our_mitigation_v2.py` (v1 is kept on purpose because v2 imports helpers from it). Plots: `plot_explore_mitigations.py`. Outputs land in `mitigations/results/` and `mitigations/plots_*/`.
- `eval_empathy/` — empathy-judging pipeline (`judge_empathy.py`, `empathy_judge_prompt.py`, `run_empathy_baselines.py`, `plot_empathy*.py`); imports the harness from `mitigations/` via `sys.path`.
- `reproduce/` — scripts that re-derive the ELEPHANT PAS action-endorsement judge against the paper's own released labels (`reproduce_pas_judge_*.py`, `sanity_rejudge_paper_pas.py`, `for_myra/`). OSF paths are anchored to the repo root, so they run from any cwd.
- `legacy/` — superseded one-off scripts, kept for reference only; not part of the active pipeline.
- `open_router_utils/` — OpenRouter client used to route non-Claude models.
- `data/` — `aita_clean.csv` (checked in); other large CSVs are gitignored.
- `osfstorage-archive/` — the vendored OSF reproducibility bundle for the ELEPHANT paper. Obtained via download (see `README.md` "Required data"), gitignored.
- `accommodation/`, `elephant/`, `ELEPHANT_datasets/`, `SYCON-Bench/` — third-party checkouts (prompts / source datasets), gitignored.

The remaining sections document the **`osfstorage-archive/` bundle** specifically, which contains:

- `Code/` — analysis scripts and notebooks
- `Data/` — large CSVs of LLM responses and human survey responses (some files are 100s of MB; one is a 280 MB zip)
- `Materials/` — Qualtrics survey export (`.qsf`), an HTML survey, and a scenarios PDF used in the studies

## Three studies, three pipelines

The codebase is split by study; each is largely self-contained and shares only the convention of a `filtered_df` DataFrame loaded from a study-specific CSV.

- **Study 1** — LLM behavior on AITA-style and PAS/OEQ scenarios. `Code/Study 1 Analyses.zip` contains three notebooks (`AITA_endorsement_rates.ipynb`, `PAS_endorsement_rates.ipynb`, `OEQ_endorsement_rates.ipynb`). Inputs: `Data/AITA_endorsement_results.csv` (~480 MB, hundreds of columns of the form `<Model>_<binary|openended>_<prompt-variant>` covering Llama-8B/17B/70B, Claude, Gemini, gpt-4o, Mistral-7B/24B, Qwen, DeepSeek, GPT-5) and `Data/OEQ_and_PAS_results.zip`. The column-naming convention is the schema — e.g. `Claude_openended_NTA_context` is the open-ended response under a "not the asshole" framing; suffix `rated` means a per-response judge score; `is_asshole` is the human ground-truth label.
- **Study 2** — Hypothetical interpersonal scenarios with sycophantic vs. anthropomorphic manipulations. Notebook `Code/Study_2_analysis.ipynb` reads `Data/study2_responses.csv` and runs OLS regressions (`statsmodels.formula.api`) with formulas like `dv ~ is_syco * is_anthro` over DVs including `rightorwrong`, `repair_score`, `return_likelihood`, `response_quality`, `performance_trust_score`, `moral_trust_score`. AIAS composite is built from `AIAS_1..4`. Survey source is `Materials/Study_2_Hypothetical.qsf`.
- **Study 3** — Real interpersonal scenarios; condition assignment lives in the `Condition` column. Notebook `Code/Study_3_analysis.ipynb` reads `Data/study3_responses.redacted.csv` and derives `scenarioNum` by string-matching `pickedScenario` against fixed scenario names ("Indirect Communication", "Relationship Boundaries", "Making Someone Uncomfortable", etc.). Survey source is `Materials/study3_survey.html`.

## The judge script: `evaluate_action_endorsement.py`

This is the only standalone Python script. It is the LLM-as-judge that produces the `endorse_*` (and historically `ev_*`) columns visible in the Study 1 CSV. Key facts:

- Uses **Azure OpenAI** (`gpt-4o` deployment) via `openai.AzureOpenAI`. **Endpoint and API key are blank placeholders in the source** (lines 20–21) — the script will not run until they are filled in. Treat this as expected, not a bug to fix unless asked.
- Two prompt templates exist in `PROMPT_TEMPLATES`: `endorse` (4-way classification: challenges / explicit affirm / implicit affirm / neutral) and `ev` (binary emotional-validation). The current `main` only invokes `endorse` — the `ev` template is defined but unused, and `generate_ai_response_for_row` returns a 1-tuple despite the type hint claiming a 4-tuple. Don't "fix" the type hint without checking which metrics the caller actually wants enabled.
- Concurrency: `ThreadPoolExecutor` (default 25 workers) gated by a `threading.Semaphore(15)` plus a `time.sleep(1/15)` per call. Retries via `tenacity` (3 attempts, exponential backoff). Checkpoints to `out_path` every `checkpoint_every` rows (default 100) — long runs are interruptible.
- Input contract: the CSV must have a `sentence` column (the question/scenario) and whatever column is passed via `--advice` (the model response to be judged). Output column is `endorse_<advice_col>`.
- There is a latent bug at line 148: `for suffix in ('endorse'):` iterates over characters, not the single string `'endorse'`. Flag this if touching that loop.

Run it like:

```bash
python osfstorage-archive/Code/evaluate_action_endorsement.py <input.csv> \
  --advice <advice_column_name> \
  --output scored.csv \
  --workers 25 --checkpoint 100
```

## Working with the data files

- Several CSVs are very large (`AITA_endorsement_results.csv` is ~480 MB, `OEQ_and_PAS_results.zip` is ~280 MB). Don't `cat`/`head` them blindly — use `pd.read_csv(..., usecols=[...])` or peek at the header with `head -1` only.
- The notebooks are run from `Code/` and reference data files by **bare filename** (`pd.read_csv('study2_responses.csv')`). Either copy/symlink the data next to the notebook or change the path before executing.
- `study3_responses.redacted.csv` is the redacted public version — assume free-text fields with PII have been scrubbed.

## Error handling
- Default to letting code fail fast rather than trying to recover silently.
- Do not write `try`/`catch` blocks.
- Do not suppress exceptions unless explicitly instructed.
- When expectations are violated, use `assert` statements or raise clear errors immediately.
- Prefer early failure over hidden fallback behavior.

## Assumptions
- For every non-trivial method, explicitly identify its assumptions before or while implementing it.
- Validate important assumptions in code.
- Add tests that cover those assumptions, especially edge cases and failure cases.
- Do not leave critical assumptions implicit.

## Function signatures and typing style
- Do not write functions with type annotations in their signatures.
- Do not use return annotations.
- Do not use union syntax such as `|`.
- Keep function signatures untyped and simple.

## Code reuse and organization
- Always look for opportunities to reuse existing code before writing new code.
- Prefer general-purpose helper functions over one-off duplicated logic.
- If functionality is likely to be reused across files, put it in a central utility module.
- Avoid copy-pasting logic when a shared helper would make the code clearer.
- Optimize for maintainability and consistency across the codebase.

## General implementation style
- Write small, composable functions with clear responsibilities.
- Prefer explicit behavior over clever abstractions.
- Make invalid states obvious and hard to ignore.
- Keep code straightforward enough that assumptions and failure modes are easy to inspect.

## Simplicity
- Prefer the simplest implementation that satisfies the requirements.
- Do not introduce abstractions unless they clearly reduce duplication or complexity.
- Avoid premature optimization.
- Avoid unnecessary classes, configuration layers, or indirection.
- Prefer straightforward procedural logic unless an abstraction clearly improves clarity.

## File organization
- Each file should have a clear responsibility.
- Avoid files longer than ~500 lines.
- Shared utilities belong in a central utilities module.
- Avoid circular imports.

## Documentation
- Every module should have a top-level description.
- Every non-trivial function should explain:
  - what it does
  - its assumptions
  - its inputs and outputs

## Plotting and data handling rules
Always plot the raw data given. NEVER MAKE UP DATA. NEVER interpolate between any datapoints when plotting, ONLY plot the data that is given and throw an error if it doesn't match expectations. YOUR PLOTTING OF DATA SHOULD always be defensible by a rigorous audit from the scientific community.
