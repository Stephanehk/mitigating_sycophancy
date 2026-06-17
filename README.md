# sycophancy

Research code for studying and mitigating sycophancy in LLM responses to
interpersonal-advice scenarios. Builds on the ELEPHANT social-sycophancy
benchmark (arxiv:2510.01395). It has two pipelines:

- **Single-turn mitigations** (`mitigations/`) — prompt- and multi-call
  mitigations applied at inference time, swept against a no-mitigation
  baseline across the AITA / OEQ / PAS settings.
- **Post-training** (`training_mitigations/`) — an iterative DPO+LoRA method
  that fine-tunes the policy itself to be less sycophantic, using a frozen
  *simulated-stakeholder veto* as the preference signal. This is the main
  contribution and is described in detail below.

## Layout

- `training_mitigations/` — the post-training pipeline (see
  "Post-training" below). Entry point `train_dpo_lora.py`; runtime library in
  the `core/` package; supporting scripts under `eval/`, `data_build/`,
  `plots/`, `labeling/`, `tests/`; cluster jobs in `slurm_scripts/`.
- `mitigations/` — the single-turn harness. The sweep entrypoint is
  `explore_mitigations.py` (envs: `PAS`, `AITA`, `OEQ`, `AITA-NTA-FLIP`,
  `OEQ_2`); its shared library lives in the `mitigations/core/` package
  (`config`, `models`, `prompts`, `data`, `llm`, `checkpoint`, `metrics`,
  `run`) and is re-exported by `explore_mitigations`, so
  `from explore_mitigations import ...` keeps working. The proposed
  stakeholder-concerns method is in `our_mitigation.py` /
  `our_mitigation_v2.py` — these same modules supply the stakeholder
  identify/filter/concerns pipeline reused by the post-training veto judge.
  Plot scripts (`plot_*.py`) read the CSVs in `mitigations/results/` and write
  into `mitigations/plots_*/`.
- `eval_empathy/` — the empathy-evaluation pipeline: `judge_empathy.py`
  scores harness responses on the Lend-an-Ear sub-components, with
  `empathy_judge_prompt.py` (prompt builder), `run_empathy_baselines.py`,
  and the `plot_empathy*.py` figures. Imports the harness from
  `mitigations/` via `sys.path`.
- `reproduce/` — standalone scripts that re-derive the ELEPHANT PAS
  action-endorsement judge against the paper's own released labels
  (`reproduce_pas_judge_*.py`, `sanity_rejudge_paper_pas.py`, and the
  `for_myra/` colleague export). OSF paths are anchored to the repo root,
  so these run from any working directory.
- `legacy/` — superseded one-off scripts kept for reference. Not part of the
  active pipeline.
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

## Post-training: iterative DPO+LoRA with a simulated-stakeholder veto

The post-training pipeline fine-tunes a policy (Qwen3-8B) to be less
sycophantic by turning a *simulated-stakeholder veto* into DPO preference
pairs and iterating. A second, **frozen** copy of the model plays the role of
the stakeholders and the judge, so no human labels and no external API are
needed for the training signal.

### The method

For each of `K` outer iterations (default 3):

1. **Identify stakeholders.** For every training prompt the frozen "simulated
   stakeholders" pipeline (`our_mitigation_v2`, reused from `mitigations/`)
   produces a small set of stakeholders (≈3 per prompt) and elicits each
   one's *concerns* in their own voice. This is policy-independent, so it is
   computed once and cached across all iterations.
2. **Sample candidates.** Draw `N` responses per prompt (default 10) from the
   *current* policy. With mixed sampling the `N` are split across sources —
   bare post, post + "Be critical.", and stakeholder-voiced prompts (default
   2 / 2 / 6) — to diversify the candidate pool.
3. **Veto.** Each stakeholder judges each candidate and *vetoes* it if it
   conflicts with that stakeholder's concerns (the `P4` veto prompt). A
   response is vetoed if **any** stakeholder vetoes it.
4. **Build preference pairs.** Every vetoed response (rejected) is paired
   against every surviving response (chosen). Optionally (`--self-preference`)
   the policy is also asked to pick the better of two non-vetoed responses,
   adding those pairs on top.
5. **Train.** Run one DPO epoch (`trl.DPOTrainer`) on a LoRA adapter over the
   iteration's pairs, then snapshot the adapter to `adapter_iter_<k>` so every
   iteration is independently evaluable.
6. **Repeat** — the next iteration re-samples from the now-updated policy.

Because the veto signal marks the responses stakeholders reject (the
sycophantic ones) as *rejected*, DPO pushes the policy away from them.

### Infrastructure

- **Two models, two GPUs.** The trainable policy (Qwen3-8B + LoRA) lives on
  one GPU; the frozen sim judge (same base) is served by a **vLLM** server on
  a second GPU so the per-(stakeholder, response) veto calls fan out
  concurrently. With `--policy-on-vllm`, candidate sampling also runs on the
  shared vLLM engine (base + the current LoRA, hot-loaded each iteration).
- **Per-stage batched build loop.** `core/trainer.py::_build_iteration`
  processes prompts in chunks and runs each stage — identify → sample → veto →
  self-preference — **breadth-first across the whole chunk**, fanning the LLM
  calls out concurrently (`--chunk-size`, `--stage-concurrency`) so vLLM's
  continuous batching saturates the GPU. This is ~10× faster than the former
  per-prompt loop while producing identical training data (the judges are
  greedy/deterministic and the reduce runs in global prompt order; see
  `tests/test_batched_pipeline.py`).
- **Resumable.** Adapters and the per-prompt `responses_iter_<k>.jsonl` are
  written incrementally. A run resumes from a snapshot with
  `--resume-adapter <dir>/adapter_iter_<k> --start-iteration <k>`.

### `training_mitigations/` layout

- `train_dpo_lora.py` — entry point / CLI (`--env OEQ|PAS`). Stays at the
  package root; the cluster jobs invoke it by this path.
- `core/` — the runtime library: `trainer.py` (the iterative DPO loop),
  `simulated_stakeholders.py` (identify/concerns/veto + the batched
  `prime_stakeholder_cache` / `veto_responses_batch`), `vllm_sim.py` (vLLM
  server + thread-safe call/sampler closures), `llama_local.py`,
  `mem_probe.py`, `weak_pipeline.py`, `concurrency.py` (the `concurrent_map`
  fan-out helper).
- `eval/` — scoring trained adapters on held-out prompts
  (`eval_endorsement_heldout.py`) and related judge/aggregation scripts.
- `data_build/` — dataset/input builders. `plots/` — training-summary
  figures. `labeling/` — human-annotation tooling. `tests/` — unit tests.
  `legacy/` — one-off checks kept for reference. `slurm_scripts/` — sbatch
  jobs. Training outputs land in `runs/<run-name>/` (gitignored).

### Running it

The runs are GPU-heavy and launched via SLURM (2 GPUs each). Set `HF_TOKEN`
(for the Qwen3-8B download/cache) and submit one of the sbatch scripts, e.g.

```
sbatch training_mitigations/slurm_scripts/run_train_oeq_n1000_mixed_P4.sbatch
```

Or run the entry point directly (see `--help` for the full flag set):

```
HF_TOKEN=... /next/u/stephhk/miniconda3/envs/llm/bin/python \
  training_mitigations/train_dpo_lora.py \
  --env OEQ --output-dir training_mitigations/runs/dpo_oeq_demo \
  --policy-on-vllm --veto-prompt P4 \
  --device cuda:0 --sim-device cuda:1 --sim-backend vllm \
  --num-iterations 3 --train-n 1000 --test-n 500 --n-samples 10 \
  --mixed-sampling --n-source-model 2 --n-source-critical 2 \
  --chunk-size 64 --stage-concurrency 128 \
  --beta 0.1 --learning-rate 5e-5 --lora-r 16 --lora-alpha 32
```

Evaluate the per-iteration adapters on the held-out split with
`training_mitigations/eval/eval_endorsement_heldout.py`, and plot training
diagnostics with the scripts in `training_mitigations/plots/`.

## Required data

Download the OSF archive from

  https://osf.io/smvw7/?view_only=ad71a7201c71477d921000c90c565da7

and extract it as `osfstorage-archive/` at the repo root. Both pipelines read
OEQ prompts from
`osfstorage-archive/Data/OEQ_results/endorse4_elephant_oeq3k_human.csv`
and PAS prompts from
`osfstorage-archive/Data/PAS_results/endorse4_elephant_oeq_pas_claude.csv`;
without the archive in place, OEQ and PAS runs will fail at CSV-read time.
AITA runs only need `data/aita_clean.csv`, which is checked in.

`osfstorage-archive/Code/CLAUDE.md` documents the three-study layout of the
OSF bundle (AITA / Study 2 / Study 3) and the judge script
`evaluate_action_endorsement.py` that produced the `endorse_*` columns
visible in the OSF CSVs.

## Environment

Use the project Python interpreter:

```
/next/u/stephhk/miniconda3/envs/llm/bin/python
```

The base environment lacks `matplotlib` / `anthropic` (and the training deps
`torch` / `trl` / `peft` / `vllm`); use the `llm` env above. Set the keys each
pipeline needs in the shell first:

- Single-turn sweeps and the endorsement/empathy judges: `ANTHROPIC_API_KEY`
  and/or `OPENROUTER_API_KEY`.
- Post-training (`training_mitigations/`): `HF_TOKEN` for the Qwen3-8B
  weights. The veto signal runs entirely on the local vLLM sim, so the
  training loop itself needs no external API key.

## Typical runs

**Single-turn sweep** on AITA with Claude as the generator and GPT-5.4 as the
action-endorsement judge:

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

**Post-training run** — see "Running it" above; outputs land in
`training_mitigations/runs/<run-name>/` with per-iteration adapters
(`adapter_iter_<k>`), the raw `responses_iter_<k>.jsonl`, and `iter_stats.json`.
