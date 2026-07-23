"""Fine-tune a local HuggingFace model via DPO + LoRA against preference
pairs derived from the `SimulatedStakeholders` veto signal, on a single
dataset env (OEQ, PAS, AITA, or AITA-NTA-FLIP).

This is the shared training entry point. The
`run_train_<env>.sbatch` wrappers invoke it with env-specific defaults.

What it does:
  1. Load the dataset for `--env` from disk.
  2. Hold out `--test-n` prompts as the eval split. OEQ/PAS/AITA-NTA-FLIP use
     the harness's `sample_uniform`; AITA uses a balanced 50/50 YTA/NTA split.
     For AITA-NTA-FLIP, `--test-n`/`--train-n` count (original, flipped) PAIRS.
     Their ids are written to `<output_dir>/test_ids.json` so a downstream
     eval script can
     reproduce the same split.
  3. From the disjoint complement, sample `--train-n` prompts as the DPO
     training prompts. Train ids → `<output_dir>/train_ids.json`.
  4. Stand up the policy and the sim judge:
       - `policy` on `--device`: a trainable HF model (LoRA-adapted),
         default `Qwen/Qwen3-8B` (`--model-id`). ALWAYS local — DPO
         needs token-level reference log-probs from its own base
         weights (trl computes them by disabling the LoRA adapter), and
         no API can serve that.
       - the sim judge: frozen; never updated. Served by a vLLM server
         subprocess on `--sim-device` by default (`--sim-backend vllm`),
         whose continuous batching lets the per-(stakeholder, response)
         veto calls run concurrently (`--veto-concurrency`) instead of
         one batch-1 HF generate at a time; `--sim-backend hf` keeps
         the original in-process frozen HF copy as a fallback; and
         `--sim-backend openrouter` runs the sim on an API model
         (`--sim-model`, e.g. openai/gpt-5-mini) with no sim GPU and no
         local weights.
     The policy is what gets sampled from each iteration and what DPO
     updates. The sim judge drives every LLM call inside the
     stakeholder pipeline (identify / filter / concerns / veto) so
     the veto judge's behavior does not drift with the policy.
     `--policy-on-vllm` puts the policy on a local vLLM engine
     independently of where the sim lives, so it composes with every
     sim backend except `hf` (which would contend for the same GPU).
  5. Construct `SimulatedStakeholders` wired to sim_base;
     `use_weak_model=True` routes the identify/filter/concerns calls
     through `weak_pipeline.py`.
  6. Construct `Trainer(sim_stake, policy, tokenizer, n_samples,
     temperature, max_new_tokens)` and call `trainer.train(train_prompts,
     output_dir, num_iterations=K, ...)`, which:
       - wraps the policy with a LoRA adapter once;
       - for each of K iterations: samples N candidate responses per
         prompt from the current policy, vetoes via the (frozen) sim,
         cross-pairs vetoed × non-vetoed into a `datasets.Dataset`,
         logs the count of all-vetoed prompts, runs one DPO epoch on
         the iteration's fresh dataset;
       - all-vetoed prompts contribute 0 pairs that iteration; the
         next iteration re-samples them from the now-updated policy
         and may produce non-vetoed responses that contribute pairs;
       - snapshots the adapter to `output_dir/adapter_iter_<k>` after
         each iteration so every iteration is independently loadable
         for evaluation;
       - saves the final LoRA adapter to `output_dir`.

Per-env user prompt format used for sampling AND for the DPO prompt
column (so DPO sees the same surface the responses were sampled
against): the bare post_text for every env (matches the harness's
`PROMPT_TEMPLATE_OEQ = "{post}"`). AITA / AITA-NTA-FLIP no longer carry
any output-format nudge — the YTA/NTA verdict is read from the response
prose by an eval-time gpt-5-mini classifier, not self-emitted.
AITA-NTA-FLIP expands each pair into two independent train prompts
(original + flipped).

The script does not call any external API. Both model copies run
locally on the configured GPUs.
"""

import argparse
import json
import os
import re
import sys

import torch

# Make `mitigations/` (the harness) and the local `training_mitigations/`
# importable so `our_mitigation*`, `core.*`, `weak_pipeline`,
# `simulated_stakeholders`, and `trainer` all resolve.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, "mitigations"))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "core"))

import our_mitigation
import our_mitigation_v2
import weak_pipeline
import gemini_pipeline

from explore_mitigations import DEFAULT_MAX_TOKENS, DEFAULT_SEED

# Shared dataset / split helpers (also used by train_grpo_lora.py) so a DPO run
# and a GRPO run on the same env/seed/split train and eval on identical prompts.
from train_data import (resolve_csv as _resolve_csv,
                        load_env_data as _load_data,
                        split_train_test as _split_train_test,
                        build_train_prompts as _build_train_prompts,
                        subset_in_id_order as _subset_in_id_order,
                        write_ids as _write_ids,
                        detect_resume_point as _detect_resume_point)

from cost import CostTracker
from llama_local import DEFAULT_HF_MODEL_ID, load_llama, make_call_llm
from mem_probe import start_mem_sampler
from openrouter_sim import make_openrouter_call_llm
from gemini_sim import make_gemini_call_llm
from simulated_stakeholders import (SimulatedStakeholders, VETO_PROMPTS,
                                    ENDORSE_CLASSIFY_PROMPTS,
                                    ASSISTANT_ROLE_VETO_PROMPT,
                                    ASSISTANT_ROLE_VETO_PROMPT_QWEN)
from trainer import (Trainer, CRITICAL_SUFFIX, NEUTRAL_SUFFIX,
                     STAKEHOLDER_RESPONSE_PROMPT,
                     STAKEHOLDER_RESPONSE_PROMPT_VOICE)
from vllm_sim import (launch_vllm_server, make_vllm_call_llm, wait_for_vllm_server,
                      make_vllm_policy_sampler, load_lora_adapter, unload_lora_adapter)


# Same tolerant stakeholder parser used in `test_v2_llama.py`. The harness
# default `parse_stakeholders` asserts strictly fewer than `MAX_STAKEHOLDERS
# =10` items; Qwen3-8B occasionally emits exactly 10 (or theme-shaped
# strings) on OEQ-style prompts, which would abort the entire train run.
# Truncate at a higher local cap instead.
_LOCAL_MAX_STAKEHOLDERS = 15


def _tolerant_parse_stakeholders(text, min_count=None, max_count=None):
    """Drop the strict-cap assert from `our_mitigation.parse_stakeholders`
    so a single out-of-spec model emission doesn't kill the whole train
    run. Truncates at `_LOCAL_MAX_STAKEHOLDERS`. Behavior is otherwise
    identical: strip bullet/number prefixes and whitespace, keep non-
    empty lines.
    """
    items = []
    for raw in (text or "").splitlines():
        s = raw.strip()
        s = re.sub(r"^(?:[\-\*•]|\d+[\.\)])\s*", "", s).strip()
        s = s.strip("*_`").strip()
        if s:
            items.append(s)
    if len(items) > _LOCAL_MAX_STAKEHOLDERS:
        items = items[:_LOCAL_MAX_STAKEHOLDERS]
    return items


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--env", required=True,
        choices=("OEQ", "PAS", "AITA", "AITA-NTA-FLIP"),
        help="dataset env to train on. AITA splits balanced 50/50 on "
             "is_asshole; AITA-NTA-FLIP counts (original,flipped) PAIRS for "
             "--train-n/--test-n and expands each pair to two train prompts.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="directory the LoRA adapter and split-id files are written to",
    )
    parser.add_argument(
        "--model-id", default=DEFAULT_HF_MODEL_ID,
        help=f"HF model id of the policy under training (default: {DEFAULT_HF_MODEL_ID})",
    )
    parser.add_argument(
        "--cache-dir", default=os.environ.get("HF_HOME"),
        help="HF cache dir; defaults to $HF_HOME if set, else HF's default",
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="torch device for the trainable policy model (default: cuda:0). "
             "Ignored for placement when --policy-device-map is set (the model "
             "is then sharded across the listed GPUs); still used as the "
             "input/main device, which must be the lowest shard index.",
    )
    parser.add_argument(
        "--policy-device-map", default=None,
        help="comma-separated CUDA indices to pipeline-shard the trainable "
             "policy across (e.g. '0,2'). When set, the policy weights are "
             "split across these GPUs via a balanced device_map (the ~7GB "
             "full-vocab logits/CE spike then sits on only one card), which "
             "lets long-sequence OEQ distill runs keep --max-length 3072 on "
             "48GB A6000s instead of OOMing a single card. The --sim-device "
             "GPU (vLLM server) must NOT be in this list. Default: None "
             "(single-device training on --device).",
    )
    parser.add_argument(
        "--policy-shard-mem", default="40GiB",
        help="per-GPU memory cap for each card in --policy-device-map "
             "(default: 40GiB). Excluded GPUs are pinned to 0GiB so the "
             "balanced map only uses the listed cards.",
    )
    parser.add_argument(
        "--sim-device", default="cuda:1",
        help="GPU for the frozen sim judge (default: cuda:1). With "
             "--sim-backend hf this is a torch device the sim_base model is "
             "loaded onto (set equal to --device to share one GPU). With "
             "--sim-backend vllm the GPU is given exclusively to the vLLM "
             "server subprocess and must NOT be shared with --device.",
    )
    parser.add_argument(
        "--sim-backend", default="vllm", choices=("vllm", "hf", "openrouter", "gemini"),
        help="how the frozen sim judge is served (default: vllm). 'vllm' "
             "launches a vLLM OpenAI-compatible server subprocess on "
             "--sim-device and fans veto calls out concurrently "
             "(--veto-concurrency); 'hf' is the original in-process "
             "HF-transformers path (sequential, batch-1 — slow; kept as a "
             "fallback); 'openrouter' runs the WHOLE stakeholder pipeline "
             "(identify/filter/concerns + veto panel) on the API model given "
             "by --sim-model instead of a local copy of --model-id, needs no "
             "sim GPU, and requires OPENROUTER_API_KEY; 'gemini' does the same "
             "on the DIRECT Google Gemini API (GEMINI_API_KEY, not OpenRouter) "
             "and should be paired with --prompt-variant gemini. The policy and "
             "the DPO reference log-probs stay local under every backend.",
    )
    parser.add_argument(
        "--sim-model", default="openai/gpt-5-mini",
        help="model id for the frozen sim under an API --sim-backend. "
             "OpenRouter style for 'openrouter' (default: openai/gpt-5-mini); "
             "plain Gemini id for 'gemini' (e.g. gemini-3.1-flash-lite — NOT "
             "'google/...'). Ignored by the local backends, which serve "
             "--model-id.",
    )
    parser.add_argument(
        "--prompt-variant", default=None, choices=("gemini",),
        help="override the identify/filter/concerns PROMPTS. Default (unset) "
             "keeps weak_pipeline, i.e. every existing arm is byte-identical. "
             "'gemini' uses core/gemini_pipeline.py, retuned because Gemini "
             "obeys the weak identify prompt literally (~2.9 candidates vs "
             "Qwen's 9.8), which starved the filter into fabricating "
             "stakeholders to reach its 3-item target.",
    )
    parser.add_argument(
        "--sim-concurrency", type=int, default=16,
        help="max in-flight API sim calls under --sim-backend openrouter/gemini "
             "(default: 16). Caps EVERY sim fan-out, overriding "
             "--stage-concurrency (default 128), which is tuned to saturate a "
             "local vLLM server and will rate-limit an API.",
    )
    parser.add_argument(
        "--sim-thinking-budget", type=int, default=0,
        help="Gemini thinking token budget for the sim (default: 0 = OFF). "
             "Google bills thinking at the full OUTPUT rate, and ~70%% of the "
             "run's calls answer with a single word (VETO/ACCEPT), so leaving "
             "it on multiplies the bill for no benefit. -1 leaves the model "
             "default.",
    )
    parser.add_argument(
        "--sim-max-retries", type=int, default=5,
        help="retry attempts for TRANSIENT Gemini sim faults (429/5xx/timeout) "
             "(default: 5). Safety refusals are never retried — they are "
             "deterministic; see --sim-blocked-action.",
    )
    parser.add_argument(
        "--sim-backoff-max", type=float, default=300.0,
        help="ceiling (seconds) on the Gemini sim's exponential retry backoff "
             "(default: 300 = 5 min). Jittered, so a --sim-concurrency-wide "
             "fan-out does not re-synchronize into the spike that caused the "
             "rate limit.",
    )
    parser.add_argument(
        "--sim-reasoning", default="minimal",
        help="OpenRouter reasoning effort for the sim (default: minimal, "
             "matching the v5 judge); 'none' to omit the field for "
             "non-reasoning models that reject it.",
    )
    parser.add_argument(
        "--sim-temp", type=float, default=0.0,
        help="OpenRouter sim decoding temperature (default: 0.0). The frozen "
             "sim is greedy on the local backends too, and several equivalence "
             "claims (resume-without-re-vetoing, batched==per-prompt, "
             "sequential==concurrent veto) hold only for a deterministic "
             "judge. Negative omits the field; expect a drifting veto signal.",
    )
    parser.add_argument(
        "--sim-max-tokens-floor", type=int, default=4000,
        help="raise any sim max_tokens request to at least this (default: "
             "4000, matching --judge-max-tokens). A reasoning model spends "
             "this budget on hidden reasoning FIRST, so the harness's "
             "128-token veto ceiling would return empty content. This is a "
             "cap, not a target — it does not raise cost on short replies.",
    )
    parser.add_argument(
        "--sim-price-in", type=float, default=0.25,
        help="OpenRouter sim input price $/1M tokens (default: 0.25, "
             "gpt-5-mini). Used only for --sim-max-cost accounting.",
    )
    parser.add_argument(
        "--sim-price-out", type=float, default=2.00,
        help="OpenRouter sim output price $/1M tokens (default: 2.00).",
    )
    parser.add_argument(
        "--sim-max-cost", type=float, default=None,
        help="hard cap (USD) on cumulative OpenRouter SIM spend across the run "
             "(default: none). Separate from --judge-max-cost. On breach the "
             "run aborts (fail-fast); the last completed iteration's adapter "
             "snapshot remains loadable. The sim is the dominant call volume "
             "of the run — set this.",
    )
    parser.add_argument(
        "--rewrite-on-sim", action="store_true",
        help="route the --perspective-shift-sampling REWRITE call through the "
             "sim backend instead of the policy. The rewriter becomes frozen "
             "(and, under --sim-backend openrouter, closer to ELEPHANT, which "
             "used GPT-4o for the rewrite). The answers drawn from the rewrite "
             "stay on-policy. NOTE: this changes what the perspshift arm "
             "means — results are not comparable to runs without it.",
    )
    parser.add_argument(
        "--veto-concurrency", type=int, default=16,
        help="max in-flight (response, stakeholder) veto calls when "
             "--sim-backend vllm (default: 16). Ignored (forced to 1) for "
             "--sim-backend hf, which cannot batch concurrent calls.",
    )
    parser.add_argument(
        "--vllm-gpu-mem-util", type=float, default=0.90,
        help="vLLM gpu_memory_utilization on the sim GPU (default: 0.90)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=64,
        help="prompts processed per chunk in the per-stage batched build "
             "loop (default: 64). Each chunk runs stages identify/sample/veto/"
             "preference breadth-first across its prompts.",
    )
    parser.add_argument(
        "--stage-concurrency", type=int, default=128,
        help="max in-flight LLM calls per stage in the batched build loop "
             "(default: 128) — the width of the fan-out the vLLM server "
             "continuous-batches. This is the knob that matters for the "
             "batched pipeline; --veto-concurrency only affects the legacy "
             "per-prompt veto path.",
    )
    parser.add_argument(
        "--veto-prompt", default="baseline", choices=tuple(VETO_PROMPTS),
        help="which stakeholder veto prompt to use (default: baseline). "
             "'P4' = proposal 1 (reframe to 'would you be okay with') + "
             "proposal 2 (expanded examples). 'P5' = action-endorsement framing "
             "(mirrors the v5 endorse judge): veto only if the response affirms "
             "the narrator's ACTION and you think it should not be endorsed. "
             "'P6' = BIDIRECTIONAL (stance-mismatch): each voter states its own "
             "preferred stance and vetoes the response on disagreement (endorse-"
             "when-they'd-challenge OR challenge-when-they'd-endorse), so it can "
             "demote over-critical responses too, not just affirming ones. "
             "See simulated_stakeholders.py. Ignored when --judge is not "
             "'stakeholder'.",
    )
    parser.add_argument(
        "--user-stakeholder", action="store_true",
        help="add a 4th 'user'/narrator voice to the veto panel (P6 only). It "
             "votes on the veto via VETO_PROMPT_USER but is NOT sampled for "
             "candidate responses (not added to the generation entries). Intended "
             "to lift the panel's endorsement toward the target by demoting "
             "over-critical responses. NOTE: as a stance veto this is a blunt "
             "pro-endorsement lever (demotes challenges wholesale); watch for "
             "re-introduced sycophancy.",
    )
    parser.add_argument(
        "--judge", default="stakeholder",
        choices=("stakeholder", "neutrality-v5"),
        help="which veto judge produces the (chosen, rejected) split (default: "
             "stakeholder). 'stakeholder' = the SimulatedStakeholders veto "
             "(--veto-prompt). 'neutrality-v5' = the NEUTRALITY BASELINE: the v5 "
             "gpt-5-mini endorse judge vetoes every response it does NOT label 3 "
             "(neutral), so chosen=neutral, rejected=challenge/affirm. The "
             "neutrality judge needs OPENROUTER_API_KEY, has no stakeholder "
             "pipeline (so --mixed-sampling is unsupported), and does not require "
             "a local sim model — the sim vLLM server is only launched when "
             "--policy-on-vllm is set.",
    )
    parser.add_argument(
        "--judge-model", default="openai/gpt-5-mini",
        help="judge model id for --judge neutrality-v5 (default: "
             "openai/gpt-5-mini, the v5 eval judge).",
    )
    parser.add_argument(
        "--judge-reasoning", default="minimal",
        help="OpenRouter reasoning effort for the neutrality judge (default: "
             "minimal); 'none' to omit.",
    )
    parser.add_argument(
        "--judge-max-tokens", type=int, default=4000,
        help="judge output token budget incl. reasoning for the neutrality "
             "judge (default: 4000, matching the v5 eval runner).",
    )
    parser.add_argument(
        "--judge-temp", type=float, default=0.0,
        help="neutrality judge sampling temperature (default: 0.0 for "
             "deterministic labels; negative omits temperature).",
    )
    parser.add_argument(
        "--judge-concurrency", type=int, default=16,
        help="max in-flight OpenRouter judge calls for --judge neutrality-v5 "
             "(default: 16). Bounds the judge's fan-out independently of "
             "--stage-concurrency (which is tuned for the local vLLM server).",
    )
    parser.add_argument(
        "--judge-price-in", type=float, default=0.25,
        help="neutrality judge input price $/1M tokens (default: 0.25, "
             "gpt-5-mini via OpenRouter). Used only for the spend cap.",
    )
    parser.add_argument(
        "--judge-price-out", type=float, default=2.00,
        help="neutrality judge output price $/1M tokens (default: 2.00).",
    )
    parser.add_argument(
        "--judge-max-cost", type=float, default=None,
        help="hard cap (USD) on cumulative v5-judge spend across the whole run "
             "(default: none). On breach the run aborts (fail-fast); the last "
             "completed iteration's adapter snapshot remains loadable.",
    )
    parser.add_argument(
        "--policy-on-vllm", action="store_true",
        help="sample candidate responses + greedy preference labels on the "
             "shared vLLM engine (base + current LoRA) instead of HF "
             "model.generate. Requires --sim-backend vllm. The engine is "
             "launched with --enable-lora and the policy adapter is hot-loaded "
             "each iteration. Validated to HF logprob parity (build_vllm_parity_check.py).",
    )
    parser.add_argument(
        "--resume-adapter", default=None,
        help="path to a saved adapter (e.g. <output_dir>/adapter_iter_1) to "
             "resume training FROM; combined with --start-iteration. Reuses the "
             "existing <output_dir>/train_ids.json split instead of resampling.",
    )
    parser.add_argument(
        "--start-iteration", type=int, default=0,
        help="0-based outer iteration to start at when resuming (e.g. 1 means "
             "iteration 1 is already done; begin at the 2nd iteration). "
             "Default 0 (fresh run).",
    )
    parser.add_argument(
        "--allow-preempt", action="store_true",
        help="make the run idempotently restartable for pause/resume under "
             "SLURM preemption. On startup, auto-detect the resume point from "
             "--output-dir: the highest FINISHED iteration (adapter_iter_<k> "
             "with a .complete sentinel) sets --start-iteration and "
             "--resume-adapter, and a partial responses_iter_<k+1>.jsonl is "
             "reloaded so that interrupted iteration's already-sampled prompts "
             "are not re-sampled. Resubmit the SAME command after a kill to "
             "continue; a cold start (empty output_dir) runs normally. Mutually "
             "exclusive with manual --resume-adapter / --start-iteration.",
    )
    parser.add_argument(
        "--num-iterations", type=int, default=3,
        help="number of outer training iterations. Each iteration "
             "re-samples all training prompts from the current policy, "
             "rebuilds the (chosen, rejected) preference dataset, and "
             "runs one DPO epoch on it. Prompts whose responses are all "
             "vetoed in iter K contribute 0 pairs that iter; iter K+1 "
             "re-samples them with the updated policy. Default: 3.",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"split / sampler seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--train-n", type=int, required=True,
        help="number of training prompts (IGNORED under --use-final-data)",
    )
    parser.add_argument(
        "--test-n", type=int, required=True,
        help="number of held-out test prompts (IGNORED under --use-final-data)",
    )
    parser.add_argument(
        "--use-final-data", action="store_true",
        help="train/eval on the frozen final_training_eval_data/ split for "
             "--env instead of the seed-derived sample; --train-n/--test-n/"
             "--seed are ignored (sizes come from the id files).",
    )
    parser.add_argument(
        "--n-samples", type=int, default=10,
        help="N — candidate responses sampled per training prompt (default: 10)",
    )
    parser.add_argument(
        "--mixed-sampling", action="store_true",
        help="variation 1: split the N candidates per prompt across sources "
             "instead of all from the bare post — --n-source-model from the "
             "bare post, --n-source-critical from the post + 'Be critical.', "
             "and the remaining N budget round-robin across the prompt's "
             "identified stakeholders (each voiced with its concerns). The "
             "DPO prompt column stays the bare post.",
    )
    parser.add_argument(
        "--n-source-model", type=int, default=2,
        help="mixed sampling: candidates drawn from the bare post (default: 2)",
    )
    parser.add_argument(
        "--n-source-critical", type=int, default=2,
        help="mixed sampling: candidates drawn from the post + 'Be critical.' "
             "(default: 2). The stakeholder budget is n_samples minus these two.",
    )
    parser.add_argument(
        "--stakeholder-voice", action="store_true",
        help="mixed sampling: generate stakeholder candidates with the "
             "second-person VOICE-constrained prompt "
             "(STAKEHOLDER_RESPONSE_PROMPT_VOICE) instead of the first-person "
             "role-play prompt — surfaces the same stakeholder perspective as "
             "direct advice to the narrator, dropping the role-play surface from "
             "the chosen DPO exemplars (the mixed_P5_nocrit_distill_gp arm). The "
             "DPO prompt column stays the bare post.",
    )
    parser.add_argument(
        "--frozen-stakeholder", action="store_true",
        help="mixed sampling: generate the stakeholder candidates from the "
             "FROZEN BASE model (a fixed teacher) instead of the evolving "
             "policy — removes the on-policy feedback loop in the stakeholder "
             "exemplars (the mixed_P5_nocrit_distill_gp_frozenstake arm). Only "
             "the stakeholder source is frozen; the model/critical sources stay "
             "on-policy. On vLLM this routes to the base model_id already served "
             "on the engine; on HF it disables the PEFT adapter for the call.",
    )
    parser.add_argument(
        "--perspective-shift-sampling", action="store_true",
        help="mixed sampling: replace the 'Be critical.' critical source with "
             "the ELEPHANT (Cheng et al. 2025, App G.2) perspective-shift "
             "rewrite-then-answer source. The post is rewritten first- to "
             "third-person (greedy, via the policy) and the --n-source-critical "
             "samples are drawn from the REWRITTEN post; the DPO prompt column "
             "stays the bare first-person post. Requires --mixed-sampling; "
             "mutually exclusive with --neutral-mixed-sampling. This is the "
             "mixed_P5_perspective_shift arm.",
    )
    parser.add_argument(
        "--future-user-sampling", action="store_true",
        help="mixed sampling: replace the 'Be critical.' critical source with "
             "the FUTURE-USER voice — the --n-source-critical samples are drawn "
             "on-policy from the narrator's own future self, who must live with "
             "the long-term consequences of the present decision "
             "(FUTURE_USER_RESPONSE_PROMPT). The DPO prompt column stays the "
             "bare first-person post. Requires --mixed-sampling with "
             "--n-source-critical > 0; mutually exclusive with "
             "--perspective-shift-sampling and --neutral-mixed-sampling. This is "
             "the mixed_P5_futureuser arm.",
    )
    parser.add_argument(
        "--addressee-filter", action="store_true",
        help="veto pipeline: before the veto panel, drop any candidate that "
             "critiques the FORM of the user's message (grammar, wording, "
             "structure, a 'more polished version') instead of engaging their "
             "situation (ADDRESSEE_FILTER_PROMPT, on the veto backend). A "
             "filtered response becomes neither chosen nor rejected. Composes "
             "with any --veto-prompt. This is the mixed_P5_addrfilter arm.",
    )
    parser.add_argument(
        "--assistant-role-veto", action="store_true",
        help="veto pipeline: alongside the veto panel, classify every candidate "
             "with ASSISTANT_ROLE_VETO_PROMPT and AUTO-REJECT any that fails to "
             "answer the user as an AI assistant — critiquing the message as a "
             "text, speaking from inside the user's story ('how I feel'), or "
             "discussing the user in the third person. Unlike --addressee-filter "
             "(which DROPS), a flagged response becomes a REJECTED exemplar, "
             "paired with a randomly chosen never-vetoed response. At most "
             "AUTO_REJECT_MAX_PAIRS (3) such pairs per prompt, on a track "
             "independent of --preference-max-pairs. Mutually exclusive with "
             "--future-user-sampling.",
    )
    parser.add_argument(
        "--rewrite-max-tokens", type=int, default=1536,
        help="--perspective-shift-sampling: max new tokens for the greedy "
             "first->third-person rewrite (default: 1536, matching the eval "
             "perspshift baseline; long narratives need room or the rewrite "
             "truncates).",
    )
    parser.add_argument(
        "--perspshift-assess", action="store_true",
        help="--perspective-shift-sampling: append the paper's OEQ addendum "
             "'Assess this narrative.' after the third-person rewrite (the "
             "paper adds it for OEQ). Off by default.",
    )
    parser.add_argument(
        "--neutral-mixed-sampling", action="store_true",
        help="neutral-mixed (steer-only) sampling for the --judge neutrality-v5 "
             "baseline: split the N candidates between the bare post "
             "(--n-source-model) and the post + 'Be neutral...' (the rest), "
             "with NO stakeholder source. Seeds judge-passing NEUTRAL 'chosen' "
             "responses on the many prompts where the bare policy rarely emits "
             "one. Defaults to a half/half split unless --n-source-model + "
             "--n-source-critical already sum to N. The DPO prompt column stays "
             "the bare post. Requires --judge neutrality-v5; mutually exclusive "
             "with --mixed-sampling.",
    )
    parser.add_argument(
        "--self-preference", action="store_true",
        help="variation 2: for every unordered pair of NON-vetoed responses, "
             "ask the policy (the model under training) to pick the better "
             "advice (post in context) and add the resulting preference pair "
             "to the DPO dataset, on top of the veto-based pairs.",
    )
    parser.add_argument(
        "--preference-max-tokens", type=int, default=128,
        help="max new tokens per self-preference label call (default: 128)",
    )
    parser.add_argument(
        "--preference-max-pairs", type=int, default=0,
        help="per-prompt cap on self-preference pairs (0 = no cap, i.e. all "
             "C(k,2) pairs over k non-vetoed responses). Default: 0.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"max new tokens for stakeholder / veto LLM calls (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--sample-max-tokens", type=int, default=512,
        help="max new tokens per sampled candidate response (default: 512)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="sampling temperature for candidate generation (default: 1.0)",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.95,
        help="nucleus sampling top-p for candidate generation (default: 0.95)",
    )
    parser.add_argument(
        "--beta", type=float, default=0.1,
        help="DPO beta (default: 0.1)",
    )
    parser.add_argument(
        "--rpo-alpha", type=float, default=None,
        help="RPO / 'distill' variant. If set (> 0), add an SFT/NLL term on the "
             "CHOSEN (non-vetoed) responses to the DPO loss: total = DPO + "
             "rpo_alpha * mean-token CE(chosen). Anchors log p(chosen) UP in "
             "absolute terms instead of only the chosen-rejected margin. "
             "Implemented via trl's multi-loss loss_type=['sigmoid','sft'], "
             "loss_weights=[1.0, rpo_alpha]. Default: None (plain DPO, no SFT "
             "term). 1.0 is the canonical RPO value.",
    )
    parser.add_argument(
        "--distill-self-only", action="store_true",
        help="TARGETED distill. Mask the RPO/distill SFT term (requires "
             "--rpo-alpha) so it imitates ONLY the responses sampled on-policy "
             "from the model itself (mixed-sampling source 'model') that "
             "survived the veto, at most --distill-max-per-prompt distinct per "
             "prompt. The DPO cross-pairs are unchanged (all sources still "
             "become chosen/rejected). Isolates the distill target from the "
             "critical-skewed stakeholder/perspshift/frozen-base exemplars.",
    )
    parser.add_argument(
        "--distill-max-per-prompt", type=int, default=4,
        help="with --distill-self-only, the max number of distinct on-policy "
             "'model'-source non-vetoed responses distilled per prompt "
             "(default: 4).",
    )
    parser.add_argument(
        "--dpo-save-steps", type=int, default=0,
        help="if > 0, checkpoint the DPO train phase every this many optimizer "
             "steps and resume it from the latest checkpoint after a kill. "
             "Combined with the per-prompt sampling checkpoints, this makes the "
             "whole iteration resumable, so a short (e.g. preempt 4h) window is "
             "enough to make progress. 0 (default) keeps the train step atomic "
             "(fine for long windows). 25 is a reasonable value.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-5,
        help="DPO learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--num-train-epochs", type=float, default=1.0,
        help="DPO epochs over the preference dataset (default: 1.0)",
    )
    parser.add_argument(
        "--per-device-batch-size", type=int, default=1,
        help="DPO per-device batch size (default: 1)",
    )
    parser.add_argument(
        "--grad-accum", type=int, default=8,
        help="DPO gradient accumulation steps (default: 8)",
    )
    parser.add_argument(
        "--precompute-ref-batch-size", type=int, default=8,
        help="batch size for trl's reference log-prob precompute pass "
             "(default: 8). The pass is no-grad inference, so this can be "
             "much larger than --per-device-batch-size; at batch 1 it took "
             "2h+ per iteration on the n1000 runs.",
    )
    parser.add_argument(
        "--max-length", type=int, default=3072,
        help="DPO total prompt+completion sequence length cap (default: 3072). "
             "trl>=1.x has no separate prompt cap; sequences over this are "
             "truncated per DPOConfig.truncation_mode (keep_start).",
    )
    parser.add_argument(
        "--lora-r", type=int, default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=32,
        help="LoRA alpha (default: 32)",
    )
    parser.add_argument(
        "--lora-dropout", type=float, default=0.05,
        help="LoRA dropout (default: 0.05)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --policy-on-vllm needs a vLLM engine for the POLICY. That engine also
    # serves the sim under --sim-backend vllm, but the sim is free to live
    # elsewhere: 'openrouter' has no local weights at all, so the engine is
    # launched for the policy alone (same as the --judge neutrality-v5 case).
    # 'hf' is still incompatible — it would load a second in-process copy of
    # the model on the GPU the vLLM server has already claimed.
    assert not (args.policy_on_vllm and args.sim_backend == "hf"), (
        "--policy-on-vllm is incompatible with --sim-backend hf (the in-process "
        "HF sim would contend with the vLLM engine for the same GPU); use "
        "--sim-backend vllm to share the engine, or openrouter for an API sim"
    )

    # --prompt-variant gemini retunes identify/filter for Gemini's terse identify
    # step. Pointing it at a local Qwen sim would swap the prompts a measured
    # 9.8-candidate model depends on, silently changing an existing arm's signal.
    assert not (args.prompt_variant == "gemini" and args.sim_backend != "gemini"), (
        f"--prompt-variant gemini is tuned for the Gemini sim but "
        f"--sim-backend is {args.sim_backend!r}; pass --sim-backend gemini "
        f"(or drop --prompt-variant to keep weak_pipeline)"
    )
    # The converse is a footgun, not an error: a Gemini sim on the weak prompts
    # is exactly the configuration measured to fabricate stakeholders on 16/16
    # audited prompts. Warn loudly rather than block — it is a legitimate
    # ablation, just never what you want by default.
    if args.sim_backend == "gemini" and args.prompt_variant is None:
        print("[warn] --sim-backend gemini WITHOUT --prompt-variant gemini: the "
              "weak identify prompt yields ~2.9 candidates on Gemini, which "
              "starves the filter into fabricating stakeholders (measured 16/16 "
              "audited prompts). Pass --prompt-variant gemini unless this is a "
              "deliberate ablation.", flush=True)
    # A Gemini sim id is a plain name; an OpenRouter-style "google/..." id would
    # 404 against the direct API only once the first call fires, hours in.
    if args.sim_backend == "gemini":
        assert "/" not in args.sim_model, (
            f"--sim-model {args.sim_model!r} looks like an OpenRouter id; "
            f"--sim-backend gemini needs the plain Gemini name "
            f"(e.g. 'gemini-3.1-flash-lite')"
        )

    assert not (args.rewrite_on_sim and not args.perspective_shift_sampling), (
        "--rewrite-on-sim only affects the perspective-shift rewrite; pass "
        "--perspective-shift-sampling or drop it"
    )

    # --judge neutrality-v5 builds no sim backend at all (it IS the judge), so
    # there is nothing for the rewrite to route to.
    assert not (args.rewrite_on_sim and args.judge == "neutrality-v5"), (
        "--rewrite-on-sim needs a sim backend, but --judge neutrality-v5 "
        "replaces the stakeholder sim entirely"
    )

    assert not (args.user_stakeholder and args.veto_prompt != "P6"), (
        "--user-stakeholder (4th narrator veto voice) is only wired for the "
        f"bidirectional P6 veto; got --veto-prompt {args.veto_prompt}"
    )

    # --allow-preempt: auto-detect where to continue from --output-dir so the
    # SAME command can be resubmitted after a SLURM preemption / kill. Derives
    # --start-iteration / --resume-adapter from the durable adapter_iter_<k>
    # snapshots, and reloads a partial responses_iter file for the interrupted
    # iteration (resume_partial). Mutually exclusive with the manual flags.
    resume_partial = False
    if args.allow_preempt:
        assert args.resume_adapter is None and args.start_iteration == 0, (
            "--allow-preempt auto-detects the resume point; do not combine it "
            "with manual --resume-adapter / --start-iteration"
        )
        start_iteration, resume_adapter, resume_partial, is_cold = (
            _detect_resume_point(args.output_dir, args.num_iterations))
        args.start_iteration = start_iteration
        args.resume_adapter = resume_adapter
        print(f"[preempt] resume detect: start_iteration={start_iteration} "
              f"resume_adapter={resume_adapter} resume_partial={resume_partial} "
              f"cold_start={is_cold}")

    # Neutrality baseline (--judge neutrality-v5): the veto signal is the v5
    # gpt-5-mini endorse judge (OpenRouter), not the local SimulatedStakeholders
    # pipeline. It has no stakeholder world-model, so the mixed sampler — which
    # voices candidates as identified stakeholders — has nothing to draw from.
    judge_is_neutrality = args.judge == "neutrality-v5"
    if judge_is_neutrality:
        assert not args.mixed_sampling, (
            "--mixed-sampling (stakeholder-voiced) is unsupported with --judge "
            "neutrality-v5 (the neutrality judge has no stakeholders to voice "
            "candidates from); use --neutral-mixed-sampling (steer-only, no "
            "stakeholder source) instead, or sample all N from the bare post."
        )
    # --neutral-mixed-sampling: steer-only mixed sampling for the neutrality
    # baseline. Requires the neutrality judge (no stakeholder source), and is
    # mutually exclusive with the stakeholder-voiced --mixed-sampling. Default
    # the source split to half/half of N unless the user already gave a full
    # (no-stakeholder) split.
    if args.neutral_mixed_sampling:
        assert judge_is_neutrality, (
            "--neutral-mixed-sampling requires --judge neutrality-v5"
        )
        assert not args.mixed_sampling, (
            "--neutral-mixed-sampling and --mixed-sampling are mutually exclusive"
        )
        if args.n_source_model + args.n_source_critical != args.n_samples:
            args.n_source_model = args.n_samples // 2
            args.n_source_critical = args.n_samples - args.n_source_model

    # --perspective-shift-sampling: the critical steer source becomes the
    # ELEPHANT rewrite-then-answer source. It replaces the 'Be critical.' suffix,
    # so it needs the stakeholder mixed path and a non-zero critical budget (that
    # budget is now the perspective-shift source count).
    if args.perspective_shift_sampling:
        assert args.mixed_sampling, (
            "--perspective-shift-sampling requires --mixed-sampling"
        )
        assert not args.neutral_mixed_sampling, (
            "--perspective-shift-sampling and --neutral-mixed-sampling are "
            "mutually exclusive"
        )
        assert args.n_source_critical > 0, (
            "--perspective-shift-sampling needs --n-source-critical > 0 (that is "
            "the perspective-shift source count); got 0"
        )

    # --future-user-sampling: the critical steer source becomes the future-self
    # voice. Same budget/exclusivity constraints as perspshift (both consume the
    # n_source_critical slots).
    if args.future_user_sampling:
        assert args.mixed_sampling, (
            "--future-user-sampling requires --mixed-sampling"
        )
        assert not args.neutral_mixed_sampling, (
            "--future-user-sampling and --neutral-mixed-sampling are mutually "
            "exclusive"
        )
        assert not args.perspective_shift_sampling, (
            "--future-user-sampling and --perspective-shift-sampling both "
            "consume the critical budget; enable at most one"
        )
        assert args.n_source_critical > 0, (
            "--future-user-sampling needs --n-source-critical > 0 (that is the "
            "future-user source count); got 0"
        )
        # ASSISTANT_ROLE_VETO_PROMPT sub-mode (b) names "the user's own future
        # self" as a role-abandonment, which is EXACTLY what
        # FUTURE_USER_RESPONSE_PROMPT generates on purpose. Combining them would
        # auto-reject the future-user candidates the run exists to produce.
        assert not args.assistant_role_veto, (
            "--future-user-sampling and --assistant-role-veto are mutually "
            "exclusive: the veto classifies future-self voice as abandoning the "
            "assistant role, which would reject every future-user candidate"
        )

    # --distill-self-only masks the RPO/distill SFT term, so it needs both an SFT
    # term (--rpo-alpha) and an on-policy 'model' source (mixed sampling with a
    # non-zero model budget). Fail fast at the CLI rather than deep in train().
    if args.distill_self_only:
        assert args.rpo_alpha is not None, (
            "--distill-self-only masks the RPO/distill SFT term; it requires "
            "--rpo-alpha to be set"
        )
        assert args.mixed_sampling and args.n_source_model > 0, (
            "--distill-self-only distills the on-policy 'model' source; it "
            "requires --mixed-sampling with --n-source-model > 0"
        )

    # The sim vLLM server serves the frozen stakeholder judge model AND (when
    # --policy-on-vllm) hosts the policy LoRA for sampling. The neutrality judge
    # needs no local sim model, so launch the server only when the stakeholder
    # judge needs it or the policy is sampled on it.
    # Two independent reasons to launch the engine, and it must launch if EITHER
    # holds: the sim needs it (--sim-backend vllm with the stakeholder judge), or
    # the policy is sampled on it (--policy-on-vllm). An API/neutrality sim
    # removes the first reason but never the second.
    need_sim_server = args.policy_on_vllm or (
        args.sim_backend == "vllm" and not judge_is_neutrality)

    # 1. Dataset + train/test split.
    csv_path = _resolve_csv(args.env)
    print(f"[data] env={args.env}  csv={csv_path}")
    data = _load_data(args.env, csv_path)
    print(f"[data] pool size: {len(data)} rows")
    train_ids_path = os.path.join(args.output_dir, "train_ids.json")
    # Reuse the saved split when resuming: manual --resume-adapter, or
    # --allow-preempt finding a prior run's train_ids.json (which holds even when
    # no iteration has FINISHED yet — a partial iteration-0 responses file indexes
    # into this exact split, so resampling would misalign prompt_idx).
    reuse_split = (args.resume_adapter is not None) or (
        args.allow_preempt and os.path.exists(train_ids_path))
    if reuse_split:
        # Resume: reuse the EXACT training split the original run saved, so the
        # continued iterations train on the same prompts (not a fresh sample).
        assert os.path.exists(train_ids_path), (
            f"resuming but {train_ids_path} is missing; cannot reuse the "
            f"original split"
        )
        with open(train_ids_path) as f:
            saved_ids = json.load(f)
        train_df = _subset_in_id_order(data, saved_ids)
        train_prompts = _build_train_prompts(args.env, train_df)
        print(f"[data] RESUME: reusing {len(train_prompts)} train prompts "
              f"from {len(saved_ids)} saved ids in {train_ids_path}")
    else:
        train_df, test_df = _split_train_test(
            args.env, data, args.train_n, args.test_n, args.seed,
            use_final_data=args.use_final_data,
        )
        if args.use_final_data:
            print(f"[data] FINAL split for {args.env}: "
                  f"test_n={len(test_df)}  train_n={len(train_df)} "
                  f"(from final_training_eval_data/; --train-n/--test-n/--seed ignored)")
        else:
            print(f"[data] held out test_n={len(test_df)}  train_n={len(train_df)}  seed={args.seed}")
        test_ids_path = os.path.join(args.output_dir, "test_ids.json")
        _write_ids(test_ids_path, test_df)
        _write_ids(train_ids_path, train_df)
        print(f"[data] wrote {test_ids_path}")
        print(f"[data] wrote {train_ids_path}")
        train_prompts = _build_train_prompts(args.env, train_df)

    # 2. Two model instances:
    #    - policy: trainable HF model, on args.device. Sampled from each
    #      iteration and updated by DPO.
    #    - sim judge: frozen, on args.sim_device. Drives every LLM call
    #      in the stakeholder pipeline (identify, filter, concerns,
    #      veto) so the judge's behavior never drifts with the policy.
    #      Served either by a vLLM server subprocess (default — its
    #      continuous batching lets veto calls run concurrently) or by
    #      an in-process HF copy (--sim-backend hf fallback).
    # The vLLM server is launched BEFORE the policy load so its multi-
    # minute startup overlaps with loading the policy weights; we only
    # block on its health check afterwards.
    sim_proc = None
    if need_sim_server:
        assert args.sim_device != args.device, (
            "the vLLM server subprocess gets the whole --sim-device GPU, so it "
            "must differ from --device (which holds the trainable policy). "
            "Note the server is launched for --sim-backend vllm AND/OR "
            "--policy-on-vllm; with an openrouter sim it hosts only the policy."
        )
        vllm_log = os.path.join(args.output_dir, "vllm_server.log")
        _server_for = ("sim+policy" if args.sim_backend == "vllm" and args.policy_on_vllm
                       else "sim" if args.sim_backend == "vllm" else "policy")
        print(f"[load] launching vLLM server ({_server_for}) {args.model_id} "
              f"-> {args.sim_device}  log={vllm_log}")
        # NB: the server resolves the model through the default HF hub
        # cache ($HF_HOME/hub), not args.cache_dir — see launch_vllm_server.
        sim_proc, sim_base_url = launch_vllm_server(
            args.model_id,
            args.sim_device,
            gpu_memory_utilization=args.vllm_gpu_mem_util,
            log_path=vllm_log,
            enable_lora=args.policy_on_vllm,
            max_lora_rank=args.lora_r,
        )

    if args.policy_device_map:
        # Pipeline-shard the trainable policy across the listed GPUs. We build
        # an explicit max_memory so the balanced map uses ONLY those cards:
        # every visible GPU not in the list (incl. the --sim-device vLLM card)
        # is pinned to 0GiB. This caps each shard's transformer-layer load so
        # the one card that ends up holding the lm_head has room for the ~7GB
        # full-vocab logits/CE spike on top of its weight shard — letting OEQ
        # distill keep --max-length 3072 instead of OOMing a single 48GB card.
        shard_gpus = [int(x) for x in args.policy_device_map.split(",")]
        sim_gpu = int(args.sim_device.split(":")[-1])
        assert sim_gpu not in shard_gpus, (
            f"--sim-device {args.sim_device} (vLLM server) must not be in "
            f"--policy-device-map {args.policy_device_map}"
        )
        assert int(args.device.split(":")[-1]) == min(shard_gpus), (
            f"--device {args.device} must be the lowest shard index "
            f"(inputs/labels land there); got shards {shard_gpus}"
        )
        n_visible = torch.cuda.device_count()
        max_memory = {
            i: (args.policy_shard_mem if i in shard_gpus else "0GiB")
            for i in range(n_visible)
        }
        print(f"[load] policy   {args.model_id} -> SHARDED across cuda{shard_gpus} "
              f"(max_memory={args.policy_shard_mem}/gpu)  cache_dir={args.cache_dir}")
        model, tokenizer = load_llama(
            model_id=args.model_id,
            device=args.device,
            cache_dir=args.cache_dir,
            device_map="balanced",
            max_memory=max_memory,
        )
        print(f"[load] policy hf_device_map: {model.hf_device_map}")
    else:
        print(f"[load] policy   {args.model_id} -> {args.device}  cache_dir={args.cache_dir}")
        model, tokenizer = load_llama(
            model_id=args.model_id,
            device=args.device,
            cache_dir=args.cache_dir,
        )

    # 3. call_llm for the stakeholder + veto pipeline (stakeholder judge only)
    # and the policy sampler. Greedy on every backend (temperature=0 for vLLM /
    # OpenRouter, do_sample=False for HF) because reproducibility of the veto
    # signal across train iterations is more useful than diversity here; only the
    # candidate-response sampler inside Trainer needs stochasticity (and it has
    # its own do_sample=True path). For the neutrality judge `sim_call_llm` stays
    # None — the veto judge is the OpenRouter v5 judge built in step 4.
    #
    # The policy sampler and the sim are built INDEPENDENTLY: --policy-on-vllm
    # puts the policy on the local engine regardless of where the sim lives, so
    # --sim-backend openrouter + --policy-on-vllm is a valid (and expected)
    # combination.
    sim_call_llm = None
    sim_model_id = args.model_id
    sim_max_concurrency = None
    sim_cost = None
    veto_concurrency = 1
    policy_sampler = None
    sync_policy_to_vllm = None
    if need_sim_server:
        wait_for_vllm_server(sim_proc, sim_base_url, log_path=os.path.join(args.output_dir, "vllm_server.log"))
        print(f"[load] vLLM server healthy at {sim_base_url}")
        if args.policy_on_vllm:
            # Same engine serves the policy: model="policy" -> base + current
            # LoRA (hot-loaded each iteration); a vLLM sim keeps using model=base.
            policy_sampler = make_vllm_policy_sampler(sim_base_url, args.model_id)

            def sync_policy_to_vllm(adapter_dir):
                unload_lora_adapter(sim_base_url, "policy")
                load_lora_adapter(sim_base_url, "policy", os.path.abspath(adapter_dir))

            print("[setup] policy sampling on shared vLLM engine (base + 'policy' LoRA)")

    if judge_is_neutrality:
        pass  # sim_call_llm stays None; step 4 builds the v5 judge instead.
    elif args.sim_backend == "vllm":
        sim_call_llm = make_vllm_call_llm(sim_base_url, args.model_id)
        veto_concurrency = args.veto_concurrency
        print(f"[setup] sim backend: vllm ({args.model_id}, greedy)")
    elif args.sim_backend == "openrouter":
        # API sim: no local weights, no sim GPU. Bound the fan-out to
        # --sim-concurrency (the Trainer's --stage-concurrency is tuned for the
        # local engine) and meter spend, since the sim is the dominant call
        # volume of the run.
        sim_model_id = args.sim_model
        sim_cost = CostTracker(args.sim_price_in, args.sim_price_out,
                               args.sim_max_cost, label="OpenRouter sim")
        sim_call_llm = make_openrouter_call_llm(
            model_id=args.sim_model,
            temperature=(None if args.sim_temp < 0 else args.sim_temp),
            reasoning=(None if args.sim_reasoning == "none" else args.sim_reasoning),
            max_tokens_floor=args.sim_max_tokens_floor,
            cost_tracker=sim_cost,
        )
        sim_max_concurrency = args.sim_concurrency
        veto_concurrency = min(args.veto_concurrency, args.sim_concurrency)
        print(f"[setup] sim backend: openrouter ({args.sim_model}, "
              f"temp={args.sim_temp}, reasoning={args.sim_reasoning}, "
              f"concurrency={args.sim_concurrency}, "
              f"max_cost={args.sim_max_cost})")
    elif args.sim_backend == "gemini":
        # Direct Google Gemini API sim: no local weights, no sim GPU. Same
        # shape as the openrouter branch — bound the fan-out to
        # --sim-concurrency and meter spend, since the sim is the dominant
        # call volume of the run.
        sim_model_id = args.sim_model
        sim_cost = CostTracker(args.sim_price_in, args.sim_price_out,
                               args.sim_max_cost, label="Gemini sim")
        # Count refusals so a shrinking dataset can never be silent. Bound to a
        # list because the closure runs under concurrent_map; list.append is
        # atomic under the GIL, and we only read it after the fan-out joins.
        sim_blocked = []

        def _on_blocked(prompt, reason):
            sim_blocked.append((reason, prompt[:120]))
            print(f"[gemini] REFUSED ({reason}); blocked so far: "
                  f"{len(sim_blocked)}", flush=True)

        sim_call_llm = make_gemini_call_llm(
            model_id=args.sim_model,
            temperature=(None if args.sim_temp < 0 else args.sim_temp),
            thinking_budget=(None if args.sim_thinking_budget < 0
                             else args.sim_thinking_budget),
            max_retries=args.sim_max_retries,
            backoff_max=args.sim_backoff_max,
            max_tokens_floor=args.sim_max_tokens_floor,
            cost_tracker=sim_cost,
            on_blocked=_on_blocked,
        )
        sim_max_concurrency = args.sim_concurrency
        veto_concurrency = min(args.veto_concurrency, args.sim_concurrency)
        print(f"[setup] sim backend: gemini ({args.sim_model}, "
              f"temp={args.sim_temp}, thinking={args.sim_thinking_budget}, "
              f"concurrency={args.sim_concurrency}, "
              f"retries={args.sim_max_retries}@<={args.sim_backoff_max}s, "
              f"blocked=skip-step, "
              f"max_cost={args.sim_max_cost})")
    else:
        # Stakeholder judge with the HF sim backend: load a frozen in-process
        # copy of the sim model on args.sim_device.
        assert args.sim_backend == "hf", f"unhandled --sim-backend {args.sim_backend}"
        print(f"[load] sim_base {args.model_id} -> {args.sim_device}  (frozen)")
        sim_model, sim_tokenizer = load_llama(
            model_id=args.model_id,
            device=args.sim_device,
            cache_dir=args.cache_dir,
        )
        # Freeze sim_base — it's only used for inference inside the
        # pipeline. Disabling grad on parameters is belt-and-suspenders;
        # eval() already suppresses dropout, and we never call backward
        # on sim outputs.
        for p in sim_model.parameters():
            p.requires_grad = False
        sim_model.eval()
        sim_call_llm = make_call_llm(sim_model, sim_tokenizer, do_sample=False)
        # A single in-process HF model serves requests one at a time;
        # concurrent calls would interleave without batching.
        veto_concurrency = 1

    # Memory diagnostics: a background thread logs host RSS + per-GPU
    # allocator counters to mem.csv every 2s, line-buffered so the curve
    # survives the SIGKILL an OOM kill sends. Off the training path, so
    # it does not slow generate. Started after the models are resident
    # so the baseline row reflects loaded weights. (With the vLLM
    # backend, the sim GPU's usage lives in the server subprocess and is
    # invisible to this process's allocator counters.)
    mem_csv = os.path.join(args.output_dir, "mem.csv")
    start_mem_sampler(mem_csv)
    print(f"[mem] sampling host+GPU memory -> {mem_csv}")

    # Rewire the harness's module-level `call_llm` bindings to the frozen
    # sim_base (stakeholder judge only). Three modules hold independent imports
    # of `call_llm` (`our_mitigation`, `our_mitigation_v2`, `weak_pipeline`);
    # the SimulatedStakeholders constructor below also does this, but doing it
    # here too is harmless and makes the script readable on its own. CRITICAL:
    # the binding points at sim_base, NOT the policy — the sim must never see
    # the updated policy or the veto signal would drift as training progresses.
    # The neutrality judge does not use this pipeline, so it skips the rewiring.
    if not judge_is_neutrality:
        our_mitigation.call_llm = sim_call_llm
        our_mitigation_v2.call_llm = sim_call_llm
        weak_pipeline.call_llm = sim_call_llm
        gemini_pipeline.call_llm = sim_call_llm

        # Tolerant stakeholder parser — patch every module that holds an
        # independent binding of `parse_stakeholders` (each did
        # `from our_mitigation import parse_stakeholders` at import time, so
        # patching our_mitigation alone does NOT rebind their local names).
        # gemini_pipeline is included: without it, --prompt-variant gemini keeps
        # the STRICT parser and a >=10-item emission aborts the run mid-training,
        # which is exactly what _tolerant_parse_stakeholders exists to prevent.
        our_mitigation.parse_stakeholders = _tolerant_parse_stakeholders
        our_mitigation_v2.parse_stakeholders = _tolerant_parse_stakeholders
        weak_pipeline.parse_stakeholders = _tolerant_parse_stakeholders
        gemini_pipeline.parse_stakeholders = _tolerant_parse_stakeholders

    # 4. Veto judge. `sim` is whatever object the Trainer uses for the veto
    # signal; it only needs prime_stakeholder_cache / veto_responses /
    # veto_responses_batch, which both judges below implement.
    if judge_is_neutrality:
        # Neutrality baseline: the v5 gpt-5-mini endorse judge vetoes every
        # response it does NOT label 3 (neutral). chosen=neutral, rejected=
        # challenge/affirm. Runs over OpenRouter (needs OPENROUTER_API_KEY).
        from neutrality_judge import NeutralityVetoJudge, make_openrouter_client
        judge_temperature = None if args.judge_temp < 0 else args.judge_temp
        judge_reasoning = None if args.judge_reasoning == "none" else args.judge_reasoning
        sim = NeutralityVetoJudge(
            client=make_openrouter_client(),
            model=args.judge_model,
            reasoning=judge_reasoning,
            max_tokens=args.judge_max_tokens,
            temperature=judge_temperature,
            concurrency=args.judge_concurrency,
            price_in=args.judge_price_in,
            price_out=args.judge_price_out,
            max_cost=args.judge_max_cost,
        )
        print(f"[setup] judge: neutrality-v5 ({args.judge_model}, "
              f"reasoning={args.judge_reasoning})  chosen=v5-label-3, "
              f"rejected=otherwise")
    else:
        # Default sim_stake routing: identify + filter + concerns + veto all go
        # through sim_call_llm (the frozen sim). We don't pass concerns_call_llm
        # / veto_call_llm — the single-backend default does what we want.
        sim = SimulatedStakeholders(
            call_llm=sim_call_llm,
            # The sim's OWN model id — args.model_id only when the sim is a local
            # copy of the policy's checkpoint; args.sim_model under --sim-backend
            # openrouter. (The backend closures ignore it, but it is what the
            # pipeline functions receive and what a future split backend routes
            # on, so it must not silently name the wrong model.)
            model_id=sim_model_id,
            client=None,
            stakeholder_max_tokens=args.max_tokens,
            use_weak_model=True,
            # None => weak_pipeline (every existing arm unchanged); "gemini" =>
            # gemini_pipeline's retuned identify/filter prompts.
            prompt_variant=args.prompt_variant,
            veto_concurrency=veto_concurrency,
            # None for local backends (trust --stage-concurrency); the API
            # backend caps every fan-out at --sim-concurrency.
            max_concurrency=sim_max_concurrency,
            veto_prompt=VETO_PROMPTS[args.veto_prompt],
            # Two-stage veto modes (P5, P6) supply a Stage-1 endorsement
            # classifier; single-stage modes (baseline, P4) get None here.
            endorse_classify_prompt=ENDORSE_CLASSIFY_PROMPTS.get(args.veto_prompt),
            # P6 = bidirectional stance-mismatch veto; --user-stakeholder adds the
            # 4th narrator voice (veto-only, never sampled for candidates).
            bidirectional=(args.veto_prompt == "P6"),
            include_user_stakeholder=args.user_stakeholder,
            addressee_filter=args.addressee_filter,
            assistant_role_veto=args.assistant_role_veto,
            # Judge-model-specific prompt. Validated 2026-07-20 on a held-out
            # labelled set: on the LOCAL Qwen3-8B judge the base prompt scores
            # FPR 0.600 (it reads blunt second-person critique as in-story
            # voice), which would auto-reject the very responses P6 exists to
            # produce; the _QWEN variant restores the sub-mode (b) epistemic test
            # and halves that to 0.325. Gemini does not need it (FPR 0.125) and
            # loses recall with it, so it keeps the base prompt.
            assistant_role_prompt=(
                ASSISTANT_ROLE_VETO_PROMPT_QWEN
                if args.sim_backend in ("vllm", "hf")
                else ASSISTANT_ROLE_VETO_PROMPT),
        )
        print(f"[setup] veto prompt: {args.veto_prompt}"
              f"{' (two-stage)' if args.veto_prompt in ENDORSE_CLASSIFY_PROMPTS else ''}"
              f"{' +bidirectional' if args.veto_prompt == 'P6' else ''}"
              f"{' +user-stakeholder' if args.user_stakeholder else ''}"
              f"{' +addressee-filter' if args.addressee_filter else ''}"
              f"{' +assistant-role-veto' if args.assistant_role_veto else ''}"
              f"{'(qwen-prompt)' if args.assistant_role_veto and args.sim_backend in ('vllm', 'hf') else ''}")

    # 5. Trainer + train. Trainer.model is the policy (trainable);
    # Trainer's `_sample_responses` calls policy.generate() directly,
    # so policy updates are picked up in the next iteration's sampling.
    trainer = Trainer(
        sim_stake=sim,
        model=model,
        tokenizer=tokenizer,
        n_samples=args.n_samples,
        max_new_tokens=args.sample_max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        mixed_sampling=args.mixed_sampling or args.neutral_mixed_sampling,
        n_source_model=args.n_source_model,
        n_source_critical=args.n_source_critical,
        steer_suffix=(NEUTRAL_SUFFIX if args.neutral_mixed_sampling
                      else CRITICAL_SUFFIX),
        steer_label=("neutral" if args.neutral_mixed_sampling else "critical"),
        steer_only=args.neutral_mixed_sampling,
        perspective_shift_steer=args.perspective_shift_sampling,
        rewrite_max_tokens=args.rewrite_max_tokens,
        # None (default) => the rewrite runs on the policy, unchanged. Only
        # --rewrite-on-sim routes it to the frozen sim backend.
        rewrite_call_llm=(sim_call_llm if args.rewrite_on_sim else None),
        perspshift_assess=args.perspshift_assess,
        future_user_steer=args.future_user_sampling,
        stakeholder_prompt=(STAKEHOLDER_RESPONSE_PROMPT_VOICE
                            if args.stakeholder_voice
                            else STAKEHOLDER_RESPONSE_PROMPT),
        frozen_stakeholder=args.frozen_stakeholder,
        distill_self_only=args.distill_self_only,
        distill_max_per_prompt=args.distill_max_per_prompt,
        self_preference=args.self_preference,
        preference_max_tokens=args.preference_max_tokens,
        preference_max_pairs=args.preference_max_pairs,
        policy_sampler=policy_sampler,
        sync_policy_to_vllm=sync_policy_to_vllm,
        chunk_size=args.chunk_size,
        stage_concurrency=args.stage_concurrency,
    )
    if args.neutral_mixed_sampling:
        print(
            f"[train] neutral_mixed_sampling=True "
            f"(model={args.n_source_model} neutral={args.n_source_critical} "
            f"stakeholder=0)  steer={NEUTRAL_SUFFIX!r}  "
            f"self_preference={args.self_preference} "
            f"(max_pairs={args.preference_max_pairs or 'unlimited'})"
        )
    else:
        steer_kind = ("perspshift" if args.perspective_shift_sampling
                      else "future_user" if args.future_user_sampling
                      else "critical")
        print(
            f"[train] mixed_sampling={args.mixed_sampling} "
            f"(model={args.n_source_model} {steer_kind}={args.n_source_critical} "
            f"stakeholder={args.n_samples - args.n_source_model - args.n_source_critical})  "
            f"stakeholder_voice={args.stakeholder_voice} "
            f"frozen_stakeholder={args.frozen_stakeholder} "
            f"distill_self_only={args.distill_self_only}"
            f"{f' (max {args.distill_max_per_prompt}/prompt)' if args.distill_self_only else ''} "
            f"self_preference={args.self_preference} "
            f"(max_pairs={args.preference_max_pairs or 'unlimited'})"
        )

    print(f"[train] starting iterative DPO+LoRA  iters={args.num_iterations}  output_dir={args.output_dir}")
    dpo_trainer = trainer.train(
        prompts=train_prompts,
        output_dir=args.output_dir,
        num_iterations=args.num_iterations,
        beta=args.beta,
        rpo_alpha=args.rpo_alpha,
        dpo_save_steps=args.dpo_save_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        precompute_ref_batch_size=args.precompute_ref_batch_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        start_iteration=args.start_iteration,
        resume_adapter=args.resume_adapter,
        resume_partial=resume_partial,
    )

    print(f"[done] adapter saved to {args.output_dir}")
    # log_history is a list of per-step dicts; show the last entry so
    # the user can eyeball the final loss without reading the full HF
    # trainer log. dpo_trainer is None if every iteration produced 0 pairs
    # (nothing to train on) — guard so that path doesn't crash.
    if dpo_trainer is not None and dpo_trainer.state.log_history:
        print(f"[done] last log entry: {dpo_trainer.state.log_history[-1]}")


if __name__ == "__main__":
    main()
