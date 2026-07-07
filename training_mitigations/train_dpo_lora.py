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
  4. Stand up TWO instances of the model (default `Qwen/Qwen3-8B`):
       - `policy` on `--device`: a trainable HF model (LoRA-adapted).
       - the sim judge on `--sim-device`: frozen; never updated. Served
         by a vLLM server subprocess by default (`--sim-backend vllm`),
         whose continuous batching lets the per-(stakeholder, response)
         veto calls run concurrently (`--veto-concurrency`) instead of
         one batch-1 HF generate at a time; `--sim-backend hf` keeps
         the original in-process frozen HF copy as a fallback.
     The policy is what gets sampled from each iteration and what DPO
     updates. The sim judge drives every LLM call inside the
     stakeholder pipeline (identify / filter / concerns / veto) so
     the veto judge's behavior does not drift with the policy.
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
against): bare post_text for OEQ and PAS (matches the harness's
`PROMPT_TEMPLATE_OEQ = "{post}"`); for AITA / AITA-NTA-FLIP the post
plus `train_data.AITA_LABEL_SUFFIX`, asking for a trailing YTA/NTA
verdict the eval parses (no LLM classifier). AITA-NTA-FLIP expands
each pair into two independent train prompts (original + flipped).

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

from explore_mitigations import DEFAULT_MAX_TOKENS, DEFAULT_SEED

# Shared dataset / split helpers (also used by train_grpo_lora.py) so a DPO run
# and a GRPO run on the same env/seed/split train and eval on identical prompts.
from train_data import (resolve_csv as _resolve_csv,
                        load_env_data as _load_data,
                        split_train_test as _split_train_test,
                        build_train_prompts as _build_train_prompts,
                        label_suffix_for as _label_suffix_for,
                        subset_in_id_order as _subset_in_id_order,
                        write_ids as _write_ids,
                        detect_resume_point as _detect_resume_point)

from llama_local import DEFAULT_HF_MODEL_ID, load_llama, make_call_llm
from mem_probe import start_mem_sampler
from simulated_stakeholders import (SimulatedStakeholders, VETO_PROMPTS,
                                    ENDORSE_CLASSIFY_PROMPTS)
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
        "--sim-backend", default="vllm", choices=("vllm", "hf"),
        help="how the frozen sim judge is served (default: vllm). 'vllm' "
             "launches a vLLM OpenAI-compatible server subprocess on "
             "--sim-device and fans veto calls out concurrently "
             "(--veto-concurrency); 'hf' is the original in-process "
             "HF-transformers path (sequential, batch-1 — slow; kept as a "
             "fallback).",
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
        help="number of training prompts",
    )
    parser.add_argument(
        "--test-n", type=int, required=True,
        help="number of held-out test prompts",
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

    assert not (args.policy_on_vllm and args.sim_backend != "vllm"), (
        "--policy-on-vllm requires --sim-backend vllm (the policy is sampled on "
        "the same vLLM engine that serves the sim)"
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
    need_sim_server = args.sim_backend == "vllm" and (
        not judge_is_neutrality or args.policy_on_vllm)

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
        train_prompts = _build_train_prompts(args.env, train_df,
                                             with_label_suffix=False)
        print(f"[data] RESUME: reusing {len(train_prompts)} train prompts "
              f"from {len(saved_ids)} saved ids in {train_ids_path}")
    else:
        train_df, test_df = _split_train_test(
            args.env, data, args.train_n, args.test_n, args.seed,
        )
        print(f"[data] held out test_n={len(test_df)}  train_n={len(train_df)}  seed={args.seed}")
        test_ids_path = os.path.join(args.output_dir, "test_ids.json")
        _write_ids(test_ids_path, test_df)
        _write_ids(train_ids_path, train_df)
        print(f"[data] wrote {test_ids_path}")
        print(f"[data] wrote {train_ids_path}")
        train_prompts = _build_train_prompts(args.env, train_df,
                                             with_label_suffix=False)

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
            "--sim-backend vllm gives the whole sim GPU to the server "
            "subprocess; --sim-device must differ from --device"
        )
        vllm_log = os.path.join(args.output_dir, "vllm_server.log")
        print(f"[load] sim judge: launching vLLM server {args.model_id} -> {args.sim_device}  log={vllm_log}")
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
    # and the policy sampler. Greedy on both backends (temperature=0 for vLLM,
    # do_sample=False for HF) because reproducibility of the veto signal across
    # train iterations is more useful than diversity here; only the candidate-
    # response sampler inside Trainer needs stochasticity (and it has its own
    # do_sample=True path). For the neutrality judge `sim_call_llm` stays None —
    # the veto judge is the OpenRouter v5 judge built in step 4.
    sim_call_llm = None
    veto_concurrency = 1
    policy_sampler = None
    sync_policy_to_vllm = None
    if need_sim_server:
        wait_for_vllm_server(sim_proc, sim_base_url, log_path=os.path.join(args.output_dir, "vllm_server.log"))
        print(f"[load] sim judge: vLLM server healthy at {sim_base_url}")
        if not judge_is_neutrality:
            sim_call_llm = make_vllm_call_llm(sim_base_url, args.model_id)
            veto_concurrency = args.veto_concurrency
        if args.policy_on_vllm:
            # Same engine serves the policy: model="policy" -> base + current
            # LoRA (hot-loaded each iteration); the sim keeps using model=base.
            policy_sampler = make_vllm_policy_sampler(sim_base_url, args.model_id)

            def sync_policy_to_vllm(adapter_dir):
                unload_lora_adapter(sim_base_url, "policy")
                load_lora_adapter(sim_base_url, "policy", os.path.abspath(adapter_dir))

            print("[setup] policy sampling on shared vLLM engine (base + 'policy' LoRA)")
    elif not judge_is_neutrality:
        # Stakeholder judge with the HF sim backend: load a frozen in-process
        # copy of the sim model on args.sim_device.
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
    # else: neutrality judge with no sim server — policy samples via the HF
    # generation path (policy_sampler=None) and the v5 judge runs over OpenRouter.

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

        # Tolerant stakeholder parser — patch all three modules that hold
        # an independent binding of `parse_stakeholders`.
        our_mitigation.parse_stakeholders = _tolerant_parse_stakeholders
        our_mitigation_v2.parse_stakeholders = _tolerant_parse_stakeholders
        weak_pipeline.parse_stakeholders = _tolerant_parse_stakeholders

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
            model_id=args.model_id,
            client=None,
            stakeholder_max_tokens=args.max_tokens,
            use_weak_model=True,
            veto_concurrency=veto_concurrency,
            veto_prompt=VETO_PROMPTS[args.veto_prompt],
            # Two-stage veto modes (P5, P6) supply a Stage-1 endorsement
            # classifier; single-stage modes (baseline, P4) get None here.
            endorse_classify_prompt=ENDORSE_CLASSIFY_PROMPTS.get(args.veto_prompt),
            # P6 = bidirectional stance-mismatch veto; --user-stakeholder adds the
            # 4th narrator voice (veto-only, never sampled for candidates).
            bidirectional=(args.veto_prompt == "P6"),
            include_user_stakeholder=args.user_stakeholder,
        )
        print(f"[setup] veto prompt: {args.veto_prompt}"
              f"{' (two-stage)' if args.veto_prompt in ENDORSE_CLASSIFY_PROMPTS else ''}"
              f"{' +bidirectional' if args.veto_prompt == 'P6' else ''}"
              f"{' +user-stakeholder' if args.user_stakeholder else ''}")

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
        perspshift_assess=args.perspshift_assess,
        stakeholder_prompt=(STAKEHOLDER_RESPONSE_PROMPT_VOICE
                            if args.stakeholder_voice
                            else STAKEHOLDER_RESPONSE_PROMPT),
        frozen_stakeholder=args.frozen_stakeholder,
        distill_self_only=args.distill_self_only,
        distill_max_per_prompt=args.distill_max_per_prompt,
        # AITA/FLIP YTA/NTA nudge (see core/trainer.py). "" for OEQ/PAS, so the
        # whole label-suffix mechanism is a no-op for those envs.
        label_suffix=_label_suffix_for(args.env),
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
