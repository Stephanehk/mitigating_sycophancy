"""Trainer — orchestrates the SimulatedStakeholders veto signal into a
preference dataset for HuggingFace DPO training, and runs an iterative
DPO+LoRA fine-tune on top.

The trainer holds a reference to the *policy* model — the one being
fine-tuned and sampled from. The sim (`sim_stake`) holds an independent
closure over a separate frozen model (sim_base) wired in by the caller;
nothing in this module touches sim_base directly, and the policy's
LoRA updates never reach it.

Per prompt, per iteration:
  1. Sample N candidate responses from the *current* policy (one
     `model.generate` call with `num_return_sequences=N`, stochastic
     decoding so the N samples diverge).
  2. Pass the responses to `sim_stake.veto_responses(prompt, responses)`,
     which returns the subset any stakeholder would object to.
  3. Build preference pairs: every vetoed response (DPO `rejected`) is
     paired with every non-vetoed response (DPO `chosen`). So if k of
     N responses are vetoed, this prompt contributes k * (N-k) pairs.
     (Pairs are emitted per-occurrence, not deduped — if the same text
     appears twice in `responses` and once gets vetoed, that's two
     legitimate samples and we keep both.)

A prompt where 0 responses are vetoed contributes nothing (no
`rejected` to dispreference). A prompt where ALL responses are vetoed
also contributes nothing (no `chosen` partner). In the latter case the
prompt id is logged so the caller can see which prompts the policy is
currently unable to produce acceptable responses for; on the next
iteration we re-sample those prompts from the updated policy and may
get a chosen partner.

`train(prompts, output_dir, num_iterations=K)`:
  - Wraps `self.model` with a LoRA adapter once.
  - For K outer iterations:
      a. Re-samples N candidates per prompt from the current policy.
      b. Builds a fresh preference dataset from this iteration's
         samples (no accumulation across iterations — fresh pairs
         match the current policy distribution).
      c. Logs the per-iteration count of all-vetoed prompts.
      d. Runs `trl.DPOTrainer` for one inner epoch on the dataset.
      e. Snapshots the adapter to `output_dir/adapter_iter_<k>` so the
         policy state at each iteration is independently loadable.
  - Saves the final LoRA adapter to `output_dir` (== adapter_iter_<K>).
    The base weights are not changed on disk.

Sampling uses the same chat-template path as `llama_local.make_call_llm`
(via `llama_local.build_chat_input`) and Qwen3's `enable_thinking=False`
quirk so the sampled responses are in the same surface format the sim
saw at probe time; the `train` method re-templates the prompt column
the same way so DPO sees the same surface that produced the responses.
"""

import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llama_local import build_chat_input
from mem_probe import mem_line
from simulated_stakeholders import _format_bullets


# ---------------------------------------------------------------------------
# Candidate-generation templates for the mixed-source sampler (variation 1).
#
# When `mixed_sampling=True`, the N candidate responses for a prompt are NOT
# all drawn from the bare post. Instead the budget is split across sources so
# the veto judge sees a more diverse candidate pool (and the chosen/rejected
# split is less degenerate): some plain model samples, some "be critical"
# samples, and some samples conditioned on each identified stakeholder's
# voice + concerns. The stakeholder/critical prompts are used ONLY at
# generation time to elicit diverse texts — the DPO `prompt` column is always
# the bare post (see `_build_iteration`), so training still teaches the
# bare-prompt policy to prefer the less-sycophantic candidates.
# ---------------------------------------------------------------------------

# Appended verbatim to the bare post for the "be critical" source. Kept as a
# named constant so the exact injected text is explicit and easy to change.
CRITICAL_SUFFIX = "\n\nBe critical."

# Stakeholder-voiced candidate-generation prompt. Mirrors the role-play
# framing of `our_mitigation_v2.CONCERNS_PROMPT_V2` / the veto prompt so the
# persona surface is consistent with the cached stakeholder data, but here we
# ask for an actual advice response addressed to the narrator rather than a
# concerns list.
STAKEHOLDER_RESPONSE_PROMPT = """You are role-playing as the following stakeholder in the situation described below: {stakeholder}.

The situation:
---
{post}
---

As {stakeholder}, write a response addressed to the narrator reflecting your perspective. Write only the response to the narrator and nothing else."""


# ---------------------------------------------------------------------------
# Preference-elicitation template (variation 2).
#
# When `self_preference=True`, for every unordered pair of NON-vetoed
# responses we ask the *policy* model (the one being updated) to pick the
# better response, with the post in context. The winner becomes DPO `chosen`
# and the loser DPO `rejected`; these pairs are ADDED to the veto-based pairs.
# Because the labeler is the policy itself, this is a self-preference /
# self-rewarding signal layered on top of the stakeholder veto signal.
# ---------------------------------------------------------------------------
PREFERENCE_PROMPT = """You are comparing two candidate responses to the situation below, to decide which is the better response for the narrator.

The situation:
---
{post}
---

Response A:
---
{response_a}
---

Response B:
---
{response_b}
---

Question: Which response is better?

Reply with a single token A or B on the first line, followed by a one-sentence justification on the second line. Use this exact format and nothing else:

A or B
<one-sentence justification>"""


class Trainer:
    """Pairs a sampler (HF model + tokenizer) with a SimulatedStakeholders
    veto judge to produce a DPO-ready preference dataset.

    The model under training is held by reference, not copied. Putting
    it in eval mode for sampling is the caller's responsibility — DPO
    typically alternates train/sample anyway.

    Decoding defaults (temperature=1.0, top_p=0.95) are deliberately
    permissive so the N samples diverge enough to give the veto judge
    something to discriminate. If they collapse, the dataset will be
    empty (no veto disagreements → no pairs) — that's a sampling
    problem, not a Trainer bug.
    """

    def __init__(self,
                 sim_stake,
                 model,
                 tokenizer,
                 n_samples=10,
                 max_new_tokens=512,
                 temperature=1.0,
                 top_p=0.95,
                 mixed_sampling=False,
                 n_source_model=2,
                 n_source_critical=2,
                 self_preference=False,
                 preference_max_tokens=128,
                 preference_max_pairs=0):
        """
        sim_stake:     a SimulatedStakeholders instance (see
                       simulated_stakeholders.py).
        model:         a HuggingFace AutoModelForCausalLM already loaded
                       onto a CUDA device.
        tokenizer:     the matching tokenizer; must have a chat template.
        n_samples:     N — number of candidate responses sampled per
                       prompt (default 10). When mixed_sampling is off, all
                       N share the bare prompt and only the sampling RNG
                       differs. When on, N is split across sources (see
                       mixed_sampling below) but the total stays N.
        max_new_tokens: generation cap per response. 512 is enough for
                       OEQ-style advice replies; bump if responses get
                       truncated.
        temperature, top_p: standard HF generation knobs. Both apply iff
                       do_sample=True, which we force on (greedy would
                       make all N samples identical).
        mixed_sampling: variation 1. When True, the N candidates per prompt
                       are drawn from multiple sources instead of all from
                       the bare post: `n_source_model` from the bare post,
                       `n_source_critical` from the post with CRITICAL_SUFFIX
                       appended, and the remaining
                       `n_samples - n_source_model - n_source_critical`
                       distributed round-robin across the prompt's identified
                       stakeholders (each generated under
                       STAKEHOLDER_RESPONSE_PROMPT). The bare post is still
                       the DPO prompt column for every resulting pair.
        n_source_model / n_source_critical: per-source counts for the mixed
                       sampler. Must sum to <= n_samples; the leftover is the
                       stakeholder budget. Ignored when mixed_sampling is off.
        self_preference: variation 2. When True, for every unordered pair of
                       NON-vetoed responses, ask the policy (this model) to
                       pick the better one via PREFERENCE_PROMPT and add the
                       resulting (chosen, rejected) pair to the dataset, on
                       top of the veto-based pairs.
        preference_max_tokens: max new tokens for each preference-label call
                       (one A/B token + a one-sentence reason fits in 128).
        preference_max_pairs: per-prompt cap on preference pairs (0 = no cap,
                       i.e. all C(k,2) pairs over k non-vetoed responses). A
                       safety valve since C(k,2) grows quadratically.
        """
        assert temperature > 0, (
            f"temperature must be > 0 for stochastic sampling (got {temperature}); "
            f"otherwise all {n_samples} samples are identical and no pairs are produced"
        )
        assert n_source_model >= 0 and n_source_critical >= 0, (
            "per-source counts must be non-negative"
        )
        if mixed_sampling:
            assert n_source_model + n_source_critical <= n_samples, (
                f"n_source_model ({n_source_model}) + n_source_critical "
                f"({n_source_critical}) exceeds n_samples ({n_samples}); "
                f"no budget left for stakeholder samples"
            )
        self.sim_stake = sim_stake
        self.model = model
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.mixed_sampling = mixed_sampling
        self.n_source_model = n_source_model
        self.n_source_critical = n_source_critical
        self.self_preference = self_preference
        self.preference_max_tokens = preference_max_tokens
        self.preference_max_pairs = preference_max_pairs

    def _generate_n(self, user_prompt, n):
        """Draw `n` stochastic responses from the policy for an arbitrary
        `user_prompt` (already the full user-message text; the chat
        template is applied here). Returns a list of `n` strings.

        Uses `num_return_sequences=n` so HF batches the n forward passes
        — much cheaper than n sequential `generate` calls. Each sample is
        decoded with `skip_special_tokens=True`; the prompt tokens are
        sliced off before decoding so only the new text is returned.
        Returns `[]` for n <= 0.
        """
        if n <= 0:
            return []
        device = next(self.model.parameters()).device
        input_ids = build_chat_input(self.tokenizer, user_prompt, system=None, device=device)
        in_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=int(self.max_new_tokens),
                do_sample=True,
                temperature=float(self.temperature),
                top_p=float(self.top_p),
                num_return_sequences=int(n),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        responses = []
        for i in range(int(n)):
            new_ids = output_ids[i, in_len:]
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            responses.append(text)
        return responses

    def _sample_responses(self, prompt):
        """Draw `n_samples` stochastic responses from the policy for the
        bare `prompt`. Thin wrapper over `_generate_n`; this is the
        non-mixed sampling path. Returns a list of length `n_samples`.
        """
        return self._generate_n(prompt, self.n_samples)

    def _sample_responses_mixed(self, prompt, entries):
        """Variation 1: draw `n_samples` candidates split across sources.

        Budget split (totals exactly `n_samples`):
          - `n_source_model`    from the bare post.
          - `n_source_critical` from the post + CRITICAL_SUFFIX.
          - the remainder       distributed round-robin across `entries`
            (the prompt's identified stakeholders), each generated under
            STAKEHOLDER_RESPONSE_PROMPT with that stakeholder's cached
            concerns.

        Round-robin distribution means: with a 6-sample stakeholder budget
        and 3 stakeholders, each gets 2; with 2 stakeholders, 3 each; with
        7 stakeholders, the first 6 get 1 and the 7th gets 0. So "2 from
        each stakeholder" holds exactly at 3 stakeholders and degrades
        gracefully otherwise while keeping the total at `n_samples`.

        Inputs:
          prompt:  the bare post (also the DPO prompt column downstream).
          entries: list of {"stakeholder", "concerns"} dicts from the sim's
                   `_stakeholder_data_for`. Must be non-empty when the
                   stakeholder budget is > 0.

        Returns (responses, sources): parallel lists of length `n_samples`.
        `sources[i]` is "model", "critical", or
        "stakeholder:<name>" — persisted for auditability, not used by DPO.
        """
        n_model = min(self.n_source_model, self.n_samples)
        n_critical = min(self.n_source_critical, self.n_samples - n_model)
        n_stake = self.n_samples - n_model - n_critical

        responses = []
        sources = []

        for r in self._generate_n(prompt, n_model):
            responses.append(r)
            sources.append("model")

        for r in self._generate_n(prompt + CRITICAL_SUFFIX, n_critical):
            responses.append(r)
            sources.append("critical")

        if n_stake > 0:
            assert entries, (
                "mixed sampling has a stakeholder budget but no stakeholders "
                "were identified for this prompt"
            )
            # Round-robin: counts[j] = number of samples for entries[j].
            counts = [0] * len(entries)
            for j in range(n_stake):
                counts[j % len(entries)] += 1
            for entry, c in zip(entries, counts):
                if c == 0:
                    continue
                stake_prompt = STAKEHOLDER_RESPONSE_PROMPT.format(
                    stakeholder=entry["stakeholder"],
                    post=prompt,
                    concerns_block=_format_bullets(entry["concerns"]),
                )
                for r in self._generate_n(stake_prompt, c):
                    responses.append(r)
                    sources.append(f"stakeholder:{entry['stakeholder']}")

        return responses, sources

    def _policy_generate_greedy(self, user_prompt, max_new_tokens):
        """Greedy single completion from the policy for `user_prompt`.
        Used by the preference labeler so the A/B verdict is deterministic
        given the current policy weights. Returns the decoded new text.
        """
        device = next(self.model.parameters()).device
        input_ids = build_chat_input(self.tokenizer, user_prompt, system=None, device=device)
        in_len = input_ids.shape[1]
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = output_ids[0, in_len:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def _elicit_preference(self, prompt, response_a, response_b):
        """Ask the policy which of two responses is the better advice for
        `prompt`, with the post in context (PREFERENCE_PROMPT). Returns
        "A" or "B". Asserts the model emits one of the two tokens — a
        mis-parsed verdict would corrupt the preference dataset, so we fail
        loud rather than guess.
        """
        pref_prompt = PREFERENCE_PROMPT.format(
            post=prompt,
            response_a=response_a,
            response_b=response_b,
        )
        text = self._policy_generate_greedy(pref_prompt, self.preference_max_tokens)
        m = re.search(r"\b([AB])\b", text)
        assert m is not None, (
            f"preference judge produced no A/B token; raw={text!r}"
        )
        return m.group(1).upper()

    def _policy_preference_pairs(self, prompt, non_vetoed):
        """Variation 2: build DPO pairs from the policy's preferences over
        every unordered pair of NON-vetoed responses.

        For each pair (i, j), i < j, ask the policy which is better. The
        winner is `chosen`, the loser `rejected`. To balance the policy's
        position bias deterministically (no RNG — RNG is unavailable in the
        reproducible train path and would desync resume), the presentation
        order of A/B alternates by pair index; the verdict is mapped back to
        the actual responses regardless of presentation order.

        Honors `preference_max_pairs` (0 = unlimited). Returns a list of
        {"prompt", "chosen", "rejected"} dicts; `prompt` is the bare post.
        """
        pairs = []
        k = len(non_vetoed)
        if k < 2:
            return pairs
        pair_idx = 0
        for i in range(k):
            for j in range(i + 1, k):
                if self.preference_max_pairs and len(pairs) >= self.preference_max_pairs:
                    return pairs
                # Alternate presentation order to balance position bias.
                if pair_idx % 2 == 0:
                    shown_a, shown_b = non_vetoed[i], non_vetoed[j]
                else:
                    shown_a, shown_b = non_vetoed[j], non_vetoed[i]
                pair_idx += 1
                choice = self._elicit_preference(prompt, shown_a, shown_b)
                chosen = shown_a if choice == "A" else shown_b
                rejected = shown_b if choice == "A" else shown_a
                pairs.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                })
        return pairs

    def create_prompt_sample(self, x):
        """Build the DPO preference pairs contributed by a single prompt.

        Steps:
          1. Sample N responses.
          2. Ask the sim which ones to veto.
          3. Cross-pair: every vetoed × every non-vetoed becomes a
             (rejected, chosen) pair under this prompt.

        Returns a list of dicts in the standard DPO schema:
          {"prompt": x, "chosen": <non-vetoed>, "rejected": <vetoed>}

        Edge cases:
          - All N vetoed   → no `chosen` partner → 0 pairs.
          - 0 vetoed       → nothing to dispreference → 0 pairs.
          - In both cases the prompt contributes nothing to the dataset
            (and we don't waste downstream training compute on a
            degenerate split). No warning — these are expected and
            common on easy/hard ends of the prompt distribution.
        """
        responses = self._sample_responses(x)
        vetoed = self.sim_stake.veto_responses(x, responses)

        # Set membership compares by string equality, which is the
        # behavior we want: if two sampled responses happen to be
        # identical and only one was vetoed, both copies are excluded
        # from the chosen pool.
        vetoed_set = set(vetoed)
        non_vetoed = [r for r in responses if r not in vetoed_set]

        pairs = []
        for rejected in vetoed:
            for chosen in non_vetoed:
                pairs.append({
                    "prompt": x,
                    "chosen": chosen,
                    "rejected": rejected,
                })
        return pairs

    def create_dataset(self, prompts):
        """Concatenate `create_prompt_sample` across `prompts` into a
        single `datasets.Dataset` ready for `trl.DPOTrainer`.

        Inputs:
          prompts: iterable of prompt strings.

        Returns: `datasets.Dataset` with three string columns —
          `prompt`, `chosen`, `rejected`. Empty rows are not inserted;
          prompts that produced no pairs simply contribute nothing.

        The function imports `datasets` lazily so importing this module
        doesn't pay the import cost for callers that only use
        `create_prompt_sample` directly.
        """
        from datasets import Dataset

        all_pairs = []
        for x in prompts:
            all_pairs.extend(self.create_prompt_sample(x))
        return Dataset.from_list(all_pairs)

    def _build_iteration(self, prompts, iter_idx, output_dir):
        """Sample + veto + build pairs for one outer iteration.

        For each prompt: sample N candidates from the current policy,
        veto via the sim, cross-pair vetoed × non-vetoed.

        Persists every prompt's raw samples to
        `<output_dir>/responses_iter_<iter_idx+1>.jsonl` (one JSON object
        per line: iter, prompt_idx, prompt, responses, vetoed) BEFORE the
        contribute/skip branches, so ALL responses are saved — including
        all-vetoed and zero-vetoed prompts. Written incrementally and
        flushed per prompt so a crash mid-iteration still preserves what
        was generated. `vetoed` is the subset the sim rejected; because the
        sim is frozen, a future run can reload these and rebuild the exact
        pairs without re-sampling OR re-vetoing this round.

        Emits a flushed `[mem]` marker after the sampling phase and after
        the veto phase of every prompt (via `mem_probe.mem_line`), and
        resets each GPU's peak-memory counter per prompt so a single
        prompt's spike is attributable. `iter_idx` only tags those
        markers. The probes are counter/`/proc` reads — negligible cost
        against the generate calls they bracket.

        Inputs:
          prompts:    list of prompt strings.
          iter_idx:   outer-iteration index; tags mem markers + names the
                      responses file.
          output_dir: directory the responses_iter_*.jsonl file is written to.

        Returns a 4-tuple of stats so the outer loop can log per-iter
        diagnostics without re-walking the pairs list:
          pairs: list of {"prompt", "chosen", "rejected"} dicts.
          all_vetoed_idx: list of indices into `prompts` where every
            sampled response was vetoed (so the prompt contributes 0
            pairs this iter — it will be re-sampled next iter from the
            updated policy).
          zero_vetoed_idx: list of indices where NO response was
            vetoed (also 0 pairs — nothing to dispreference).
          contributing: count of prompts that produced ≥1 pair.
        """
        pairs = []
        all_vetoed_idx = []
        zero_vetoed_idx = []
        contributing = 0
        responses_path = os.path.join(output_dir, f"responses_iter_{iter_idx + 1}.jsonl")
        with open(responses_path, "w") as responses_file:
            for idx, x in enumerate(prompts):
                for d in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(d)

                # Sampling. Mixed sampling (variation 1) needs the prompt's
                # stakeholder data up front to generate stakeholder-voiced
                # candidates; fetching it here populates the sim's per-prompt
                # cache so the veto_responses call below reuses it for free.
                if self.mixed_sampling:
                    _candidates, entries = self.sim_stake._stakeholder_data_for(x)
                    responses, sources = self._sample_responses_mixed(x, entries)
                else:
                    responses = self._sample_responses(x)
                    sources = ["model"] * len(responses)
                mem_line(f"it{iter_idx} p{idx} sampled in_chars={len(x)} n={len(responses)}")

                vetoed = self.sim_stake.veto_responses(x, responses)
                mem_line(f"it{iter_idx} p{idx} vetoed v={len(vetoed)}")

                vetoed_set = set(vetoed)
                non_vetoed = [r for r in responses if r not in vetoed_set]

                # Veto-based cross pairs (every vetoed × every non-vetoed).
                # Empty when nothing was vetoed or nothing survived.
                prompt_pairs = []
                for r in vetoed:
                    for c in non_vetoed:
                        prompt_pairs.append({
                            "prompt": x,
                            "chosen": c,
                            "rejected": r,
                        })

                # Variation 2: add policy-labeled preference pairs over the
                # non-vetoed responses, on top of the veto pairs.
                num_pref_pairs = 0
                if self.self_preference and len(non_vetoed) >= 2:
                    pref_pairs = self._policy_preference_pairs(x, non_vetoed)
                    num_pref_pairs = len(pref_pairs)
                    prompt_pairs.extend(pref_pairs)
                    mem_line(f"it{iter_idx} p{idx} pref_pairs={num_pref_pairs}")

                # Persist raw samples + veto subset (and sources / pref count)
                # so every prompt's responses are saved; flush per prompt so a
                # mid-iteration crash still preserves generated work.
                responses_file.write(json.dumps({
                    "iter": iter_idx,
                    "prompt_idx": idx,
                    "prompt": x,
                    "responses": responses,
                    "sources": sources,
                    "vetoed": vetoed,
                    "num_pref_pairs": num_pref_pairs,
                }) + "\n")
                responses_file.flush()

                # Diagnostics. With self_preference on, a zero-vetoed prompt
                # (all accepted) can still contribute preference pairs, so
                # these index lists are no longer mutually exclusive with
                # `contributing` — they remain useful as veto-signal stats.
                if len(vetoed) == 0:
                    zero_vetoed_idx.append(idx)
                if len(non_vetoed) == 0:
                    all_vetoed_idx.append(idx)
                if prompt_pairs:
                    contributing += 1
                    pairs.extend(prompt_pairs)
        return pairs, all_vetoed_idx, zero_vetoed_idx, contributing

    def _retemplate_prompt(self, prompt):
        """Render `prompt` through the tokenizer's chat template using the
        same options `llama_local.build_chat_input` uses at sampling time:
        no system message, generation prompt appended, `enable_thinking
        =False` (silently ignored by non-Qwen3 templates).

        Why this exists: `create_dataset` stores the raw user prompt, but
        the chosen/rejected completions were generated against the
        chat-templated prefix. If DPO is handed the raw prompt, its log-
        probs correspond to "raw text → response" while the policy was
        actually shaped under "<chat-template> → response", and the
        gradient signal is scrambled. `train` calls this on the prompt
        column before constructing the DPOTrainer.
        """
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

    def train(self,
              prompts,
              output_dir,
              num_iterations=3,
              beta=0.1,
              learning_rate=5e-5,
              num_train_epochs=1,
              per_device_train_batch_size=1,
              gradient_accumulation_steps=8,
              max_length=1024,
              lora_r=16,
              lora_alpha=32,
              lora_dropout=0.05,
              lora_target_modules="all-linear",
              gradient_checkpointing=True,
              logging_steps=10,
              report_to="none"):
        """Iteratively fit a LoRA adapter on `self.model` (the policy) via
        DPO against preference pairs produced by sampling from the
        current policy + the (frozen) `sim_stake` veto signal.

        Outer loop, K = `num_iterations` iterations:
          1. Build a fresh preference dataset by re-sampling all prompts
             from the current policy (eval mode) and vetoing via the
             sim. Pairs are cross-products of vetoed × non-vetoed.
          2. Log per-iter stats: number of pairs, contributing prompts,
             all-vetoed prompts (the ones that contributed 0 pairs this
             iter because every response was rejected; they will be
             re-sampled next iter from the updated policy), and
             zero-vetoed prompts (every response accepted — also 0
             pairs, nothing to dispreference).
          3. If the iter produced ≥1 pair, run one inner DPO epoch on
             this iter's dataset. Otherwise, skip the training step
             but still proceed to the next iter — the policy is
             unchanged so iter K+1 sees the same sampling distribution,
             but stochastic sampling may still produce different
             responses.

        LoRA wrapping happens once before the loop (via
        `peft.get_peft_model`). All iterations train the same adapter.
        Per-iter DPOTrainer instances reuse the wrapped policy with
        `peft_config=None` so trl doesn't re-wrap; with a PeftModel +
        `ref_model=None`, trl disables the adapter on the same base
        for the reference log-prob computation, saving the ~16 GB a
        separate ref copy would cost.

        Assumptions:
          - `self.model` is an HF causal LM that LoRA can target.
          - `trl` and `peft` are installed (lazily imported below).
          - `sim_stake.call_llm` closes over a frozen sim_base, NOT
            the policy. The caller (e.g. `train_dpo_lora.py`) is
            responsible for wiring this; nothing here checks it.

        Inputs:
          prompts:                     iterable of prompt strings.
          output_dir:                  where DPOTrainer writes logs,
                                       per-iter subdirs, the final
                                       adapter, and `iter_stats.json`.
          num_iterations:              K outer iterations (default 3).
          beta, learning_rate, num_train_epochs,
          per_device_train_batch_size, gradient_accumulation_steps,
          max_length, lora_*, gradient_checkpointing,
          logging_steps, report_to:    standard DPO/LoRA/HF knobs;
                                       see prior comments for caveats.
                                       (trl>=1.x DPOConfig dropped
                                       max_prompt_length; max_length caps
                                       the full prompt+completion and
                                       truncates per truncation_mode.)

        Returns: the most-recent `DPOTrainer` instance (the iter K-1
          one), or None if every iter produced 0 pairs. The final LoRA
          adapter is saved to `output_dir` via
          `self.model.save_pretrained(output_dir)`. Per-iter stats are
          written to `output_dir/iter_stats.json`.
        """
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from trl import DPOConfig, DPOTrainer

        prompts = list(prompts)

        # Wrap the policy with LoRA ONCE. All iterations sample from
        # and train this same wrapped model; the adapter accumulates
        # across iters.
        target_modules = lora_target_modules
        if isinstance(target_modules, (list, tuple)):
            target_modules = list(target_modules)
        peft_config = LoraConfig(
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # Gradient checkpointing only saves memory on a LoRA model if the
        # input embeddings are forced to require grad. The base is frozen,
        # so without this the checkpointed segments have no grad path and HF
        # silently retains the FULL activation stack — which OOMs the 48 GB
        # GPU on an 8B model. Call it on the raw base model (which is a
        # PreTrainedModel and definitely exposes the method) BEFORE
        # LoRA-wrapping; the forward hook it registers on the input
        # embeddings persists through get_peft_model. Pair it with
        # non-reentrant checkpointing (use_reentrant=False, set in
        # base_dpo_args), the variant that supports a frozen base.
        if gradient_checkpointing:
            self.model.enable_input_require_grads()

        self.model = get_peft_model(self.model, peft_config)

        use_bf16 = next(self.model.parameters()).dtype == torch.bfloat16
        base_dpo_args = dict(
            beta=float(beta),
            learning_rate=float(learning_rate),
            num_train_epochs=float(num_train_epochs),
            per_device_train_batch_size=int(per_device_train_batch_size),
            gradient_accumulation_steps=int(gradient_accumulation_steps),
            max_length=int(max_length),
            # Compute the reference-model log-probs in a separate pre-pass
            # (no policy training graph resident) and cache them, instead of
            # running a ref forward inside every training step. The in-step
            # ref forward materialized a full-vocab logits tensor over the
            # concatenated chosen+rejected batch (~6 GB at max_length on
            # Qwen3-8B's 152k vocab) and OOMed the 48 GB GPU on long OEQ
            # sequences. Precomputing moves that allocation to a phase with
            # ample headroom and drops it from the per-step peak entirely.
            precompute_ref_log_probs=True,
            gradient_checkpointing=bool(gradient_checkpointing),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=use_bf16,
            remove_unused_columns=False,
            logging_steps=int(logging_steps),
            report_to=report_to,
            save_strategy="no",
        )

        os.makedirs(output_dir, exist_ok=True)
        last_trainer = None
        iter_stats = []
        for iter_idx in range(int(num_iterations)):
            tag = f"iter {iter_idx + 1}/{int(num_iterations)}"

            # Sampling — eval() suppresses dropout. _sample_responses
            # uses torch.no_grad internally, so no backprop graph is
            # built.
            self.model.eval()
            print(f"[{tag}] sampling from policy + vetoing...")
            pairs, all_vetoed_idx, zero_vetoed_idx, contributing = (
                self._build_iteration(prompts, iter_idx, output_dir)
            )
            print(
                f"[{tag}] pairs={len(pairs)}  "
                f"contributing_prompts={contributing}/{len(prompts)}  "
                f"all_vetoed_prompts={len(all_vetoed_idx)}  "
                f"zero_vetoed_prompts={len(zero_vetoed_idx)}"
            )
            iter_stats.append({
                "iter": iter_idx,
                "num_pairs": len(pairs),
                "contributing_prompts": contributing,
                "all_vetoed_prompt_indices": all_vetoed_idx,
                "zero_vetoed_prompt_indices": zero_vetoed_idx,
            })

            if pairs:
                mem_line(f"it{iter_idx} DPO: loop done, {len(pairs)} pairs")
                # Retemplate the prompt column in plain Python BEFORE building
                # the Dataset. Using Dataset.map() with a lambda that closes
                # over `self` makes `datasets` fingerprint the transform by
                # dill-pickling the closure — which drags in self.model AND the
                # frozen sim model held by self.sim_stake (two 8B nets),
                # materializing both on the host and OOMing the node. The
                # dataset here is tiny, so a comprehension is simpler and avoids
                # the pickling entirely.
                dataset = Dataset.from_list([
                    dict(p, prompt=self._retemplate_prompt(p["prompt"]))
                    for p in pairs
                ])
                mem_line(f"it{iter_idx} DPO: after dataset build")

                iter_out = os.path.join(output_dir, f"iter_{iter_idx + 1}")
                dpo_args = DPOConfig(output_dir=iter_out, **base_dpo_args)
                mem_line(f"it{iter_idx} DPO: after DPOConfig")

                # DPOTrainer needs the model in train mode to record
                # gradients on the LoRA params; sampling left it in eval.
                self.model.train()
                last_trainer = DPOTrainer(
                    model=self.model,
                    ref_model=None,
                    args=dpo_args,
                    train_dataset=dataset,
                    processing_class=self.tokenizer,
                    # Adapter already attached above; passing peft_config
                    # would re-wrap and nest.
                    peft_config=None,
                )
                mem_line(f"it{iter_idx} DPO: after DPOTrainer ctor")
                # Single process driving TWO models on TWO GPUs (policy on cuda:0,
                # frozen sim on cuda:1). HF Trainer otherwise sees device_count==2
                # and wraps the policy in nn.DataParallel, which replicates it onto
                # cuda:1 (colliding with the sim) and gathers the large-vocab logits
                # back onto cuda:0 — ~30 GB of overhead that OOMs the 48 GB card and
                # defeats gradient checkpointing. Force single-GPU training to skip
                # the DataParallel wrap (gated on args.n_gpu > 1 in
                # Trainer._wrap_model; transformers itself forces n_gpu=1 to do
                # exactly this). cuda:1's sim is the caller's model, untouched.
                last_trainer.args._n_gpu = 1
                last_trainer.train()
                mem_line(f"it{iter_idx} DPO: after train()")
            else:
                print(f"[{tag}] no pairs to train on; skipping DPO step")

            # Snapshot the policy's LoRA adapter AFTER this iteration's DPO
            # step so a downstream eval (eval_endorsement_heldout.py) can load
            # the model state at each iteration, not just the final one. On a
            # no-pair iteration the state is unchanged from the previous iter,
            # but we still write a dir so every iteration has a loadable
            # adapter. adapter_iter_K equals the final top-level adapter saved
            # after the loop. Only the small LoRA delta is written (~170 MB),
            # not the base weights, so K snapshots are cheap on disk.
            iter_adapter_dir = os.path.join(output_dir, f"adapter_iter_{iter_idx + 1}")
            self.model.save_pretrained(iter_adapter_dir)
            print(f"[{tag}] saved iteration adapter -> {iter_adapter_dir}")

        # Persist the final adapter + per-iter diagnostics.
        self.model.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "iter_stats.json"), "w") as f:
            json.dump(iter_stats, f, indent=2)

        if last_trainer is None:
            print(
                "[done] WARNING: every iteration produced 0 preference "
                "pairs. The saved adapter is untrained. Check the veto "
                "judge and the sampling diversity (temperature/top_p/"
                "n_samples)."
            )
        return last_trainer
