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
  - Saves the final LoRA adapter to `output_dir`. The base weights
    are not changed on disk.

Sampling uses the same chat-template path as `llama_local.make_call_llm`
(via `llama_local.build_chat_input`) and Qwen3's `enable_thinking=False`
quirk so the sampled responses are in the same surface format the sim
saw at probe time; the `train` method re-templates the prompt column
the same way so DPO sees the same surface that produced the responses.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llama_local import build_chat_input
from mem_probe import mem_line


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
                 top_p=0.95):
        """
        sim_stake:     a SimulatedStakeholders instance (see
                       simulated_stakeholders.py).
        model:         a HuggingFace AutoModelForCausalLM already loaded
                       onto a CUDA device.
        tokenizer:     the matching tokenizer; must have a chat template.
        n_samples:     N — number of candidate responses sampled per
                       prompt (default 10). All N share the same prompt
                       and chat-template prefix; only the sampling RNG
                       differs.
        max_new_tokens: generation cap per response. 512 is enough for
                       OEQ-style advice replies; bump if responses get
                       truncated.
        temperature, top_p: standard HF generation knobs. Both apply iff
                       do_sample=True, which we force on (greedy would
                       make all N samples identical).
        """
        assert temperature > 0, (
            f"temperature must be > 0 for stochastic sampling (got {temperature}); "
            f"otherwise all {n_samples} samples are identical and no pairs are produced"
        )
        self.sim_stake = sim_stake
        self.model = model
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def _sample_responses(self, prompt):
        """Draw `n_samples` stochastic responses from the model for
        `prompt`. Returns a list of strings of length `n_samples`.

        Uses `num_return_sequences=n` so HF batches the N forward passes
        — much cheaper than N sequential `generate` calls. Each sample
        is decoded with `skip_special_tokens=True`; the prompt tokens
        are sliced off before decoding so only the new text is returned.
        """
        device = next(self.model.parameters()).device
        input_ids = build_chat_input(self.tokenizer, prompt, system=None, device=device)
        in_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=int(self.max_new_tokens),
                do_sample=True,
                temperature=float(self.temperature),
                top_p=float(self.top_p),
                num_return_sequences=int(self.n_samples),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        responses = []
        for i in range(self.n_samples):
            new_ids = output_ids[i, in_len:]
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            responses.append(text)
        return responses

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
                responses = self._sample_responses(x)
                mem_line(f"it{iter_idx} p{idx} sampled in_chars={len(x)} n={len(responses)}")
                vetoed = self.sim_stake.veto_responses(x, responses)
                mem_line(f"it{iter_idx} p{idx} vetoed v={len(vetoed)}")
                # Persist raw samples + veto subset before the skip branches
                # so every prompt's responses are saved; flush per prompt so a
                # mid-iteration crash still preserves generated work.
                responses_file.write(json.dumps({
                    "iter": iter_idx,
                    "prompt_idx": idx,
                    "prompt": x,
                    "responses": responses,
                    "vetoed": vetoed,
                }) + "\n")
                responses_file.flush()
                vetoed_set = set(vetoed)
                non_vetoed = [r for r in responses if r not in vetoed_set]
                if len(vetoed) == 0:
                    zero_vetoed_idx.append(idx)
                    continue
                if len(non_vetoed) == 0:
                    all_vetoed_idx.append(idx)
                    continue
                contributing += 1
                for r in vetoed:
                    for c in non_vetoed:
                        pairs.append({
                            "prompt": x,
                            "chosen": c,
                            "rejected": r,
                        })
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

            if not pairs:
                print(f"[{tag}] no pairs to train on; skipping DPO step")
                continue

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
