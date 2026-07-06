"""Targeted ("self") distill for the RPO/distill DPO variant.

Background: the distill runs add an SFT/NLL term on the DPO `chosen` responses
(trl multi-loss `loss_type=["sigmoid", "sft"]`, `loss_weights=[1.0, rpo_alpha]`).
trl's `sft` term pools EVERY `chosen` completion in the micro-batch into one
cross-entropy, so it imitates the whole veto-surviving pool — including the
stakeholder-voice / perspshift / frozen-base exemplars, which skew critical.
Empirically the over-criticality only appears once that SFT term is on.

This module narrows the SFT/distill target WITHOUT touching the DPO preference
signal: the DPO cross-pairs stay exactly as the parent recipe builds them (all
sources as `chosen`), but the SFT cross-entropy is masked to only the responses
sampled on-policy from the model itself (source == "model") and non-vetoed, at
most `distill_max_per_prompt` distinct ones per prompt. The mask rides a per-row
`sft_mask` column threaded through trl's data pipeline.

Two pure helpers (`select_sft_targets`, `annotate_pairs_with_sft_mask`) decide
the per-row mask from a prompt's (responses, sources, non_vetoed); the trainer
subclass `SelfDistillDPOTrainer` consumes the `sft_mask` column at loss time.

Design note (why this is a thin subclass, not a re-implemented loss): the DPO
loss is computed by `super()._compute_loss` with the `sft` component removed,
which returns the DPO loss AND the model's forward `outputs`. We then add
`rpo_alpha * masked_CE(chosen)` computed from those SAME logits — one forward,
and the DPO math is trl's verbatim. With an all-ones mask this is numerically
identical to trl's built-in `sft` term (see tests/test_self_distill.py).
"""

import dataclasses

import torch
import torch.nn.functional as F


def select_sft_targets(responses, sources, non_vetoed, max_per_prompt):
    """Return the list of distinct `chosen` strings eligible as SFT/distill
    targets for one prompt: responses sampled on-policy from the model itself
    (source == "model") that survived the veto, in sample order, deduped, and
    capped at `max_per_prompt`.

    Assumptions:
      - `responses` and `sources` are parallel lists (same length); `sources[i]`
        is the source tag recorded by the mixed sampler ("model", "critical",
        "perspshift", "stakeholder:<name>", "stakeholder-base:<name>").
      - `non_vetoed` is this prompt's surviving responses (already excludes
        vetoed and unparsable-veto responses).
      - `max_per_prompt >= 1`.
    """
    assert len(responses) == len(sources), (
        f"responses ({len(responses)}) and sources ({len(sources)}) must be "
        f"parallel lists"
    )
    assert max_per_prompt >= 1, f"max_per_prompt must be >= 1 (got {max_per_prompt})"
    non_vetoed_set = set(non_vetoed)
    targets = []
    seen = set()
    for r, s in zip(responses, sources):
        if s == "model" and r in non_vetoed_set and r not in seen:
            seen.add(r)
            targets.append(r)
            if len(targets) >= max_per_prompt:
                break
    return targets


def annotate_pairs_with_sft_mask(pairs, targets):
    """Set `sft_mask` on every pair dict in `pairs` (mutates in place).

    Exactly one row per distinct target `chosen` gets `sft_mask=1` (the first
    row whose `chosen` is that target); every other row gets `sft_mask=0`. This
    deduplicates the distill target so a model survivor is imitated once, not
    once per vetoed partner it was cross-paired with.
    """
    targets_set = set(targets)
    marked = set()
    for p in pairs:
        c = p["chosen"]
        if c in targets_set and c not in marked:
            p["sft_mask"] = 1
            marked.add(c)
        else:
            p["sft_mask"] = 0
    return pairs


def masked_chosen_ce(shift_logits, shift_labels, shift_completion_mask, sft_mask):
    """Cross-entropy over the CHOSEN completion tokens of the SFT-eligible rows.

    Mirrors trl's `sft` term (`F.cross_entropy` over all chosen completion
    tokens, mean reduction) but restricted to rows where `sft_mask` is truthy.
    Returns a 0-dim tensor; when no eligible token is present it returns a
    graph-connected zero so the backward pass stays well-defined.

    Shapes: `shift_logits` [2B, T, V] (chosen concatenated with rejected, as trl
    lays out the batch), `shift_labels`/`shift_completion_mask` [2B, T],
    `sft_mask` [B] (one entry per pair, aligned with the chosen half). `sft_mask`
    may be None (all rows eligible → identical to trl's unmasked term).
    """
    chosen_logits, _ = shift_logits.chunk(2, dim=0)
    chosen_labels, _ = shift_labels.chunk(2, dim=0)
    chosen_cmask, _ = shift_completion_mask.chunk(2, dim=0)
    tok = chosen_cmask.bool()
    if sft_mask is not None:
        tok = tok & sft_mask.bool().to(tok.device).unsqueeze(1)
    if tok.any():
        return F.cross_entropy(chosen_logits[tok], chosen_labels[tok])
    return chosen_logits.sum() * 0.0


class SelfDistillDPOTrainer:
    """Factory for a DPOTrainer subclass that masks the SFT/distill term.

    Kept as a factory rather than a module-level subclass so `trl` is only
    imported where the trainer is constructed (matching core/trainer.py, which
    imports trl lazily inside `train`). The pure helpers above stay import-light
    (torch only) so the pair-building code can use them without pulling in trl.
    """

    def __new__(cls, *args, **kwargs):
        raise TypeError("use SelfDistillDPOTrainer.build() to get the class")

    @staticmethod
    def build():
        from trl import DPOTrainer
        from trl.trainer.dpo_trainer import DataCollatorForPreference

        class _PreferenceCollatorWithSFTMask(DataCollatorForPreference):
            """trl's preference collator + a per-row `sft_mask` tensor."""

            def torch_call(self, examples):
                output = super().torch_call(examples)
                if "sft_mask" in examples[0]:
                    output["sft_mask"] = torch.tensor(
                        [ex["sft_mask"] for ex in examples], dtype=torch.long
                    )
                return output

        class _SelfDistillDPOTrainer(DPOTrainer):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                assert not getattr(self, "use_liger_kernel", False), (
                    "SelfDistillDPOTrainer does not support the liger kernel "
                    "path (the masked SFT term reuses the standard forward)."
                )
                # Swap in the sft_mask-aware collator, copying the config trl
                # built for the base one.
                base = self.data_collator
                self.data_collator = _PreferenceCollatorWithSFTMask(
                    **dataclasses.asdict(base)
                )

            def _set_signature_columns_if_needed(self):
                # Keep `sft_mask` from being stripped by remove_unused_columns.
                super()._set_signature_columns_if_needed()
                if "sft_mask" not in self._signature_columns:
                    self._signature_columns.append("sft_mask")

            def compute_ref_log_probs(self, inputs):
                # The ref-logp precompute forward must not see sft_mask.
                if "sft_mask" in inputs:
                    inputs = {k: v for k, v in inputs.items() if k != "sft_mask"}
                return super().compute_ref_log_probs(inputs)

            def _compute_loss(self, model, inputs, return_outputs):
                sft_mask = inputs.get("sft_mask")
                model_inputs = {k: v for k, v in inputs.items() if k != "sft_mask"}

                # DPO loss via trl verbatim, with the sft component removed so we
                # can re-add it masked. super() returns the model forward outputs
                # too, so the masked SFT reuses the SAME forward (no extra pass).
                saved_types = self.loss_types
                saved_weights = self.loss_weights
                sft_weight = None
                keep_types, keep_weights = [], []
                for t, w in zip(saved_types, saved_weights):
                    if t == "sft":
                        sft_weight = w
                    else:
                        keep_types.append(t)
                        keep_weights.append(w)
                try:
                    self.loss_types = keep_types
                    self.loss_weights = keep_weights
                    dpo_loss, outputs = super()._compute_loss(
                        model, model_inputs, True
                    )
                finally:
                    self.loss_types = saved_types
                    self.loss_weights = saved_weights

                if sft_weight is None:
                    loss = dpo_loss
                else:
                    input_ids = model_inputs["input_ids"]
                    completion_mask = model_inputs["completion_mask"]
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    shift_completion_mask = completion_mask[..., 1:].contiguous()
                    sft_loss = masked_chosen_ce(
                        shift_logits, shift_labels, shift_completion_mask, sft_mask
                    )
                    loss = dpo_loss + sft_weight * sft_loss

                return (loss, outputs) if return_outputs else loss

        return _SelfDistillDPOTrainer
