"""Tests for core/self_distill_trainer.py pure helpers + the masked SFT CE.

Run: /projects/m000254/envs/llm126/bin/python -m pytest \
        training_mitigations/tests/test_self_distill.py -q
(or execute directly; a __main__ runner is provided.)
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

from self_distill_trainer import (  # noqa: E402
    select_sft_targets,
    annotate_pairs_with_sft_mask,
    masked_chosen_ce,
)


def test_select_targets_model_only_dedup_cap():
    responses = ["A", "B", "C", "A", "S1", "P1", "V1"]
    sources = ["model", "model", "model", "model", "stakeholder:x",
               "perspshift", "model"]
    # V1 (model) is vetoed -> excluded from non_vetoed; A duplicated -> deduped.
    non_vetoed = ["A", "B", "C", "S1", "P1"]
    targets = select_sft_targets(responses, sources, non_vetoed, max_per_prompt=4)
    assert targets == ["A", "B", "C"], targets
    # Cap binds:
    assert select_sft_targets(responses, sources, non_vetoed, 2) == ["A", "B"]
    # Non-model sources never selected even if non-vetoed:
    assert "S1" not in targets and "P1" not in targets


def test_select_targets_excludes_vetoed_model():
    responses = ["A", "B"]
    sources = ["model", "model"]
    non_vetoed = ["A"]  # B was vetoed
    assert select_sft_targets(responses, sources, non_vetoed, 4) == ["A"]


def test_annotate_one_row_per_distinct_target():
    # A survives-veto (model), cross-paired with two vetoed V1,V2; B another
    # model survivor. Only ONE row per distinct chosen gets sft_mask=1.
    pairs = [
        {"prompt": "p", "chosen": "A", "rejected": "V1"},
        {"prompt": "p", "chosen": "A", "rejected": "V2"},
        {"prompt": "p", "chosen": "B", "rejected": "V1"},
        {"prompt": "p", "chosen": "S1", "rejected": "V1"},  # stakeholder chosen
    ]
    annotate_pairs_with_sft_mask(pairs, targets=["A", "B"])
    assert [p["sft_mask"] for p in pairs] == [1, 0, 1, 0]


def test_annotate_no_targets_all_zero():
    pairs = [{"prompt": "p", "chosen": "S1", "rejected": "V1"}]
    annotate_pairs_with_sft_mask(pairs, targets=[])
    assert pairs[0]["sft_mask"] == 0


def _toy_batch(batch_pairs=3, seq=5, vocab=7, prompt_len=2):
    """Build a trl-shaped (chosen||rejected) toy batch. Returns
    (shift_logits, shift_labels, shift_completion_mask) already shifted."""
    torch.manual_seed(0)
    two_b = 2 * batch_pairs
    logits = torch.randn(two_b, seq, vocab, requires_grad=True)
    input_ids = torch.randint(0, vocab, (two_b, seq))
    # completion_mask: prompt tokens 0, completion tokens 1
    completion_mask = torch.zeros(two_b, seq, dtype=torch.long)
    completion_mask[:, prompt_len:] = 1
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_completion_mask = completion_mask[..., 1:].contiguous()
    return shift_logits, shift_labels, shift_completion_mask


def test_masked_ce_all_ones_matches_trl_unmasked():
    """With every row eligible, masked_chosen_ce must equal trl's own sft term:
    F.cross_entropy over ALL chosen completion tokens."""
    sl, lab, cm = _toy_batch()
    b = sl.size(0) // 2
    chosen_logits, _ = sl.chunk(2, 0)
    chosen_labels, _ = lab.chunk(2, 0)
    chosen_mask, _ = cm.chunk(2, 0)
    trl_ref = F.cross_entropy(
        chosen_logits[chosen_mask.bool()], chosen_labels[chosen_mask.bool()])

    ours_none = masked_chosen_ce(sl, lab, cm, sft_mask=None)
    ours_ones = masked_chosen_ce(sl, lab, cm, sft_mask=torch.ones(b, dtype=torch.long))
    assert torch.allclose(ours_none, trl_ref), (ours_none, trl_ref)
    assert torch.allclose(ours_ones, trl_ref), (ours_ones, trl_ref)


def test_masked_ce_subset_matches_manual():
    sl, lab, cm = _toy_batch()
    b = sl.size(0) // 2
    mask = torch.zeros(b, dtype=torch.long)
    mask[0] = 1  # only the first pair's chosen is a distill target
    chosen_logits, _ = sl.chunk(2, 0)
    chosen_labels, _ = lab.chunk(2, 0)
    chosen_mask, _ = cm.chunk(2, 0)
    tok = chosen_mask.bool().clone()
    tok[1:] = False  # only row 0 tokens
    manual = F.cross_entropy(chosen_logits[tok], chosen_labels[tok])
    ours = masked_chosen_ce(sl, lab, cm, sft_mask=mask)
    assert torch.allclose(ours, manual), (ours, manual)


def test_masked_ce_all_zero_is_zero_and_differentiable():
    sl, lab, cm = _toy_batch()
    b = sl.size(0) // 2
    ours = masked_chosen_ce(sl, lab, cm, sft_mask=torch.zeros(b, dtype=torch.long))
    assert ours.item() == 0.0
    ours.backward()  # graph-connected zero -> no error


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} tests passed")
