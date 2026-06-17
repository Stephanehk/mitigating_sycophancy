"""Tests for `trainer._preference_pair_indices` — the enumeration order
behind the self-preference per-prompt pair cap.

Assumptions covered:
  1. Uncapped (max_pairs=0): exactly the full C(k, 2) set of unordered
     index pairs, each with i < j, no duplicates.
  2. Capped: at most max_pairs pairs are returned.
  3. Capped subsets are balanced: no single response index dominates the
     kept pairs (the lexicographic nested-loop order this replaced kept
     response 0 in 9 of 10 pairs at k=10, cap=10).
  4. Degenerate k (0 or 1): empty list.

Run with the project interpreter:
  /next/u/stephhk/miniconda3/envs/llm/bin/python training_mitigations/test_preference_pairs.py
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

from trainer import _preference_pair_indices


def test_uncapped_is_complete():
    for k in (2, 3, 5, 10):
        pairs = _preference_pair_indices(k, 0)
        expected = {(i, j) for i in range(k) for j in range(i + 1, k)}
        assert all(i < j for i, j in pairs), f"k={k}: pair with i >= j"
        assert len(pairs) == len(set(pairs)) == k * (k - 1) // 2, f"k={k}: wrong count"
        assert set(pairs) == expected, f"k={k}: missing pairs"


def test_cap_respected():
    assert len(_preference_pair_indices(10, 10)) == 10
    assert len(_preference_pair_indices(10, 100)) == 45  # cap above C(10,2)
    assert len(_preference_pair_indices(3, 1)) == 1


def test_capped_subset_is_balanced():
    # k=10, cap=10: offset-1 contributes 9 pairs covering all indices,
    # then one offset-2 pair. Every index must appear, and none may
    # appear more than 3 times (lexicographic order would give index 0
    # a count of 9).
    pairs = _preference_pair_indices(10, 10)
    counts = Counter(i for p in pairs for i in p)
    assert set(counts) == set(range(10)), f"some index never paired: {counts}"
    assert max(counts.values()) <= 3, f"unbalanced capped subset: {counts}"


def test_degenerate_k():
    assert _preference_pair_indices(0, 0) == []
    assert _preference_pair_indices(1, 10) == []


if __name__ == "__main__":
    test_uncapped_is_complete()
    test_cap_respected()
    test_capped_subset_is_balanced()
    test_degenerate_k()
    print("all tests passed")
