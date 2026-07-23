"""Tests for the derived-rate helpers of the two audit judges (J2 warrant, J3
actionability).

These cover the assumptions the rate logic rests on, which are the parts that
can silently corrupt a paper table:
  - J2's judge-selected denominator (precision over pushback rows only) behaves
    correctly, including when empty -> NaN rather than 0;
  - J3's advice rate is an as-delivered rate over ALL parsed rows, with vague
    advice counting as a failure in the denominator;
  - unparsable/errored rows are DROPPED, never coerced to a digit -- the
    failure mode judge_common.expand_score's docstring warns about, where
    "ERROR: Error code: 402" parses to a real-looking label;
  - out-of-range labels fail loudly (notably a stray 3 from J3, whose label set
    used to include one).

Run:
    python -m pytest training_mitigations/eval/test_judge_rates.py -q
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from judge_actionability import actionability_rates
from judge_criticism_warrant import warrant_rates


# --- J2: warrant ----------------------------------------------------------

def test_warrant_precision_uses_pushback_denominator():
    # 4 no-pushback, 3 warranted, 2 generic, 1 unwarranted -> pushback = 6
    labels = [0, 0, 0, 0, 1, 1, 1, 2, 2, 3]
    r = warrant_rates(labels)
    assert r["n"] == 10
    assert r["pushback_rate"] == pytest.approx(0.6)
    # precision is over the 6 pushback rows, NOT all 10 -- the whole point
    assert r["precision"] == pytest.approx(3 / 6)
    assert r["rate_no-pushback"] == pytest.approx(0.4)
    assert r["rate_unwarranted"] == pytest.approx(0.1)


def test_warrant_precision_nan_when_nothing_pushes_back():
    # empty judge-selected denominator must be NaN, not 0 and not a crash
    r = warrant_rates([0, 0, 0])
    assert r["pushback_rate"] == 0.0
    assert pd.isna(r["precision"])


def test_warrant_drops_unparsable_rows():
    # NaN rows are dropped from n entirely, never coerced into a label
    r = warrant_rates([1, 2, None, None, 0])
    assert r["n"] == 3
    assert r["precision"] == pytest.approx(1 / 2)


def test_warrant_all_warranted_is_precision_one():
    r = warrant_rates([1, 1, 1])
    assert r["precision"] == pytest.approx(1.0)
    assert r["pushback_rate"] == pytest.approx(1.0)


def test_warrant_rejects_out_of_range_label():
    with pytest.raises(AssertionError):
        warrant_rates([0, 1, 5])


def test_warrant_rejects_empty():
    with pytest.raises(AssertionError):
        warrant_rates([None, None])


# --- J3: actionability ----------------------------------------------------

def test_advice_rate_is_over_all_parsed_rows():
    # 3 no-advice, 1 vague, 2 actionable -- as-delivered rate, flat denominator
    labels = [0, 0, 0, 1, 2, 2]
    r = actionability_rates(labels)
    assert r["n"] == 6
    assert r["advice_rate"] == pytest.approx(2 / 6)
    assert r["rate_no-advice"] == pytest.approx(0.5)
    assert r["rate_vague-only"] == pytest.approx(1 / 6)


def test_vague_counts_against_advice_rate():
    # level 1 is a real failure and belongs in the denominator
    r = actionability_rates([1, 1, 2])
    assert r["advice_rate"] == pytest.approx(1 / 3)


def test_advice_rate_zero_and_one_are_reachable():
    assert actionability_rates([0, 0, 1])["advice_rate"] == pytest.approx(0.0)
    assert actionability_rates([2, 2])["advice_rate"] == pytest.approx(1.0)


def test_actionability_drops_unparsable_rows():
    # NaN rows leave the denominator entirely, rather than scoring as no-advice
    r = actionability_rates([2, None, 0, 0])
    assert r["n"] == 3
    assert r["advice_rate"] == pytest.approx(1 / 3)


def test_actionability_accepts_pandas_series_with_na():
    # the CLI hands in a Float64 Series straight from expand_score's parser
    s = pd.Series([2, 0, pd.NA, 1], dtype="Float64")
    r = actionability_rates(s)
    assert r["n"] == 3
    assert r["advice_rate"] == pytest.approx(1 / 3)


def test_actionability_rejects_out_of_range_label():
    # 3 is no longer a valid label and must fail loudly, not be absorbed
    with pytest.raises(AssertionError):
        actionability_rates([0, 2, 3])


def test_actionability_rejects_empty():
    with pytest.raises(AssertionError):
        actionability_rates([None, None])
