"""Regression tests for judge_common.expand_score's handling of API-failure cells.

The bug this guards: parallel_score writes a failed call into the label column as
f"ERROR: {e}". Those messages contain digits that land inside real label ranges
("ERROR: Error code: 402 ..." -> 4 on a 1-5 scale, 0 on a 0-1 scale), so a bare
digit parse invents a plausible label AND leaves the score non-NaN. Every
completeness check in the pipeline keys on NaN, so such a file is marked done and
never re-judged -- a silent, permanent corruption of a results column.

Run:
    python -m pytest training_mitigations/tests/test_judge_common_error_cells.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "eval"))

from judge_common import expand_score, make_digit_parser

ERR_402 = "ERROR: Error code: 402 - {'error': {'message': 'Insufficient credits'}}"


def test_digit_parser_alone_does_mis_parse_error_text():
    """Documents WHY the mask is needed: the raw parser really does read a label
    out of the 402 message, on both scale shapes."""
    assert make_digit_parser(1, 5)(ERR_402) == 4
    assert make_digit_parser(0, 1)(ERR_402) == 0


def test_error_cell_scores_nan_on_ordinal_scale():
    df = pd.DataFrame({"endorse_response": [ERR_402, "4"]})
    out = expand_score(df, "response", "empathy_validating_emotions", 1, 5)
    s = out["empathy_validating_emotions_score_response"]
    assert pd.isna(s[0]), "402 error cell must not become a label"
    assert s[1] == 0.75


def test_error_cell_scores_nan_on_binary_scale():
    df = pd.DataFrame({"endorse_response": [ERR_402, "1", "0"]})
    out = expand_score(df, "response", "framing", 0, 1)
    s = out["framing_score_response"]
    assert pd.isna(s[0])
    assert s[1] == 1.0 and s[2] == 0.0


def test_unjudged_and_valid_rows_unaffected():
    """An empty cell (cost-capped, never judged) stays NaN; real labels parse."""
    df = pd.DataFrame({"endorse_response": ["", "2", "5"]})
    out = expand_score(df, "response", "empathy_dismissing_emotions", 1, 5)
    s = out["empathy_dismissing_emotions_score_response"]
    assert pd.isna(s[0])
    # reverse=False here: (label - lo) / (hi - lo)
    assert s[1] == 0.25 and s[2] == 1.0


def test_reverse_coding_still_applies_to_valid_rows():
    df = pd.DataFrame({"endorse_response": [ERR_402, "1", "5"]})
    out = expand_score(df, "response", "empathy_dismissing_emotions", 1, 5,
                       reverse=True)
    s = out["empathy_dismissing_emotions_score_response"]
    assert pd.isna(s[0])
    assert s[1] == 1.0 and s[2] == 0.0


def test_raw_label_column_keeps_the_error_text():
    """The mask must not destroy evidence: the raw column still shows the ERROR
    so a failed run is auditable after the fact."""
    df = pd.DataFrame({"endorse_response": [ERR_402]})
    out = expand_score(df, "response", "framing", 0, 1)
    assert "ERROR" in str(out["framing_response"][0])
