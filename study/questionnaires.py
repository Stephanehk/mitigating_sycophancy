"""Scale definitions for the sycophancy user study.

Every item, anchor label and response option is VERBATIM from
`syco_label_mitigations_-_Copy_for_stephane.qsf`. The QSF question id and its
`DataExportTag` are recorded on each page so the instruments can be diffed
against the source, and so the exported column names match the original
study's tags wherever the item is unchanged.

Two question shapes are used, matching the QSF's two Qualtrics types:
  "matrix" — a Likert grid (QSF `Matrix / Likert / SingleAnswer`): several
             `items`, one shared set of `options`.
  "single" — one horizontal radio row (QSF `MC / SAHR`): one `options` list,
             no items.

`options` values are the QSF choice ids, so a stored value of 5 means the same
thing here as in the original study's data. This matters for the DV pages,
where the QSF's choice ids are NOT always 1..7 — `response_objectivity`,
`response_quality` and `rightorwrong` skip ids, and those gaps are preserved
below rather than renumbered. `scale_point` gives the 1..7 position for
analysis; export_responses.py writes that alongside the raw value.

PAGE ORDER reproduces the QSF's "The DVs" block, including its page breaks:
  [fk, MDMT] | [response_objectivity, response_quality] | [rightorwrong] |
  [Q11] | [return_likelihood]
then the attention check, AI use, and Big Five blocks. The demographics block
(QID127/128/130) is intentionally absent.

WORDING NOTE: several DV stems say "in this conversation" or "responses"
(plural), which the source survey earned with an 8-turn chat. This study shows
one response. The stems are left verbatim so the instrument stays comparable
to the source study; see README.md "Items needing your sign-off".

DEVIATIONS from the QSF, both made on 2026-07-27 at the researcher's request:

  1. EVERY question is mandatory (`required: True`). The source survey left the
     attention check, AI use and Big Five blocks with ForceResponse OFF.
     static/widgets.js enforces this — it highlights the specific unanswered
     question and blocks Continue. The server does NOT re-reject at submit time:
     by then the participant has finished, and losing a whole session over one
     blank cell would be worse than recording the gap, so
     flow.validate_questionnaire_answers reports it in `missing_keys` instead.
  2. The two chatbot checklists are REMOVED — QID131 ("which ... have you heard
     of?") and QID132 ("which ... have you used?"). `frequency` and `AIAS` still
     carry the AI-exposure signal. No `multi` question remains in the study as a
     result, though flow.py and widgets.js still support the shape.
"""

# --- Shared anchor sets ----------------------------------------------------

AGREE_7 = [
    {"value": 1, "label": "Strongly disagree", "scale_point": 1},
    {"value": 2, "label": "Disagree", "scale_point": 2},
    {"value": 3, "label": "Somewhat disagree", "scale_point": 3},
    {"value": 4, "label": "Neither agree nor disagree", "scale_point": 4},
    {"value": 5, "label": "Somewhat agree", "scale_point": 5},
    {"value": 6, "label": "Agree", "scale_point": 6},
    {"value": 7, "label": "Strongly agree", "scale_point": 7},
]

# QID116's anchors: endpoints and midpoint labelled, 2/3/5/6 bare.
AGREE_7_NUMBERED = [
    {"value": 1, "label": "Strongly disagree", "sublabel": "1", "scale_point": 1},
    {"value": 2, "label": "2", "scale_point": 2},
    {"value": 3, "label": "3", "scale_point": 3},
    {"value": 4, "label": "Neither agree nor disagree", "sublabel": "4", "scale_point": 4},
    {"value": 5, "label": "5", "scale_point": 5},
    {"value": 6, "label": "6", "scale_point": 6},
    {"value": 7, "label": "Strongly agree", "sublabel": "7", "scale_point": 7},
]


def _numeric_row(low_label, high_label, values):
    """Build a 7-point horizontal radio row with only its endpoints labelled.

    `values` are the QSF choice ids in display order; they are NOT assumed to be
    1..7 because several QSF questions skip ids. `scale_point` is the 1-based
    display position, which is what analysis should use.

    Assumption: exactly 7 points, matching every such item in the source survey.
    """
    assert len(values) == 7, "expected a 7-point row, got %d" % (len(values),)
    row = []
    for position, value in enumerate(values, start=1):
        option = {"value": value, "label": str(position), "scale_point": position}
        if position == 1:
            option["endpoint"] = low_label
        elif position == 7:
            option["endpoint"] = high_label
        row.append(option)
    return row


# --- The DVs (QSF block 9) -------------------------------------------------

DV_PAGES = [
    {
        "page": "dv_felt_understood_trust",
        "questions": [
            {
                "id": "fk",
                "qsf_qid": "QID196",
                "type": "matrix",
                "required": True,
                "text": (
                    "To what extent do you agree with the following statements? "
                    "In this conversation, this AI chatbot..."
                ),
                "items": [
                    {"id": "fk_1", "label": "...made me feel understood"},
                    {"id": "fk_2", "label": "...made me feel validated"},
                    {"id": "fk_3", "label": "...made me feel affirmed"},
                    {"id": "fk_4", "label": "...made me feel seen"},
                    {"id": "fk_8", "label": "...made me feel accepted"},
                    {"id": "fk_9", "label": "...made me feel cared for"},
                ],
                "options": AGREE_7,
            },
            {
                "id": "MDMT",
                "qsf_qid": "QID248",
                "type": "matrix",
                "required": True,
                "text": (
                    "To what extent do you agree with the following statements? "
                    "This AI chatbot..."
                ),
                "items": [
                    {"id": "MDMT_1", "label": "...is moral"},
                    {"id": "MDMT_2", "label": "...is truthful"},
                    {"id": "MDMT_3", "label": "...is considerate"},
                    {"id": "MDMT_4", "label": "...has integrity"},
                    {"id": "MDMT_10", "label": "...is genuine"},
                    {"id": "MDMT_11", "label": "...has goodwill"},
                    {"id": "MDMT_8", "label": "...is dependable"},
                    {"id": "MDMT_12", "label": "...is capable"},
                    {"id": "MDMT_13", "label": "...is consistent"},
                    {"id": "MDMT_14", "label": "...is meticulous"},
                ],
                "options": AGREE_7,
            },
        ],
    },
    {
        "page": "dv_objectivity_quality",
        "questions": [
            {
                "id": "response_objectivity",
                "qsf_qid": "QID201",
                "type": "single",
                "required": True,
                "text": "Rate how objective the AI chatbot’s responses seemed in this conversation.",
                # QSF choice ids: 1,2,3,4,5,7,8 (6 is skipped).
                "options": _numeric_row("Not at all", "Very", [1, 2, 3, 4, 5, 7, 8]),
            },
            {
                "id": "response_quality",
                "qsf_qid": "QID202",
                "type": "single",
                "required": True,
                "text": "Rate the overall quality of the AI chatbot’s responses in this conversation.",
                # QSF choice ids: 1,2,3,4,5,7,8 (6 is skipped).
                "options": _numeric_row(
                    "Extremely low quality", "Extremely high quality", [1, 2, 3, 4, 5, 7, 8]),
            },
        ],
    },
    {
        "page": "dv_rightorwrong",
        "questions": [
            {
                "id": "rightorwrong",
                "qsf_qid": "QID143",
                "type": "single",
                "required": True,
                "text": "In your opinion, was your behavior in your situation in the right or in the wrong?",
                # QSF choice ids: 7,14,15,16,17,18,19.
                "options": [
                    {"value": 7, "label": "1", "endpoint": "Completely wrong", "scale_point": 1},
                    {"value": 14, "label": "2", "scale_point": 2},
                    {"value": 15, "label": "3", "scale_point": 3},
                    {"value": 16, "label": "4", "endpoint": "Neither right nor wrong", "scale_point": 4},
                    {"value": 17, "label": "5", "scale_point": 5},
                    {"value": 18, "label": "6", "scale_point": 6},
                    {"value": 19, "label": "7", "endpoint": "Completely right", "scale_point": 7},
                ],
            },
        ],
    },
    {
        "page": "dv_repair",
        "questions": [
            {
                "id": "repair",
                "qsf_qid": "QID116",
                "type": "matrix",
                "required": True,
                "text": (
                    "Please rate how much you agree with the following statements "
                    "about the situation you described."
                ),
                "items": [
                    {"id": "repair_1", "label": "I should apologize for what happened."},
                    {"id": "repair_2", "label": "I should do something after this incident to make it better."},
                    {"id": "repair_3",
                     "label": "I should change certain aspects of myself so that this wouldn’t happen again."},
                ],
                "options": AGREE_7_NUMBERED,
            },
        ],
    },
    {
        "page": "dv_return_likelihood",
        "questions": [
            {
                "id": "return_likelihood",
                "qsf_qid": "QID204",
                "type": "single",
                "required": True,
                "text": "How likely would you be to ask the AI chatbot for similar questions in the future?",
                # QSF choice ids here ARE 1..7.
                "options": _numeric_row("Not likely at all", "Very likely", [1, 2, 3, 4, 5, 6, 7]),
            },
        ],
    },
]


# --- Attention check (QSF block 10, QID126) --------------------------------
# DEVIATION: not required in the QSF (ForceResponse OFF), but mandatory here —
# every question in this study is (see the module note below), at the
# researcher's request on 2026-07-27. A non-response is therefore no longer
# possible except by abandoning the study.
# `attention_expected` records the correct answer for scoring in the export.

ATTENTION_CHECK_PAGE = {
    "page": "attention_check",
    "questions": [
        {
            "id": "attention_check",
            "qsf_qid": "QID126",
            "type": "matrix",
            "required": True,
            "text": "As an attention check, please select option 0 for each category below.",
            "items": [
                {"id": "attention_check_1", "label": "Your situation"},
                {"id": "attention_check_2", "label": "The A.I. responses"},
                {"id": "attention_check_3", "label": "Your responses"},
            ],
            "options": [
                {"value": 1, "label": "Extremely Low", "sublabel": "0", "scale_point": 1},
                {"value": 2, "label": "1", "scale_point": 2},
                {"value": 3, "label": "2", "scale_point": 3},
                {"value": 4, "label": "3", "scale_point": 4},
                {"value": 5, "label": "4", "scale_point": 5},
                {"value": 6, "label": "Extremely High", "sublabel": "5", "scale_point": 6},
            ],
            "attention_expected": 1,
        },
    ],
}


# --- AI use (QSF block 12, QID131-134) -------------------------------------

AI_USE_PAGE = {
    "page": "ai_use",
    "questions": [
        # REMOVED (2026-07-27, at the researcher's request): the two chatbot
        # checklists, QID131 "which ... have you heard of?" and QID132
        # "which ... have you used?". `frequency` and `AIAS` below still carry
        # the AI-exposure signal. Their `_CHATBOT_CHOICES` option list went with
        # them, as nothing else used it.
        {
            "id": "frequency",
            "qsf_qid": "QID133",
            "type": "vertical",
            "required": True,
            "text": "How often do you use AI chatbots?",
            "options": [
                {"value": 1, "label": "Daily"},
                {"value": 2, "label": "A few times a week"},
                {"value": 3, "label": "A few times a month"},
                {"value": 4, "label": "Once every few months"},
                {"value": 5, "label": "Never"},
            ],
        },
        {
            "id": "AIAS",
            "qsf_qid": "QID134",
            "type": "matrix",
            "required": True,
            "text": "How much do you agree with the following statements?",
            "items": [
                {"id": "AIAS_1", "label": "I believe that AI will improve my life."},
                {"id": "AIAS_2", "label": "I believe that AI will improve my work."},
                {"id": "AIAS_3", "label": "I think I will use AI technology in the future."},
                {"id": "AIAS_4", "label": "I think AI technology is positive for humanity."},
                {"id": "AIAS_5", "label": "As an attention check, please select 10 (Completely agree)"},
            ],
            "options": (
                [{"value": 1, "label": "Not at all", "sublabel": "1", "scale_point": 1}]
                + [{"value": v, "label": str(v), "scale_point": v} for v in range(2, 10)]
                + [{"value": 10, "label": "Completely Agree", "sublabel": "10", "scale_point": 10}]
            ),
            "attention_item": "AIAS_5",
            "attention_expected": 10,
        },
    ],
}


# --- Big Five (QSF block 13, QID135) ---------------------------------------

BIG_FIVE_PAGE = {
    "page": "big_five",
    "questions": [
        {
            "id": "big_five",
            "qsf_qid": "QID135",
            "type": "matrix",
            "required": True,
            "text": "I see myself as someone who ...",
            "items": [
                {"id": "big_five_1", "label": "... is reserved"},
                {"id": "big_five_2", "label": "... is generally trusting"},
                {"id": "big_five_3", "label": "... tends to be lazy"},
                {"id": "big_five_4", "label": "... is relaxed, handles stress well"},
                {"id": "big_five_5", "label": "... has few artistic interests"},
                {"id": "big_five_6", "label": "... is outgoing, sociable"},
                {"id": "big_five_7", "label": "... tends to find fault with others"},
                {"id": "big_five_8", "label": "... does a thorough job"},
                {"id": "big_five_9", "label": "... gets nervous easily"},
                {"id": "big_five_10", "label": "... has an active imagination"},
            ],
            "options": [
                {"value": 1, "label": "Disagree strongly", "scale_point": 1},
                {"value": 2, "label": "Disagree a little", "scale_point": 2},
                {"value": 3, "label": "Neither agree nor disagree", "scale_point": 3},
                {"value": 4, "label": "Agree a little", "scale_point": 4},
                {"value": 5, "label": "Agree strongly", "scale_point": 5},
            ],
        },
    ],
}


# Every questionnaire page, in presentation order.
QUESTIONNAIRE_PAGES = DV_PAGES + [ATTENTION_CHECK_PAGE, AI_USE_PAGE, BIG_FIVE_PAGE]

QUESTIONNAIRE_PAGE_IDS = [p["page"] for p in QUESTIONNAIRE_PAGES]


def get_page(page_id):
    """Return one questionnaire page by id, or raise if unknown."""
    for page in QUESTIONNAIRE_PAGES:
        if page["page"] == page_id:
            return page
    raise AssertionError("unknown questionnaire page: %r" % (page_id,))


def all_questions():
    """Yield every question dict across all pages, in presentation order."""
    for page in QUESTIONNAIRE_PAGES:
        for question in page["questions"]:
            yield question


def response_keys(question):
    """The storage keys a single question produces.

    A matrix question stores one value per item, keyed by the item id; every
    other shape stores a single value keyed by the question id. Used by both
    the server's validation and the analysis export so the two cannot drift.
    """
    if question["type"] == "matrix":
        return [item["id"] for item in question["items"]]
    return [question["id"]]


def all_response_keys():
    """Every storage key the questionnaire can produce, in presentation order."""
    keys = []
    for question in all_questions():
        keys.extend(response_keys(question))
    return keys
