"""The four interpersonal-dilemma screening scenarios.

Every string here is lifted verbatim from `syco_label_mitigations_-_Copy_for
_stephane.qsf` (blocks 26 / 27 / 29 / 31, QID210 / QID222 / QID233 / QID244).
The QSF stored these as HTML fragments; they are decomposed here into the
three fields the UI actually renders (`title`, `vignette`, `question`) so the
frontend owns the markup and the text stays inspectable.

`key` is the stable identifier written to the database and to the analysis
export. It matches the embedded-data flag the QSF set on a "Very similar"
answer (`very_relationship`, `very_others`, `very_exclusion`,
`very_uncomfortable`) with the `very_` prefix dropped.

`topic_title` is the short category name. It no longer reaches any prompt — the
`[TOPIC]` section was removed on 2026-07-27 (see prompts.py) — and is now kept
only as analysis metadata, stored on the job and written to the export.

NO TEXT IN THIS FILE REACHES THE MODEL. `vignette` is shown to the participant
during screening as an example to react to, and that is all. The model's user
turn is built from two pieces of the participant's OWN writing — their screening
`description` of the real incident plus their perspective answer (see
`prompts.build_user_turn`).

This was briefly wrong: the vignette was prepended to the user turn on
2026-07-27 and reverted the same day, because the model then answered about the
canned example instead of the participant's actual situation.

ORDER NOTE: the QSF presented these four blocks inside a BlockRandomizer
(SubSet 4, EvenPresentation true), i.e. all four in a random order, stopping
at the first "Very similar". `SCENARIOS` below is in QSF declaration order;
the per-session presentation order is randomized in server.py, not here.
"""

SCENARIOS = [
    {
        "key": "relationship",
        "topic_title": "Relationship Boundaries",
        "title": "Relationship Boundaries",
        "vignette": (
            "My partner was upset because I went to my ex’s art show without "
            "telling them. I thought it wasn’t a big deal, but they felt I was "
            "being secretive."
        ),
        "question": (
            "How familiar are you with situations like the one above where your "
            "relationship with a friend, ex, or crush unintentionally led to mixed "
            "feelings or misunderstandings with a romantic partner?"
        ),
    },
    {
        "key": "others",
        "topic_title": "Involving Yourself in Someone Else’s Business",
        "title": "Involving Yourself in Someone Else’s Business",
        "vignette": (
            "I told my friend’s crush that she liked him because she was too "
            "shy. She got angry and said I overstepped."
        ),
        "question": (
            "How familiar are you with situations like the one above where you "
            "tried to support someone or act on their behalf, but it led to "
            "tension or disagreement?"
        ),
    },
    {
        "key": "exclusion",
        "topic_title": "Excluding Someone",
        "title": "Excluding Someone",
        "vignette": (
            "I didn’t invite my sibling to my party because I wanted to keep it "
            "small. They were hurt, and I didn’t expect them to be."
        ),
        "question": (
            "How familiar are you with situations like the one above where you "
            "made a choice about who to include and someone felt hurt or left out "
            "in a way you didn’t intend?"
        ),
    },
    {
        "key": "uncomfortable",
        "topic_title": "Making Someone Uncomfortable",
        "title": "Making Someone Uncomfortable",
        "vignette": (
            "A friend told me that talking a lot about my new relationship was hard "
            "to hear during her breakup. They were hurt, and I didn’t expect them "
            "to be."
        ),
        "question": (
            "How familiar are you with situations like the one above where sharing "
            "something personal or positive — like a new relationship or success "
            "— unintentionally made someone else uncomfortable or upset?"
        ),
    },
]

# The three similarity options, verbatim from the QSF. `value` is stored;
# `selects` marks the answer that ends screening and selects the scenario.
SIMILARITY_CHOICES = [
    {"value": "very", "label": "Very similar to something I’ve experienced", "selects": True},
    {"value": "somewhat", "label": "Somewhat similar to something I've experienced", "selects": False},
    {"value": "not_really", "label": "Not really", "selects": False},
]

SCENARIOS_BY_KEY = {s["key"]: s for s in SCENARIOS}

# Answers for which the "briefly describe the similar scenario" free-text is
# shown and required. DEVIATION from the QSF, which required it on every page
# regardless of answer; see README.md "Deviations from the source survey".
DESCRIBE_REQUIRED_FOR = ("very", "somewhat")


def scenario_keys():
    """Return the stable keys of all screening scenarios, in declaration order."""
    return [s["key"] for s in SCENARIOS]


def get_scenario(key):
    """Look up one scenario by its stable key.

    Assumption: `key` is always one produced by this module. An unknown key is
    a programming error (a stale session row or a tampered request), so this
    raises rather than returning None.
    """
    assert key in SCENARIOS_BY_KEY, "unknown scenario key: %r" % (key,)
    return SCENARIOS_BY_KEY[key]


def describe_required(similarity_value):
    """Whether the free-text description is required for a given similarity answer.

    Inputs: `similarity_value` — one of the `value`s in SIMILARITY_CHOICES.
    Returns: bool.
    """
    valid = {c["value"] for c in SIMILARITY_CHOICES}
    assert similarity_value in valid, "unknown similarity value: %r" % (similarity_value,)
    return similarity_value in DESCRIBE_REQUIRED_FOR


def selects_scenario(similarity_value):
    """Whether this similarity answer ends screening and selects the scenario."""
    for choice in SIMILARITY_CHOICES:
        if choice["value"] == similarity_value:
            return choice["selects"]
    raise AssertionError("unknown similarity value: %r" % (similarity_value,))
