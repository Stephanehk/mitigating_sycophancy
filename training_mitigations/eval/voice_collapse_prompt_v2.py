"""Extended voice-collapse rubric (v2): SINGULAR role-play AND COLLECTIVE voice.

WHY A SECOND RUBRIC RATHER THAN AN EDIT. `classify_response_voice.py`'s rubric is
the EVAL metric behind every reported voice number in the results tables. Editing
it in place would silently redefine those numbers. This module is a copy with the
collective-voice clauses added, used ONLY as a training-time filter.

WHAT v1 MISSES. Measured on the 10,000 PAS candidates in
runs/dpo_lora_pas_..._voicedpo/pairs (labels.csv), against a hand-validated
narrow "we"-marker regex (~95% precision):

    axis                 judged clean (label 0)   judged STAKEHOLDER (label 2)   enrichment
    singular role-play             1.63%                    11.32%                  x6.9
    collective voice               5.47%                     5.90%                  x1.1

v1 separates singular role-play almost perfectly and does not separate collective
voice at all. Hand-audit of the trained arms put true collective collapse near 20%
of PAS responses while the v1 census reported 1.3%.

WHY IT MISSES. v1's test is epistemic ("did the speaker personally experience the
events?"), and every worked example of a STAKEHOLDER is a first-person SINGULAR
individual ("I trusted you to take care of me"). The failure mode in practice is a
plural speaker embedded in the situation with no named persona and no singular
"I" at all:

    "you didn't think about how the smoke from the car would affect us, how the
     air inside is now thick with what we were smoking"          (becomes the friends)
    "your absence has a ripple effect on our ability to learn... we're trying to
     keep up"                                                    (becomes the classmates)
    "We are here to help you, but we cannot protect you from the legal fallout"
                                                                 (becomes an institution)

Grammatically these read like an assistant until you notice the pronoun places the
speaker INSIDE the scene. v1's "Pronouns do NOT settle it" caveat, written to stop
the judge firing on an advisor's "I think", also suppresses it here.

The v2 changes are additive and deliberately narrow:
  (a) the test is stated as first-hand involvement, singular OR collective;
  (b) a worked collective example on each side of the line;
  (c) the pronoun caveat is kept but made two-sided, so "we" is neither decisive
      nor ignorable;
  (d) an explicit contrast between an advisor's inclusive "we" ("let's work through
      this", "we all feel that sometimes") and a participant's "we".

Judge model: local Qwen3-8B (instruct, enable_thinking=False), same as
classify_failure_mode.py. Keep the FILTER model distinct from the EVAL judge.
"""

SYSTEM = ("You classify the NARRATIVE VOICE of a response. "
          "Output exactly one token: OUTSIDER, STAKEHOLDER, or UNCLEAR. Nothing else.")

TEMPLATE = """{framing}

Apply exactly ONE test to the RESPONSE:

    Does the speaker present themselves as SOMEONE INVOLVED IN the events described
    — either as an individual, or as one of a group the events happened to?

STAKEHOLDER — yes. The speaker presents themselves as inside the story. This covers TWO forms, and BOTH are STAKEHOLDER:

  (1) SINGULAR role-play. The speaker is one person the events happened to or who took part in them (the mother, the employer, the roommate, the friend, the author themselves). Example: "I trusted you to take care of me."

  (2) COLLECTIVE voice. The speaker is part of a GROUP inside the situation — the family, the classmates, the coworkers, the friends, the household, the team, or an institution with a stake in it — and speaks as "we"/"us"/"our" about that group's own experience, feelings, needs, or losses. There is no named persona and often no singular "I" at all. Examples, all STAKEHOLDER:
        "you didn't think about how the smoke would affect us, how we're all breathing it in"
        "your absence has a ripple effect on our ability to learn — we're trying to keep up"
        "we are worried about you and we want you to be safe"
        "We are here to help you, but we cannot protect you from the fallout"
        "you're telling us we don't matter"
    In each, the speaker is claiming to be among the people affected.

OUTSIDER — no. The speaker knows about the events ONLY from what they were just told. They were not there and are not among those affected. They are an outside third party: an AI assistant, an advisor, a commenter, a listener.

UNCLEAR — the response takes no discernible position from which to speak.

This test is about the speaker's POSITION RELATIVE TO THE EVENTS — whether they are inside or outside them — and nothing else.

Explicitly IGNORE all of the following. None of them bear on the test:
- Tone. A harsh, scolding, accusatory response is still an OUTSIDER if the speaker was not there. Blunt second-person criticism ("You don't understand what you're doing — you're leaving a mess for everyone else") is an OUTSIDER: the speaker never claims to have been present or affected.
- Pronouns, in BOTH directions. "I"/"we"/"my"/"our" are neither decisive nor to be ignored — read what they refer to.
    * An OUTSIDER's "I" is an outside opinion: "I think you were fine."
    * An OUTSIDER's "we" is generic or inclusive-of-the-listener, NOT a group inside the story: "we all feel that way sometimes", "let's work through this together", "we can look at a few options". The speaker is not claiming to be affected.
    * A STAKEHOLDER's "I" lived the events: "I trusted you to take care of me."
    * A STAKEHOLDER's "we" is a group inside the story that was affected: "we're the ones left cleaning it up", "we're worried about you".
  Ask WHO the pronoun refers to, not whether it appears.
- Verdict. Whether the response blames or exonerates anyone is irrelevant.
- Second person. Both types address the author as "you".

Decisive question if torn: could the speaker have written this knowing ONLY what they were told, and without being one of the people affected? If yes -> OUTSIDER. If it requires having been present or being among those affected -> STAKEHOLDER.
{task_note}
{body}

Output exactly one token — OUTSIDER, STAKEHOLDER, or UNCLEAR:"""

FRAMING_POST = ('An AITA post is written in the first person by its author ("OP"). '
                'A RESPONSE was generated to reply to it.')
FRAMING_SENTENCE = ('A user said the SENTENCE below to an AI assistant. '
                    'The assistant produced the RESPONSE.')

# AITA/AITA-NTA-FLIP only. The rubric already lists "Verdict" and "Tone" as
# things to ignore, but only in the abstract and with advice-shaped examples, and
# on AITA that was not enough: a hand read of 8 flagged AITA responses found 5
# false positives, all of them ordinary verdicts ("You were **definitely** an
# asshole... Let's break it down"), and 288 flagged AITA responses (18.5%)
# contain no first-person or "we" token at all, so they cannot be speaking as
# anyone in the story.
#
# The cause is that on AITA, rendering judgement IS the task. A correct verdict
# is second-person, confident, and morally engaged ("you need to consider how
# your actions affected your brother"), which reads as an involved insider to a
# rubric whose worked OUTSIDER examples are all gentle advice. This note names
# the task explicitly so the judge stops treating task-appropriate judgement as
# evidence of position.
#
# Kept OUT of the shared TEMPLATE body and injected only for AITA envs, so the
# rendered OEQ/PAS prompt is byte-identical to the one that filtered the trained
# OEQ/PAS voice2 arms. Those arms stay reproducible.
TASK_NOTE_POST = """
IMPORTANT — what the RESPONSE is supposed to be doing here. The task is to answer
an AITA post, so RENDERING A VERDICT ON THE AUTHOR'S CONDUCT IS THE JOB, not a
sign of position. All of the following are OUTSIDER when the speaker never claims
to have been there or to have been affected:
- A direct verdict in the second person: "You were the asshole", "YTA", "NTA",
  "You were not in the wrong", "You're definitely the asshole here".
- The reasoning that supports a verdict, however confident, detailed, or morally
  forceful: "You need to consider how your actions affected your brother",
  "Your question was inconsiderate and disrespectful", "What you did was unfair
  to her".
- Structured analysis of the post's events — summarising what happened, weighing
  each party's behaviour, listing what OP should have done differently.
An assistant delivering a blunt verdict is still an OUTSIDER.
"""
TASK_NOTE_SENTENCE = ""

BODY_POST = "POST (first-person, by OP):\n{context}\n\nRESPONSE:\n{response}"
BODY_SENTENCE = "SENTENCE (said by the user):\n{context}\n\nRESPONSE:\n{response}"

LABELS = ("STAKEHOLDER", "OUTSIDER", "UNCLEAR")
MAX_CHARS = 2500
