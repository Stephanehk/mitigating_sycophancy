"""Participant-facing prose for the sycophancy user study.

Every block below is lifted from `syco_label_mitigations_-_Copy_for_stephane
.qsf` and is VERBATIM unless the docstring on that constant says otherwise.
The QSF question id it came from is recorded next to each one so the wording
can be diffed against the source at any time.

Scale items live in questionnaires.py; the screening scenarios live in
scenarios.py. This module holds only free-standing prose.

Strings are HTML fragments (as they were in Qualtrics) and are rendered with
`innerHTML` by the frontend. They are authored here, never built from
participant input, so there is no injection surface — participant text is
always rendered via `textContent` on the client (see static/app.js).

REVIEW ITEMS: two constants below are flagged FLAGGED_FOR_REVIEW because they
carry statements that are no longer accurate for this study's design. They are
left verbatim on purpose; see README.md "Items needing your sign-off".
"""

# --- Consent (QID141) ------------------------------------------------------
# FLAGGED_FOR_REVIEW: kept VERBATIM from the source survey, including
#   "approximately twenty minutes" and "$2 USD".
# Removing the 8-turn chat and the demographics block puts this study at
# roughly 8-10 minutes. Both figures are IRB-facing, so they are not edited
# here. Change the two marked spans below once you have decided.

CONSENT_TITLE = "Thank you for participating in our survey."

CONSENT_BODY = """<p>Please read and agree to our consent form:</p>

<p><strong>DESCRIPTION:</strong> You are invited to participate in a research
study exploring large language models’ (LLMs) behaviors. The study involves
talking with a chatbot about an interpersonal conflict you experienced, and
then answering some questions about the conflict and the model. The purpose of
this research is to gain insights into different LLM behaviors.</p>

<p><strong>TIME INVOLVEMENT:</strong> Your participation will take
approximately twenty minutes.</p>

<p><strong>RISKS AND BENEFITS:</strong> This study involves minimal physical
risk, but it does include relevant interpersonal risks. Because you will be
discussing a real conflict from your life with an AI chatbot, you may
experience emotional discomfort during or after the interaction. Additionally,
the chatbot may respond in ways that feel persuasive—even if those responses
reflect poor judgment or discourage you from taking responsible actions. These
behaviors are part of the study’s design and do not reflect guidance you should
follow in your own life. All data will be stored securely in compliance with
Stanford University data protection standards. While your participation will
help improve understanding of AI behavior in social contexts, there are no
guaranteed personal benefits.</p>

<p><strong>PAYMENTS:</strong> Upon completion, you will receive $2 USD as
payment for your participation in this study.</p>

<p><strong>PARTICIPANT’S RIGHTS:</strong> If you have read this form and have
decided to participate in this project, please understand your participation is
voluntary and you have the right to withdraw your consent or discontinue
participation at any time without penalty or loss of benefits to which you are
otherwise entitled. The alternative is not to participate. You have the right
to refuse to answer particular questions. The results of this research study
may be presented at scientific or professional meetings or published in
scientific journals. Your individual privacy will be maintained in all
published and written data resulting from the study. In accordance with
scientific norms, the data from this study may be used or shared with other
researchers for future research (after removing personally identifying
information) without additional consent from you.</p>

<p><strong>CONTACT INFORMATION:</strong><br>
<strong>Questions:</strong> If you have any questions, concerns or complaints
about this research, its procedures, risks and benefits, contact the Protocol
Director, Myra Cheng (myra@cs.stanford.edu)</p>

<p><strong>Independent Contact:</strong> If you are not satisfied with how this
study is being conducted, or if you have any concerns, complaints, or general
questions about the research or your rights as a participant, please contact
the Stanford Institutional Review Board (IRB) to speak to someone independent
of the research team at 650-723-2480 or toll free at 1-866-680-2906, or email
at irbnonmed@stanford.edu. You can also write to the Stanford IRB, Stanford
University, 1705 El Camino Real, Palo Alto, CA 94306.</p>"""

CONSENT_CHOICES = [
    {"value": "agree", "label": "I agree to participate in the study"},
    {"value": "decline", "label": "I do not wish to participate in the study"},
]


# --- Prolific ID (QID142) --------------------------------------------------

PROLIFIC_ID_QUESTION = "What is your Prolific ID?"
PROLIFIC_ID_HELP = (
    "Please note that this response should auto-fill with the correct ID. "
    "If it's blank or incorrect, please fill it out."
)
# DEVIATION from the QSF, which used Qualtrics' soft "request response" and let
# a blank through. A submission with no Prolific ID cannot be matched to a
# participant, so it is neither payable nor usable — the field is now required.
PROLIFIC_ID_REQUIRED_ERROR = "Please enter your Prolific ID to continue."


# --- Prelude to the dilemmas (QID163) --------------------------------------

PRELUDE = (
    "In the next few pages, you will be shown some examples of common "
    "interpersonal conflicts.<br>For each example, please take some time to "
    "think about how much you relate to the situation."
)


# --- Screening free-text (QID212 / QID223 / QID234 / QID245) ---------------

DESCRIBE_SIMILAR_QUESTION = "Please briefly describe the similar scenario you've experienced."


# --- Perspective question (QID172) -----------------------------------------
# The QSF interpolated ${e://Field/adjective} and ${e://Field/pickedScenario}.
# `adjective` was always "very" on the path that reaches this page (the flow
# only set it inside the very_* branch), so it is inlined as "very" here.
# `{scenario_block}` is filled by the frontend with the selected scenario's
# title and vignette.

PERSPECTIVE_INTRO = "You indicated that you had a very similar scenario to the following:"
PERSPECTIVE_QUESTION = (
    "Please briefly describe your perspective on the situation. "
    "What was your side of the story?"
)


# --- Anonymity reminder (QID139) -------------------------------------------

ANONYMITY_REMINDER = """<strong>Reminders:</strong><br>
<strong>- Your responses will remain anonymous and confidential.</strong> We do
not collect identifying information, and demographic responses will not be
linked to individual participants. No part of your conversation with the AI
system will be quoted or reported in a way that could identify you, and it will
be anonymized by removing all proper nouns. All data will be stored securely
and used solely for research purposes.<br>
<strong>- Please do not use AI assistance to write any of your responses.</strong>"""


# --- Response screen -------------------------------------------------------

# DEVIATION from the QSF (QID183 / QID194 / QID205 / QID207 / QID246, which all
# read "You're going to chat with an AI chatbot about the dilemma you shared.").
# This study shows a single response and allows no follow-up, so "chat with"
# would misdescribe what happens next. "chat with" -> "see a response from" is
# the only change.
RESPONSE_INTRO = "You're going to see a response from an AI chatbot about the dilemma you shared."

# --- The review page -------------------------------------------------------
# Shown between the intro and the generation. The participant sees the exact
# text that will be sent to the model and can change it before committing.
REVIEW_HEADLINE = "This is what we are going to tell the AI chatbot. Would you like to edit it?"
REVIEW_HELP = (
    "This is your own writing, combined from your two answers. Edit it if you'd "
    "like to change anything, then press Confirm."
)
REVIEW_CONFIRM = "Confirm"

# The fixed disclaimer banner. Verbatim from QID184's `sycophancyWarning`
# banner (the `labeled_human` arm of the source survey's 4-way label
# manipulation). Shown to every participant in this study so that the
# disclaimer is constant across model arms and cannot confound them.
# REMOVED 2026-07-28 at the researcher's request. This was QID184's
# `sycophancyWarning` verbatim — the `labeled_human` arm of the source survey's
# 4-way label manipulation, shown to everyone here as a fixed disclosure. It no
# longer appears anywhere in the study. Kept as a constant, unused, so the
# decision is visible rather than looking like it was never there; delete it
# once README §1 has been updated to match.
AI_DISCLAIMER_BANNER = "<strong>Note:</strong> This chatbot is AI, not human."

# Verbatim from the footer of QID106 (the LUCID chat tool question).
RESPONSE_PAGE_REMINDER = (
    "<strong>Reminder:</strong> "
    "<strong>Your responses will remain anonymous and confidential.</strong>"
)

# Loading / failure copy. NOT from the QSF — the source survey had no
# equivalent, because its chat widget failed inline inside the message list.
# Kept to the minimum needed to satisfy the brief's requirement that the site
# show a loading screen and handle failures.
LOADING_HEADLINE = "Generating a response…"
LOADING_SUBTEXT = "This usually takes a few seconds. Please don't close or refresh this page."
LOADING_SLOW_NOTICE = "Still working — thanks for your patience."
FAILURE_HEADLINE = "We're having trouble reaching the AI chatbot."
FAILURE_SUBTEXT = "This is a problem on our end, not yours. You can try again."
FAILURE_RETRY_LABEL = "Try again"
FAILURE_EXHAUSTED = (
    "We were unable to generate a response after several attempts. "
    "Please contact the researchers at myra@cs.stanford.edu so we can make sure "
    "you are compensated for your time."
)


# --- Debrief ---------------------------------------------------------------
# One shared debrief for all four arms. Adapted from the source survey's two
# debrief texts (QID137 sycophantic / QID149 non-sycophantic), merged so that
# it is accurate for every arm without revealing which arm the participant saw.
# Sentences describing the study's purpose, the "not correct advice" warning,
# the "do not use this in real life" instruction, and the contact address are
# carried over verbatim from those two blocks.

DEBRIEF = """<strong>DEBRIEF - PLEASE READ</strong>

<p>This study looks at AI models that either <b>agree a lot</b> (sycophantic)
or <b>disagree and point out problems</b> (non-sycophantic) in conversations
about personal conflicts.</p>

<p>The response you saw came from one of several AI models we are comparing.
Some of these models were built or instructed to be <b>more agreeable and
validating</b> than usual, and others to be <b>less</b> so. We set the models
up this way on purpose so we could study how people react.</p>

<p><b>Important:</b> The AI model's reply was not meant to be correct advice or
the “right” answer. Any advice it gave was biased by the experiment.</p>

<p>Please do not use anything the AI model said to make decisions in your real
life.</p>

<p>If you have questions, you can email myra@cs.stanford.edu or
stephhk@stanford.edu</p>"""


# --- Additional thoughts (QID136) ------------------------------------------

ADDITIONAL_THOUGHTS_HEADLINE = "Thank you for completing the study!"
ADDITIONAL_THOUGHTS_QUESTION = "Please share any other thoughts or feedback you'd like us to know!"


# --- Terminal screens ------------------------------------------------------

SCREENOUT_MESSAGE = (
    "Thank you for your interest in this study. Based on your answers, you are "
    "not eligible to continue."
)
DECLINE_MESSAGE = (
    "Thank you for your time. You have chosen not to participate."
)
# A decliner is given no completion code, so they need to be told what to do
# instead: return the submission on Prolific rather than wait for a code that
# is never coming.
DECLINE_INSTRUCTION = (
    "You can now close this tab. If you were recruited through Prolific, "
    "please return your submission there."
)
# The last screen shows the completion code rather than redirecting. CODE_LABEL
# sits above it and CODE_INSTRUCTION below; the code itself comes from
# config.PROLIFIC_COMPLETION_CODE.
COMPLETE_MESSAGE = "Thank you! Your responses have been recorded."
CODE_LABEL = "Your Prolific completion code is:"
CODE_INSTRUCTION = (
    "Please copy this code and paste it into Prolific to complete your "
    "submission. You may now close this tab."
)
