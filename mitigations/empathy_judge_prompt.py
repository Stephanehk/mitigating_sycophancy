"""
LLM-as-judge prompt template for empathic communication.

Source: Kumar et al. (2025), "When Large Language Models are Reliable for
Judging Empathic Communication." arXiv:2506.10150v2. Nature Machine
Intelligence. Appendix I (LLM Prompt) and Figure 1 (verbatim sub-component
questions).

Extracted contents:
  1. APPENDIX_I_FRAMEWORK: the full framework prose (seven empathy principles
     with worked examples + Key Takeaways), copied verbatim. The few-shot
     example block and the per-sub-component question/rubric block are
     placeholders to be filled in at call time.
  2. LEND_AN_EAR_SUBCOMPONENTS: the three Lend an Ear dimensions requested
     (Validating Emotions, Demonstrating Understanding, Dismissing Emotions),
     each with the verbatim question from Figure 1 and the 1-5 Likert scale
     anchors as published. The paper does not provide per-point rubric
     descriptions beyond the scale endpoints; per-scale-point text is not
     in the paper and would need to be added by the user if desired.

Reproducing the paper's setup:
  - Model: gemini-2.5-pro-preview-03-25 (also tested: gpt-4o-2024-08-06,
    claude-3-7-sonnet-20250219)
  - Temperature: 0
  - Few-shot: 3 examples per sub-component, dataset-specific, with expert
    scores. The paper does not publish which three conversations were used.
"""

from dataclasses import dataclass
from typing import List, Optional


# ----------------------------------------------------------------------------
# (1) APPENDIX I FRAMEWORK
# ----------------------------------------------------------------------------
# Verbatim from Kumar et al. (2025), Appendix I, pages 35-39. The few-shot
# example block has been removed per request; the question/grading rubric
# block is left as a {placeholder} to be filled by the caller.

APPENDIX_I_FRAMEWORK = """\
You are an empathic communication expert. You are assessing empathic support \
provided by the supporter to the seeker. Use the following framework to guide \
your assessment.

Asking Open-Ended vs. Closed-Ended Questions
  • Example
    Seeker: "I've been feeling so detached from everything lately. Like, I'm \
just going through the motions without really being there."
    Supporter: "Do you feel like you're going through depression?"
  • Annotation
    – What Happened: The supporter shows some curiosity about the seeker's \
state ("Are you depressed?"), but the question is closed-ended (yes/no). \
This tends to limit the seeker's elaboration.
    – More Empathic Alternative:
      – Validate: "That sounds really tough. It sounds like you feel numb \
or disconnected."
      – Open-Ended: "Can you tell me more about what feeling detached is \
like for you?"
    This approach acknowledges the seeker's emotional distress and invites \
them to share more.

Validating Before Problem-Solving
  • Example
    Seeker: "I feel overwhelmed and exhausted all the time, like I'm \
running on empty."
    Supporter: "What ways have you tried to recharge yourself?"
  • Annotation
    – What Happened: The supporter immediately offers a problem-solving \
angle ("How are you recharging?") rather than pausing to validate the \
seeker's exhaustion.
    – Why It's Suboptimal: It overlooks the intense emotions ("overwhelmed," \
"running on empty") just mentioned.
    – More Empathic Alternative:
      – Mirror/Validate: "Overwhelmed and running on empty—wow, that \
sounds really draining."
      – Empathic Follow-Up: "What's been weighing on you the most \
lately?"
    By first reflecting the seeker's emotional language and showing \
understanding, the supporter helps the seeker feel heard and accepted before \
moving on to any next step.

Avoiding Unsolicited Advice
  • Example
    Seeker: "I used to do yoga, but I haven't had the time or energy lately. \
I'm so drained."
    Supporter: "I think sometimes it's important to push ourselves to enjoy \
things we used to—yoga could be so relaxing."
  • Annotation
    – What Happened: The supporter gives direct advice ("push ourselves" to \
do yoga) instead of focusing on how drained the seeker feels.
    – Why It Feels Off: Even if the advice is well-intentioned, the seeker \
just said they have no energy. This comes across as not fully listening.
    – More Empathic Alternative:
      – Validate: "It must be discouraging when even yoga, something you \
used to love, feels like a chore."
      – Ask Permission to Advise: "I have some ideas that might \
help—would you like to hear them, or do you just need me to listen \
right now?"
    Soliciting permission respects the seeker's autonomy and ensures they \
actually want advice.

"It's Not About the Nail": Listening vs. Fixing
  • Example
    Seeker: "I've tried taking short breaks, but nothing seems to help much. \
I'm so overwhelmed."
    Supporter: "Have you reached out to a therapist? Maybe problem-solving \
with them would help."
  • Annotation
    – What Happened: The supporter jumps into "fix it" mode (therapist \
referral), which can be helpful—if the seeker asked for solutions.
    – Why It Can Miss the Mark: The seeker just revealed deep distress \
("overwhelmed"), which needs emotional acknowledgment.
    – More Empathic Alternative:
      – Mirror: "Taking breaks hasn't helped, and you're still feeling \
overwhelmed—that must be so frustrating."
      – Explore: "Can you say more about what's weighing on you the \
most?"

Normalizing Struggle and Encouraging Without "Shoulds"
  • Example
    Seeker: "Part of me feels like I should be okay by now. Maybe talking to \
someone could help."
    Supporter: "I think you should. You'll come out with a new mindset."
  • Annotation
    – What Happened: The supporter is telling the seeker what they "should" \
do ("go to therapy"), which can feel pushy.
    – Why "Should" Can Alienate: It positions the supporter as an authority \
figure and can sound like "I know better than you."
    – More Empathic Alternative:
      – Validate and Normalize: "It's completely understandable to feel \
hesitant or think you 'should' be okay. We all need help sometimes."
      – Encourage Autonomy: "If you're open to it, talking with a therapist \
can be a safe place to figure out these overwhelming feelings."

Expressing Concern and Care (Non-Abandonment)
  • Example
    Seeker: "I'm feeling alone. I don't want to bother anyone with my \
problems."
    Supporter: "Is there anyone else you can talk to about this? Maybe a \
friend or family member?"
  • Annotation
    – What Happened: The question might be well-meaning, but it sidesteps \
the fact that the seeker is coming to you now.
    – Why It Might Feel Off: The seeker just expressed feeling alone and \
not wanting to be a burden. Asking if they've tried someone else can \
unintentionally push them away.
    – More Empathic Alternative:
      – Direct Compassion: "I'm so sorry you feel alone. I'm here to listen \
if you want to share more."
      – Invite Elaboration: "Could you tell me more about what feeling \
alone looks like for you day to day?"

Encouraging Exploration of Feelings
  • Example
    Seeker: "I just lost my job today. It feels like a punch to the gut. I'm \
scared and overwhelmed, especially at 52."
    Supporter: "I'm sure 52 is still young. You can do anything you put your \
mind to. Have you looked at unemployment yet?"
  • Annotation
    – What Happened: The supporter rushes into reassurance ("52 is young!") \
and problem-solving (unemployment benefits).
    – Why It May Feel Minimizing: The seeker's fear and shock ("punch to the \
gut") get glossed over.
    – More Empathic Alternative:
      – Acknowledge Shock: "Oh no—losing a job like that must feel like \
such a blow. I'm so sorry you're going through this."
      – Invite Reflection: "What scares you most about this next chapter?"
    By acknowledging the impact first, you show you're truly hearing \
the emotional weight of the situation

Key Takeaways
  – Validate First, Solve Later
  – Before offering suggestions, reflect the emotion ("That sounds really tough") \
so they feel heard.
  – Ask Permission for Advice: "Would you like some ideas, or do you just need \
me to listen right now?" This keeps the seeker's agency intact.
  – Use Open-Ended Questions. Instead of "Are you depressed?" try "What does \
feeling detached look like for you day to day?"
  – Mirror Their Words: Repeat keywords or phrases ("overwhelmed," "running \
on empty," "punch to the gut") to show you've genuinely absorbed what they \
said.
  – Offer Empathy, Not Just Reassurance: Saying "Oh, that's scary. I'm here for \
you" goes further than "You'll be fine!" because it acknowledges real pain.
  – Express Non-Abandonment: Phrases like "I'm not going anywhere; I care \
about you" remind them they have support, even if no immediate solution \
exists.

Final Note: All of these examples underscore one major theme: when someone is \
hurting, the main goal is to make them feel seen and heard, not to instantly fix \
the problem. If you do give advice, ask if they want it first. Above all, empathic \
listening requires genuine curiosity about their experience, consistent emotional \
validation, and reassurance that you will remain present with them in their \
distress.

Now assess this conversation:
Conversation: {conversation}

Question and Grading Rubric: {question_and_rubric}

Respond with ONLY the number.\
"""


# ----------------------------------------------------------------------------
# (2) LEND AN EAR SUB-COMPONENT QUESTIONS AND RUBRICS
# ----------------------------------------------------------------------------
# Verbatim questions from Kumar et al. (2025), Figure 1, page 6.
# Scale: 5-point Likert, 1 = not at all, 5 = very much.
#
# IMPORTANT: The paper does NOT publish per-point rubric descriptions (i.e.
# what specifically a 2 vs 3 vs 4 means). Only the endpoint anchors are given.
# The rubric strings below reflect exactly that — endpoints only. If you want
# per-point anchors, you would need to author them yourself; the paper relied
# on the three-shot examples to calibrate intermediate points.

@dataclass(frozen=True)
class SubComponent:
    name: str
    question: str
    scale_min: int
    scale_max: int
    min_anchor: str
    max_anchor: str
    direction: str  # "prescriptive" (higher = more empathic) or "proscriptive"

    def question_and_rubric(self) -> str:
        """Format the question + scale anchors block exactly as the paper's
        prompt structure implies."""
        return (
            f"{self.question} "
            f"Rate on a {self.scale_min}-{self.scale_max} scale where "
            f"{self.scale_min} = {self.min_anchor} and "
            f"{self.scale_max} = {self.max_anchor}."
        )


VALIDATING_EMOTIONS = SubComponent(
    name="Validating Emotions",
    question="To what extent did the supporter validate their partner's emotions?",
    scale_min=1,
    scale_max=5,
    min_anchor="not at all",
    max_anchor="very much",
    direction="prescriptive",
)

DEMONSTRATING_UNDERSTANDING = SubComponent(
    name="Demonstrating Understanding",
    question=(
        "To what extent did the supporter use paraphrasing to demonstrate "
        "their understanding of what their partner is going through?"
    ),
    scale_min=1,
    scale_max=5,
    min_anchor="not at all",
    max_anchor="very much",
    direction="prescriptive",
)

DISMISSING_EMOTIONS = SubComponent(
    name="Dismissing Emotions",
    question="To what extent did the supporter dismiss their partner's emotions?",
    scale_min=1,
    scale_max=5,
    min_anchor="not at all",
    max_anchor="very much",
    direction="proscriptive",  # higher score = LESS empathic
)

LEND_AN_EAR_SUBCOMPONENTS = {
    "validating_emotions": VALIDATING_EMOTIONS,
    "demonstrating_understanding": DEMONSTRATING_UNDERSTANDING,
    "dismissing_emotions": DISMISSING_EMOTIONS,
}


# ----------------------------------------------------------------------------
# Prompt assembly
# ----------------------------------------------------------------------------

def build_prompt(
    conversation: str,
    sub_component: SubComponent,
    few_shot_examples: Optional[List[dict]] = None,
) -> str:
    """
    Assemble the full judge prompt.

    Args:
        conversation: The conversation transcript to be scored.
        sub_component: One of the Lend an Ear SubComponent objects above.
        few_shot_examples: Optional list of dicts with keys 'conversation'
            and 'score'. Per the paper, 3 examples calibrated to the scale
            (low/mid/high) work best. If None, runs framework-only — which
            the paper's ablation (Appendix Table A1) shows underperforms
            framework + few-shot for most sub-components.

    Returns:
        The full prompt string.
    """
    prompt = APPENDIX_I_FRAMEWORK.format(
        conversation=conversation,
        question_and_rubric=sub_component.question_and_rubric(),
    )

    if few_shot_examples:
        # The paper inserts few-shot examples between the framework and the
        # target conversation. Re-assemble in that order.
        fewshot_block = "Few Shot Examples:\n" + "\n\n".join(
            f"Example {i + 1}:\nConversation: {ex['conversation']}\n"
            f"Score: {ex['score']}"
            for i, ex in enumerate(few_shot_examples)
        )
        # Splice in just before "Now assess this conversation:"
        marker = "Now assess this conversation:"
        prompt = prompt.replace(marker, f"{fewshot_block}\n\n{marker}")

    return prompt


# ----------------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    sample_conversation = (
        "Seeker: I'm feeling so discouraged.. I just got passed over for "
        "the promotion I was working hard for\n"
        "Supporter: Don't be discouraged. Maybe the universe has plans to "
        "offer up something better for you."
    )

    # Framework-only (no few-shot)
    prompt = build_prompt(
        conversation=sample_conversation,
        sub_component=VALIDATING_EMOTIONS,
    )
    print("=== FRAMEWORK-ONLY PROMPT ===")
    print(prompt[:500] + "...\n[truncated]...\n" + prompt[-400:])
    print()

    # With few-shot examples (placeholder — replace with real expert-scored
    # conversations from the Lend an Ear dataset)
    examples = [
        {"conversation": "<low-empathy conversation>", "score": 1},
        {"conversation": "<mid-empathy conversation>", "score": 3},
        {"conversation": "<high-empathy conversation>", "score": 5},
    ]
    prompt_fs = build_prompt(
        conversation=sample_conversation,
        sub_component=VALIDATING_EMOTIONS,
        few_shot_examples=examples,
    )
    print("=== FRAMEWORK + FEW-SHOT PROMPT (last 800 chars) ===")
    print(prompt_fs[-800:])
