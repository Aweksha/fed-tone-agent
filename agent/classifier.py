
import json
import re
from typing import Literal

from groq import BadRequestError
from pydantic import BaseModel, Field, model_validator

from agent.llm import get_groq_client
from agent.tools import fetch_full_text


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "openai/gpt-oss-120b"

TEST_URL = (
    "https://www.federalreserve.gov/"
    "newsevents/speech/jefferson20261001a.htm"
)


# ============================================================
# STRUCTURED CLASSIFICATION RESULT
# ============================================================

class ToneResult(BaseModel):
    """
    Validated monetary policy tone classification.
    """

    relevant: bool

    label: Literal[
        "strongly dovish",
        "dovish",
        "neutral",
        "hawkish",
        "strongly hawkish",
        "not applicable",
    ]

    score: int | None = Field(
        ge=-2,
        le=2,
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    reason: str = Field(
        min_length=10,
        max_length=500,
    )

    evidence: str = Field(
        min_length=15,
        max_length=1500,
    )

    @model_validator(mode="after")
    def validate_tone(self):
        """
        Ensure the relevance, score, and label agree.
        """

        score_labels = {
            -2: "strongly dovish",
            -1: "dovish",
            0: "neutral",
            1: "hawkish",
            2: "strongly hawkish",
        }

        if not self.relevant:
            if self.score is not None:
                raise ValueError(
                    "Irrelevant speeches must have a null score."
                )

            if self.label != "not applicable":
                raise ValueError(
                    "Irrelevant speeches must use "
                    "'not applicable' as the label."
                )

        else:
            if self.score is None:
                raise ValueError(
                    "Relevant speeches must have a tone score."
                )

            if self.label != score_labels[self.score]:
                raise ValueError(
                    "Tone label does not match the score."
                )

        return self


# ============================================================
# STRICT JSON SCHEMA
# ============================================================

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {
            "type": "boolean",
        },
        "label": {
            "type": "string",
            "enum": [
                "strongly dovish",
                "dovish",
                "neutral",
                "hawkish",
                "strongly hawkish",
                "not applicable",
            ],
        },
        "score": {
            "type": ["integer", "null"],
        },
        "confidence": {
            "type": "number",
        },
        "reason": {
            "type": "string",
        },
        "evidence": {
            "type": "string",
        },
    },
    "required": [
        "relevant",
        "label",
        "score",
        "confidence",
        "reason",
        "evidence",
    ],
    "additionalProperties": False,
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a careful monetary policy research analyst.

Your task is to classify the monetary policy tone expressed
in Federal Reserve speeches.

Analyze ONLY the supplied speech text.

STEP 1: DETERMINE RELEVANCE

A speech is relevant if it meaningfully discusses:
- Federal Reserve monetary policy
- Interest rate decisions or the policy outlook
- Inflation and price stability in a policy context
- Employment and the Federal Reserve's dual mandate
- FOMC decisions or monetary policy strategy

If the speech is not relevant, return:
- relevant: false
- label: "not applicable"
- score: null
- confidence: a number between 0 and 1
- reason: a brief explanation
- evidence: an exact passage showing the speech's subject

STEP 2: CLASSIFY MONETARY POLICY TONE

If relevant, use this scale:

-2 = strongly dovish
-1 = dovish
 0 = neutral
+1 = hawkish
+2 = strongly hawkish

Definitions:

STRONGLY DOVISH (-2):
Clearly advocates substantial monetary easing or expresses
strong concern about economic weakness and employment risks.

DOVISH (-1):
Leans toward easier monetary policy, lower rates, or
greater concern about employment and growth risks.

NEUTRAL (0):
Expresses a balanced stance, uncertainty, or no clear
preference toward tighter or easier policy.

HAWKISH (+1):
Leans toward maintaining restrictive policy, delaying
rate cuts, or prioritizing inflation control.

STRONGLY HAWKISH (+2):
Clearly advocates additional tightening or emphasizes
strong inflation risks requiring restrictive action.

IMPORTANT RULES:

1. Mentioning high inflation does not automatically
   make a speech hawkish.

2. Distinguish economic observations from actual
   monetary policy preferences.

3. Focus on the speaker's expressed policy stance.

4. Consider the entire speech, not isolated sentences.

5. Do not invent statements or economic statistics.

6. If the evidence is mixed or unclear, prefer neutral.

7. Confidence should reflect how clearly the speech
   supports the selected classification.

8. Explain the classification in one concise sentence.

9. The label MUST correspond to the score.

EVIDENCE REQUIREMENTS:

- Include an "evidence" field containing an exact passage
  copied from the supplied speech.
- Choose a passage that directly supports the monetary
  policy classification.
- Do not paraphrase or fabricate the evidence.
- Prefer the speaker's own statements about policy.
- Keep the passage concise: one to three consecutive sentences.
- The passage must be a single contiguous span of text.
  Do not use ellipses ("...") to skip words or join
  separate passages.
- Copy sentences in full; do not cut a sentence short.
- For irrelevant speeches, quote a passage demonstrating
  the speech's primary non-policy subject.
- Do not include quotation marks around the evidence
  unless they appear in the original speech.

OUTPUT REQUIREMENTS:

Return exactly these JSON fields:
relevant, label, score, confidence, reason, evidence.

Return valid JSON only.
"""


# ============================================================
# GROQ CLASSIFICATION
# ============================================================

def classify_speech(url: str) -> ToneResult:
    """
    Fetch a Federal Reserve speech and classify its tone.

    Args:
        url: Official Federal Reserve speech URL.

    Returns:
        Validated ToneResult object.
    """

    print("Fetching full speech...")

    speech_text = fetch_full_text(url)

    print(f"Retrieved {len(speech_text)} characters.")

    return classify_text(speech_text)


def classify_text(speech_text: str) -> ToneResult:
    """
    Classify the tone of already-fetched speech text.

    Args:
        speech_text: Full text of a Federal Reserve speech.

    Returns:
        Validated ToneResult object with verified evidence.
    """

    print("Analyzing monetary policy tone with Groq...\n")

    client = get_groq_client()

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        "Classify the monetary policy tone "
                        "of the following Federal Reserve speech.\n\n"
                        f"{speech_text}"
                    ),
                },
            ],
            temperature=0,
            max_completion_tokens=1200,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "fed_tone_classification",
                    "strict": True,
                    "schema": CLASSIFICATION_SCHEMA,
                },
            },
        )

    except BadRequestError as exc:
        raise RuntimeError(
            "Groq rejected the structured-output request. "
            "Check model support and schema compatibility. "
            f"Details: {exc}"
        ) from exc

    # Extract the model's JSON response.
    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "Groq returned an empty classification response."
        )

    # Parse JSON.
    try:
        parsed = json.loads(content)

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Groq returned invalid JSON."
        ) from exc

    # Validate the model's classification.
    result = ToneResult.model_validate(parsed)

    # ========================================================
    # EVIDENCE VERIFICATION
    # ========================================================

    if not verify_evidence(result.evidence, speech_text):
        raise ValueError(
            "Evidence verification failed: "
            "the quoted passage was not found in the original speech.\n"
            f"Returned evidence: {result.evidence}"
        )

    print("Evidence verification passed!")

    return result


# ============================================================
# EVIDENCE HELPERS
# ============================================================

# Ellipses the model may use to mark omitted text.
ELLIPSIS_PATTERN = re.compile(r"\s*(?:\.\s*\.\s*\.|…)\s*")

# Fragments shorter than this are too trivial to verify.
MIN_FRAGMENT_LENGTH = 15


def normalize_text(text: str) -> str:
    """
    Normalize whitespace, quotes, and spacing around punctuation.
    """

    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
    )

    text = " ".join(text.split())

    # Remove spaces introduced before punctuation
    # when HTML elements are extracted separately.
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    return text


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")


def split_sentences(fragment: str) -> list[str]:
    """
    Split a fragment into sentences, stripped of edge punctuation.

    Short pieces (e.g. from abbreviations like "U.S.") are merged
    with the following sentence so they keep their context.
    """

    sentences = []
    buffer = ""

    for sentence in SENTENCE_BOUNDARY.split(fragment):
        buffer = f"{buffer} {sentence}".strip() if buffer else sentence

        if len(buffer) >= MIN_FRAGMENT_LENGTH:
            sentences.append(buffer)
            buffer = ""

    if buffer:
        if sentences:
            sentences[-1] = f"{sentences[-1]} {buffer}"
        else:
            sentences.append(buffer)

    return [s.strip(" .,;:") for s in sentences]


def verify_evidence(evidence: str, speech_text: str) -> bool:
    """
    Confirm the evidence is quoted verbatim from the speech.

    The model sometimes elides text with "..." or ends a quote
    with a period mid-sentence, or silently skips sentences.
    Each sentence of the evidence must appear verbatim in the
    speech, in order. Edge punctuation is ignored.
    """

    normalized_speech = normalize_text(speech_text)

    fragments = [
        normalize_text(fragment).strip(" .,;:")
        for fragment in ELLIPSIS_PATTERN.split(evidence)
    ]
    fragments = [f for f in fragments if f]

    if not fragments:
        return False

    if any(len(f) < MIN_FRAGMENT_LENGTH for f in fragments):
        return False

    # The model may also skip sentences without marking it,
    # so verify each sentence individually, in order.
    pieces = [
        piece
        for fragment in fragments
        for piece in split_sentences(fragment)
    ]

    position = 0

    for fragment in pieces:
        index = normalized_speech.find(fragment, position)

        if index == -1:
            return False

        position = index + len(fragment)

    return True


# ============================================================
# LOCAL TEST
# ============================================================

def main():
    print("=" * 60)
    print("FED TONE AGENT - SPEECH CLASSIFICATION TEST")
    print("=" * 60)

    result = classify_speech(TEST_URL)

    print("\nCLASSIFICATION RESULT")
    print("=" * 60)

    print(result.model_dump_json(indent=2))

    print("\nValidation successful!")


if __name__ == "__main__":
    main()
