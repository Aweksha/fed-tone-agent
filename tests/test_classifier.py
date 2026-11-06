"""
Offline tests for evidence verification and tone validation.

No network or Groq calls: the speech excerpt below mirrors the
scraped text of jefferson20261001a, including the line breaks
the HTML extraction inserts around the "figure 8" link.
"""

import pytest
from pydantic import ValidationError

from agent.classifier import (
    ToneResult,
    normalize_text,
    split_sentences,
    verify_evidence,
)


SPEECH = (
    "than-anticipated aggregate demand.\n"
    "Monetary Policy\n"
    "Considering the economic conditions that persisted through "
    "mid-September, the FOMC voted last month to raise the target "
    "range for the federal funds rate by a 1/4 percentage point to "
    "3-3/4 to 4 percent as shown in\n"
    "figure 8\n"
    ". I supported that decision because I saw it as the appropriate "
    "policy to pursue the Fed's dual mandate. This was an important "
    "step to ensure longer-term inflation expectations remain well "
    "anchored and to validate the public's confidence that we will "
    "achieve our 2 percent inflation objective in a timely manner.\n"
    "As I described, I see economic activity and labor market "
    "conditions remaining broadly solid while inflation remains "
    "above our target. The U.S. economy has proven resilient."
)

S1 = (
    "Considering the economic conditions that persisted through "
    "mid-September, the FOMC voted last month to raise the target "
    "range for the federal funds rate by a 1/4 percentage point to "
    "3-3/4 to 4 percent"
)
S1_FULL = S1 + " as shown in figure 8."
S2 = (
    "I supported that decision because I saw it as the appropriate "
    "policy to pursue the Fed's dual mandate."
)
S3_TRUNCATED = (
    "This was an important step to ensure longer-term inflation "
    "expectations remain well anchored."
)
S4 = (
    "As I described, I see economic activity and labor market "
    "conditions remaining broadly solid while inflation remains "
    "above our target."
)


# ============================================================
# EVIDENCE VERIFICATION: ACCEPTED
# ============================================================

class TestVerifyEvidenceAccepts:

    def test_contiguous_passage(self):
        assert verify_evidence(f"{S1_FULL} {S2}", SPEECH)

    def test_ellipsis_elision(self):
        # First failure: "as shown in figure 8" replaced by "...".
        evidence = f"{S1} ... {S2} {S3_TRUNCATED}"
        assert verify_evidence(evidence, SPEECH)

    def test_unicode_ellipsis(self):
        assert verify_evidence(f"{S1}… {S2}", SPEECH)

    def test_silently_skipped_sentence(self):
        # Second failure: S3 omitted with no ellipsis.
        evidence = f"{S1_FULL} {S2} {S4}"
        assert verify_evidence(evidence, SPEECH)

    def test_sentence_cut_short_with_period(self):
        # Speech continues "anchored and to validate ...".
        assert verify_evidence(S3_TRUNCATED, SPEECH)

    def test_curly_apostrophe(self):
        evidence = S2.replace("Fed's", "Fed’s")
        assert verify_evidence(evidence, SPEECH)

    def test_abbreviation_not_split(self):
        assert verify_evidence(
            "The U.S. economy has proven resilient.", SPEECH
        )


# ============================================================
# EVIDENCE VERIFICATION: REJECTED
# ============================================================

class TestVerifyEvidenceRejects:

    def test_fabricated_passage(self):
        assert not verify_evidence(
            "The FOMC will cut rates aggressively next year.", SPEECH
        )

    def test_fabricated_sentence_mixed_with_real(self):
        evidence = f"{S2} We will cut rates aggressively next year."
        assert not verify_evidence(evidence, SPEECH)

    def test_paraphrased_sentence(self):
        evidence = S2.replace("appropriate", "right")
        assert not verify_evidence(evidence, SPEECH)

    def test_real_sentences_out_of_order(self):
        assert not verify_evidence(f"{S2} {S1_FULL}", SPEECH)

    def test_trivial_fragment(self):
        assert not verify_evidence(f"{S1} ... the Fed", SPEECH)

    @pytest.mark.parametrize("evidence", ["", "   ", "...", " … "])
    def test_empty_evidence(self, evidence):
        assert not verify_evidence(evidence, SPEECH)


# ============================================================
# HELPERS
# ============================================================

def test_normalize_text_removes_space_before_punctuation():
    assert normalize_text("in\nfigure 8\n. I") == "in figure 8. I"


def test_split_sentences_merges_short_pieces():
    assert split_sentences("The U.S. economy grew. Inflation fell.") == [
        "The U.S. economy grew",
        "Inflation fell",
    ]


# ============================================================
# TONE RESULT VALIDATION
# ============================================================

def make_result(**overrides):
    data = {
        "relevant": True,
        "label": "hawkish",
        "score": 1,
        "confidence": 0.8,
        "reason": "Endorses a rate hike to anchor inflation.",
        "evidence": S2,
    }
    data.update(overrides)
    return ToneResult.model_validate(data)


class TestToneResult:

    def test_valid_relevant(self):
        assert make_result().score == 1

    def test_valid_irrelevant(self):
        result = make_result(
            relevant=False, label="not applicable", score=None
        )
        assert result.score is None

    def test_label_score_mismatch(self):
        with pytest.raises(ValidationError, match="does not match"):
            make_result(label="dovish", score=1)

    def test_relevant_requires_score(self):
        with pytest.raises(ValidationError, match="must have a tone score"):
            make_result(score=None)

    def test_irrelevant_requires_null_score(self):
        with pytest.raises(ValidationError, match="null score"):
            make_result(relevant=False, label="not applicable", score=0)

    def test_irrelevant_requires_not_applicable_label(self):
        with pytest.raises(ValidationError, match="not applicable"):
            make_result(relevant=False, label="neutral", score=None)

    @pytest.mark.parametrize("score", [-3, 3])
    def test_score_out_of_range(self, score):
        with pytest.raises(ValidationError):
            make_result(score=score)

    @pytest.mark.parametrize("confidence", [-0.1, 1.1])
    def test_confidence_out_of_range(self, confidence):
        with pytest.raises(ValidationError):
            make_result(confidence=confidence)
