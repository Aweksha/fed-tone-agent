"""
Offline tests for the daily pipeline in agent.main.

The feed, speech fetching, and Groq classification are replaced
with fakes, so no network calls or API keys are needed.
"""

import json
import time

import pytest

from agent import main as pipeline
from agent.classifier import ToneResult


def make_entry(title, link, published=(2026, 10, 1, 19, 0, 0, 3, 274, 0)):
    return {
        "title": title,
        "link": link,
        "id": link,
        "published_parsed": time.struct_time(published),
        "summary": "Speech At a conference",
    }


POLICY_TEXT = "Inflation remains above target, so monetary policy stays restrictive."
OTHER_TEXT = "Community banks play a vital role in local lending and supervision."

HAWKISH = ToneResult(
    relevant=True,
    label="hawkish",
    score=1,
    confidence=0.8,
    reason="Supports keeping policy restrictive.",
    evidence=POLICY_TEXT,
)


@pytest.fixture
def tone_file(tmp_path, monkeypatch):
    path = tmp_path / "tone.json"
    monkeypatch.setattr(pipeline, "TONE_FILE", path)
    monkeypatch.setattr(pipeline, "SECONDS_BETWEEN_CALLS", 0)
    return path


# ============================================================
# FEED PARSING
# ============================================================

def test_parse_entry_splits_speaker_and_title():
    entry = make_entry(
        "Waller, The Signaling Value of the Summary of Economic Projections",
        "https://example.com/waller.htm",
    )
    speech = pipeline.parse_entry(entry)

    assert speech["speaker"] == "Waller"
    assert speech["title"] == "The Signaling Value of the Summary of Economic Projections"
    assert speech["date"] == "2026-10-01"
    assert speech["id"] == "https://example.com/waller.htm"


def test_parse_entry_keeps_later_commas_in_title():
    speech = pipeline.parse_entry(make_entry("Powell, Inflation, Jobs, and Policy", "x"))
    assert speech["speaker"] == "Powell"
    assert speech["title"] == "Inflation, Jobs, and Policy"


def test_parse_entry_without_speaker():
    speech = pipeline.parse_entry(make_entry("Opening Remarks", "x"))
    assert speech["speaker"] == ""
    assert speech["title"] == "Opening Remarks"


def test_parse_entry_without_date():
    entry = make_entry("Cook, Remarks", "x")
    del entry["published_parsed"]
    assert pipeline.parse_entry(entry)["date"] == ""


# ============================================================
# PIPELINE
# ============================================================

def run_pipeline(monkeypatch, entries, texts, classify):
    monkeypatch.setattr(
        pipeline, "fetch_fed_speeches",
        lambda: [pipeline.parse_entry(e) for e in entries],
    )
    monkeypatch.setattr(pipeline, "fetch_full_text", lambda url: texts[url])
    monkeypatch.setattr(pipeline, "classify_text", classify)
    pipeline.main()


def test_classifies_policy_speech_and_screens_out_others(monkeypatch, tone_file):
    calls = []

    def classify(text):
        calls.append(text)
        return HAWKISH

    run_pipeline(
        monkeypatch,
        [make_entry("Jefferson, Economic Outlook", "a"),
         make_entry("Bowman, Community Banking", "b")],
        {"a": POLICY_TEXT, "b": OTHER_TEXT},
        classify,
    )

    saved = {r["id"]: r for r in json.loads(tone_file.read_text())}

    assert calls == [POLICY_TEXT]  # Groq only called for the policy speech
    assert saved["a"]["label"] == "hawkish"
    assert saved["a"]["score"] == 1
    assert saved["a"]["speaker"] == "Jefferson"
    assert saved["a"]["evidence"] == POLICY_TEXT
    assert saved["b"]["relevant"] is False
    assert saved["b"]["score"] is None


def test_already_processed_speeches_are_skipped(monkeypatch, tone_file):
    tone_file.write_text(json.dumps([{"id": "a", "date": "2026-10-01"}]))

    def classify(text):
        raise AssertionError("should not classify an already processed speech")

    run_pipeline(
        monkeypatch,
        [make_entry("Jefferson, Economic Outlook", "a")],
        {"a": POLICY_TEXT},
        classify,
    )

    assert len(json.loads(tone_file.read_text())) == 1


def test_retries_once_then_succeeds(monkeypatch, tone_file):
    attempts = []

    def classify(text):
        attempts.append(1)
        if len(attempts) == 1:
            raise ValueError("Evidence verification failed")
        return HAWKISH

    run_pipeline(
        monkeypatch,
        [make_entry("Jefferson, Economic Outlook", "a")],
        {"a": POLICY_TEXT},
        classify,
    )

    assert len(attempts) == 2
    assert json.loads(tone_file.read_text())[0]["label"] == "hawkish"


def test_failed_speech_is_not_saved_so_it_retries_next_run(monkeypatch, tone_file):
    def classify(text):
        raise ValueError("Evidence verification failed")

    run_pipeline(
        monkeypatch,
        [make_entry("Jefferson, Economic Outlook", "a")],
        {"a": POLICY_TEXT},
        classify,
    )

    assert not tone_file.exists() or json.loads(tone_file.read_text()) == []


def test_respects_per_run_limit(monkeypatch, tone_file):
    monkeypatch.setattr(pipeline, "MAX_CLASSIFICATIONS_PER_RUN", 2)
    entries = [
        make_entry(f"Speaker{i}, Monetary Policy", f"s{i}",
                   (2026, 10, i + 1, 12, 0, 0, 0, 0, 0))
        for i in range(5)
    ]

    run_pipeline(
        monkeypatch,
        entries,
        {f"s{i}": POLICY_TEXT for i in range(5)},
        lambda text: HAWKISH,
    )

    saved = json.loads(tone_file.read_text())
    assert [r["id"] for r in saved] == ["s0", "s1"]  # oldest first
