import json
import time
from calendar import timegm
from datetime import datetime, timezone
from pathlib import Path

import feedparser

from agent.classifier import classify_text
from agent.filter import is_potentially_relevant
from agent.tools import fetch_full_text


FED_SPEECHES_RSS = "https://www.federalreserve.gov/feeds/speeches.xml"

BASE_DIR = Path(__file__).resolve().parent.parent
TONE_FILE = BASE_DIR / "data" / "tone.json"

# Cap Groq calls per run to stay within free-tier limits.
# Anything left over is picked up on the next run.
MAX_CLASSIFICATIONS_PER_RUN = 10

# The model's output varies between calls, so one failed
# evidence check is worth a second try.
CLASSIFY_ATTEMPTS = 2

# Pause between Groq calls to avoid per-minute token limits.
SECONDS_BETWEEN_CALLS = 5


# ============================================================
# FEED
# ============================================================

def parse_entry(entry) -> dict:
    """Turn an RSS entry into a speech record."""

    raw_title = entry.get("title", "Unknown")

    # Fed titles look like "Bowman, Supervisory Modernization ...".
    speaker, sep, title = raw_title.partition(", ")
    if not sep:
        speaker, title = "", raw_title

    parsed = entry.get("published_parsed")
    date = (
        datetime.fromtimestamp(timegm(parsed), tz=timezone.utc)
        .date()
        .isoformat()
        if parsed
        else ""
    )

    link = entry.get("link", "")

    return {
        "id": entry.get("id") or link,
        "title": title.strip(),
        "speaker": speaker.strip(),
        "date": date,
        "link": link,
        "summary": entry.get("summary", ""),
    }


def fetch_fed_speeches() -> list[dict]:
    """Fetch the latest speeches from the Federal Reserve."""

    print("Fetching Federal Reserve speeches...\n")

    feed = feedparser.parse(FED_SPEECHES_RSS)

    if feed.bozo and not feed.entries:
        raise RuntimeError(
            f"Unable to parse Federal Reserve RSS feed: {feed.bozo_exception}"
        )

    return [parse_entry(entry) for entry in feed.entries]


# ============================================================
# RESULTS STORAGE
# ============================================================

def load_results() -> list[dict]:
    """Load previously processed speeches."""

    if not TONE_FILE.exists():
        return []

    with open(TONE_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_results(results: list[dict]) -> None:
    """Save processed speeches, oldest first."""

    results = sorted(results, key=lambda r: (r.get("date", ""), r["id"]))

    with open(TONE_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
        file.write("\n")


def speech_record(speech: dict, **fields) -> dict:
    """Build the stored record for a processed speech."""

    return {
        "id": speech["id"],
        "title": speech["title"],
        "speaker": speech["speaker"],
        "date": speech["date"],
        "link": speech["link"],
        **fields,
    }


# ============================================================
# PROCESSING
# ============================================================

def classify_with_retry(text: str):
    """Classify speech text, retrying once on failure."""

    for attempt in range(1, CLASSIFY_ATTEMPTS + 1):
        try:
            return classify_text(text)
        except Exception as exc:
            print(f"  Attempt {attempt} failed: {exc}")
            if attempt == CLASSIFY_ATTEMPTS:
                raise
            time.sleep(SECONDS_BETWEEN_CALLS)


def process_speech(speech: dict) -> tuple[dict, bool]:
    """
    Fetch, screen, and classify one speech.

    Returns:
        (record, used_groq)
    """

    text = fetch_full_text(speech["link"])

    # Cheap keyword screen first: skip speeches with no
    # monetary policy terms at all, without calling Groq.
    if not is_potentially_relevant(speech["title"], text):
        print("  Skipped: no monetary policy terms.")
        return (
            speech_record(
                speech,
                relevant=False,
                label="not applicable",
                score=None,
                confidence=None,
                reason="Skipped by keyword screen: no monetary policy terms.",
                evidence="",
            ),
            False,
        )

    result = classify_with_retry(text)
    print(f"  {result.label} ({result.score}), confidence {result.confidence}")

    return speech_record(speech, **result.model_dump()), True


def main():
    speeches = fetch_fed_speeches()
    results = load_results()
    processed = {r["id"] for r in results}

    new_speeches = [s for s in speeches if s["id"] not in processed]

    print(f"Speeches in feed: {len(speeches)}")
    print(f"Already processed: {len(processed)}")
    print(f"New: {len(new_speeches)}\n")

    groq_calls = 0
    failures = 0

    # Oldest first, so a capped run fills in history in order.
    for speech in sorted(new_speeches, key=lambda s: s["date"]):
        if groq_calls >= MAX_CLASSIFICATIONS_PER_RUN:
            print("Reached the per-run limit; the rest will run next time.")
            break

        print(f"{speech['date']}  {speech['speaker']}: {speech['title']}")

        if groq_calls:
            time.sleep(SECONDS_BETWEEN_CALLS)

        try:
            record, used_groq = process_speech(speech)
        except Exception as exc:
            # Not saved, so it is retried on the next run.
            print(f"  Failed, will retry next run: {exc}")
            failures += 1
            groq_calls += 1
            continue

        groq_calls += used_groq
        results.append(record)

        # Save after each speech so progress survives a crash.
        save_results(results)

    print(f"\nDone. {failures} failed and will be retried.")


if __name__ == "__main__":
    main()
