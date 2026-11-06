import json
import feedparser
from pathlib import Path

FED_SPEECHES_RSS = "https://www.federalreserve.gov/feeds/speeches.xml"

BASE_DIR = Path(__file__).resolve().parent.parent
SEEN_FILE = BASE_DIR / "data" / "seen.json"


def fetch_fed_speeches():
    """Fetch the latest speeches from the Federal Reserve."""

    print("Fetching Federal Reserve speeches...\n")

    feed = feedparser.parse(FED_SPEECHES_RSS)

    if feed.bozo and not feed.entries:
        raise RuntimeError(
            f"Unable to parse Federal Reserve RSS feed: {feed.bozo_exception}"
        )

    speeches = []

    for entry in feed.entries:
        speech = {
            "id": entry.get("id") or entry.get("link"),
            "title": entry.get("title", "Unknown"),
            "link": entry.get("link", ""),
            "published": entry.get("published", "Unknown"),
            "summary": entry.get("summary", ""),
        }

        speeches.append(speech)

    return speeches


def load_seen():
    """Load previously processed speech IDs."""

    if not SEEN_FILE.exists():
        return set()

    with open(SEEN_FILE, "r", encoding="utf-8") as file:
        return set(json.load(file))


def save_seen(seen):
    """Save processed speech IDs."""

    with open(SEEN_FILE, "w", encoding="utf-8") as file:
        json.dump(sorted(seen), file, indent=2)


def main():
    speeches = fetch_fed_speeches()
    seen = load_seen()

    print(f"Total speeches in feed: {len(speeches)}")
    print(f"Previously processed: {len(seen)}\n")

    new_speeches = [
        speech for speech in speeches
        if speech["id"] not in seen
    ]

    print(f"New speeches found: {len(new_speeches)}\n")

    for speech in new_speeches:
        print(f"NEW: {speech['title']}")
        seen.add(speech["id"])

    # Temporary ingestion-only checkpoint.
    # Later, we'll mark items processed only after analysis succeeds.
    save_seen(seen)

    print("\nSpeech tracking completed!")


if __name__ == "__main__":
    main()