
import os

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

load_dotenv()

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

ALLOWED_SERIES = {
    "FEDFUNDS",   # Effective Federal Funds Rate
    "CPIAUCSL",   # Consumer Price Index
    "UNRATE",     # Unemployment Rate
}


# ============================================================
# TOOL 1: FETCH FULL FEDERAL RESERVE SPEECH TEXT
# ============================================================

def fetch_full_text(url: str) -> str:
    """
    Retrieve the full text of a Federal Reserve speech.

    Args:
        url: URL of the Federal Reserve speech.

    Returns:
        Extracted speech text.

    Raises:
        requests.RequestException: If the HTTP request fails.
        ValueError: If speech content cannot be extracted.
    """

    if not url.startswith(
        "https://www.federalreserve.gov/newsevents/"
    ):
        raise ValueError("Unsupported Federal Reserve URL")

    response = requests.get(
        url,
        timeout=20,
        headers={
            "User-Agent": "FedToneAgent/1.0 (academic research project)"
        },
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Locate the main speech content.
    content = soup.select_one("#article")

    if content is None:
        content = soup.select_one("main")

    if content is None:
        raise ValueError(
            "Could not locate speech content on the page."
        )

    # Remove unnecessary HTML elements.
    for element in content.select(
        "script, style, nav, footer, aside"
    ):
        element.decompose()

    text = content.get_text(
        separator="\n",
        strip=True,
    )

    if len(text) < 200:
        raise ValueError(
            "Extracted speech text is unexpectedly short."
        )

    return text


# ============================================================
# TOOL 2: FETCH ECONOMIC DATA FROM FRED
# ============================================================

def get_fred_data(
    series_id: str,
    limit: int = 12,
) -> list[dict]:
    """
    Retrieve recent economic observations from the FRED API.

    Supported series:
        FEDFUNDS - Effective Federal Funds Rate (%)
        CPIAUCSL - Consumer Price Index (index level)
        UNRATE   - Unemployment Rate (%)

    Args:
        series_id: FRED economic series identifier.
        limit: Maximum number of observations to retrieve.

    Returns:
        List of dictionaries containing dates and values.

    Raises:
        ValueError: If the series or limit is invalid.
        RuntimeError: If the API key is missing.
        requests.RequestException: If the API request fails.
    """

    if series_id not in ALLOWED_SERIES:
        raise ValueError(
            f"Unsupported FRED series: {series_id}"
        )

    if not 1 <= limit <= 120:
        raise ValueError(
            "limit must be between 1 and 120"
        )

    api_key = os.getenv("FRED_API_KEY")

    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY is missing. "
            "Add it to your .env file."
        )

    response = requests.get(
        FRED_URL,
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        },
        timeout=20,
    )

    response.raise_for_status()

    observations = response.json().get(
        "observations", []
    )

    results = []

    for item in observations:
        value = item.get("value", ".")

        # FRED represents missing values using ".".
        if value == ".":
            continue

        results.append(
            {
                "date": item["date"],
                "value": float(value),
            }
        )

    return results
