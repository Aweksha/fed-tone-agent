import re

POLICY_KEYWORDS = [
    "monetary policy",
    "federal funds rate",
    "interest rates",
    "inflation",
    "price stability",
    "employment",
    "fomc",
    "economic outlook",
    "summary of economic projections",
    "balance sheet",
    "rate cuts",
    "rate hikes",
]


def is_potentially_relevant(title: str, summary: str = "") -> bool:
    """
    Identify speeches potentially related to monetary policy.

    This is a preliminary filter, not a final classification.
    """

    text = f"{title} {summary}".lower()

    return any(
        re.search(r"\b" + re.escape(keyword) + r"\b", text)
        for keyword in POLICY_KEYWORDS
    )