
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def get_groq_client():
    """Initialize the Groq API client."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing from your .env file."
        )

    return Groq(api_key=api_key)


def test_groq_connection():
    """Test whether the Groq API is working."""

    client = get_groq_client()

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial research assistant "
                    "specializing in Federal Reserve monetary policy."
                ),
            },
            {
                "role": "user",
                "content": (
                    "In one sentence, explain what a hawkish "
                    "Federal Reserve monetary policy stance means."
                ),
            },
        ],
        temperature=0,
        max_completion_tokens=150,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    print("Testing Groq API connection...\n")

    result = test_groq_connection()

    print(result)
