import json
import os

from google import genai
from dotenv import load_dotenv


load_dotenv()

client = genai.Client(
    api_key=os.getenv(
        "GEMINI_API_KEY"
    )
)


def decide_agents(query):

    prompt = f"""
You are an AI supervisor.

Choose ONLY the minimum agents required.

Available agents:

1. sentiment
   - overall sentiment analysis

2. topic
   - complaint themes
   - topic modeling

3. memory
   - historical comparisons
   - previous runs

4. trend
   - trend analysis
   - changes over time

Examples:

Question:
What is customer sentiment?

Answer:
{{"agents":["sentiment"]}}

Question:
What are customers complaining about recently?

Answer:
{{"agents":["topic","trend"]}}

Question:
How have complaints changed over time?

Answer:
{{"agents":["topic","memory","trend"]}}

Question:
Generate a complete business report.

Answer:
{{"agents":["sentiment","topic","memory","trend"]}}

Return ONLY valid JSON.

User Question:
{query}
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        text = response.text.strip()

        print(
            "\nGemini Raw Response:"
        )

        print(text)

        # ------------------------------
        # Remove markdown fences
        # ------------------------------

        if "```json" in text:

            text = (
                text
                .split("```json")[1]
                .split("```")[0]
                .strip()
            )

        elif "```" in text:

            text = (
                text
                .replace(
                    "```",
                    ""
                )
                .strip()
            )

        print(
            "\nParsed JSON:"
        )

        print(text)

        return json.loads(text)

    except Exception as e:

        print(
            f"\nGemini Error: {e}"
        )

        return {
            "agents": [
                "sentiment",
                "topic",
                "memory",
                "trend"
            ]
        }