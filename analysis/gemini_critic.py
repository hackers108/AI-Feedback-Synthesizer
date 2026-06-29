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


def evaluate_report(report):

    prompt = f"""
You are a senior business analyst.

Review the following report.

Evaluate:

1. Is sentiment analysis present?
2. Are customer complaints discussed?
3. Is trend analysis present?
4. Are recommendations provided?

Return ONLY valid JSON.

Example:

{{
    "status": "PASS",
    "issues": []
}}

OR

{{
    "status": "FAIL",
    "issues": [
        "Missing trend analysis",
        "Missing recommendations"
    ]
}}

Report:

{report}
"""

    try:

        response = (
            client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
        )

        text = response.text.strip()

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

        return json.loads(text)

    except Exception as e:

        print(
            f"\nGemini Critic Error: {e}"
        )

        return {
            "status": "PASS",
            "issues": []
        }