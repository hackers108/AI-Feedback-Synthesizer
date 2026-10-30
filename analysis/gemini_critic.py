import json
import os


def load_environment():

    try:

        from dotenv import load_dotenv

        load_dotenv()

    except Exception:

        pass


def get_gemini_client():

    from google import genai

    load_environment()

    return genai.Client(
        api_key=os.getenv(
            "GEMINI_API_KEY"
        )
    )


def evaluate_report_locally(report):

    required_sections = {
        "sentiment": "Missing sentiment analysis",
        "complaint": "Missing customer complaints",
        "recommend": "Missing recommendations"
    }

    report_lower = report.lower()

    issues = [
        issue
        for keyword, issue in required_sections.items()
        if keyword not in report_lower
    ]

    return {
        "status": "FAIL" if issues else "PASS",
        "issues": issues
    }


def evaluate_report(report):

    prompt = f"""
You are a senior business analyst.

Review the following report.

Evaluate:

1. Is sentiment analysis present?
2. Are customer complaints discussed?
3. Are recommendations provided?

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
        "Missing recommendations"
    ]
}}

Report:

{report}
"""

    try:

        client = get_gemini_client()

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
            **evaluate_report_locally(
                report
            )
        }
