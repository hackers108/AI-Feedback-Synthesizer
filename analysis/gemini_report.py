import os

from analysis.local_report import (
    generate_local_report
)


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


def generate_business_report(

    sentiment_summary,

    topic_summary,

    user_question="Generate a complete business report."

):

    prompt = f"""
You are a Senior Customer Experience Analyst.

Create a professional business report.

User Question:
{user_question}

Use the following data.

Sentiment Summary:
{sentiment_summary}

Topic Summary:
{topic_summary}

If the user requested a complete report, generate:

1. Executive Summary

2. Sentiment Analysis

3. Key Customer Complaints

4. Business Recommendations

Otherwise, answer the user's exact question directly and include only the
sections needed for that answer. Do not force a full report for a focused
question. Never include a report quality review, score, critique, or evaluation.
Keep the response concise and business friendly.
Use only the supplied current-upload data. Do not discuss historical trends,
previous runs, changes over time, or future trend monitoring.
"""

    try:

        client = get_gemini_client()

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text
    except Exception as e:

        print(
            f"\nGemini Report Error: {e}"
        )

        print(
            "\nUsing Local Fallback Report..."
        )

        return generate_local_report(

            sentiment_summary,
            topic_summary
        )
