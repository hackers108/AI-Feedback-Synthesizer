from google import genai

from dotenv import load_dotenv

import os

from analysis.local_report import (
    generate_local_report
)


load_dotenv()

client = genai.Client(
    api_key=os.getenv(
        "GEMINI_API_KEY"
    )
)


def generate_business_report(

    sentiment_summary,

    topic_summary,

    memory_summary,

    trend_summary

):

    prompt = f"""
You are a Senior Customer Experience Analyst.

Create a professional business report.

Use the following data.

Sentiment Summary:
{sentiment_summary}

Topic Summary:
{topic_summary}

Memory Summary:
{memory_summary}

Trend Summary:
{trend_summary}

Generate:

1. Executive Summary

2. Sentiment Analysis

3. Key Customer Complaints

4. Trend Analysis

5. Business Recommendations

Keep the report concise and business friendly.
"""

    try:

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

            topic_summary,

            memory_summary,

            trend_summary
        )
