import os
import pandas as pd
from dotenv import load_dotenv
from google import genai
import time


def run_gemini_summary():

    # --------------------------------------------------
    # Load Environment Variables
    # --------------------------------------------------

    load_dotenv()

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "GEMINI_API_KEY not found in .env file"
        )

    client = genai.Client(
        api_key=api_key
    )

    # --------------------------------------------------
    # Load Topic Summary
    # --------------------------------------------------

    INPUT_FILE = (
        "data/topic_summary.csv"
    )

    if not os.path.exists(
        INPUT_FILE
    ):

        raise FileNotFoundError(
            f"{INPUT_FILE} not found"
        )

    topic_df = pd.read_csv(
        INPUT_FILE
    )

    required_columns = [
        "Topic",
        "Count",
        "Name",
        "Representative_Reviews"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in topic_df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print(
        f"\nTopics Loaded: {len(topic_df)}"
    )

    # --------------------------------------------------
    # Remove Noise Topic
    # --------------------------------------------------

    topic_df = topic_df[
        topic_df["Topic"] != -1
    ]

    if len(topic_df) == 0:

        print(
            "\nNo valid topics found."
        )

        return None

    # --------------------------------------------------
    # Sort Topics
    # --------------------------------------------------

    topic_df = topic_df.sort_values(
        by="Count",
        ascending=False
    )

    top_topics = topic_df.head(15)

    print(
        f"Topics Sent To Gemini: {len(top_topics)}"
    )

    # --------------------------------------------------
    # Build Prompt
    # --------------------------------------------------

    topic_text = ""

    for _, row in top_topics.iterrows():

        topic_text += (
            f"\n{'=' * 60}\n"
            f"Topic ID: {row['Topic']}\n"
            f"Users Affected: {row['Count']}\n"
            f"Keywords: {row['Name']}\n\n"
            f"Representative Reviews:\n"
            f"{row['Representative_Reviews']}\n"
        )

    prompt = f"""
    You are a Senior Product Analyst.

    Analyze the customer feedback topics below.

    {topic_text}

    Create a professional business report.

    Include:

    1. Executive Summary

    2. Top Customer Pain Points
    - Rank by severity
    - Mention affected users

    3. Root Cause Analysis

    4. Business Impact

    5. Product Improvement Recommendations

    6. Quick Wins

    7. Long-Term Improvements

    8. Priority Matrix
    - High Priority
    - Medium Priority
    - Low Priority

    9. Final Conclusion

    Return clean markdown.
    """

    # --------------------------------------------------
    # Gemini Analysis
    # --------------------------------------------------

    print(
        "\nGenerating Executive Report..."
    )

    MAX_RETRIES = 2

    report = None

    for attempt in range(MAX_RETRIES):

        try:

            print(
                f"\nGemini Attempt {attempt + 1}/{MAX_RETRIES}"
            )

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )

            report = response.text

            print(
                "\nGemini Report Generated Successfully!"
            )

            break

        except Exception as error:

            print(
                f"\nAttempt {attempt + 1} Failed"
            )

            print(error)

            if attempt < MAX_RETRIES - 1:

                print(
                    "\nWaiting 10 seconds before retry..."
                )

                time.sleep(10)

    # --------------------------------------------------
    # Fallback
    # --------------------------------------------------

    if report is None:

        print(
            "\nAll Gemini attempts failed."
        )

        print(
            "\nGenerating Fallback Report..."
        )

        report = f"""
    # Executive Feedback Report

    ## Executive Summary

    Gemini API was unavailable after multiple retries.

    A fallback report has been generated from BERTopic output.

    ## Top Customer Pain Points

    {top_topics[['Topic', 'Count', 'Name']].to_markdown(index=False)}

    ## Business Impact

    The most frequently occurring topics affect the largest portion of users and should be prioritized.

    ## Recommendations

    1. Address the highest-volume topics first.
    2. Investigate recurring bugs and crashes.
    3. Improve account access and login reliability.
    4. Review messaging and reels functionality.
    5. Monitor future feedback trends.

    ## Conclusion

    Topic modeling completed successfully.

    Review the top topics above to guide product decisions.
    """
        
    # --------------------------------------------------
    # Display Report
    # --------------------------------------------------

    print("\n")
    print("=" * 80)
    print(
        "EXECUTIVE FEEDBACK REPORT"
    )
    print("=" * 80)

    print(report)

    # --------------------------------------------------
    # Save Report
    # --------------------------------------------------

    OUTPUT_FILE = (
        "data/executive_report.md"
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    print("\nSaved:")
    print(OUTPUT_FILE)

    print(
        "\nGemini Summary Completed!"
    )

    return OUTPUT_FILE


if __name__ == "__main__":

    run_gemini_summary()