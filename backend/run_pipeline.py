from analysis.sentiment import run_sentiment
from analysis.topic_modeling import run_topic_modeling

from memory.memory_manager import (
    save_sentiment_history,
    save_topic_history
)

from tools.trend_detector import detect_trends


# --------------------------------------------------
# Helper
# --------------------------------------------------

def update_progress(callback, value, message):
    if callback:
        callback(value, message)


# --------------------------------------------------
# Full Pipeline
# --------------------------------------------------

def run_full_pipeline(
    input_file,
    progress_callback=None
):

    print("\nStarting Pipeline...")

    # ----------------------------------
    # Step 1
    # ----------------------------------

    update_progress(
        progress_callback,
        0.05,
        "📂 Preparing Dataset..."
    )

    # ----------------------------------
    # Step 2
    # ----------------------------------

    update_progress(
        progress_callback,
        0.15,
        "Running Sentiment Analysis..."
    )

    sentiment_result = run_sentiment(
        input_file
    )

    print(
        "\nSentiment Analysis Complete"
    )

    # run_sentiment may return either a string path or a dict
    if isinstance(sentiment_result, dict):
        sentiment_file = sentiment_result.get(
            "sentiment_file",
            "data/sentiment_feedback.csv"
        )
    else:
        sentiment_file = sentiment_result

    # ----------------------------------
    # Step 3
    # ----------------------------------

    update_progress(
        progress_callback,
        0.35,
        "Loading BERTopic..."
    )

    topic_result = run_topic_modeling(
        sentiment_file,
        progress_callback=progress_callback
    )

    print(
        "\nTopic Modeling Complete"
    )

    if isinstance(topic_result, dict):
        topic_file = topic_result.get(
            "topic_summary_file",
            "data/topic_summary.csv"
        )
    else:
        topic_file = topic_result

    # ----------------------------------
    # Step 4
    # ----------------------------------

    update_progress(
        progress_callback,
        0.90,
        "Updating Memory..."
    )

    save_sentiment_history()

    save_topic_history()

    print(
        "\nMemory Updated"
    )

    # ----------------------------------
    # Step 5
    # ----------------------------------

    update_progress(
        progress_callback,
        0.95,
        "Detecting Trends..."
    )

    trends = detect_trends()

    print(
        "\nTrend Analysis Complete"
    )

    # ----------------------------------
    # Finish
    # ----------------------------------

    update_progress(
        progress_callback,
        1.0,
        "Pipeline Completed"
    )

    print(
        "\nPipeline Completed"
    )

    return {

        "sentiment_file": sentiment_file,

        "topic_file": topic_file,

        "trends": trends

    }