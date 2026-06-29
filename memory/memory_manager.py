import os
import pandas as pd
from datetime import datetime

# ==========================================
# Memory Files
# ==========================================

SENTIMENT_HISTORY_FILE = (
    "memory/sentiment_history.csv"
)

TOPIC_HISTORY_FILE = (
    "memory/topic_history.csv"
)

# ==========================================
# Sentiment Memory
# ==========================================

def save_sentiment_history():

    source_file = (
        "data/sentiment_feedback.csv"
    )

    if not os.path.exists(
        source_file
    ):

        print(
            "No sentiment file found."
        )

        return

    current_df = pd.read_csv(
        source_file
    )

    current_df["run_date"] = (
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    if os.path.exists(
        SENTIMENT_HISTORY_FILE
    ):

        history_df = pd.read_csv(
            SENTIMENT_HISTORY_FILE
        )

        combined_df = pd.concat(
            [history_df, current_df],
            ignore_index=True
        )

    else:

        combined_df = current_df

    combined_df.to_csv(
        SENTIMENT_HISTORY_FILE,
        index=False
    )

    print(
        "\nSentiment history updated."
    )

# ==========================================
# Topic Memory (V3 Ready)
# ==========================================

def save_topic_history():

    source_file = (
        "data/topic_summary.csv"
    )

    if not os.path.exists(
        source_file
    ):

        print(
            "No topic summary file found."
        )

        return

    topic_df = pd.read_csv(
        source_file
    )

    required_columns = [
        "Topic",
        "Count",
        "Name"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in topic_df.columns
    ]

    if missing_columns:

        print(
            f"Missing columns: {missing_columns}"
        )

        return

    # Remove BERTopic noise cluster
    topic_df = topic_df[
        topic_df["Topic"] != -1
    ]

    run_date = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    current_df = pd.DataFrame({
        "run_date": run_date,
        "topic_id": topic_df["Topic"],
        "topic": topic_df["Name"],
        "count": topic_df["Count"]
    })

    if os.path.exists(
        TOPIC_HISTORY_FILE
    ):

        history_df = pd.read_csv(
            TOPIC_HISTORY_FILE
        )

        combined_df = pd.concat(
            [history_df, current_df],
            ignore_index=True
        )

    else:

        combined_df = current_df

    combined_df.to_csv(
        TOPIC_HISTORY_FILE,
        index=False
    )

    print(
        "\nTopic history updated."
    )

# ==========================================
# Load Memory
# ==========================================

def load_sentiment_history():

    if os.path.exists(
        SENTIMENT_HISTORY_FILE
    ):

        return pd.read_csv(
            SENTIMENT_HISTORY_FILE
        )

    return pd.DataFrame()

def load_topic_history():

    if os.path.exists(
        TOPIC_HISTORY_FILE
    ):

        return pd.read_csv(
            TOPIC_HISTORY_FILE
        )

    return pd.DataFrame()

# ==========================================
# Memory Statistics
# ==========================================

def get_memory_stats():

    sentiment_rows = 0
    topic_rows = 0

    if os.path.exists(
        SENTIMENT_HISTORY_FILE
    ):

        sentiment_rows = len(
            pd.read_csv(
                SENTIMENT_HISTORY_FILE
            )
        )

    if os.path.exists(
        TOPIC_HISTORY_FILE
    ):

        topic_rows = len(
            pd.read_csv(
                TOPIC_HISTORY_FILE
            )
        )

    return {
        "sentiment_records":
        sentiment_rows,

        "topic_records":
        topic_rows
    }

# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    save_sentiment_history()

    save_topic_history()

    stats = get_memory_stats()

    print(
        "\nMemory Updated!"
    )

    print(
        f"\nSentiment Records: "
        f"{stats['sentiment_records']}"
    )

    print(
        f"Topic Records: "
        f"{stats['topic_records']}"
    )