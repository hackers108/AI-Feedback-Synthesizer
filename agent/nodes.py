from ingestion.normalize import (
    normalize_file
)

from analysis.sentiment import (
    run_sentiment
)

from analysis.topic_modeling import (
    run_topic_modeling
)

from analysis.gemini_summary import (
    run_gemini_summary
)

from memory.memory_manager import (
    save_sentiment_history,
    save_topic_history
)


def normalize_node(state):

    normalize_file(
        state["input_file"]
    )

    state["normalized_file"] = (
        "data/normalized_feedback.csv"
    )

    return state


def sentiment_node(state):

    state["sentiment_file"] = (
        run_sentiment()
    )

    return state


def topic_node(state):

    state["topic_file"] = (
        run_topic_modeling()
    )

    return state


def gemini_node(state):

    state["report_file"] = (
        run_gemini_summary()
    )

    return state


def memory_node(state):

    save_sentiment_history()

    save_topic_history()

    return state