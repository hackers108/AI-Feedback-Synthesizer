import pandas as pd


def sentiment_agent(state):

    df = pd.read_csv(
        "data/sentiment_feedback.csv"
    )

    sentiment_counts = (
        df["sentiment"]
        .value_counts()
        .to_dict()
    )

    state["sentiment_result"] = (
        sentiment_counts
    )

    print(
        "\nSentiment Agent Completed"
    )

    return state