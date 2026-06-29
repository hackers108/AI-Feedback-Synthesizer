import pandas as pd


def memory_agent(state):

    try:

        sentiment_df = pd.read_csv(
            "memory/sentiment_history.csv"
        )

        topic_df = pd.read_csv(
            "memory/topic_history.csv"
        )

        memory_result = {

            "total_sentiment_records":
            len(sentiment_df),

            "total_topic_records":
            len(topic_df),

            "total_runs":
            len(
                topic_df[
                    "run_date"
                ].unique()
            )
        }

    except Exception as e:

        memory_result = {
            "error": str(e)
        }

    state[
        "memory_result"
    ] = memory_result

    print(
        "\nMemory Agent Completed"
    )

    return state