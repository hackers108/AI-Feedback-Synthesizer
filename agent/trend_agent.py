import pandas as pd


def trend_agent(state):

    try:

        topic_df = pd.read_csv(
            "memory/topic_history.csv"
        )

        runs = sorted(
            topic_df[
                "run_date"
            ].unique()
        )

        if len(runs) < 2:

            trend_result = {
                "status":
                "Not enough runs"
            }

        else:

            current_run = runs[-1]

            previous_run = runs[-2]

            current_topics = len(
                topic_df[
                    topic_df["run_date"]
                    == current_run
                ]
            )

            previous_topics = len(
                topic_df[
                    topic_df["run_date"]
                    == previous_run
                ]
            )

            trend_result = {

                "current_run":
                current_run,

                "previous_run":
                previous_run,

                "current_topics":
                current_topics,

                "previous_topics":
                previous_topics,

                "topic_change":
                current_topics
                - previous_topics
            }

    except Exception as e:

        trend_result = {
            "error": str(e)
        }

    state[
        "trend_result"
    ] = trend_result

    print(
        "\nTrend Agent Completed"
    )

    return state