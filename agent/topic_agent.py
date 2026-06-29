import pandas as pd


def topic_agent(state):

    df = pd.read_csv(
        "data/topic_summary.csv"
    )

    topics = (
        df[
            [
                "Business_Label",
                "Count"
            ]
        ]
        .head(10)
        .to_dict(
            orient="records"
        )
    )

    state["topic_result"] = topics

    print(
        "\nTopic Agent Completed"
    )

    return state