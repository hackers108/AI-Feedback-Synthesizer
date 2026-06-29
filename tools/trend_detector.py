import pandas as pd


def detect_trends():

    history_file = (
        "memory/topic_history.csv"
    )

    history_df = pd.read_csv(
        history_file
    )

    runs = (
        history_df["run_date"]
        .unique()
    )

    runs = sorted(runs)

    if len(runs) < 2:

        return (
            "Need at least two runs "
            "for trend analysis."
        )

    current_run = runs[-1]
    previous_run = runs[-2]

    current_df = history_df[
        history_df["run_date"]
        == current_run
    ]

    previous_df = history_df[
        history_df["run_date"]
        == previous_run
    ]

    trends = []

    for _, row in current_df.iterrows():

        topic = row["topic"]

        current_count = row["count"]

        previous_match = previous_df[
            previous_df["topic"]
            == topic
        ]

        if len(previous_match) == 0:

            continue

        previous_count = (
            previous_match
            .iloc[0]["count"]
        )

        growth = (
            (
                current_count
                - previous_count
            )
            /
            max(previous_count, 1)
        ) * 100

        trends.append({
            "topic": topic,
            "previous_count":
            previous_count,
            "current_count":
            current_count,
            "growth_percent":
            round(growth, 2)
        })

    trends = sorted(
        trends,
        key=lambda x:
        x["growth_percent"],
        reverse=True
    )

    return trends

if __name__ == "__main__":

    trends = detect_trends()

    if isinstance(trends, str):

        print(trends)

    elif len(trends) == 0:

        print("No trends found.")

    else:

        print("\nTrend Analysis:\n")

        for trend in trends:

            print(
                f"Topic: {trend['topic']}"
            )

            print(
                f"Previous Count: "
                f"{trend['previous_count']}"
            )

            print(
                f"Current Count: "
                f"{trend['current_count']}"
            )

            print(
                f"Growth: "
                f"{trend['growth_percent']}%"
            )

            print("-" * 40)