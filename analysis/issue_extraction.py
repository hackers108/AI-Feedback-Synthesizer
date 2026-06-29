import pandas as pd


def run_issue_extraction():

    df = pd.read_csv(
        "data/sentiment_feedback.csv"
    )

    negative_df = df[
        df["sentiment"] == "negative"
    ]

    print(
        "Negative Reviews:",
        len(negative_df)
    )

    for i, review in enumerate(
        negative_df["text"].head(10),
        start=1
    ):

        print(
            f"\nIssue {i}:"
        )

        print(review)

        print("-" * 100)

    return negative_df


if __name__ == "__main__":

    run_issue_extraction()