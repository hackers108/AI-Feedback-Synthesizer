import pandas as pd

from ingestion.detector import detect_source
from normalization.schema import create_normalized_df


def normalize_file(file_path):

    df = pd.read_csv(file_path)

    source = detect_source(df)

    print(f"\nDetected Source: {source}")

    # =====================================
    # Play Store
    # =====================================

    if source == "playstore":

        normalized_df = create_normalized_df(
            ids=df.index.astype(str),
            texts=df["review_description"],
            source="playstore",
            dates=df["review_date"],
            ratings=df["rating"]
        )

    # =====================================
    # Email
    # =====================================

    elif source == "email":

        normalized_df = create_normalized_df(
            ids=df["email_id"],
            texts=df["email_body"],
            source="email",
            dates=df["created_at"],
            users=df["sender"]
        )

    # =====================================
    # Survey
    # =====================================

    elif source == "survey":

        normalized_df = create_normalized_df(
            ids=df["response_id"],
            texts=df["feedback"],
            source="survey",
            dates=df["submitted_at"],
            users=df["respondent"],
            ratings=df.get("rating")
        )

    # =====================================
    # Ticket
    # =====================================

    elif source == "ticket":

        normalized_df = create_normalized_df(
            ids=df["ticket_id"],
            texts=df["description"],
            source="ticket",
            dates=df["created_at"],
            users=df["customer"]
        )

    # =====================================
    # Twitter
    # =====================================

    elif source == "twitter":

        normalized_df = create_normalized_df(
            ids=df["tweet_id"],
            texts=df["tweet"],
            source="twitter",
            dates=df["created_at"],
            users=df["username"]
        )

    else:

        raise ValueError(
            "Unsupported CSV format."
        )

    normalized_df.to_csv(
        "data/normalized_feedback.csv",
        index=False
    )

    print(
        "\nNormalized file saved:"
    )

    print(
        "data/normalized_feedback.csv"
    )

    return normalized_df


if __name__ == "__main__":

    normalize_file(
        "data/instagram.csv"
    )