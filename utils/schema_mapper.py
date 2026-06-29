import pandas as pd

TEXT_COLUMNS = [
    "text",
    "Text",
    "review",
    "Review",
    "review_description",
    "Review_Text",
    "comment",
    "feedback",
    "content",
    "message"
]

RATING_COLUMNS = [
    "rating",
    "Rating",
    "stars",
    "score"
]


def normalize_schema(df: pd.DataFrame):

    df.columns = [c.strip() for c in df.columns]

    text_column = None

    for col in TEXT_COLUMNS:
        if col in df.columns:
            text_column = col
            break

    if text_column is None:
        raise ValueError(
            f"No supported text column found.\nAvailable columns: {df.columns.tolist()}"
        )

    df["text"] = df[text_column].astype(str)

    for col in RATING_COLUMNS:
        if col in df.columns:
            df["rating"] = df[col]
            break

    return df