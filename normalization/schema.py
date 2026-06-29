import pandas as pd

# ==================================================
# Universal Feedback Schema
# ==================================================

COMMON_COLUMNS = [
    "id",
    "text",
    "source",
    "date",
    "user",
    "rating"
]


def create_normalized_df(
    ids,
    texts,
    source,
    dates=None,
    users=None,
    ratings=None
):
    """
    Create a normalized dataframe
    following the common schema.
    """

    row_count = len(texts)

    if dates is None:
        dates = [None] * row_count

    if users is None:
        users = [None] * row_count

    if ratings is None:
        ratings = [None] * row_count

    normalized_df = pd.DataFrame({
        "id": ids,
        "text": texts,
        "source": source,
        "date": dates,
        "user": users,
        "rating": ratings
    })

    return normalized_df[COMMON_COLUMNS]