def normalize_schema(df):

    column_map = {

        "review_description": "text",
        "review": "text",
        "Review": "text",
        "comment": "text",
        "feedback": "text",
        "Feedback": "text",
        "content": "text"

    }

    df = df.rename(columns=column_map)

    if "text" not in df.columns:

        raise ValueError(
            f"No supported text column found.\n"
            f"Available columns:\n{df.columns.tolist()}"
        )

    return df