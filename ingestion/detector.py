import pandas as pd


def detect_source(df):

    columns = [
        col.lower().strip()
        for col in df.columns
    ]

    # Play Store
    if (
        "review_description" in columns
        and "rating" in columns
    ):
        return "playstore"

    # Email
    elif (
        "email_body" in columns
        or "sender" in columns
    ):
        return "email"

    # Survey
    elif (
        "feedback" in columns
        or "response_id" in columns
    ):
        return "survey"

    # Support Ticket
    elif (
        "ticket_id" in columns
        or "description" in columns
    ):
        return "ticket"

    # Twitter
    elif (
        "tweet" in columns
        or "tweet_id" in columns
    ):
        return "twitter"

    else:
        return "unknown"