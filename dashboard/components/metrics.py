import streamlit as st

from dashboard.ui_utils.helpers import format_number


def render_metrics(df):

    st.subheader(" Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    rating = "N/A"

    for col in [
        "rating",
        "Rating",
        "score",
        "Score"
    ]:

        if col in df.columns:

            rating = round(
                df[col].mean(),
                2
            )

            break

    col1.metric(
        "Reviews",
        format_number(len(df))
    )

    col2.metric(
        "Average Rating",
        rating
    )

    col3.metric(
        "Columns",
        len(df.columns)
    )

    col4.metric(
        "AI Engine",
        "BERTopic"
    )