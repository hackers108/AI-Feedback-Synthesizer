import streamlit as st

from dashboard.components.charts import (
    sentiment_bar,
    sentiment_pie
)

from dashboard.components.downloads import (
    sentiment_download
)


def render_sentiment(sentiment_df):

    st.header("😊 Sentiment Analysis")

    sentiment_counts = (
        sentiment_df["sentiment"]
        .value_counts()
    )

    positive = sentiment_counts.get(
        "positive",
        0
    )

    neutral = sentiment_counts.get(
        "neutral",
        0
    )

    negative = sentiment_counts.get(
        "negative",
        0
    )

    total = (
        positive +
        neutral +
        negative
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "😊 Positive",
        positive,
        f"{positive*100/total:.1f}%"
    )

    c2.metric(
        "😐 Neutral",
        neutral,
        f"{neutral*100/total:.1f}%"
    )

    c3.metric(
        "😡 Negative",
        negative,
        f"{negative*100/total:.1f}%"
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        sentiment_bar(
            sentiment_df
        )

    with right:

        sentiment_pie(
            sentiment_df
        )

    st.divider()

    st.subheader(
        "Filter Reviews"
    )

    selected = st.multiselect(

        "Select Sentiments",

        options=[
            "positive",
            "neutral",
            "negative"
        ],

        default=[
            "positive",
            "neutral",
            "negative"
        ]

    )

    filtered = sentiment_df[

        sentiment_df[
            "sentiment"
        ].isin(
            selected
        )

    ]

    st.dataframe(

        filtered,

        use_container_width=True,

        height=500

    )

    st.divider()

    sentiment_download(
        filtered
    )