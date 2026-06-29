import plotly.express as px
import streamlit as st


# ==========================================
# Sentiment Bar Chart
# ==========================================

def sentiment_bar(sentiment_df):

    counts = (
        sentiment_df["sentiment"]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "Sentiment",
        "Count"
    ]

    fig = px.bar(
        counts,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        text="Count",
        title="Sentiment Distribution"
    )

    fig.update_layout(
        height=450,
        xaxis_title="",
        yaxis_title="Reviews",
        template="plotly_dark",
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================
# Sentiment Pie Chart
# ==========================================

def sentiment_pie(sentiment_df):

    counts = (
        sentiment_df["sentiment"]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "Sentiment",
        "Count"
    ]

    fig = px.pie(
        counts,
        values="Count",
        names="Sentiment",
        hole=0.45,
        title="Sentiment Share"
    )

    fig.update_layout(
        height=450,
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================
# Topic Frequency
# ==========================================

def topic_chart(topic_df):

    topic_df = (
        topic_df
        .sort_values(
            "Count",
            ascending=False
        )
        .head(10)
    )

    fig = px.bar(

        topic_df,

        x="Business_Label",

        y="Count",

        color="Count",

        text="Count",

        title="Top 10 Complaint Topics"

    )

    fig.update_layout(

        template="plotly_dark",

        xaxis_title="",

        yaxis_title="Reviews",

        height=500,

        coloraxis_showscale=False

    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_xaxes(
        tickangle=-35
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================
# Rating Distribution
# ==========================================

def rating_histogram(df):

    rating_column = None

    for col in [

        "Rating",

        "rating",

        "Score",

        "score"

    ]:

        if col in df.columns:

            rating_column = col

            break

    if rating_column is None:

        return

    fig = px.histogram(

        df,

        x=rating_column,

        nbins=5,

        title="Rating Distribution"

    )

    fig.update_layout(

        template="plotly_dark",

        height=450

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )


# ==========================================
# Trend Growth
# ==========================================

def trend_chart(trends):

    if len(trends) == 0:

        st.info(
            "No trend data available."
        )

        return

    import pandas as pd

    trend_df = pd.DataFrame(
        trends
    )

    fig = px.bar(

        trend_df,

        x="topic",

        y="growth_percent",

        color="growth_percent",

        text="growth_percent",

        title="Topic Growth"

    )

    fig.update_layout(

        template="plotly_dark",

        height=500

    )

    fig.update_xaxes(

        tickangle=-35

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )