import streamlit as st


def sentiment_download(df):

    st.download_button(

        "⬇ Download Sentiment CSV",

        df.to_csv(index=False),

        "sentiment_results.csv",

        "text/csv"

    )


def topic_download(df):

    st.download_button(

        "⬇ Download Topic Summary",

        df.to_csv(index=False),

        "topic_summary.csv",

        "text/csv"

    )