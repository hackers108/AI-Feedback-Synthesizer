import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import streamlit as st
import pandas as pd

from backend.run_pipeline import run_full_pipeline

from dashboard.components.theme import apply_theme
from dashboard.components.sidebar import render_sidebar
from dashboard.components.progress import ProgressManager

from dashboard.pages.overview import render_overview
from dashboard.pages.sentiment import render_sentiment
from dashboard.pages.topics import render_topics
from dashboard.pages.trends import render_trends


# ==========================================
# Page Config
# ==========================================

st.set_page_config(
    page_title="Customer Feedback Intelligence Platform",
    page_icon="🤖",
    layout="wide"
)

apply_theme()


# ==========================================
# Cache
# ==========================================

@st.cache_data
def load_csv(path):

    return pd.read_csv(path)


# ==========================================
# Title
# ==========================================

st.title(
    "🤖 Customer Feedback Intelligence Platform"
)

st.caption(
    "AI Powered Customer Feedback Analytics using "
    "RoBERTa • BERTopic • Gemini"
)

st.divider()


# ==========================================
# Sidebar
# ==========================================

uploaded_file = render_sidebar()


# ==========================================
# Main
# ==========================================

if uploaded_file is not None:

    os.makedirs(
        "uploads",
        exist_ok=True
    )

    upload_path = "uploads/input.csv"

    with open(
        upload_path,
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )

    with st.spinner(
        "Loading dataset..."
    ):

        df = load_csv(
            upload_path
        )

    st.success(
        "Dataset Uploaded Successfully"
    )

    render_overview(
        df
    )

    st.divider()

    if st.button(

        "🚀 Run AI Pipeline",

        use_container_width=True

    ):

        progress = ProgressManager()

        def callback(
            value,
            message
        ):

            progress.update(
                value,
                message
            )

        try:

            result = run_full_pipeline(

                upload_path,

                progress_callback=callback

            )

            progress.success(
                "Pipeline Completed"
            )

            sentiment_df = load_csv(
                "data/sentiment_feedback.csv"
            )

            topic_df = load_csv(
                "data/topic_summary.csv"
            )

            trends = result.get(
                "trends",
                []
            )

            tab1, tab2, tab3 = st.tabs(

                [

                    "😊 Sentiment",

                    "🧠 Topics",

                    "📈 Trends"

                ]

            )

            with tab1:

                render_sentiment(
                    sentiment_df
                )

            with tab2:

                render_topics(
                    topic_df
                )

            with tab3:

                render_trends(
                    trends
                )

        except Exception as e:

            progress.error(
                str(e)
            )

else:

    st.info(
        "Upload a CSV dataset to begin analysis."
    )