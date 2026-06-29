import streamlit as st
import pandas as pd
import shutil
import sys
import time
from pathlib import Path

# --------------------------------------------------
# Project Path
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from agent.graph import graph

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Feedback Synthesizer",
    layout="wide"
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title(
    "AI Feedback Synthesizer"
)

st.sidebar.markdown(
    """
    ### Workflow

    1. Source Detection
    2. Data Normalization
    3. Sentiment Analysis
    4. Topic Modeling
    5. Executive Reporting
    6. Memory Update
    """
)

st.sidebar.divider()

st.sidebar.write(
    "LangGraph Powered Feedback Intelligence Platform"
)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title(
    "AI Feedback Synthesizer"
)

st.write(
    """
    Analyze customer feedback using
    Sentiment Analysis,
    Topic Modeling,
    Executive Reporting,
    and Memory Tracking.
    """
)

st.divider()

# --------------------------------------------------
# Upload Section
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Feedback CSV",
    type=["csv"]
)

if uploaded_file:

    save_path = "uploads/input.csv"

    with open(
        save_path,
        "wb"
    ) as file:

        shutil.copyfileobj(
            uploaded_file,
            file
        )

    st.success(
        "File uploaded successfully."
    )

    # --------------------------------------------------
    # Run Analysis Button
    # --------------------------------------------------

    if st.button(
        "Run Analysis",
        use_container_width=True
    ):

        start_time = time.time()

        progress_bar = st.progress(0)

        status_box = st.empty()

        try:

            status_box.info(
                "Initializing workflow..."
            )

            progress_bar.progress(10)

            result = graph.invoke(
                {
                    "input_file":
                    save_path
                }
            )

            progress_bar.progress(100)

            elapsed_time = round(
                time.time() - start_time,
                2
            )

            status_box.success(
                f"Analysis completed in {elapsed_time} seconds."
            )

        except Exception as error:

            st.error(
                f"Pipeline failed:\n{error}"
            )

            st.stop()

        # --------------------------------------------------
        # Load Generated Files
        # --------------------------------------------------

        sentiment_df = pd.read_csv(
            "data/sentiment_feedback.csv"
        )

        topic_df = pd.read_csv(
            "data/topic_summary.csv"
        )

        topic_df = topic_df[
            topic_df["Topic"] != -1
        ]

        # --------------------------------------------------
        # Summary Metrics
        # --------------------------------------------------

        st.header(
            "Summary Metrics"
        )

        total_reviews = len(
            sentiment_df
        )

        positive_reviews = len(
            sentiment_df[
                sentiment_df["sentiment"]
                == "positive"
            ]
        )

        neutral_reviews = len(
            sentiment_df[
                sentiment_df["sentiment"]
                == "neutral"
            ]
        )

        negative_reviews = len(
            sentiment_df[
                sentiment_df["sentiment"]
                == "negative"
            ]
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Reviews",
            f"{total_reviews:,}"
        )

        col2.metric(
            "Positive",
            f"{positive_reviews:,}"
        )

        col3.metric(
            "Neutral",
            f"{neutral_reviews:,}"
        )

        col4.metric(
            "Negative",
            f"{negative_reviews:,}"
        )

        st.divider()

        # --------------------------------------------------
        # Sentiment Analysis
        # --------------------------------------------------

        st.header(
            "Sentiment Distribution"
        )

        sentiment_counts = (
            sentiment_df["sentiment"]
            .value_counts()
        )

        st.bar_chart(
            sentiment_counts
        )

        st.divider()

        # --------------------------------------------------
        # Top Customer Issues
        # --------------------------------------------------

        st.header(
            "Top Customer Issues"
        )

        st.dataframe(
            topic_df[
                [
                    "Topic",
                    "Count",
                    "Name"
                ]
            ]
            .sort_values(
                by="Count",
                ascending=False
            )
            .head(10),
            use_container_width=True
        )

        st.divider()

        # --------------------------------------------------
        # Topic Distribution
        # --------------------------------------------------

        st.header(
            "Topic Distribution"
        )

        topic_chart_df = (
            topic_df[
                [
                    "Name",
                    "Count"
                ]
            ]
            .head(10)
            .set_index(
                "Name"
            )
        )

        st.bar_chart(
            topic_chart_df
        )

        st.divider()

        # --------------------------------------------------
        # Executive Report
        # --------------------------------------------------

        st.header(
            "Executive Report"
        )

        try:

            with open(
                "data/executive_report.md",
                "r",
                encoding="utf-8"
            ) as file:

                report = file.read()

            st.markdown(
                report
            )

            st.download_button(
                label="Download Executive Report",
                data=report,
                file_name="executive_report.md",
                mime="text/markdown"
            )

        except Exception:

            st.warning(
                "Executive report not found."
            )

        st.divider()

        # --------------------------------------------------
        # Pipeline Outputs
        # --------------------------------------------------

        with st.expander(
            "Pipeline Output Files"
        ):

            st.json(
                result
            )