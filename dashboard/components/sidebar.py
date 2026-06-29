import streamlit as st


def render_sidebar():

    with st.sidebar:

        st.title("🤖 AI Dashboard")

        uploaded_file = st.file_uploader(
            "Upload Feedback CSV",
            type=["csv"]
        )

        st.divider()

        st.success(
            """
### Pipeline

✅ Sentiment Analysis

✅ Topic Modeling

✅ Trend Detection

✅ Report Generation
"""
        )

        st.divider()

        st.info(
            """
### AI Models

• RoBERTa

• BERTopic

• Gemini

Fallback supported.
"""
        )

        st.divider()

        st.caption(
            "Customer Feedback Intelligence Platform v1.0"
        )

    return uploaded_file