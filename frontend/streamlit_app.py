import shutil #Copies uploaded CSV to uploads folder.
import sys
import time
from pathlib import Path #file handling

import pandas as pd
import streamlit as st


# --------------------------------------------------
# Project Path
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Feedback Synthesizer",
    layout="wide"
)

#to stop reload of LangGraph on every run
@st.cache_resource
def load_agent_graph():

    from agent.graph import graph  #Loads LangGraph

    return graph


def initialize_session():

    defaults = {
        "active_file_path": None,
        "active_file_name": None,
        "sentiment_file": None,
        "topic_file": None,
        "last_result": None
    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value


def reset_analysis_cache():

    #Reset when new csv uploaded
    st.session_state.sentiment_file = None
    st.session_state.topic_file = None
    st.session_state.last_result = None


def render_outputs(result):

    #converts Ai outputs to dashboard components

    result = {
        key: value
        for key, value in result.items()
        if key != "progress_callback"
    }

    if result.get(
        "unsupported_query"
    ):

        st.warning(
            result.get(
                "report",
                "This workflow only supports customer feedback analysis."
            )
        )

        return

    completed_agents = result.get(
        "completed_agents",
        []
    )

    show_sentiment = "sentiment" in completed_agents
    show_topics = "topic" in completed_agents

    try:

        sentiment_df = None
        topic_df = None

        if show_sentiment:

            sentiment_df = pd.read_csv(
                result.get(
                    "sentiment_file",
                    "data/sentiment_feedback.csv"
                )
            )

        if show_topics:

            topic_df = pd.read_csv(
                result.get(
                    "topic_file",
                    "data/topic_summary.csv"
                )
            )

    except Exception as error:

        st.info(
            f"Run completed, but there are no dashboard files to display yet: {error}"
        )

        with st.expander(
            "Agent Result"
        ):

            st.json(
                result
            )

        return

    if show_sentiment and sentiment_df is not None:

        st.header(
            "Sentiment Result"
        )

        sentiment_result = result.get(
            "sentiment_result",
            {}
        )

        total_reviews = sum(
            sentiment_result.values()
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Analyzed Reviews",
            f"{total_reviews:,}"
        )

        col2.metric(
            "Positive",
            f"{sentiment_result.get('positive', 0):,}"
        )

        col3.metric(
            "Neutral",
            f"{sentiment_result.get('neutral', 0):,}"
        )

        col4.metric(
            "Negative",
            f"{sentiment_result.get('negative', 0):,}"
        )

        sentiment_counts = (
            sentiment_df["sentiment"]
            .value_counts()
        )

        st.bar_chart(
            sentiment_counts
        )

        st.divider()

    if show_topics and topic_df is not None:

        topic_df = topic_df[
            topic_df["Topic"] != -1
        ]

        st.header(
            "Topic Result"
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

    if result.get(
        "report"
    ):

        st.header(
            "Executive Report"
        )

        st.markdown(
            result["report"]
        )

        st.download_button(
            label="Download Executive Report",
            data=result["report"],
            file_name="executive_report.md",
            mime="text/markdown"
        )

        st.divider()

    with st.expander(
        "Agent Result"
    ):

        st.json(
            result
        )


initialize_session()

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1180px;
        padding-top: 2rem;
    }
    div[data-testid="stMetric"] {
        background: #151923;
        border: 1px solid #2b3342;
        border-radius: 8px;
        padding: 14px 16px;
    }
    div[data-testid="stFileUploader"] section {
        border-radius: 8px;
        border-color: #394150;
    }
    .status-card {
        background: #111722;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 16px 18px;
        margin: 12px 0 18px 0;
    }
    .status-card strong {
        color: #f8fafc;
    }
    .muted {
        color: #94a3b8;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True
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

    1. Upload feedback CSV
    2. Ask multiple questions
    3. Supervisor routes agents
    4. Dashboard displays current-file results
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

st.caption(
    """
    Upload a feedback CSV and analyze only the currently selected file.
    """
)

st.divider()


SUGGESTED_PROMPTS = [
    "Generate a complete business report and review its quality.",
    "What is the overall customer sentiment?",
    "Find the top customer complaint themes.",
    "Create an executive summary with recommendations."
]


# --------------------------------------------------
# Upload Section
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Feedback CSV",
    type=["csv"]
)

if uploaded_file:

    save_path = "uploads/input.csv"
    Path("uploads").mkdir(
        exist_ok=True
    )

    if (
        st.session_state.active_file_name
        != uploaded_file.name
    ):

        reset_analysis_cache()

    with open(
        save_path,
        "wb"
    ) as file:

        shutil.copyfileobj(
            uploaded_file,
            file
        )

    st.session_state.active_file_path = save_path
    st.session_state.active_file_name = uploaded_file.name


if st.session_state.active_file_path:

    st.markdown(
        f"""
        <div class="status-card">
            <strong>Active file:</strong> {st.session_state.active_file_name}
            <div class="muted">Only this currently uploaded file will be used for the analysis.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    task_col, ask_col = st.columns(
        [1, 1.4]
    )

    with task_col:

        selected_prompt = st.selectbox(
            "Suggested agent tasks",
            SUGGESTED_PROMPTS
        )

    with ask_col:

        custom_query = st.text_input(
            "Ask the AI workflow what to do",
            placeholder="Example: Find the top complaint themes and generate recommendations."
        )

    query = custom_query.strip() or selected_prompt

    st.caption(
        "Each result is based only on the active uploaded CSV."
    )

    if st.button(
        "Run Analysis",
        use_container_width=True
    ):

        start_time = time.time()

        progress_bar = st.progress(0)

        progress_percent = st.empty()

        status_box = st.empty()

        try:

            def update_progress(value, message):

                value = int(value)

                value = max(
                    0,
                    min(
                        value,
                        100
                    )
                )

                progress_bar.progress(
                    value
                )

                progress_percent.markdown(
                    f"**{value}% complete**"
                )

                status_box.info(
                    message
                )

            update_progress(
                3,
                "Preparing workflow"
            )

            workflow = load_agent_graph()

            graph_input = {
                "input_file":
                st.session_state.active_file_path,

                "query":
                query,

                "progress_callback":
                update_progress
            }

            if st.session_state.sentiment_file:

                graph_input[
                    "sentiment_file"
                ] = st.session_state.sentiment_file

            if st.session_state.topic_file:

                graph_input[
                    "topic_file"
                ] = st.session_state.topic_file

            result = workflow.invoke(
                graph_input
            )

            result.pop(
                "progress_callback",
                None
            )

            st.session_state.last_result = result

            st.session_state.sentiment_file = result.get(
                "sentiment_file",
                st.session_state.sentiment_file
            )

            st.session_state.topic_file = result.get(
                "topic_file",
                st.session_state.topic_file
            )

            update_progress(
                100,
                "Analysis completed"
            )

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

    if st.session_state.last_result:

        render_outputs(
            st.session_state.last_result
        )

else:

    st.info(
        "Upload a CSV file to analyze its current feedback data."
    )
