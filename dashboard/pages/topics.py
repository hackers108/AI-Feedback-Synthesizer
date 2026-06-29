import streamlit as st

from dashboard.components.charts import (
    topic_chart
)

from dashboard.components.downloads import (
    topic_download
)


def render_topics(topic_df):

    st.header("🧠 Topic Modeling")

    if topic_df.empty:

        st.warning(
            "No topics available."
        )

        return

    # ----------------------------------------
    # KPI Cards
    # ----------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Topics Found",
        len(topic_df)
    )

    col2.metric(
        "Largest Topic",
        topic_df["Count"].max()
    )

    col3.metric(
        "Average Reviews / Topic",
        round(
            topic_df["Count"].mean(),
            1
        )
    )

    st.divider()

    # ----------------------------------------
    # Topic Chart
    # ----------------------------------------

    topic_chart(
        topic_df
    )

    st.divider()

    # ----------------------------------------
    # Search Topics
    # ----------------------------------------

    st.subheader(
        "🔍 Search Topics"
    )

    query = st.text_input(
        "Search Business Labels"
    )

    filtered = topic_df

    if query:

        filtered = topic_df[
            topic_df[
                "Business_Label"
            ]
            .astype(str)
            .str.contains(
                query,
                case=False,
                na=False
            )
        ]

    st.dataframe(
        filtered,
        use_container_width=True,
        height=500
    )

    st.divider()

    # ----------------------------------------
    # Executive Summary
    # ----------------------------------------

    st.subheader(
        "📋 Executive Summary"
    )

    top5 = (
        topic_df
        .sort_values(
            "Count",
            ascending=False
        )
        .head(5)
    )

    for i, row in top5.iterrows():

        st.info(
            f"""
### {row['Business_Label']}

Reviews : {row['Count']}

Representative Review

{row['Representative_Reviews']}
"""
        )

    st.divider()

    topic_download(
        filtered
    )