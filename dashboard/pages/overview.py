import streamlit as st

from dashboard.components.metrics import (
    render_metrics
)

from dashboard.components.search import (
    render_search
)

from dashboard.components.charts import (
    rating_histogram
)


def render_overview(df):

    st.header(
        "📊 Dataset Overview"
    )

    render_metrics(
        df
    )

    st.divider()

    render_search(
        df
    )

    st.divider()

    with st.expander(
        "Dataset Preview",
        expanded=False
    ):

        st.dataframe(

            df.head(20),

            use_container_width=True

        )

    st.divider()

    rating_histogram(
        df
    )

    st.divider()

    st.subheader(
        "Dataset Statistics"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(

            "Missing Values",

            int(
                df.isnull()
                .sum()
                .sum()
            )

        )

        st.metric(

            "Duplicate Rows",

            int(
                df.duplicated()
                .sum()
            )

        )

    with col2:

        st.metric(

            "Memory Usage",

            f"{df.memory_usage(deep=True).sum()/1024/1024:.2f} MB"

        )

        st.metric(

            "Features",

            len(df.columns)

        )

    st.divider()

    st.subheader(
        "Column Types"
    )

    dtype_df = (
        df.dtypes
        .reset_index()
    )

    dtype_df.columns = [

        "Column",

        "Data Type"

    ]

    st.dataframe(

        dtype_df,

        use_container_width=True

    )