import streamlit as st


def render_search(df):

    review_column = None

    for col in [

        "text",

        "Text",

        "review",

        "Review",

        "feedback",

        "Feedback"

    ]:

        if col in df.columns:

            review_column = col

            break

    st.subheader("🔍 Search Reviews")

    query = st.text_input(
        "Search customer feedback"
    )

    if review_column is None:

        st.warning(
            "No review column found."
        )

        return

    if query:

        filtered = df[

            df[
                review_column
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

            use_container_width=True

        )