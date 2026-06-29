import os
import streamlit as st


def load_css():

    css_path = os.path.join(
        "dashboard",
        "styles",
        "theme.css"
    )

    if os.path.exists(css_path):

        with open(
            css_path,
            encoding="utf-8"
        ) as f:

            st.markdown(

                f"<style>{f.read()}</style>",

                unsafe_allow_html=True

            )


def format_number(value):

    try:

        return f"{int(value):,}"

    except:

        return value