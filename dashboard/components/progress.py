import streamlit as st


class ProgressManager:

    def __init__(self):

        self.progress = st.progress(0)

        self.status = st.empty()

    def update(
        self,
        value,
        message
    ):

        self.progress.progress(
            int(value * 100)
        )

        self.status.info(
            message
        )

    def success(
        self,
        message
    ):

        self.progress.progress(
            100
        )

        self.status.success(
            message
        )

    def error(
        self,
        message
    ):

        self.status.error(
            message
        )