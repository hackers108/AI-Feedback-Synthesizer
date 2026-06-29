from typing import TypedDict


class FeedbackState(TypedDict):

    input_file: str

    query: str

    required_agents: list

    sentiment_result: dict

    topic_result: dict

    memory_result: dict

    trend_result: dict

    report: str

    critic: dict