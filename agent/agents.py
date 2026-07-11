import os
from typing import NotRequired, TypedDict

import pandas as pd


class FeedbackState(TypedDict):
    input_file: str
    query: NotRequired[str]
    required_agents: NotRequired[list[str]]
    completed_agents: NotRequired[list[str]]
    sentiment_file: NotRequired[str]
    sentiment_result: NotRequired[dict]
    topic_file: NotRequired[str]
    topic_result: NotRequired[object]
    memory_result: NotRequired[dict]
    trend_result: NotRequired[object]
    report: NotRequired[str]
    critic: NotRequired[dict]
    unsupported_query: NotRequired[bool]
    progress_callback: NotRequired[object]


AGENT_ORDER = [
    "sentiment",
    "topic",
    "memory",
    "trend",
    "report",
    "critic"
]

AGENT_DEPENDENCIES = {
    "topic": ["sentiment"],
    "memory": ["sentiment", "topic"],
    "trend": ["sentiment", "topic", "memory"],
    "report": ["sentiment", "topic", "memory", "trend"],
    "critic": ["report"]
}

SUPPORTED_KEYWORDS = [
    "feedback",
    "review",
    "customer",
    "sentiment",
    "complaint",
    "topic",
    "theme",
    "issue",
    "trend",
    "memory",
    "history",
    "compare",
    "report",
    "summary",
    "recommendation",
    "analysis",
    "quality"
]

UNSUPPORTED_MESSAGE = (
    "I can only help with customer feedback analysis. "
    "Ask for sentiment, complaint topics, trends, comparison, or a report."
)


def update_progress(state, value, message):
    callback = state.get("progress_callback")

    if callback:
        callback(value, message)


def mark_completed(state, agent_name):
    completed = list(state.get("completed_agents", []))

    if agent_name not in completed:
        completed.append(agent_name)

    state["completed_agents"] = completed
    return state


def is_supported_query(query):
    query = query.lower()
    return any(keyword in query for keyword in SUPPORTED_KEYWORDS)


def choose_agents(query):
    query = query.lower()

    if any(word in query for word in ["quality", "critic", "critique", "missing"]):
        return ["critic"]

    if any(word in query for word in ["report", "executive", "summary", "recommendation"]):
        return ["report"]

    if any(word in query for word in ["trend", "rising", "changed", "over time"]):
        return ["trend"]

    if any(word in query for word in ["memory", "history", "historical", "compare", "previous"]):
        return ["memory"]

    if any(word in query for word in ["complaint", "topic", "theme", "issue", "pain point"]):
        return ["topic"]

    if "sentiment" in query:
        return ["sentiment"]

    return ["report"]


def expand_dependencies(selected_agents):
    expanded = set()

    def add_agent(agent_name):
        for dependency in AGENT_DEPENDENCIES.get(agent_name, []):
            add_agent(dependency)

        expanded.add(agent_name)

    for agent_name in selected_agents:
        add_agent(agent_name)

    return [
        agent_name
        for agent_name in AGENT_ORDER
        if agent_name in expanded
    ]


def route_next_agent(state):
    required = state.get("required_agents", [])
    completed = state.get("completed_agents", [])

    for agent_name in AGENT_ORDER:
        if agent_name in required and agent_name not in completed:
            return agent_name

    return "end"


def supervisor_agent(state):
    query = state.get("query") or "Generate a complete business report."
    update_progress(state, 5, "Supervisor is reading the request")

    if not is_supported_query(query):
        state["required_agents"] = []
        state["completed_agents"] = []
        state["unsupported_query"] = True
        state["report"] = UNSUPPORTED_MESSAGE
        update_progress(state, 100, "Unsupported question")
        return state

    selected_agents = choose_agents(query)
    state["required_agents"] = expand_dependencies(selected_agents)
    state["completed_agents"] = []

    update_progress(state, 10, f"Selected agents: {', '.join(state['required_agents'])}")
    return state


def sentiment_agent(state):
    update_progress(state, 15, "Running RoBERTa sentiment analysis")

    sentiment_file = state.get("sentiment_file")

    if not sentiment_file:
        from analysis.sentiment import run_sentiment

        sentiment_file = run_sentiment(state["input_file"])
        state["sentiment_file"] = sentiment_file

    df = pd.read_csv(sentiment_file)
    state["sentiment_result"] = df["sentiment"].value_counts().to_dict()

    update_progress(state, 35, "Sentiment analysis completed")
    return mark_completed(state, "sentiment")


def topic_agent(state):
    update_progress(state, 40, "Running BERTopic complaint clustering")

    topic_file = state.get("topic_file")

    if not topic_file:
        from analysis.topic_modeling import run_topic_modeling

        result = run_topic_modeling(state.get("sentiment_file", "data/sentiment_feedback.csv"))

        if isinstance(result, dict):
            topic_file = result.get("topic_summary_file")
        else:
            topic_file = result

        state["topic_file"] = topic_file

    if not topic_file:
        state["topic_result"] = "Not enough negative reviews for topic modeling."
        update_progress(state, 60, "Topic modeling skipped")
        return mark_completed(state, "topic")

    df = pd.read_csv(topic_file)
    state["topic_result"] = (
        df[["Business_Label", "Count"]]
        .head(10)
        .to_dict(orient="records")
    )

    update_progress(state, 60, "Topic modeling completed")
    return mark_completed(state, "topic")


def memory_agent(state):
    update_progress(state, 70, "Updating memory from this run")

    try:
        from memory.memory_manager import save_sentiment_history, save_topic_history

        save_sentiment_history()
        save_topic_history()

        sentiment_df = pd.read_csv("memory/sentiment_history.csv")
        topic_df = pd.read_csv("memory/topic_history.csv")

        state["memory_result"] = {
            "sentiment_records": len(sentiment_df),
            "topic_records": len(topic_df),
            "total_runs": len(topic_df["run_date"].unique())
        }

    except Exception as error:
        state["memory_result"] = {"error": str(error)}

    update_progress(state, 78, "Memory updated")
    return mark_completed(state, "memory")


def trend_agent(state):
    update_progress(state, 82, "Detecting complaint trends")

    try:
        from tools.trend_detector import detect_trends

        state["trend_result"] = detect_trends()

    except Exception as error:
        state["trend_result"] = {"error": str(error)}

    update_progress(state, 88, "Trend detection completed")
    return mark_completed(state, "trend")


def report_agent(state):
    update_progress(state, 92, "Generating executive report")

    from analysis.gemini_report import generate_business_report

    report = generate_business_report(
        state.get("sentiment_result", {}),
        state.get("topic_result", {}),
        state.get("memory_result", {}),
        state.get("trend_result", {})
    )

    state["report"] = report

    os.makedirs("data", exist_ok=True)

    with open("data/executive_report.md", "w", encoding="utf-8") as file:
        file.write(report)

    update_progress(state, 96, "Report generated")
    return mark_completed(state, "report")


def critic_agent(state):
    update_progress(state, 98, "Checking report quality")

    from analysis.gemini_critic import evaluate_report

    state["critic"] = evaluate_report(state.get("report", ""))

    update_progress(state, 100, "Workflow completed")
    return mark_completed(state, "critic")
