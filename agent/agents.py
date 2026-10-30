import json
import os #for file operations
from typing import NotRequired, TypedDict

import pandas as pd


class FeedbackState(TypedDict):#Typedict has predefined keys and their types, which helps in maintaining a consistent structure for the state dictionary used across different agents.
    input_file: str #it stores a file path.
    batch_size: NotRequired[int | None] #number of uploaded reviews to analyze
    query: NotRequired[str] #user asked question
    required_agents: NotRequired[list[str]]#notrequired means key may or may not be present in the dictionary
    completed_agents: NotRequired[list[str]]
    sentiment_file: NotRequired[str]
    sentiment_result: NotRequired[dict]
    topic_file: NotRequired[str]
    topic_result: NotRequired[object]
    report: NotRequired[str]
    critic: NotRequired[dict]
    query_understanding: NotRequired[dict]
    unsupported_query: NotRequired[bool]
    progress_callback: NotRequired[object]


AGENT_ORDER = [
    "sentiment",
    "topic",
    "report",
    "critic"
]

AGENT_DEPENDENCIES = {
    "topic": ["sentiment"],
    "report": ["sentiment", "topic"],
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
    "report",
    "summary",
    "recommendation",
    "quality",
    "recommend",
    "improve",
    "insight",
    "root cause",
    "analysis",
    "rating",
    "positive",
    "negative",
    "neutral",
    "satisfaction",
    "dissatisfaction",
    "pain point",
    "bug",
    "feature",
    "request",
    "priority",
    "action"
]

HISTORICAL_QUERY_KEYWORDS = [
    "historical",
    "previous upload",
    "previous run",
    "over time",
    "trend",
    "compare with previous"
]

UNSUPPORTED_MESSAGE = (
    "I can only help with customer feedback analysis. "
    "Ask for sentiment, complaint topics, or a report based on the uploaded file."
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
    #extracting keyword from query and checking if it is present in the list of supported keywords
    query = query.lower()

    if any(
        keyword in query
        for keyword in HISTORICAL_QUERY_KEYWORDS
    ):

        return False

    return any(keyword in query for keyword in SUPPORTED_KEYWORDS)


# ==================================================
# Local Query Understanding
# ==================================================
# This is the reliable fallback when Gemini is not
# available. It can select more than one agent.

def choose_agents(query):
    #choosing agents based on the keywords
    query = query.lower()

    if any(word in query for word in ["quality", "critic", "critique", "missing"]):
        return ["critic"]

    if any(word in query for word in [
        "report",
        "executive",
        "summary",
        "recommendation",
        "recommend",
        "priority",
        "action",
        "improve",
        "root cause",
        "insight"
    ]):
        return ["report"]

    selected_agents = []

    if any(word in query for word in [
        "sentiment",
        "rating",
        "positive",
        "negative",
        "neutral",
        "satisfaction",
        "dissatisfaction"
    ]):
        selected_agents.append("sentiment")

    if any(word in query for word in [
        "complaint",
        "topic",
        "theme",
        "issue",
        "pain point",
        "bug",
        "feature request"
    ]):
        selected_agents.append("topic")

    if selected_agents:
        return selected_agents

    return ["report"]


# ==================================================
# Gemini Query Understanding
# ==================================================
# Gemini understands natural-language questions that
# may not contain the exact local routing keywords.
# It only returns agent names supported by this graph.

def understand_query_with_llm(query):

    prompt = f"""
You are the supervisor for a customer-feedback analysis application.

Decide whether the user's question can be answered using only the currently
uploaded feedback CSV. Supported requests include sentiment, ratings,
complaint themes, customer pain points, product issues, feature requests,
priorities, recommendations, summaries, business reports, and internal report
quality checks.

Historical trends, comparisons with previous uploads, unrelated general
knowledge, and requests requiring unavailable external data are unsupported.

Choose one or more final agents from this exact list:
- sentiment: sentiment counts or positive/neutral/negative questions
- topic: complaint themes, issues, pain points, bugs, or feature requests
- report: explanations, summaries, priorities, recommendations, root causes,
  business questions, or questions combining multiple analyses
- critic: internally review report quality; its review is not shown to users

Dependencies are added automatically. Return ONLY valid JSON in this format:
{{
  "supported": true,
  "agents": ["report"],
  "intent": "short description of the user's request"
}}

User question:
{query}
"""

    try:

        from analysis.gemini_report import get_gemini_client

        response = get_gemini_client().models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        response_text = response.text.strip()

        if "```" in response_text:

            response_text = (
                response_text
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        understanding = json.loads(
            response_text
        )

        allowed_agents = set(
            AGENT_ORDER
        )

        selected_agents = [
            agent_name
            for agent_name in understanding.get(
                "agents",
                []
            )
            if agent_name in allowed_agents
        ]

        supported = (
            understanding.get("supported")
            is True
        )

        if supported and not selected_agents:

            selected_agents = ["report"]

        return {
            "supported": supported,
            "agents": selected_agents,
            "intent": str(
                understanding.get(
                    "intent",
                    "Customer feedback analysis"
                )
            )
        }

    except Exception as error:

        print(
            f"\nGemini Query Understanding Error: {error}"
        )

        print(
            "Using local query routing instead."
        )

        return None


def expand_dependencies(selected_agents):
    #it ensures all the required agent dependencies for selected agent are also included
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
    #Determine the next agent to run based on the current state, required agents, and completed agents
    required = state.get("required_agents", [])
    completed = state.get("completed_agents", [])

    for agent_name in AGENT_ORDER:
        if agent_name in required and agent_name not in completed:
            return agent_name

    return "end"


def supervisor_agent(state):
    #supervising the workflow, selecting agents based on the query, and managing dependencies
    query = state.get("query") or "Generate a complete business report."
    update_progress(state, 5, "Understanding your feedback question")

    # Hybrid routing first asks Gemini to understand
    # the question, then uses keywords as a fallback.
    understanding = understand_query_with_llm(
        query
    )

    if understanding is None:

        supported = is_supported_query(
            query
        )

        selected_agents = choose_agents(
            query
        )

        understanding = {
            "supported": supported,
            "agents": selected_agents,
            "intent": "Detected with local keyword routing",
            "method": "local_fallback"
        }

    else:

        supported = understanding["supported"]
        selected_agents = understanding["agents"]
        understanding["method"] = "gemini"

    state["query_understanding"] = understanding

    if not supported:
        state["required_agents"] = []
        state["completed_agents"] = []
        state["unsupported_query"] = True
        state["report"] = UNSUPPORTED_MESSAGE
        update_progress(state, 100, "Unsupported question")
        return state

    state["required_agents"] = expand_dependencies(selected_agents)
    state["completed_agents"] = []
    state["unsupported_query"] = False

    update_progress(state, 10, f"Selected agents: {', '.join(state['required_agents'])}")
    return state


def sentiment_agent(state):
    #Perform sentiment analysis on the input file and store the results in the state
    update_progress(state, 15, "Running RoBERTa sentiment analysis")

    sentiment_file = state.get("sentiment_file")

    if not sentiment_file:
        from analysis.sentiment import run_sentiment

        def update_sentiment_progress(progress, message):

            mapped_progress = 15 + int(
                max(0, min(progress, 1)) * 20
            )

            update_progress(
                state,
                mapped_progress,
                message
            )

        sentiment_file = run_sentiment(
            state["input_file"],
            sample_size=state.get(
                "batch_size",
                500
            ),
            progress_callback=update_sentiment_progress
        )
        state["sentiment_file"] = sentiment_file

    df = pd.read_csv(sentiment_file)
    state["sentiment_result"] = df["sentiment"].value_counts().to_dict()

    update_progress(state, 35, "Sentiment analysis completed")
    return mark_completed(state, "sentiment")


def topic_agent(state):
    #Perform topic modeling on the negative reviews and store the results in the state
    update_progress(state, 40, "Running BERTopic complaint clustering")

    topic_file = state.get("topic_file")

    if not topic_file:
        from analysis.topic_modeling import run_topic_modeling

        def update_topic_progress(progress, message):

            # Topic modeling reports progress from
            # 0.40 to 1.00. Map it to 40%-88% of
            # the complete multi-agent workflow.
            normalized_progress = (
                max(0.40, min(progress, 1.0))
                - 0.40
            ) / 0.60

            mapped_progress = 40 + int(
                normalized_progress * 48
            )

            update_progress(
                state,
                mapped_progress,
                message
            )

        result = run_topic_modeling(
            state.get(
                "sentiment_file",
                "data/sentiment_feedback.csv"
            ),
            progress_callback=update_topic_progress
        )

        if isinstance(result, dict):
            topic_file = result.get("topic_summary_file")
        else:
            topic_file = result

        state["topic_file"] = topic_file

    if not topic_file:
        state["topic_result"] = "Not enough negative reviews for topic modeling."
        update_progress(state, 88, "Topic modeling skipped")
        return mark_completed(state, "topic")

    df = pd.read_csv(topic_file)

    # Store readable labels, explanations, and counts.
    # Gemini uses this compact result for the report.
    topic_columns = ["Business_Label", "Count"]

    if "Topic_Description" in df.columns:
        topic_columns.insert(1, "Topic_Description")

    state["topic_result"] = (
        df[topic_columns]
        .head(10)
        .to_dict(orient="records")
    )

    update_progress(state, 88, "Topic modeling completed")
    return mark_completed(state, "topic")


def report_agent(state):
    #Generate an executive report based on the results of previous agents and store it in the state
    update_progress(state, 92, "Generating executive report")

    from analysis.gemini_report import generate_business_report

    report = generate_business_report(
        state.get("sentiment_result", {}),
        state.get("topic_result", {}),
        state.get("query", "Generate a customer feedback report.")
    )

    state["report"] = report

    os.makedirs("data", exist_ok=True)

    with open("data/executive_report.md", "w", encoding="utf-8") as file:
        file.write(report)

    update_progress(state, 96, "Report generated")
    return mark_completed(state, "report")


def critic_agent(state):
    # Evaluate report quality internally. The frontend
    # intentionally does not display this critic result.
    update_progress(state, 98, "Checking report quality")

    from analysis.gemini_critic import evaluate_report

    state["critic"] = evaluate_report(
        state.get("report", "")
    )

    update_progress(state, 100, "Workflow completed")
    return mark_completed(state, "critic")
