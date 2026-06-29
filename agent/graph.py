from langgraph.graph import (
    StateGraph,
    START,
    END
)

from agent.router import (
    route_from_supervisor
)

from agent.state import (
    FeedbackState
)

from agent.supervisor import (
    supervisor_agent
)

from agent.sentiment_agent import (
    sentiment_agent
)

from agent.topic_agent import (
    topic_agent
)

from agent.memory_agent import (
    memory_agent
)

from agent.trend_agent import (
    trend_agent
)

from agent.report_agent import (
    report_agent
)

from agent.critic_agent import (
    critic_agent
)

builder = StateGraph(
    FeedbackState
)

# -------------------------
# Nodes
# -------------------------

builder.add_node(
    "supervisor",
    supervisor_agent
)

builder.add_node(
    "sentiment",
    sentiment_agent
)

builder.add_node(
    "topic",
    topic_agent
)

builder.add_node(
    "memory",
    memory_agent
)

builder.add_node(
    "trend",
    trend_agent
)

builder.add_node(
    "report",
    report_agent
)

builder.add_node(
    "critic",
    critic_agent
)

# -------------------------
# Start
# -------------------------

builder.add_edge(
    START,
    "supervisor"
)

# -------------------------
# Supervisor Routing
# -------------------------

builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "sentiment": "sentiment",
        "topic": "topic",
        "memory": "memory",
        "trend": "trend",
        "report": "report"
    }
)

# -------------------------
# Fixed Flow
# -------------------------

builder.add_edge(
    "sentiment",
    "topic"
)

builder.add_edge(
    "topic",
    "memory"
)

builder.add_edge(
    "memory",
    "trend"
)

builder.add_edge(
    "trend",
    "report"
)

builder.add_edge(
    "report",
    "critic"
)

builder.add_edge(
    "critic",
    END
)

graph = builder.compile()

if __name__ == "__main__":

    query = input(
        "\nEnter Query: "
    )

    result = graph.invoke(
        {
            "input_file":
            "data/instagram.csv",

            "query":
            query
        }
    )

    print(
        "\n================="
    )

    print(
        "BUSINESS REPORT"
    )

    print(
        "=================\n"
    )

    print(
        result.get(
            "report",
            "No report generated"
        )
    )

    print(
        "\n================="
    )

    print(
        "CRITIC RESULT"
    )

    print(
        "=================\n"
    )

    print(
        result.get(
            "critic",
            {}
        )
    )