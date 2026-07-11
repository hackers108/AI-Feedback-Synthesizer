from langgraph.graph import (
    StateGraph,
    START,
    END
)

from agent.agents import (
    FeedbackState,
    critic_agent,
    memory_agent,
    report_agent,
    route_next_agent,
    sentiment_agent,
    supervisor_agent,
    topic_agent,
    trend_agent
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

route_map = {
    "sentiment": "sentiment",
    "topic": "topic",
    "memory": "memory",
    "trend": "trend",
    "report": "report",
    "critic": "critic",
    "end": END
}

for node_name in [
    "supervisor",
    "sentiment",
    "topic",
    "memory",
    "trend",
    "report",
    "critic"
]:
    builder.add_conditional_edges(
        node_name,
        route_next_agent,
        route_map
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
