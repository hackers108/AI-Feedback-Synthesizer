from langgraph.graph import (
    StateGraph,
    START,
    END
)

from agent.agents import (
    FeedbackState,
    critic_agent,
    report_agent,
    route_next_agent,
    sentiment_agent,
    supervisor_agent,
    topic_agent
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
    #Route based on the next agent to run(agent to node mapping)
    "sentiment": "sentiment",
    "topic": "topic",
    "report": "report",
    "critic": "critic",
    "end": END
}

for node_name in [
    #List of all nodes in the graph
    "supervisor",
    "sentiment",
    "topic",
    "report",
    "critic"
]:
    builder.add_conditional_edges(
        #Route based on the next agent to run(agent to node mapping)
        node_name,
        route_next_agent, #dynamically connect using the route_next_agent function
        route_map
    )

graph = builder.compile()

if __name__ == "__main__":
    #For testing

    query = input(
        "\nEnter Query: "
    )

    result = graph.invoke(
        {
            "input_file":
            "uploads/input.csv",

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
