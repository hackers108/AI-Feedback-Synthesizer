from agent.supervisor_llm import (
    decide_agents
)


def supervisor_agent(state):

    query = state.get(
        "query",
        ""
    )

    result = decide_agents(
        query
    )

    state[
        "required_agents"
    ] = result.get(
        "agents",
        []
    )

    print(
        "\nUser Query:"
    )

    print(query)

    print(
        "\nSupervisor Selected:"
    )

    print(
        state[
            "required_agents"
        ]
    )

    return state