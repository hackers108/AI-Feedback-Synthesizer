def route_from_supervisor(state):

    required = state.get(
        "required_agents",
        []
    )

    if "sentiment" in required:
        return "sentiment"

    if "topic" in required:
        return "topic"

    if "memory" in required:
        return "memory"

    if "trend" in required:
        return "trend"

    return "report"