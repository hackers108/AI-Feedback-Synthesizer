from analysis.gemini_critic import (
    evaluate_report
)


def critic_agent(state):

    report = state.get(
        "report",
        ""
    )

    result = evaluate_report(
        report
    )

    state["critic"] = result

    print(
        "\nCritic Agent Completed"
    )

    return state