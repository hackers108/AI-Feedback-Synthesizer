from analysis.gemini_report import (
    generate_business_report
)


def report_agent(state):

    report = generate_business_report(

        state.get(
            "sentiment_result",
            {}
        ),

        state.get(
            "topic_result",
            {}
        ),

        state.get(
            "memory_result",
            {}
        ),

        state.get(
            "trend_result",
            {}
        )
    )

    state[
        "report"
    ] = report

    print(
        "\nReport Agent Completed"
    )

    return state