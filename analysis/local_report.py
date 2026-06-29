def generate_local_report(
    sentiment_data,
    topic_data,
    memory_data,
    trend_data
):

    report = []

    report.append(
        "# Customer Experience Report\n"
    )

    # --------------------------
    # Sentiment
    # --------------------------

    if sentiment_data:

        report.append(
            "## Sentiment Analysis\n"
        )

        for sentiment, count in (
            sentiment_data.items()
        ):

            report.append(
                f"- {sentiment}: {count}"
            )

    # --------------------------
    # Topics
    # --------------------------

    if topic_data:

        report.append(
            "\n## Top Customer Complaints\n"
        )

        for topic in topic_data[:5]:

            report.append(
                f"- {topic['Business_Label']} "
                f"({topic['Count']})"
            )

    # --------------------------
    # Memory
    # --------------------------

    if memory_data:

        report.append(
            "\n## Historical Memory\n"
        )

        report.append(
            f"Total Runs: "
            f"{memory_data.get('total_runs', 0)}"
        )

        report.append(
            f"Total Sentiment Records: "
            f"{memory_data.get('total_sentiment_records', 0)}"
        )

    # --------------------------
    # Trend
    # --------------------------

    if trend_data:

        report.append(
            "\n## Trend Analysis\n"
        )

        report.append(
            f"Current Topics: "
            f"{trend_data.get('current_topics', 0)}"
        )

        report.append(
            f"Previous Topics: "
            f"{trend_data.get('previous_topics', 0)}"
        )

        report.append(
            f"Topic Change: "
            f"{trend_data.get('topic_change', 0)}"
        )

    report.append(
        "\n## Recommendation\n"
    )

    report.append(
        "Investigate the highest-volume "
        "customer complaints and monitor "
        "topic growth across future runs."
    )

    return "\n".join(report)