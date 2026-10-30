def generate_local_report(
    sentiment_data,
    topic_data
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

    report.append(
        "\n## Recommendation\n"
    )

    report.append(
        "Investigate the highest-volume "
        "customer complaints in the uploaded data."
    )

    return "\n".join(report)
