import streamlit as st

from dashboard.components.charts import (
    trend_chart
)


def render_trends(trends):

    st.header("📈 Trend Analysis")

    if not trends:

        st.warning(
            "No trend data available."
        )

        return

    trend_chart(
        trends
    )

    st.divider()

    st.subheader(
        "Emerging Trends"
    )

    for trend in trends:

        with st.container():

            st.markdown(
                f"""
### {trend.get('topic','Unknown')}

**Growth:** {trend.get('growth_percent',0)}%

**Current Reviews:** {trend.get('current_count',0)}

**Previous Reviews:** {trend.get('previous_count',0)}
"""
            )

            st.progress(
                min(
                    trend.get(
                        "growth_percent",
                        0
                    ) / 100,
                    1.0
                )
            )

    st.divider()

    st.subheader(
        "Business Recommendations"
    )

    recommendations = [

        "Investigate the fastest-growing complaint category.",

        "Prioritize high-volume issues for engineering review.",

        "Track topic growth weekly to detect emerging problems.",

        "Review representative customer feedback for top topics.",

        "Monitor sentiment trends after product releases."

    ]

    for rec in recommendations:

        st.success(rec)