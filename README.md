---
title: AI Feedback Synthesizer
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.36.0
app_file: app.py
pinned: false
---

# AI Feedback Synthesizer

An agentic customer feedback analysis app built with Python, Streamlit, LangGraph, RoBERTa, BERTopic, Gemini, and Pandas.

## What It Does

The app lets a user upload a customer feedback CSV and analyze only the currently selected file, such as:

- What is the overall customer sentiment?
- What are the top complaint themes?
- Generate an executive report with recommendations.

## Simple Architecture

```text
frontend/streamlit_app.py
        |
        v
agent/graph.py
        |
        v
agent/agents.py
        |
        +--> analysis/sentiment.py       RoBERTa sentiment analysis
        +--> analysis/topic_modeling.py  BERTopic complaint clustering
        +--> analysis/gemini_report.py   Gemini business report
        +--> analysis/gemini_critic.py   Report quality check
```

## Main Flow

```text
CSV Upload
 -> Schema normalization
 -> Sentiment Agent
 -> Topic Agent
 -> Report Agent
 -> Critic Agent
 -> Streamlit Results
```

The supervisor does not always run every agent. It reads the user question, chooses the minimum required agents, expands dependencies, and LangGraph routes through the needed steps.

## Important Files

- `frontend/streamlit_app.py`: Streamlit UI, file upload, suggested prompts, progress bar, result display.
- `agent/graph.py`: LangGraph wiring.
- `agent/agents.py`: Supervisor, routing, and all agent node functions.
- `analysis/sentiment.py`: RoBERTa sentiment classification.
- `analysis/topic_modeling.py`: BERTopic clustering on negative reviews.
- `analysis/gemini_report.py`: Generates business report.
- `analysis/gemini_critic.py`: Checks report quality.

## Run

```powershell
cd C:\Users\arpit\feedback-synthesizer
venv\Scripts\activate
streamlit run app.py
```

## Environment

Create a `.env` file with:

```text
GEMINI_API_KEY=your_api_key_here
```

If Gemini fails, report generation uses the local fallback in `analysis/local_report.py`.

## Interview Pitch

"I built an agentic AI feedback analysis platform. Users upload a CSV of customer reviews and ask natural-language questions. A LangGraph supervisor decides which agents are needed. The Sentiment Agent uses RoBERTa, the Topic Agent uses BERTopic, Gemini generates an executive report from the current upload, and the Critic Agent checks report quality. The system is modular, so each agent has one responsibility."
