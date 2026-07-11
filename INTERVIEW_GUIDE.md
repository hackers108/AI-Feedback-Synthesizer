# Interview Guide

## 1. One-Minute Explanation

This project analyzes customer feedback at scale. A user uploads a CSV of reviews, then asks questions like "What are customers complaining about?" or "Generate a business report." The LangGraph supervisor routes the task to the correct agents. The system performs sentiment analysis with RoBERTa, complaint clustering with BERTopic, memory tracking with CSV history, trend detection, and Gemini-based executive reporting.

## 2. Why It Is Agentic

It is agentic because the workflow is not just one fixed script. The user gives a natural-language task, the supervisor selects the required agents, LangGraph routes execution, and each agent updates shared state.

Agents:

- Supervisor Agent: understands the user query and selects required agents.
- Sentiment Agent: classifies reviews as positive, neutral, or negative.
- Topic Agent: clusters negative reviews into complaint themes.
- Memory Agent: stores and reads previous analysis runs.
- Trend Agent: compares current topics with previous runs.
- Report Agent: generates a business report.
- Critic Agent: checks whether the report has required sections.

## 3. File Architecture To Explain

```text
frontend/streamlit_app.py  User interface
agent/graph.py            LangGraph connections
agent/agents.py           All agent logic
analysis/                 ML and LLM analysis code
memory/                   Historical run data
tools/                    Trend detection helper
normalization/            Schema normalization
utils/                    CSV loading helper
data/                     Sample data and outputs
```

## 4. Main Technical Decisions

RoBERTa:
I used a transformer sentiment model because customer reviews are informal and context matters. It is better than simple keyword matching.

BERTopic:
I used BERTopic because it works well with short text. It uses embeddings, UMAP, HDBSCAN, and topic representation to find complaint clusters.

Negative-only topic modeling:
I cluster negative reviews because the business value is finding what customers complain about.

LangGraph:
I used LangGraph because it makes the workflow explicit as nodes and edges. This is easier to extend than one long script.

Memory:
I store results with timestamps so the system can compare current and previous runs.

Gemini:
Structured analytics are generated first, then Gemini converts them into a business-readable report.

## 5. Common Interview Questions

Q: Why did you normalize schema?
A: Different datasets may call the review column `review`, `comment`, `feedback`, or `content`. I standardize it to `text`, so the ML pipeline always receives the same input format.

Q: Why RoBERTa?
A: RoBERTa is a transformer model that understands context. It is suitable for short, informal review text.

Q: Why BERTopic?
A: BERTopic uses semantic embeddings and density-based clustering, so it can discover natural complaint themes without manually choosing the number of topics.

Q: What does the supervisor do?
A: It reads the user query, chooses the required agents, expands dependencies, and lets LangGraph route execution.

Q: What would you improve next?
A: I would add a database, async background jobs, authentication, richer vector memory, and production deployment.

## 6. Demo Prompts

Use these in the UI:

```text
What is the overall customer sentiment?
Find the top customer complaint themes.
Detect rising complaint trends.
Compare this run with previous feedback runs.
Generate a complete business report and review its quality.
```
