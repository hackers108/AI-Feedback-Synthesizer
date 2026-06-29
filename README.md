# AI Feedback Synthesizer

An AI-powered Customer Feedback Intelligence Platform for automated sentiment analysis, topic discovery, trend detection, and executive reporting.

---

## Overview

AI Feedback Synthesizer is an end-to-end customer feedback analytics platform that transforms unstructured customer reviews into actionable business insights.

The platform combines transformer-based sentiment analysis, unsupervised topic modeling, trend detection, and large language models to help organizations understand customer feedback at scale.

The project is designed for Product Teams, Customer Success Teams, Business Analysts, and Data Scientists who require a scalable solution for analyzing customer feedback from multiple sources.

---

## Key Features

### Sentiment Analysis

* Transformer-based sentiment classification using RoBERTa
* Positive, Neutral, and Negative classification
* Batch inference for large datasets

### Topic Modeling

* BERTopic-based topic discovery
* Automatic complaint clustering
* Representative review extraction
* Business-friendly topic labeling

### Trend Analysis

* Historical topic tracking
* Complaint growth detection
* Emerging issue identification

### Executive Reporting

* Gemini-powered executive summaries
* Local fallback reporting
* Business recommendations
* Key insight generation

### Interactive Dashboard

* Dataset upload
* Review search
* KPI dashboard
* Interactive visualizations
* Topic exploration
* CSV export

---

## System Architecture

```
Customer Feedback
        │
        ▼
Schema Normalization
        │
        ▼
RoBERTa Sentiment Analysis
        │
        ▼
BERTopic Topic Modeling
        │
        ▼
Trend Detection
        │
        ▼
Gemini Executive Summary
        │
        ▼
Interactive Streamlit Dashboard
```

---

## Project Structure

```
AI-Feedback-Synthesizer/

├── agent/
├── analysis/
├── backend/
├── dashboard/
├── ingestion/
├── memory/
├── normalization/
├── tools/
├── utils/

├── data/
├── uploads/

├── requirements.txt
├── README.md
└── LICENSE
```

---

## Dashboard

### Overview

![Overview]<img width="1918" height="911" alt="image" src="https://github.com/user-attachments/assets/97a45e9c-eb3e-484a-b26a-3be2fd63d35b" />


---

### Sentiment Analysis

![Sentiment]<img width="1919" height="909" alt="image" src="https://github.com/user-attachments/assets/de728472-a5b8-4e74-b7e0-086954d6dbbf" />


---

### Topic Modeling

![Topics]<img width="1919" height="918" alt="image" src="https://github.com/user-attachments/assets/9df77ea3-35d1-4823-a119-60574ccbccf2" />


---

### Trend Analysis

![Trends]<img width="1919" height="902" alt="image" src="https://github.com/user-attachments/assets/94afeb68-bd30-471f-9ef9-c8b99cbf514f" />


---

## Technology Stack

### Programming Language

* Python

### Machine Learning

* Transformers
* Hugging Face
* BERTopic
* Sentence Transformers
* Scikit-learn

### Large Language Model

* Google Gemini

### Dashboard

* Streamlit
* Plotly

### Data Processing

* Pandas
* NumPy

---

## Installation

Clone the repository

```bash
git clone https://github.com/hackers108/AI-Feedback-Synthesizer.git

cd AI-Feedback-Synthesizer
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Launch the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

The application will be available at

```
http://localhost:8501
```

---

## Sample Workflow

1. Upload a customer feedback dataset
2. Execute the analysis pipeline
3. Perform sentiment classification
4. Discover complaint topics
5. Analyze emerging trends
6. Generate executive insights
7. Explore results through the dashboard
8. Export reports

---

## Future Improvements

* Multi-file processing
* Authentication and user management
* REST API
* Docker deployment
* Cloud deployment
* PDF report generation
* Scheduled analysis
* Real-time monitoring

---

## Author

**Arpit Yadav**

B.Tech, Electrical Engineering
Madan Mohan Malaviya University of Technology

GitHub
https://github.com/hackers108

LinkedIn
https://linkedin.com/in/arpityadav-cpp

Email
[arpit799yadav@gmail.com](mailto:arpit799yadav@gmail.com)

---

## License

This project is distributed under the MIT License.
