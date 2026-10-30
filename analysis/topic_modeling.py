# ==================================================
# BERTopic Pipeline
# ==================================================
#
# Documents (Customer Reviews)
#        │
#        ▼
# SentenceTransformer
# (Convert each review into a semantic embedding)
#        │
#        ▼
# UMAP
# (Reduce embedding dimensions from 384 → 5)
#        │
#        ▼
# HDBSCAN
# (Cluster semantically similar reviews)
#        │
#        ▼
# CountVectorizer
# (Create a Bag-of-Words representation for each cluster)
#        │
#        ▼
# c-TF-IDF
# (Identify the most representative words in each cluster)
#        │
#        ▼
# KeyBERTInspired
# (Refine and select the best topic keywords)
#        │
#        ▼
# Representative Reviews
# (Provide real customer examples for each topic)
#        │
#        ▼
# Gemini Labeling + Local Fallback
# (Generate a clear topic name and one-line explanation)
# ==================================================

import json
import re

import pandas as pd

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

from sentence_transformers import (
    SentenceTransformer
)

from sklearn.feature_extraction.text import (
    CountVectorizer,
    ENGLISH_STOP_WORDS
)

from umap import UMAP

from hdbscan import HDBSCAN

from utils.file_loader import load_feedback


# ==================================================
# Generic Topic Words
# ==================================================
# These words appear often but do not explain the
# actual customer complaint, so they are removed
# from the topic keyword representation.

GENERIC_TOPIC_WORDS = {
    "app",
    "application",
    "instagram",
    "issue",
    "issues",
    "problem",
    "problems",
    "thing",
    "things",
    "use",
    "using",
    "work",
    "working"
}


# ==================================================
# Local Topic Label Fallback
# ==================================================
# This function keeps topic labels readable when
# Gemini is unavailable or returns invalid output.

def create_fallback_topic_metadata(
    keywords,
    reviews
):

    # Combine reviews so complaint patterns can be
    # checked across the complete topic.
    combined_text = " ".join(reviews).lower()

    # Each regex pattern maps common customer words
    # to one business-friendly complaint label.
    issue_patterns = [
        (r"crash|freez|blank screen|stopped responding|won.t open|not open", "App Crashing or Not Opening"),
        (r"hack|security|recover.*account|account.*recover", "Account Security and Recovery"),
        (r"log.?in|sign.?in|account.*access", "Login and Account Access"),
        (r"reels?.*audio|audio.*reels?", "Reels Audio Problems"),
        (r"message|\bdm\b|chat", "Messaging and Delivery Problems"),
        (r"upload|posting|can.t post|cannot post", "Upload and Posting Failures"),
        (r"refresh|recent posts|\bfeed\b", "Feed Refresh Problems"),
        (r"load|loading", "Content Loading Problems"),
        (r"notification", "Notification Problems"),
        (r"update|version", "Update and Version Problems"),
        (r"lag|slow|performance", "Slow App Performance")
    ]

    label = None
    best_score = 0

    # Score every complaint pattern. A pattern gets
    # more weight when it appears in multiple reviews.
    for pattern, candidate in issue_patterns:
        matching_reviews = sum(
            1
            for review in reviews
            if re.search(pattern, review.lower())
        )
        total_matches = len(re.findall(pattern, combined_text))
        score = (matching_reviews * 10) + total_matches

        if score > best_score:
            label = candidate
            best_score = score

    # If no known pattern matches, clean the BERTopic
    # keywords and use the two most useful phrases.
    if not label:
        useful_keywords = []

        for keyword in keywords:
            cleaned_words = [
                word
                for word in re.findall(r"[a-zA-Z]+", keyword.lower())
                if word not in GENERIC_TOPIC_WORDS
            ]
            cleaned = " ".join(cleaned_words).strip()

            if cleaned and cleaned not in useful_keywords:
                useful_keywords.append(cleaned)

        label = " and ".join(useful_keywords[:2]).title() or "Customer Experience Issue"

    return {
        "label": label,
        "description": f"Customers report problems related to {label.lower()}."
    }


# ==================================================
# Gemini Topic Labeling
# ==================================================
# Gemini receives topic keywords and representative
# reviews. It returns a short label and description.
# Local labels are prepared first as a safe fallback.

def generate_topic_metadata(
    topic_model,
    representative_docs
):

    metadata = {}
    prompt_topics = []

    # Prepare both the local fallback and the compact
    # topic data that will be sent to Gemini.
    for topic_id, reviews in representative_docs.items():
        topic_words = topic_model.get_topic(topic_id) or []
        keywords = [word for word, _ in topic_words[:8]]
        fallback = create_fallback_topic_metadata(keywords, reviews)
        metadata[int(topic_id)] = fallback
        prompt_topics.append({
            "topic": int(topic_id),
            "keywords": keywords,
            "representative_reviews": [review[:500] for review in reviews[:3]]
        })

    if not prompt_topics:
        return metadata

    # Gemini is asked for strict JSON so the labels
    # can be saved directly into the topic CSV.
    prompt = f"""
You label customer-feedback complaint clusters.

For every topic below, use its keywords and representative reviews to create:
- label: a specific, business-friendly 3-6 word issue name
- description: one plain sentence explaining what customers report

Avoid raw keyword lists, underscores, topic numbers, vague labels such as
"App Issue", and claims about historical trends. Do not merge topics. Make
labels distinct when the clusters describe different aspects of a problem.
Return ONLY a valid JSON array with objects containing topic, label, and description.

Topics:
{json.dumps(prompt_topics, ensure_ascii=False)}
"""

    try:

        from analysis.gemini_report import get_gemini_client

        # One Gemini request labels all topics. This
        # avoids making a separate API call per topic.
        response = get_gemini_client().models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        response_text = response.text.strip()

        if "```" in response_text:
            response_text = response_text.replace("```json", "").replace("```", "").strip()

        generated_items = json.loads(response_text)

        # Replace local fallback values only when both
        # Gemini fields are present and valid.
        for item in generated_items:
            topic_id = int(item.get("topic"))
            label = str(item.get("label", "")).strip()
            description = str(item.get("description", "")).strip()

            if topic_id in metadata and label and description:
                metadata[topic_id] = {
                    "label": label,
                    "description": description
                }

    except Exception as error:

        # The pipeline continues with local labels if
        # Gemini, networking, or JSON parsing fails.
        print(f"\nGemini Topic Labeling Error: {error}")
        print("Using local topic labels instead.")

    return metadata



# ==================================================
# Topic Modeling
# ==================================================

import time

start = time.time()

def run_topic_modeling(
    input_file="data/sentiment_feedback.csv",
    progress_callback=None
):

    def update(progress, message):
        if progress_callback:
            progress_callback(progress, message)

    print(
        "\nLoading Feedback Dataset..."
    )

    update(0.40, "📂 Loading feedback dataset...")

    df = load_feedback(
        input_file
    )

    required_columns = [
        "text",
        "sentiment"
    ]

    missing_columns = [

        column

        for column in required_columns

        if column not in df.columns

    ]

    if missing_columns:

        raise ValueError(

            f"\nMissing Required Columns: "

            f"{missing_columns}"

        )

    print(
        "\nDataset Loaded Successfully"
    )

    update(0.45, "Dataset loaded")

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"\nExecution Time : "
        f"{time.time()-start:.2f} sec"
    )

    # =============================================
    # Negative Reviews
    # =============================================

    negative_df = df[
        df["sentiment"] == "negative"
    ].copy()

    negative_df = negative_df[

        negative_df["text"]

        .astype(str)

        .str.len()

        > 20

    ].copy()

    print(
        f"\nNegative Reviews: "
        f"{len(negative_df)}"
    )

    if len(negative_df) < 20:

        print(
            "\nNot enough negative reviews "
            "for topic modeling."
        )

        return None

    # =============================================
    # Documents
    # =============================================

    documents = (

        negative_df["text"]

        .fillna("")

        .astype(str)

        .tolist()

    )

    # =============================================
    # Embedding Model
    # =============================================

    print(
        "\nLoading Embedding Model..."
    )

    update(0.50, "Loading embedding model...")

    embedding_model = (

        SentenceTransformer(

            "all-MiniLM-L6-v2"

        )

    )

    update(0.55, " Embedding model loaded")


    print(
        "Embedding Model Loaded."
    )

    # =============================================
    # Vectorizer
    # =============================================

    vectorizer_model = CountVectorizer(
        #use bag-of-words model to convert text into numerical features with TF-IDF weighting, which helps to identify important words in the text data

        # Remove standard stop words and generic app
        # words to produce more meaningful keywords.
        stop_words=sorted(
            set(ENGLISH_STOP_WORDS).union(GENERIC_TOPIC_WORDS)
        ),

        ngram_range=(1, 2),
        #unigram and bigram

        min_df=1,
        #minimum document frequency

        max_df=0.95
        #maximum document frequency
    )

    # =============================================
    # UMAP
    # =============================================

    umap_model = UMAP(
        #Initialize UMAP for dimensionality reduction from 384 to 5 dimensions, preserving local structure and using cosine distance metric for clustering

        # A smaller neighborhood keeps complaint
        # clusters more focused on similar reviews.
        n_neighbors=10,

        n_components=5,

        min_dist=0.0,

        metric="cosine",
        #cosine similarity for measuring distance between vectors(semantically similar documents will be closer in the reduced space)

        random_state=42

    )

    # =============================================
    # HDBSCAN
    # =============================================

    hdbscan_model = HDBSCAN(
        #HDBSCAN is a clustering algorithm that identifies clusters of varying densitiese and can find clustrs

        min_cluster_size=15,

        min_samples=5,

        metric="euclidean",
        #HDBSCAN works in the low-dimensional coordinate space, so Euclidean distance is appropriate

        # Leaf selection creates smaller and more
        # specific complaint clusters.
        cluster_selection_method="leaf",

        prediction_data=True

    )

    # =============================================
    # Topic Representation
    # =============================================

    representation_model = (
        #appropriate topic name from cluster

        KeyBERTInspired()

    )

    # =============================================
    # BERTopic
    # =============================================

    update(0.60, "⚙ Initializing BERTopic...")

    topic_model = BERTopic(

        embedding_model=embedding_model,

        vectorizer_model=vectorizer_model,

        umap_model=umap_model,

        hdbscan_model=hdbscan_model,

        representation_model=representation_model,

        calculate_probabilities=False,

        verbose=True
        #show training progress in the console

    )

    print(
        "\nTraining BERTopic..."
    )

    update(0.65, "Running BERTopic clustering...")

    topics, probabilities = (
        #It performs all the major steps of the BERTopic pipeline internally.

        topic_model.fit_transform(

            documents

        )

    )
    update(0.80, " Extracting representative reviews...")

    negative_df["topic"] = topics
    #This adds a new column to the negative_df DataFrame, storing the topic assigned to each review

    # =============================================
    # Representative Reviews
    # =============================================

    print(
        "\nExtracting Representative Reviews..."
    )

    representative_docs = {}

    for topic_id in topic_model.get_topics():

        if topic_id == -1:
            continue

        try:

            docs = (
                topic_model
                .get_representative_docs(
                    topic_id
                )
            )

            representative_docs[
                topic_id
            ] = docs[:5]

        except Exception:

            representative_docs[
                topic_id
            ] = []

    print(
        "Representative Reviews Extracted."
    )

    # =============================================
    # Topic Information
    # =============================================

    topic_info = (
        topic_model
        .get_topic_info()
    )

    # =============================================
    # Business Friendly Topic Labels
    # =============================================

    def create_business_label(
        topic_id
    ):

        if topic_id == -1:

            return "Noise"

        words = (
            topic_model.get_topic(
                topic_id
            )
        )

        if words is None:

            return "Unknown Topic"

        top_words = [

            word

            for word, score

            in words[:4]

        ]

        return " | ".join(

            word.title()

            for word in top_words

        )

    topic_info[
        "Business_Label"
    ] = (

        topic_info["Topic"]

        .apply(

            create_business_label

        )

    )

    # =============================================
    # Representative Review Column
    # =============================================

    topic_info[
        "Representative_Reviews"
    ] = (

        topic_info["Topic"]

        .map(

            lambda topic_id:

            "\n\n".join(

                representative_docs.get(

                    topic_id,

                    []

                )

            )

        )

    )

    # =============================================
    # Readable Topic Labels and Descriptions
    # =============================================
    # Hybrid approach:
    # 1. Build safe local labels.
    # 2. Improve them with Gemini when available.
    # 3. Store the final label and description.

    update(0.88, "Generating readable topic labels...")

    topic_metadata = generate_topic_metadata(
        topic_model,
        representative_docs
    )

    topic_info["Business_Label"] = topic_info.apply(
        lambda row: topic_metadata.get(
            int(row["Topic"]),
            {"label": row["Business_Label"]}
        )["label"],
        axis=1
    )

    topic_info["Topic_Description"] = topic_info.apply(
        lambda row: topic_metadata.get(
            int(row["Topic"]),
            {
                "description": (
                    f"Customers report problems related to "
                    f"{str(row['Business_Label']).lower()}."
                )
            }
        )["description"],
        axis=1
    )

    # =============================================
    # Remove Noise Cluster
    # =============================================

    topic_info = (

        topic_info[

            topic_info["Topic"] != -1

        ]

        .copy()

    )

    # =============================================
    # Sort Topics
    # =============================================

    topic_info = (

        topic_info

        .sort_values(

            by="Count",

            ascending=False

        )

        .reset_index(

            drop=True

        )

    )

    print(
        "\nTopics Generated:"
    )

    print(
        len(topic_info)
    )

    # =============================================
    # Preview Top Topic
    # =============================================

    if len(topic_info) > 0:

        top_topic = (
            topic_info.iloc[0]
        )

        print(
            "\nTop Complaint Topic"
        )

        print(
            "-" * 50
        )

        print(
            top_topic[
                "Business_Label"
            ]
        )

        print(
            "\nRepresentative Review:\n"
        )

        print(

            top_topic[
                "Representative_Reviews"
            ][:1000]

        )

        print(
            "\n" + "-" * 50
        )
    # =============================================
    # Topic Summary
    # =============================================

    update(
        0.90,
        "📊 Generating topic summary..."
    )

    print(
        "\nGenerating Topic Summary..."
    )

    topic_summary = topic_info[
        [
            "Topic",
            "Count",
            "Name",
            "Business_Label",
            "Topic_Description",
            "Representative_Reviews"
        ]
    ].copy()

    # =============================================
    # Merge Duplicate Readable Topics
    # =============================================
    # Different BERTopic clusters may receive the
    # same readable label. Merge those rows and add
    # their review counts to avoid duplicate results.

    topic_summary = (

        topic_summary

        .groupby(

            "Business_Label",

            as_index=False,

            sort=False

        )

        .agg(

            {
                "Topic": "first",
                "Count": "sum",
                "Name": "first",
                "Topic_Description": "first",
                "Representative_Reviews": (
                    lambda reviews:
                    "\n\n".join(
                        review
                        for review in reviews
                        if isinstance(review, str)
                    )
                )
            }

        )

        .sort_values(

            by="Count",

            ascending=False

        )

        .reset_index(

            drop=True

        )

    )

    topic_summary = topic_summary[
        [
            "Topic",
            "Count",
            "Name",
            "Business_Label",
            "Topic_Description",
            "Representative_Reviews"
        ]
    ]

    print(
        "\n=============================="
    )

    print(
        "TOPIC SUMMARY"
    )

    print(
        "=============================="
    )

    print(
        topic_summary[
            [
                "Topic",
                "Count",
                "Business_Label"
            ]
        ]
    )

    print(
        f"\nTotal Topics Found: "
        f"{len(topic_summary)}"
    )

    # =============================================
    # Statistics
    # =============================================

    total_negative_reviews = len(
        negative_df
    )

    total_topics = len(
        topic_summary
    )

    average_reviews_per_topic = round(

        total_negative_reviews /

        max(total_topics, 1),

        2

    )

    print(
        "\nDataset Statistics"
    )

    print(
        "-" * 35
    )

    print(
        f"Negative Reviews : "
        f"{total_negative_reviews}"
    )

    print(
        f"Topics Generated : "
        f"{total_topics}"
    )

    print(
        f"Average Reviews/Topic : "
        f"{average_reviews_per_topic}"
    )

    # =============================================
    # Save Topic Summary
    # =============================================

    update(
        0.95,
        "Saving analysis results..."
    )

    topic_summary_file = (
        "data/topic_summary.csv"
    )

    topic_summary.to_csv(

        topic_summary_file,

        index=False

    )

    print(
        "\nSaved:"
    )

    print(
        topic_summary_file
    )

    # =============================================
    # Save Topic Feedback
    # =============================================

    topic_feedback_file = (
        "data/topic_feedback.csv"
    )

    negative_df.to_csv(

        topic_feedback_file,

        index=False

    )

    print(
        topic_feedback_file
    )

    # =============================================
    # Save Representative Reviews
    # =============================================

    representative_file = (
        "data/representative_reviews.csv"
    )

    representative_data = []

    for topic_id, docs in representative_docs.items():

        for review in docs:

            representative_data.append(

                {

                    "Topic": topic_id,

                    "Review": review

                }

            )

    representative_df = pd.DataFrame(

        representative_data

    )

    representative_df.to_csv(

        representative_file,

        index=False

    )

    print(
        representative_file
    )

    # =============================================
    # Summary Preview
    # =============================================

    print(
        "\nTop 5 Topics"
    )

    print(
        "-" * 50
    )

    for _, row in topic_summary.head().iterrows():

        print(
            f"{row['Business_Label']} "
            f"({row['Count']})"
        )

    print(
        "-" * 50
    )
    # =============================================
    # Completion Message
    # =============================================

    print(
        "\nTopic Modeling Completed Successfully!"
    )

    print(
        "=" * 60
    )

    # =============================================
    # Return Results
    # =============================================

    update(
        1.0,
        "Topic Modeling Completed"
    )

    return {

        "topic_feedback_file": topic_feedback_file,

        "topic_summary_file": topic_summary_file,

        "representative_reviews_file": representative_file,

        "topic_df": negative_df,

        "summary_df": topic_summary,

        "total_topics": total_topics,

        "negative_reviews": total_negative_reviews

    }


# ==================================================
# Main
# ==================================================

if __name__ == "__main__":

    try:

        result = run_topic_modeling()

        if result is None:

            print(
                "\nTopic Modeling Skipped."
            )

        else:

            print(
                "\nPipeline Output\n"
            )

            print(
                f"Topic Summary File : "
                f"{result['topic_summary_file']}"
            )

            print(
                f"Topic Feedback File : "
                f"{result['topic_feedback_file']}"
            )

            print(
                f"Representative Reviews : "
                f"{result['representative_reviews_file']}"
            )

            print(
                f"Topics Generated : "
                f"{result['total_topics']}"
            )

            print(
                f"Negative Reviews : "
                f"{result['negative_reviews']}"
            )

    except Exception as e:

        print(
            "\nTopic Modeling Failed!"
        )

        print(
            f"\nError: {e}"
        )

        raise
