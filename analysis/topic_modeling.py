import pandas as pd

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

from sentence_transformers import (
    SentenceTransformer
)

from sklearn.feature_extraction.text import (
    CountVectorizer
)

from umap import UMAP

from hdbscan import HDBSCAN

from utils.file_loader import load_feedback



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

        stop_words="english",

        ngram_range=(1, 2),

        min_df=1,

        max_df=0.95

    )

    # =============================================
    # UMAP
    # =============================================

    umap_model = UMAP(

        n_neighbors=15,

        n_components=5,

        min_dist=0.0,

        metric="cosine",

        random_state=42

    )

    # =============================================
    # HDBSCAN
    # =============================================

    hdbscan_model = HDBSCAN(

        min_cluster_size=15,

        min_samples=5,

        metric="euclidean",

        cluster_selection_method="eom",

        prediction_data=True

    )

    # =============================================
    # Topic Representation
    # =============================================

    representation_model = (

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

    )

    print(
        "\nTraining BERTopic..."
    )

    update(0.65, "Running BERTopic clustering...")

    topics, probabilities = (

        topic_model.fit_transform(

            documents

        )

    )
    update(0.80, " Extracting representative reviews...")

    negative_df["topic"] = topics

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
            "Representative_Reviews"
        ]
    ].copy()

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