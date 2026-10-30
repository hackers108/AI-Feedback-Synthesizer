from transformers import pipeline
import pandas as pd

from normalization.schema_mapper import normalize_schema


def run_sentiment(
    input_file,
    sample_size=500,
    progress_callback=None
):

    def update(progress, message):

        if progress_callback:

            progress_callback(
                progress,
                message
            )

    # ----------------------------------
    # Load Input File
    # ----------------------------------

    update(0.0, "Loading uploaded feedback...")

    df = pd.read_csv(input_file)

    print("\nColumns Found:")
    print(df.columns.tolist())

    # ----------------------------------
    # Normalize Schema
    # ----------------------------------

    update(0.10, "Normalizing CSV columns...")

    df = normalize_schema(df)

    print("\nColumns After Mapping:")
    print(df.columns.tolist())

    # ----------------------------------
    # Sampling
    # ----------------------------------

    # sample_size=None means analyze every review.
    if (
        sample_size is not None
        and len(df) > sample_size
    ):

        df = df.sample(
            sample_size,
            random_state=42
        )

    print(f"\nTotal Reviews Selected: {len(df)}")

    # ----------------------------------
    # Load Sentiment Model
    # ----------------------------------

    update(0.20, "Loading RoBERTa sentiment model...")

    classifier = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

    print("\nSentiment Model Loaded!")

    update(0.35, "RoBERTa model loaded")

    sentiments = []

    batch_size = 32

    # ----------------------------------
    # Batch Prediction
    # ----------------------------------

    for i in range(0, len(df), batch_size):

        batch = (
            df["text"]
            .fillna("")
            .astype(str)
            .iloc[i:i + batch_size]
            .tolist()
        )

        results = classifier(batch)

        sentiments.extend(
            [result["label"] for result in results]
        )

        print(
            f"Processed {min(i + batch_size, len(df))}/{len(df)} reviews"
        )

        processed_reviews = min(
            i + batch_size,
            len(df)
        )

        update(
            0.35 + (
                0.60
                * processed_reviews
                / max(len(df), 1)
            ),
            f"Analyzed {processed_reviews}/{len(df)} reviews"
        )

    # ----------------------------------
    # Save Results
    # ----------------------------------

    update(0.98, "Saving sentiment results...")

    df["sentiment"] = sentiments

    print("\nSentiment Distribution:")
    print(df["sentiment"].value_counts())

    OUTPUT_FILE = "data/sentiment_feedback.csv"

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSentiment Analysis Completed!")

    update(1.0, "Sentiment analysis completed")

    return OUTPUT_FILE


if __name__ == "__main__":

    run_sentiment("uploads/input.csv")
