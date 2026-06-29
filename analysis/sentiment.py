from transformers import pipeline
import pandas as pd

from normalization.schema_mapper import normalize_schema


def run_sentiment(input_file):

    # ----------------------------------
    # Load Input File
    # ----------------------------------

    df = pd.read_csv(input_file)

    print("\nColumns Found:")
    print(df.columns.tolist())

    # ----------------------------------
    # Normalize Schema
    # ----------------------------------

    df = normalize_schema(df)

    print("\nColumns After Mapping:")
    print(df.columns.tolist())

    # ----------------------------------
    # Sampling
    # ----------------------------------

    SAMPLE_SIZE = 500

    if len(df) > SAMPLE_SIZE:
        df = df.sample(
            SAMPLE_SIZE,
            random_state=42
        )

    print(f"\nTotal Reviews Selected: {len(df)}")

    # ----------------------------------
    # Load Sentiment Model
    # ----------------------------------

    classifier = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

    print("\nSentiment Model Loaded!")

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

    # ----------------------------------
    # Save Results
    # ----------------------------------

    df["sentiment"] = sentiments

    print("\nSentiment Distribution:")
    print(df["sentiment"].value_counts())

    OUTPUT_FILE = "data/sentiment_feedback.csv"

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSentiment Analysis Completed!")

    return OUTPUT_FILE


if __name__ == "__main__":

    run_sentiment("uploads/input.csv")