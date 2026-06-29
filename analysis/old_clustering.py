import pandas as pd
from sentence_transformers import SentenceTransformer
import hdbscan

# Load sentiment results
df = pd.read_csv("data/sentiment_feedback.csv")

# Keep only negative reviews
negative_df = df[
    df["sentiment"] == "negative"
].copy()

print(f"\nNegative Reviews: {len(negative_df)}")

# Load embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Generate embeddings
embeddings = model.encode(
    negative_df["text"]
    .fillna("")
    .astype(str)
    .tolist(),
    show_progress_bar=True
)

print("\nEmbeddings Generated!")

# HDBSCAN Clustering
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=3,
    min_samples=1
)

clusters = clusterer.fit_predict(
    embeddings
)

negative_df["cluster"] = clusters

print("\nClustering Completed!")

# Cluster Statistics
print("\nCluster Counts:\n")

cluster_summary = (
    negative_df["cluster"]
    .value_counts()
    .reset_index()
)

cluster_summary.columns = [
    "cluster",
    "count"
]

print(cluster_summary)

# Noise reviews
noise_count = (
    negative_df["cluster"] == -1
).sum()

print(
    f"\nNoise Reviews: {noise_count}"
)

# Display sample reviews from each cluster
for cluster_id in sorted(
    negative_df["cluster"].unique()
):

    if cluster_id == -1:
        continue

    cluster_reviews = negative_df[
        negative_df["cluster"] == cluster_id
    ]

    cluster_size = len(
        cluster_reviews
    )

    print("\n" + "=" * 60)
    print(
        f"CLUSTER {cluster_id}"
    )
    print(
        f"Affected Users: {cluster_size}"
    )
    print("=" * 60)

    sample_reviews = (
        cluster_reviews["text"]
        .head(10)
    )

    for review in sample_reviews:

        print(review)
        print("-" * 40)

# Save clustered output
negative_df.to_csv(
    "data/clustered_feedback.csv",
    index=False
)

print(
    "\nClustered data saved:"
)
print(
    "data/clustered_feedback.csv"
)