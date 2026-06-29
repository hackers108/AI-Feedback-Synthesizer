import pandas as pd

from normalization.schema import (
    create_normalized_df
)

# Load Play Store reviews
df = pd.read_csv(
    "data/instagram.csv"
)

# Normalize
normalized_df = create_normalized_df(
    ids=df.index.astype(str),
    texts=df["review_description"],
    source="playstore",
    dates=df["review_date"],
    ratings=df["rating"]
)

print("\nSample Data:\n")
print(normalized_df.head())

print("\nColumns:")
print(
    normalized_df.columns.tolist()
)

# Save normalized output
normalized_df.to_csv(
    "data/normalized_feedback.csv",
    index=False
)

print(
    "\nNormalization Completed!"
)