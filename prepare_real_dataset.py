"""
Converts data/twitter_training.csv (downloaded from Kaggle: "Twitter Entity
Sentiment Analysis") into data/sentiment_data.csv, matching the format
train_model.py expects: two columns, "text" and "label".

Run this AFTER placing twitter_training.csv in the data/ folder, and BEFORE
running train_model.py.
"""
import pandas as pd

# This Kaggle file has no header row. Its columns, in order, are:
# tweet_id, entity, sentiment, text
raw = pd.read_csv(
    "data/twitter_training.csv",
    header=None,
    names=["tweet_id", "entity", "sentiment", "text"],
)

print(f"Loaded {len(raw)} raw rows")
print(raw["sentiment"].value_counts())

# Keep only the three sentiment classes we care about (drop "Irrelevant")
raw = raw[raw["sentiment"].isin(["Positive", "Negative", "Neutral"])].copy()

# Drop rows with missing/empty text
raw = raw.dropna(subset=["text"])
raw = raw[raw["text"].str.strip() != ""]

# Match the column names and lowercase labels train_model.py expects
raw["label"] = raw["sentiment"].str.lower()
out = raw[["text", "label"]]

# Balance the classes so none dominates training
min_count = out["label"].value_counts().min()
parts = [
    group.sample(min_count, random_state=42)
    for _, group in out.groupby("label")
]
balanced = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

balanced.to_csv("data/sentiment_data.csv", index=False)
print(f"\nWrote {len(balanced)} balanced rows to data/sentiment_data.csv")
print(balanced["label"].value_counts())
