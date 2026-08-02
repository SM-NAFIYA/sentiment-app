"""
Trains three classifiers on data/sentiment_data.csv:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Random Forest
Compares their accuracy, then saves the best-performing model + the
TF-IDF vectorizer to model/ so the Flask app can load them instantly.
"""
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
os.makedirs("model",exit_ok=True)
# 1. Load data
df = pd.read_csv("data/sentiment_data.csv")
print(f"Loaded {len(df)} rows")
print(df["label"].value_counts(), "\n")

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Vectorize text (TF-IDF)
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000,
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 3. Train and compare three algorithms
candidates = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

results = {}
for name, clf in candidates.items():
    clf.fit(X_train_vec, y_train)
    preds = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    results[name] = (acc, clf)
    print(f"=== {name} ===")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, preds))
    print()

# 4. Pick the best model
best_name = max(results, key=lambda n: results[n][0])
best_acc, best_model = results[best_name]
print(f"Best model: {best_name} (accuracy {best_acc:.3f})")

# 5. Save the winning model + vectorizer + a small metadata file
joblib.dump(best_model, "model/sentiment_model.joblib")
joblib.dump(vectorizer, "model/vectorizer.joblib")
with open("model/best_model.txt", "w") as f:
    f.write(f"{best_name}\n{best_acc:.3f}\n")

print("\nSaved model/sentiment_model.joblib and model/vectorizer.joblib")
