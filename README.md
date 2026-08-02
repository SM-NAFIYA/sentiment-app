# Sentiment Analysis using Machine Learning

A BSc Data Science project that classifies text (customer reviews, feedback,
social media comments) as **Positive**, **Negative**, or **Neutral** using
Machine Learning, replacing slow manual review analysis.

## Tech stack (matches project report)
- **Frontend:** HTML, CSS, Bootstrap
- **Backend:** Python (Flask)
- **Database:** MySQL
- **ML Algorithms compared:** Naive Bayes, Logistic Regression, Random Forest
- **Libraries:** pandas, numpy, scikit-learn, joblib
- **Tools:** VS Code, Jupyter Notebook, XAMPP, a web browser

## Project structure
```
sentiment_app/
├── app.py                  # Flask backend (routes + MySQL logging)
├── train_model.py          # Trains & compares NB / Logistic Regression / Random Forest
├── generate_dataset.py     # Builds the demo dataset (data/sentiment_data.csv)
├── db_setup.sql            # MySQL table creation script
├── requirements.txt
├── data/
│   └── sentiment_data.csv  # Labeled training data (generated)
├── model/
│   ├── sentiment_model.joblib   # Best trained model (saved after training)
│   ├── vectorizer.joblib        # Fitted TF-IDF vectorizer
│   └── best_model.txt           # Which algorithm won, and its accuracy
├── templates/
│   └── index.html          # Bootstrap UI
└── static/
    └── style.css
```

## Setup (Windows, matches your platform)

1. **Install Python** (3.10+) if not already installed.

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Set up MySQL** (via XAMPP):
   - Start MySQL in the XAMPP control panel.
   - Open phpMyAdmin (or the MySQL CLI) and run the contents of `db_setup.sql`.
   - If your MySQL root user has a password, update `DB_CONFIG` in `app.py`.
   - MySQL is optional — the app still runs and predicts without it; it just
     won't save history.

4. **Generate the training data:**
   ```
   python generate_dataset.py
   ```
   This creates `data/sentiment_data.csv`, a template-generated dataset of
   labeled reviews. It's meant to get the whole pipeline working end-to-end.
   **For your final submission, swap in a real dataset** (e.g. a Kaggle
   "Twitter/Amazon/IMDB sentiment" CSV with `text` and `label` columns) —
   real text is messier, so accuracy will be a more realistic 75-90% instead
   of the near-100% you'll see on the template data.

5. **Train the model:**
   ```
   python train_model.py
   ```
   This trains all three algorithms, prints an accuracy comparison and
   classification report for each, and saves the best-performing one to
   `model/`.

6. **Run the app:**
   ```
   python app.py
   ```
   Open `http://127.0.0.1:5000` in your browser. Type a review, click
   **Analyze Sentiment**, and see the predicted label + confidence. If MySQL
   is connected, past analyses appear in the "Recent Analyses" list.

## How it works
1. Text is converted to numbers using **TF-IDF** (term frequency–inverse
   document frequency), which weights words by how distinctive they are.
2. Three classifiers are trained on the same vectorized text: **Naive
   Bayes**, **Logistic Regression**, and **Random Forest**.
3. Whichever scores highest on a held-out test set is saved and used by the
   Flask app for live predictions.
4. Each prediction is optionally logged to MySQL (`predictions` table) with
   the review text, predicted sentiment, and timestamp.

## Notes for your report
- The "Problem in Existing System" (manual analysis, slow, inconsistent) is
  solved by items 1-4 in your "What will I change" section — this app does
  exactly that: instant classification, no human judgment involved.
- For the comparison table in your report, run `train_model.py` and copy
  the accuracy/precision/recall printed for each of the three algorithms.
