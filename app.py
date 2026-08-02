"""
Flask backend for the Sentiment Analysis app.

Routes:
  GET  /            -> renders the Bootstrap UI
  POST /predict      -> takes text, returns predicted sentiment (AJAX/JSON)

Every prediction is logged to a MySQL table (see db_setup.sql) so you
can show "history of analyzed reviews" in your project demo. If MySQL
isn't configured/running, the app still works -- it just skips logging
and prints a warning.
"""
from flask import Flask, render_template, request, jsonify
import joblib
import os

# Optional MySQL logging -- app still works without a DB connection
try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

app = Flask(__name__)

# --- Load the trained model + vectorizer -----------------------------------
MODEL_PATH = os.path.join("model", "sentiment_model.joblib")
VECTORIZER_PATH = os.path.join("model", "vectorizer.joblib")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# --- MySQL configuration ---------------------------------------------------
# Reads from environment variables if set (used when deployed), otherwise
# falls back to your local XAMPP defaults -- no code changes needed either way.
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", ""),
    "database": os.environ.get("DB_NAME", "sentiment_db"),
    "use_pure": True,        # avoids a C-extension bug with text params on Windows
}


def log_prediction(text, sentiment):
    """Insert a prediction into MySQL. Fails silently if DB isn't set up."""
    if not MYSQL_AVAILABLE:
        return
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO predictions (review_text, sentiment) VALUES (%s, %s)",
            (str(text), str(sentiment)),
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"[warning] Could not log to MySQL: {e}")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or request.form
    text = (data.get("review_text") or "").strip()

    if not text:
        return jsonify({"error": "Please enter some text."}), 400

    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]

    # Confidence score, if the model supports predict_proba
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        confidence = round(float(max(proba)) * 100, 1)

    log_prediction(text, prediction)

    return jsonify({
        "sentiment": prediction,
        "confidence": confidence,
    })


@app.route("/history")
def history():
    """Returns the last 20 logged predictions from MySQL (if configured)."""
    if not MYSQL_AVAILABLE:
        return jsonify([])
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT review_text, sentiment, created_at FROM predictions "
            "ORDER BY id DESC LIMIT 20"
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        # created_at isn't JSON-serializable by default
        for r in rows:
            r["created_at"] = str(r["created_at"])
        return jsonify(rows)
    except Exception as e:
        print(f"[warning] Could not read history from MySQL: {e}")
        return jsonify([])


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(debug=debug_mode)
