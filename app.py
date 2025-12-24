from flask import Flask, render_template, request
import os
import pandas as pd

from utils import clean_tweet
from models import vader_predict, roberta_predict, load_roberta
from evaluation import evaluate

app = Flask(__name__)

# ⚙️ Preload RoBERTa model once (for speed and stability)
@app.before_request
def preload_model():
    """Load RoBERTa once before handling requests."""
    if not hasattr(app, "roberta_loaded"):
        print("⚙️ Preloading RoBERTa model once...")
        load_roberta()
        app.roberta_loaded = True


CSV_NAME = "training.1600000.processed.noemoticon.csv"
PROCESSED_CSV = "processed_sample.csv"
FIRST_RUN_ROWS = 1000   # process fewer rows initially for speed
FULL_SIZE = 1000        # cached larger sample for future runs


def load_and_prepare():
    """
    Load preprocessed dataset if available, else process a small subset (100 rows)
    on first run for faster startup.
    """
    if os.path.exists(PROCESSED_CSV):
        print("✅ Loaded cached dataset from processed_sample.csv")
        return pd.read_csv(PROCESSED_CSV)

    if not os.path.exists(CSV_NAME):
        print("⚠️ Original dataset not found in project directory.")
        return None

    print(f"🔄 Processing dataset for the first time ({FIRST_RUN_ROWS} rows)...")
    raw_df = pd.read_csv(CSV_NAME, encoding="latin-1", header=None, nrows=FIRST_RUN_ROWS)
    df = raw_df[[0, 5]].copy() if raw_df.shape[1] >= 6 else raw_df.copy()
    df.columns = ["sentiment", "text"]
    df["sentiment"] = df["sentiment"].map({0: "negative", 4: "positive"}).fillna("neutral")
    df["clean_text"] = df["text"].apply(clean_tweet)

    # 💾 Cache for next time
    df.to_csv(PROCESSED_CSV, index=False)
    print("💾 Cached processed dataset for faster future loads.")
    return df


@app.route("/")
def home():
    df = load_and_prepare()
    metrics = None

    if df is not None:
        # Compute model predictions only if not already cached
        if not {"vader_pred_binary", "roberta_pred_binary"}.issubset(df.columns):
            print("⚙️ Computing model predictions (first run only)...")
            df["vader_pred"] = df["clean_text"].apply(lambda x: vader_predict(x)["label"])
            df["vader_pred_binary"] = df["vader_pred"].replace({"neutral": "negative"})

            df["roberta_pred"] = df["clean_text"].apply(lambda x: roberta_predict(x)["label"])
            df["roberta_pred_binary"] = df["roberta_pred"].replace({"neutral": "negative"})

            df.to_csv(PROCESSED_CSV, index=False)
            print("📊 Predictions saved to processed_sample.csv")

        # Evaluate models
        metrics = {
            "vader": evaluate(df["sentiment"], df["vader_pred_binary"], "VADER"),
            "roberta": evaluate(df["sentiment"], df["roberta_pred_binary"], "RoBERTa"),
        }

    return render_template("index.html", metrics=metrics)


@app.route("/predict", methods=["POST"])
def predict():
    """Handle individual text prediction requests."""
    text = request.form.get("text")
    if not text:
        return render_template("index.html", error="Please enter some text!")

    clean = clean_tweet(text)
    vader_result = vader_predict(clean)
    roberta_result = roberta_predict(clean)

    return render_template(
        "index.html",
        original=text,
        cleaned=clean,
        vader_label=vader_result["label"],
        vader_scores=vader_result["scores"],
        roberta_label=roberta_result["label"],
        roberta_scores=roberta_result["scores"],
    )


if __name__ == "__main__":
    app.run(debug=True)
