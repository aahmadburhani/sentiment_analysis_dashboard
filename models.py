import nltk
import numpy as np
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer

# --- VADER Setup ---
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

def vader_predict(text):
    """
    Return both the overall label and sentiment scores from VADER.
    Example output:
      {
        "label": "positive",
        "scores": {"positive": 0.73, "neutral": 0.22, "negative": 0.05}
      }
    """
    text = str(text)
    scores = sia.polarity_scores(text)

    if scores["compound"] >= 0.05:
        label = "positive"
    elif scores["compound"] <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    # Return all sentiment components
    return {
        "label": label,
        "scores": {
            "positive": round(scores["pos"], 3),
            "neutral": round(scores["neu"], 3),
            "negative": round(scores["neg"], 3),
        },
    }


# --- RoBERTa Setup (optimized) ---
_roberta_pipeline = None
_model_loaded = False


def load_roberta():
    """Load RoBERTa model once and reuse it."""
    global _roberta_pipeline, _model_loaded
    if _model_loaded and _roberta_pipeline is not None:
        return _roberta_pipeline

    try:
        print("🔄 Loading RoBERTa model (first time only)...")
        model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        _roberta_pipeline = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
        _model_loaded = True
        print("✅ RoBERTa model loaded successfully!")
        return _roberta_pipeline
    except Exception as e:
        print(f"⚠️ Failed to load RoBERTa: {e}")
        _roberta_pipeline = None
        return None


def roberta_predict(text):
    """
    Return both the overall label and sentiment scores from RoBERTa.
    Example output:
      {
        "label": "positive",
        "scores": {"positive": 0.88, "neutral": 0.09, "negative": 0.03}
      }
    """
    global _roberta_pipeline
    if _roberta_pipeline is None:
        _roberta_pipeline = load_roberta()

    if _roberta_pipeline is None:
        return {"label": "neutral", "scores": {"positive": 0.0, "neutral": 1.0, "negative": 0.0}}

    text = str(text)
    try:
        result = _roberta_pipeline(text[:512])[0]
        label = result["label"].lower()
        score = float(result["score"])

        # Approximate conversion into 3-class probabilities
        if "label_2" in label or "positive" in label:
            scores = {"positive": score, "neutral": 1 - score, "negative": 0.0}
        elif "label_1" in label or "neutral" in label:
            scores = {"positive": 0.0, "neutral": score, "negative": 1 - score}
        else:
            scores = {"positive": 0.0, "neutral": 1 - score, "negative": score}

        # Normalize so values sum to 1
        total = sum(scores.values())
        for k in scores:
            scores[k] = round(scores[k] / total, 3)

        # Determine final label by highest confidence
        top_label = max(scores, key=scores.get)

        return {"label": top_label, "scores": scores}

    except Exception as e:
        print(f"⚠️ RoBERTa prediction error: {e}")
        return {"label": "neutral", "scores": {"positive": 0.0, "neutral": 1.0, "negative": 0.0}}
