import pytest
import pandas as pd
from flask import Flask

# Import from your project
from utils import clean_tweet
from models import vader_predict, roberta_predict, load_roberta
from evaluation import evaluate
from app import app


# ======================
# 🔹 1. Utils Module Tests
# ======================
def test_clean_tweet():
    """Test text cleaning logic."""
    assert clean_tweet("@user Check out http://example.com #cool") == "check out cool"
    assert clean_tweet("HELLO WORLD!!!") == "hello world"
    assert clean_tweet("Spaces   here") == "spaces here"
    assert clean_tweet("") == ""
    print("✅ clean_tweet passed")


# ======================
# 🔹 2. Model Prediction Tests
# ======================
def test_vader_predict():
    """Check VADER returns proper sentiment dictionary."""
    result = vader_predict("This is a wonderful experience!")
    assert isinstance(result, dict)
    assert "label" in result and "scores" in result
    assert result["label"] in ["positive", "neutral", "negative"]
    print(f"✅ VADER output: {result}")


@pytest.mark.skipif(load_roberta() is None, reason="RoBERTa model not available")
def test_roberta_predict():
    """Check RoBERTa returns sentiment dictionary."""
    result = roberta_predict("I love this app!")
    assert isinstance(result, dict)
    assert "label" in result and "scores" in result
    assert result["label"] in ["positive", "neutral", "negative"]
    print(f"✅ RoBERTa output: {result}")


# ======================
# 🔹 3. Evaluation Function Tests
# ======================
def test_evaluate_function():
    """Validate metric calculations."""
    y_true = pd.Series(["positive", "negative", "positive", "negative"])
    y_pred = pd.Series(["positive", "negative", "negative", "positive"])

    metrics = evaluate(y_true, y_pred, "TestModel")

    assert "accuracy" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert metrics["model"] == "TestModel"
    print("✅ evaluate() metrics computed successfully")


# ======================
# 🔹 4. Flask App Route Tests
# ======================
@pytest.fixture
def client():
    """Set up Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home_route(client):
    """Test home page loads correctly."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Sentiment Analysis" in response.data
    print("✅ / route loaded successfully")


def test_predict_route(client):
    """Test /predict route with valid text."""
    response = client.post("/predict", data={"text": "This app is awesome!"})
    assert response.status_code == 200
    assert b"VADER" in response.data
    assert b"RoBERTa" in response.data
    print("✅ /predict route works fine")


def test_empty_input_prediction(client):
    """Ensure empty input handled without crash."""
    response = client.post("/predict", data={"text": ""})
    assert response.status_code == 200  # Page re-renders safely
    print("✅ Empty input handled gracefully")
