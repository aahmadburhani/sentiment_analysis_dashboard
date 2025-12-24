📊 Sentiment Analysis Dashboard (Flask + VADER + RoBERTa)

A full-stack NLP project that performs real-time sentiment analysis using both VADER (rule-based) and RoBERTa (transformer-based) models.
This dashboard includes a modern UI, dual-model comparison, animated charts, dataset evaluation, and pytest-based testing.

This project is part of my Major Project (B.Tech – CSE).

🚀 Features
🔹 Dual Sentiment Models

Both models return:

Sentiment label (Positive / Neutral / Negative)
Score distribution for all three sentiments
Emoji-based visualization
Models used:
VADER – Fast, lexicon & rule-based
RoBERTa-base – Deep learning transformer for high accuracy

🔹 Modern Interactive Dashboard

Clean & responsive interface
Light mode + Navy Blue Dark Mode toggle
Smooth animated sentiment bars
Metric comparison chart (Accuracy, Precision, Recall, F1 Score)
Professional color scheme
Accessibility-friendly UI

🔹 Dataset Processing

Reads 1000 tweets from the original dataset
Cleans text using regex via clean_tweet()
Auto-creates processed_sample.csv for fast loading
Evaluates both models using confusion-matrix based metrics

🔹 Testing (pytest)

Includes 7 automated test cases:

Test Case	Purpose
test_clean_tweet: Ensures preprocessing removes noise
test_vader_predict:	Validates VADER output structure
test_roberta_predict: Validates RoBERTa output structure
test_evaluate_function:	Checks metric calculation
test_home_route: Ensures homepage loads
test_predict_route:	Validates prediction workflow
test_empty_input_prediction: Handles empty input error

📁 Project Structure
sentiment_analysis_flask/
│── app.py                     # Flask backend + main entry
│── models.py                  # VADER + RoBERTa logic
│── utils.py                   # clean_tweet()
│── evaluation.py              # accuracy, precision, recall, F1
│── requirements.txt
│── test_sentiment.py          # pytest test suite
│
│── templates/
│     └── index.html           # UI dashboard
│
│── static/
│     └── style.css            # Full responsive UI, animations, dark mode
│
│── processed_sample.csv       # Auto-generated cached cleaned dataset
│── training_data_info.txt     # Dataset info
│── README.md

📥 Dataset Download

GitHub does NOT allow files larger than 100 MB.
The original dataset file:

training.1600000.processed.noemoticon.csv (227 MB)

must be downloaded manually.

🔗 Download Full Dataset

👉 Link (add your link):
https://drive.google.com/drive/folders/11RECJ63eFUBMl26ReVGP_ilGF6MPBL1P?usp=sharing
After downloading, place it inside your project folder:

sentiment_analysis_flask/
│── training.1600000.processed.noemoticon.csv
│── app.py
│── ...

Installation & Setup
Step 1 — Create Virtual Environment
python -m venv venv

Step 2 — Activate Environment

Windows:

venv\Scripts\activate

Mac/Linux:

source venv/bin/activate

Step 3 — Install Libraries
pip install -r requirements.txt

Step 4 — Run Your Flask App
python app.py


Your project will run at:
👉 http://127.0.0.1:5000/

How the System Works
1️⃣ User Input

User enters any sentence in the text box.

2️⃣ Text Cleaning

clean_tweet() removes:

URLs
@mentions
Punctuation
Extra spaces
Hashtags
Non-alphabetic characters

3️⃣ Model Prediction Engine

Each cleaned text is sent to:
vader_predict()
roberta_predict()
Both return:

{
  "label": "positive",
  "scores": {
      "positive": 0.85,
      "neutral": 0.12,
      "negative": 0.03
  }
}

4️⃣ Visualization

UI displays:

Animated sentiment bars
Scores
Emoji indicators

5️⃣ Dataset Evaluation

The system:
Loads 1000 tweets
Computes predictions for both models
Generates metrics using evaluate()
Displays a bar chart comparison

6️⃣ Caching

After first run:

Cleaned + predicted data stored in processed_sample.csv
Speeds up future loads significantly

📊 Metrics Computed

For both VADER & RoBERTa:

Metric	Description
Accuracy	Overall correctness
Precision	Correct positive predictions
Recall	Correctly captured actual positives
F1 Score	Harmonic mean of precision & recall
🧪 Running Test Suite

Run:

pytest -v

You will see:

7 passed, 0 failed

🌗 Dark Mode & Light Mode

The project includes:

Toggle switch

Animated transition

Theme-aware charts

Theme-aware sentiment bars

Navy blue dark mode for smooth contrast

All controlled using:

applyTheme(isDark)

🌱 Future Enhancements

Add multilingual sentiment support

Deploy on Render / Railway

Add model fine-tuning

Add custom dataset uploader

Add domain-specific sentiment (finance, healthcare, movies)

🤝 Contributing

Pull requests and suggestions are welcome!

📜 License

MIT License © 2025