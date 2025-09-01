# Task 2 — Subject Classification (Politics / Business / Health)

This project trains a robust text classifier using BBC RSS (titles + summaries) for Politics, Business, and Health.
It uses TF-IDF with word (1–2 grams) + character (3–5 grams) features and evaluates Multinomial Naïve Bayes (with sample weights) and Logistic Regression (class_weight=balanced). The best model is saved for prediction.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000