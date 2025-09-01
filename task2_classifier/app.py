# app.py
# Task 2 — Subject Classifier mini web app (Politics / Business / Health)
# Place this file at the project root: task2-classifier/app.py
# Run: python app.py   → http://127.0.0.1:5000/

import os
from typing import Dict, List, Tuple
from flask import Flask, request, render_template_string, jsonify
from src.config import MODEL_PATH, ensure_dirs
from src.models.predict import load_model

# ------------------------------
# App + lazy model loader
# ------------------------------
app = Flask(__name__)
_model = None  # lazy-loaded on first request

def get_model():
    global _model
    if _model is None:
        ensure_dirs()
        _model = load_model(MODEL_PATH)  # raises FileNotFoundError if not trained
    return _model

# ------------------------------
# Utilities
# ------------------------------
def probs_from_model(pipeline, text: str) -> List[Tuple[str, float]]:
    """Return [(class, prob), ...] descending if model supports predict_proba.
       If not supported, returns [(pred, 1.0)]."""
    clf = pipeline.named_steps.get("clf")
    vec = pipeline.named_steps.get("vec")
    if hasattr(clf, "predict_proba"):
        pr = clf.predict_proba(vec.transform([text]))[0]
        classes = clf.classes_
        pairs = sorted(zip(classes, pr), key=lambda x: -x[1])
        return pairs
    # fallback: no probabilities
    pred = pipeline.predict([text])[0]
    return [(pred, 1.0)]

# ------------------------------
# HTML (inline template)
# ------------------------------
HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Task 2 — Subject Classifier</title>
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,'Helvetica Neue',Arial,sans-serif;margin:24px;max-width:900px}
    textarea{width:100%;height:150px;padding:12px;border:1px solid #cbd5e1;border-radius:10px}
    button{padding:10px 16px;border:none;border-radius:10px;background:#0b69ff;color:#fff;cursor:pointer}
    .card{border:1px solid #e5e7eb;border-radius:12px;padding:14px 16px;margin-top:14px}
    .muted{color:#6b7280}
    .bar{height:10px;background:#e5e7eb;border-radius:999px;overflow:hidden}
    .fill{height:100%;background:#0b69ff}
    .row{display:flex;align-items:center;gap:12px}
    .cls{min-width:90px}
    .pill{display:inline-block;background:#eef2ff;padding:2px 8px;border-radius:999px;margin-left:8px;font-size:12px;color:#3730a3}
    .warn{color:#b45309}
  </style>
</head>
<body>
  <h2>Task 2 — Subject Classifier <span class="pill">Politics • Business • Health</span></h2>
  <h3>Author: Nitesh Khadka Student ID: 250207 CUID: 16542697</h3>
  <form method="GET" action="/">
    <textarea name="q" placeholder="Paste any news snippet or paragraph...">{{ q or '' }}</textarea>
    <div style="margin-top:8px">
      <button type="submit">Classify</button>
    </div>
  </form>

  {% if error %}
    <div class="card warn"><b>Note:</b> {{ error }}</div>
  {% endif %}

  {% if q is not none %}
    <div class="card">
      <div class="muted">Input</div>
      <div>{{ q }}</div>
    </div>

    {% if result %}
      <div class="card">
        <div><b>Prediction</b></div>
        <div>{{ result }}</div>
      </div>

      {% if probs %}
        <div class="card">
          <div class="muted">Confidence</div>
          {% for c, p in probs %}
            <div class="row">
              <div class="cls">{{ c }}</div>
              <div class="bar" style="flex:1">
                <div class="fill" style="width: {{ '%0.0f' | format(p*100) }}%"></div>
              </div>
              <div style="width:42px;text-align:right">{{ '%0.2f'|format(p) }}</div>
            </div>
          {% endfor %}
        </div>
      {% endif %}
    {% endif %}
  {% endif %}

  <div class="muted" style="margin-top:18px">
    Tip: train first using <code>python -m src.cli.main train</code> so <code>models/task2_model.joblib</code> exists.
  </div>
</body>
</html>
"""

# ------------------------------
# Routes
# ------------------------------
@app.route("/", methods=["GET"])
def home():
    q = request.args.get("q", default=None, type=str)
    error = ""
    result = ""
    probs = []
    if q is not None:
        q = (q or "").strip()
        if not q:
            error = "Please paste some text to classify."
        else:
            try:
                model = get_model()
                pairs = probs_from_model(model, q)
                probs = pairs
                top_c, top_p = pairs[0]
                # formatted line e.g., "Predicted: Health | Politics:0.12, Business:0.08, Health:0.80"
                prob_line = ", ".join(f"{c}: {p:.2f}" for c, p in pairs)
                result = f"Predicted: {top_c}  |  {prob_line}"
            except FileNotFoundError:
                error = "Model not found. Please run training first:  python -m src.cli.main train"
            except Exception as e:
                error = f"Classification error: {e}"
    return render_template_string(HTML, q=q, result=result, probs=probs, error=error)

@app.route("/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(silent=True) or {}
        txt = (data.get("text") or "").strip()
        if not txt:
            return jsonify({"error": "Missing 'text'"}), 400
        model = get_model()
        pairs = probs_from_model(model, txt)
        top_c, top_p = pairs[0]
        return jsonify({
            "result": f"Predicted: {top_c}",
            "probs": {c: float(p) for c, p in pairs}
        })
    except FileNotFoundError:
        return jsonify({"error": "Model not found. Train first (python -m src.cli.main train)."}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/reload", methods=["POST"])
def reload_model():
    """Reload model from disk without restarting the server (use after re-training)."""
    global _model
    _model = None
    try:
        get_model()
        return jsonify({"ok": True, "message": "Model reloaded."})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/healthz", methods=["GET"])
def health():
    try:
        get_model()
        return jsonify({"ok": True})
    except FileNotFoundError:
        # App is up, but model not trained yet
        return jsonify({"ok": True, "model": "missing"}), 200

if __name__ == "__main__":
    # You can change host/port via env vars, e.g. PORT=5001 python app.py
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    app.run(host=host, port=port, debug=False)
