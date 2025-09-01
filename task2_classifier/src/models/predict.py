import os
import joblib
from sklearn.pipeline import Pipeline

def load_model(model_path: str) -> Pipeline:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Run training first.")
    return joblib.load(model_path)

def predict_text(pipeline: Pipeline, text: str, low_conf: float = 0.45) -> str:
    """Return formatted prediction string; includes confidence if available."""
    pred = pipeline.predict([text])[0]
    conf_msg = ""
    clf = pipeline.named_steps.get("clf")
    vec = pipeline.named_steps.get("vec")

    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(vec.transform([text]))[0]
        classes = clf.classes_
        pairs = sorted(zip(classes, probs), key=lambda x: -x[1])
        top_c, top_p = pairs[0]
        conf_msg = " | " + ", ".join(f"{c}: {p:.2f}" for c, p in pairs)
        if top_p < low_conf:
            return f"Predicted: {pred} (low confidence: {top_p:.2f}){conf_msg}"
    return f"Predicted: {pred}{conf_msg}"
