import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from ..config import LABELS, RSEED
from ..features.vectorizer import build_vectorizer_union
from .plot import save_confusion_matrix

def train_and_evaluate(
    df: pd.DataFrame,
    test_size: float,
    model_out: str,
    cm_out: str,
) -> Tuple[Pipeline, str, float]:
    print(f"[data] rows={len(df)}", flush=True)
    print(f"[data] by class={df['label'].value_counts().to_dict()}", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=test_size, random_state=RSEED, stratify=df["label"]
    )
    print(f"[split] train={len(X_train)} test={len(X_test)} (test_size={test_size})", flush=True)

    vec_union = build_vectorizer_union()
    print("[model] building pipelines…", flush=True)
    # Per-class weights -> per-sample weights for NB (handles imbalance)
    classes_ = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes_, y=y_train)
    cw_map = {c: w for c, w in zip(classes_, class_weights)}
    sample_weight = np.array([cw_map[y] for y in y_train])

    models = {
        "MultinomialNB": Pipeline([("vec", vec_union), ("clf", MultinomialNB())]),
        "LogReg": Pipeline([("vec", vec_union),
                            ("clf", LogisticRegression(max_iter=2000, solver="saga",
                                                       class_weight="balanced", random_state=RSEED))]),
    }

    best_name, best_acc, best_model = None, -1.0, None
    reports = {}

    for name, pipe in models.items():
        if name == "MultinomialNB":
            pipe.fit(X_train, y_train, clf__sample_weight=sample_weight)
        else:
            pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, pred)
        rep = classification_report(y_test, pred, digits=3, zero_division=0)
        cm  = confusion_matrix(y_test, pred, labels=LABELS)
        reports[name] = (acc, rep, cm)

        if acc > best_acc:
            best_name, best_acc, best_model = name, acc, pipe

    # Print summaries
    for name, (acc, rep, _) in reports.items():
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.3f}")
        print(rep)

    # Save CM for best
    cm_best = reports[best_name][2]
    save_confusion_matrix(cm_best, LABELS, f"Confusion Matrix — {best_name}", cm_out)

    # Save the full pipeline
    joblib.dump(best_model, model_out)
    print(f"\nBest model: {best_name}  |  accuracy={best_acc:.3f}")
    print(f"Saved model → {model_out}")
    print(f"Saved confusion matrix → {cm_out}")

    return best_model, best_name, best_acc
