# src/config.py
import os
from pathlib import Path

FEEDS = {
    "Politics": "https://feeds.bbci.co.uk/news/politics/rss.xml",
    "Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "Health":   "https://feeds.bbci.co.uk/news/health/rss.xml",
}
LABELS = ["Politics", "Business", "Health"]
RSEED = 42

# Anchor to src/
SRC_DIR = Path(__file__).resolve().parent            # .../project-name/src

# Save *inside* src/
DATA_DIR    = SRC_DIR / "data"
MODELS_DIR  = SRC_DIR / "models"
REPORTS_DIR = SRC_DIR / "reports"

# Files (allow optional env override if you ever want)
DATASET_PATH = Path(os.getenv("DATASET_PATH", str(DATA_DIR / "task2_corpus.csv")))
MODEL_PATH   = Path(os.getenv("MODEL_PATH",   str(MODELS_DIR / "task2_model.joblib")))
CM_PATH      = Path(os.getenv("CM_PATH",      str(REPORTS_DIR / "task2_cm.png")))

def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
