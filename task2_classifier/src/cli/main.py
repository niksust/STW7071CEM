import argparse
from ..config import ensure_dirs, DATASET_PATH, MODEL_PATH, CM_PATH, FEEDS
from ..data.fetch import collect_corpus
from ..data.io import save_csv, load_csv
from ..models.train import train_and_evaluate
from ..models.predict import load_model, predict_text

def _parse_args():
    p = argparse.ArgumentParser(description="Task 2 — BBC Subject Classification")
    p.add_argument("mode", choices=["train", "predict"], help="train/evaluate or predict")
    p.add_argument("--dataset", default=DATASET_PATH, help="CSV dataset path")
    p.add_argument("--model",   default=MODEL_PATH,   help="Model path")
    p.add_argument("--cm",      default=CM_PATH,      help="Confusion matrix image path")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split (0–1)")
    p.add_argument("--no-fetch", action="store_true",
                   help="(train) Skip RSS fetch and reuse existing dataset")
    p.add_argument("--text", help="Text to classify (predict mode). Omit for interactive loop.")
    return p.parse_args()

def cmd_train(args):
    ensure_dirs()
    if args.no_fetch:
        df = load_csv(args.dataset)
    else:
        print("Fetching BBC RSS…")
        df = collect_corpus(FEEDS)
        print("Class counts:", df["label"].value_counts().to_dict())
        save_csv(df, args.dataset)
        print(f"Saved dataset → {args.dataset}")
    train_and_evaluate(df, test_size=args.test_size, model_out=args.model, cm_out=args.cm)

def cmd_predict(args):
    ensure_dirs()
    model = load_model(args.model)
    if args.text:
        print(predict_text(model, args.text))
        return
    print("Enter text to classify (blank line to quit):")
    while True:
        try:
            s = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye"); break
        if not s:
            print("bye"); break
        print(predict_text(model, s))

def main():
    args = _parse_args()
    if args.mode == "train":
        cmd_train(args)
    else:
        cmd_predict(args)

if __name__ == "__main__":
    main()