"""
evaluate.py — evaluation pipeline for all runs

Usage:
  python eval/evaluate.py --run sms_only     # Run A: Discord gap (Phase 1)
  python eval/evaluate.py --run naive        # Run B: naive combined
  python eval/evaluate.py --run dann         # Run C: DANN
  python eval/evaluate.py --run all          # Runs A + B + C comparison table
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "models"))

from score import get_spam_score, load_spam_model

SPLITS = ROOT / "data" / "splits"
HOLDOUT_PATH = ROOT / "data" / "holdout.csv"
THRESHOLD = 0.5


def load_discord_test() -> pd.DataFrame:
    path = SPLITS / "discord_test.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/preprocess.py first.")
    return pd.read_csv(path)[["text", "label"]].dropna()


def load_holdout() -> pd.DataFrame:
    if not HOLDOUT_PATH.exists():
        raise FileNotFoundError(
            f"{HOLDOUT_PATH} not found.\n"
            "The holdout set is committed to the repo and must never be regenerated.\n"
            "Restore it with: git checkout data/holdout.csv"
        )
    return pd.read_csv(HOLDOUT_PATH)[["text", "label"]].dropna()


def _predict(texts: list[str], threshold: float = THRESHOLD) -> np.ndarray:
    """Run get_spam_score on each text and threshold into 0/1 predictions."""
    preds = []
    for text in texts:
        cleaned = str(text)
        # match training distribution — URLs → URLTOKEN
        import re
        cleaned = re.sub(r"https?://\S+|www\.\S+", "URLTOKEN", cleaned, flags=re.IGNORECASE)
        prob = get_spam_score(cleaned)
        preds.append(1 if prob >= threshold else 0)
    return np.array(preds)


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
    }


def _print_results(run: str, trained_on: str, metrics: dict, y_true: np.ndarray, y_pred: np.ndarray):
    print(f"\n{'='*55}")
    print(f"  Run: {run}   Trained on: {trained_on}")
    print(f"{'='*55}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion matrix (Discord test set):")
    print(f"              Predicted")
    print(f"              ham   spam")
    print(f"  Actual ham  {tn:4d}   {fp:4d}")
    print(f"  Actual spam {fn:4d}   {tp:4d}")
    print()


def evaluate_run(run: str) -> dict:
    """Load checkpoint, run against Discord test split, print and return metrics."""
    trained_on = {
        "sms_only": "SMS only",
        "naive":    "SMS + Discord",
        "dann":     "SMS + Discord + GRL",
    }[run]

    print(f"\nLoading checkpoint: {run}")
    load_spam_model(run)

    df = load_discord_test()
    y_true = df["label"].values
    y_pred = _predict(df["text"].tolist())

    metrics = _scores(y_true, y_pred)
    _print_results(run, trained_on, metrics, y_true, y_pred)
    return {"run": run, "trained_on": trained_on, **metrics}


def results_table(rows: list[dict]):
    """Print the core results comparison table."""
    print("\n" + "="*55)
    print("  Core Results — Discord Test Set")
    print("="*55)
    header = f"  {'Run':<10}  {'Trained on':<22}  {'F1':>6}  {'P':>6}  {'R':>6}"
    print(header)
    print("  " + "-"*51)
    for r in rows:
        print(
            f"  {r['run']:<10}  {r['trained_on']:<22}  "
            f"{r['f1']:>6.4f}  {r['precision']:>6.4f}  {r['recall']:>6.4f}"
        )
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        required=True,
        choices=["sms_only", "naive", "dann", "all"],
        help="Which checkpoint to evaluate, or 'all' for comparison table",
    )
    args = parser.parse_args()

    if args.run == "all":
        rows = []
        for run in ["sms_only", "naive", "dann"]:
            ckpt = ROOT / "checkpoints" / run
            if not ckpt.exists():
                print(f"[{run}] Skipping — checkpoint not found at {ckpt}")
                continue
            rows.append(evaluate_run(run))
        if rows:
            results_table(rows)
    else:
        evaluate_run(args.run)


if __name__ == "__main__":
    main()
