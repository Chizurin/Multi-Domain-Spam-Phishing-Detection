"""
evaluate.py — evaluation pipeline for all runs

Usage:
  python eval/evaluate.py --run sms_only          # Run A: Discord gap (Phase 1)
  python eval/evaluate.py --run naive             # Run B: naive combined
  python eval/evaluate.py --run dann              # Run C: DANN
  python eval/evaluate.py --run all               # Runs A + B + C comparison table
  python eval/evaluate.py --run holdout           # Adversarial robustness on holdout set
  python eval/evaluate.py --run holdout --checkpoint naive
  python eval/evaluate.py --run errors            # Error analysis on Discord test set
  python eval/evaluate.py --run errors --checkpoint sms_only
"""

import argparse
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "models"))

from score import get_spam_score, load_spam_model
from behavioral import BEHAVIORAL_COLS

SPLITS = ROOT / "data" / "splits"
HOLDOUT_PATH = ROOT / "data" / "holdout.csv"
THRESHOLD = 0.5
RANDOM_SEED = 42

_URL_SUB_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def _load_behavioral_features(df: pd.DataFrame, checkpoint: str) -> "np.ndarray | None":
    """Return (N, N_BEHAVIORAL) array for a behavioral checkpoint, else None.

    Missing feature values are filled with the feature mean (maps to 0 in scaled space).
    A has_behavioral flag is appended: 1 if all features present, 0 if any missing.
    """
    ckpt_dir = ROOT / "checkpoints" / checkpoint
    scaler_path = ckpt_dir / "scaler.pkl"
    if not scaler_path.exists():
        return None
    import joblib
    scaler = joblib.load(scaler_path)
    n = len(df)
    features = np.zeros((n, len(BEHAVIORAL_COLS)), dtype=np.float32)
    for i, col in enumerate(BEHAVIORAL_COLS):
        if col in df.columns:
            features[:, i] = df[col].fillna(scaler.mean_[i]).values.astype(np.float32)
    scaled = scaler.transform(features).astype(np.float32)
    has_flag = df[BEHAVIORAL_COLS].notna().all(axis=1).values.astype(np.float32).reshape(-1, 1) \
        if all(c in df.columns for c in BEHAVIORAL_COLS) \
        else np.zeros((n, 1), dtype=np.float32)
    return np.hstack([scaled, has_flag])


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_discord_test() -> pd.DataFrame:
    path = SPLITS / "discord_test.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/preprocess.py first.")
    return pd.read_csv(path).dropna(subset=["text", "label"])


def load_holdout() -> pd.DataFrame:
    if not HOLDOUT_PATH.exists():
        raise FileNotFoundError(
            f"{HOLDOUT_PATH} not found.\n"
            "The holdout set is committed to the repo and must never be regenerated.\n"
            "Restore it with: git checkout data/holdout.csv"
        )
    return pd.read_csv(HOLDOUT_PATH)[["text", "label"]].dropna()


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _predict(texts: list[str], behavioral: "np.ndarray | None" = None, threshold: float = THRESHOLD) -> np.ndarray:
    preds = []
    for i, text in enumerate(texts):
        cleaned = _URL_SUB_RE.sub("URLTOKEN", str(text))
        bvec = behavioral[i] if behavioral is not None else None
        prob = get_spam_score(cleaned, behavioral=bvec)
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


# ---------------------------------------------------------------------------
# Evasion techniques
# ---------------------------------------------------------------------------

_SPAM_KEYWORDS = {
    "free", "win", "winner", "prize", "cash", "claim", "offer", "urgent",
    "congratulations", "selected", "reward", "bonus", "discount", "deal",
    "guaranteed", "apply", "subscribe", "click", "call", "text", "stop",
    "mobile", "ringtone", "tone", "loan", "credit", "debt", "income",
    "earn", "money", "£", "$", "voucher", "gift", "limited", "exclusive",
}

_HOMOGRAPH_MAP = {
    "a": "а", "e": "е", "o": "о", "p": "р", "c": "с",
    "x": "х", "y": "у", "i": "і", "A": "А", "E": "Е",
    "O": "О", "B": "В", "C": "С", "H": "Н", "K": "К",
    "M": "М", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
}

_LEET_MAP = {
    "a": "@", "i": "1", "l": "1", "o": "0", "e": "3",
    "s": "5", "t": "7", "b": "8", "g": "9", "z": "2",
}


def _apply_spacing(text: str) -> str:
    """Inject spaces between characters of known spam keywords."""
    words = text.split()
    result = []
    for word in words:
        if word.lower() in _SPAM_KEYWORDS:
            result.append(" ".join(word))
        else:
            result.append(word)
    return " ".join(result)


def _apply_charswap(text: str, seed: int = RANDOM_SEED) -> str:
    """Randomly swap one pair of adjacent characters per word (len >= 4)."""
    rng = random.Random(seed)
    words = text.split()
    result = []
    for word in words:
        if len(word) >= 4:
            i = rng.randint(1, len(word) - 2)
            chars = list(word)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            result.append("".join(chars))
        else:
            result.append(word)
    return " ".join(result)


def _apply_homograph(text: str) -> str:
    """Replace Latin characters with Cyrillic/Unicode lookalikes in spam keywords."""
    words = text.split()
    result = []
    for word in words:
        if word.lower() in _SPAM_KEYWORDS:
            result.append("".join(_HOMOGRAPH_MAP.get(c, c) for c in word))
        else:
            result.append(word)
    return " ".join(result)


def _apply_eda(text: str, seed: int = RANDOM_SEED) -> str:
    """EDA: random combination of word swap, deletion, and insertion."""
    rng = random.Random(seed)
    words = text.split()
    if len(words) < 2:
        return text

    # random swap
    if len(words) >= 2:
        i, j = rng.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]

    # random deletion (10% of words, at least 1 kept)
    n_delete = max(0, int(len(words) * 0.1))
    for _ in range(n_delete):
        if len(words) > 1:
            words.pop(rng.randint(0, len(words) - 1))

    # random insertion of a duplicate word
    if words:
        insert_word = rng.choice(words)
        insert_pos = rng.randint(0, len(words))
        words.insert(insert_pos, insert_word)

    return " ".join(words)


def _apply_paraphrase(text: str, seed: int = RANDOM_SEED) -> str:
    """Replace words with WordNet synonyms where available."""
    try:
        from nltk.corpus import wordnet
    except ImportError:
        return text

    rng = random.Random(seed)
    words = text.split()
    result = []
    for word in words:
        synsets = wordnet.synsets(word)
        synonyms = []
        for syn in synsets:
            for lemma in syn.lemmas():
                candidate = lemma.name().replace("_", " ")
                if candidate.lower() != word.lower() and " " not in candidate:
                    synonyms.append(candidate)
        if synonyms:
            result.append(rng.choice(synonyms))
        else:
            result.append(word)
    return " ".join(result)


def _apply_hybrid(text: str) -> str:
    """Combine spacing on spam keywords with leet-speak substitution on remaining chars."""
    words = text.split()
    result = []
    for word in words:
        if word.lower() in _SPAM_KEYWORDS:
            # spacing evasion
            result.append(" ".join(word))
        else:
            # leet substitution on non-keyword words
            result.append("".join(_LEET_MAP.get(c.lower(), c) for c in word))
    return " ".join(result)


_EVASION_TECHNIQUES = {
    "spacing":    _apply_spacing,
    "charswap":   _apply_charswap,
    "homograph":  _apply_homograph,
    "eda":        _apply_eda,
    "paraphrase": _apply_paraphrase,
    "hybrid":     _apply_hybrid,
}


# ---------------------------------------------------------------------------
# Holdout robustness evaluation
# ---------------------------------------------------------------------------

def evaluate_holdout(checkpoint: str = "naive") -> None:
    """Apply 6 evasion techniques to the holdout set and report detection rate per technique."""
    print(f"\nLoading checkpoint: {checkpoint}")
    load_spam_model(checkpoint)

    df = load_holdout()
    texts = df["text"].tolist()
    n = len(texts)

    # Baseline — no evasion
    baseline_preds = _predict(texts)
    baseline_detected = int(baseline_preds.sum())

    print(f"\n{'='*55}")
    print(f"  Adversarial Robustness — Holdout Set ({n} spam messages)")
    print(f"  Checkpoint: {checkpoint}")
    print(f"{'='*55}")
    print(f"  {'Technique':<14}  {'Detected':>8}  {'Evaded':>7}  {'Detection %':>11}  {'F1 drop':>8}")
    print(f"  {'-'*52}")
    print(f"  {'baseline':<14}  {baseline_detected:>8}  {n - baseline_detected:>7}  {baseline_detected/n*100:>10.1f}%  {'—':>8}")

    baseline_rate = baseline_detected / n

    for name, fn in _EVASION_TECHNIQUES.items():
        evaded_texts = [fn(t) for t in texts]
        preds = _predict(evaded_texts)
        detected = int(preds.sum())
        evaded = n - detected
        detection_rate = detected / n
        drop = baseline_rate - detection_rate
        print(
            f"  {name:<14}  {detected:>8}  {evaded:>7}  {detection_rate*100:>10.1f}%  {drop*100:>+7.1f}%"
        )

    print()


# ---------------------------------------------------------------------------
# Error analysis 
# ---------------------------------------------------------------------------

_DISCORD_TERMS = {
    "discord", "server", "channel", "dm", "mod", "moderator", "bot", "ping",
    "gg", "lol", "bruh", "ngl", "lmao", "tbh", "sus", "nitro", "boost",
    "guild", "role", "ban", "kick", "mute", "emoji", "react", "thread",
    "vc", "voice", "stream", "clip", "gaming", "game", "play", "gamer",
}

_LEET_PATTERN = re.compile(r"[0-9@$€][a-zA-Z]|[a-zA-Z][0-9@$€]")
_REPEAT_CHAR_PATTERN = re.compile(r"(.)\1{2,}")


def _categorize(text: str) -> list[str]:
    """Heuristically assign one or more failure-mode categories to a message."""
    categories = []
    words = str(text).lower().split()

    if len(words) <= 4:
        categories.append("short message")

    if any(w in _DISCORD_TERMS for w in words):
        categories.append("discord-specific slang")

    if _LEET_PATTERN.search(text) or _REPEAT_CHAR_PATTERN.search(text):
        categories.append("evasion tactics")

    # Behavioral signals: flagging messages that are ambiguous without account context.
    # These are short or neutral messages where join date / num_roles would resolve ambiguity.
    if len(words) <= 6 and not categories:
        categories.append("missing behavioral signals")

    if not categories:
        categories.append("other")

    return categories


def error_analysis(checkpoint: str = "naive", n_sample: int = 25) -> None:
    """Print sampled false positives and false negatives with heuristic failure-mode categories."""
    print(f"\nLoading checkpoint: {checkpoint}")
    load_spam_model(checkpoint)

    df = load_discord_test()
    texts = df["text"].tolist()
    y_true = df["label"].values

    probs = []
    for text in texts:
        cleaned = _URL_SUB_RE.sub("URLTOKEN", str(text))
        probs.append(get_spam_score(cleaned))
    probs = np.array(probs)
    y_pred = (probs >= THRESHOLD).astype(int)

    fp_idx = np.where((y_pred == 1) & (y_true == 0))[0]
    fn_idx = np.where((y_pred == 0) & (y_true == 1))[0]

    rng = random.Random(RANDOM_SEED)

    def _print_samples(label: str, indices: np.ndarray):
        sample = sorted(rng.sample(list(indices), min(n_sample, len(indices))))
        print(f"\n{'='*65}")
        print(f"  {label}  (n={len(indices)} total, showing {len(sample)})")
        print(f"  Checkpoint: {checkpoint}")
        print(f"{'='*65}")
        cat_counts: dict[str, int] = {}
        for i in sample:
            cats = _categorize(texts[i])
            for c in cats:
                cat_counts[c] = cat_counts.get(c, 0) + 1
            cat_str = ", ".join(cats)
            score_str = f"{probs[i]:.3f}"
            print(f"\n  [{score_str}] [{cat_str}]")
            print(f"  {str(texts[i])[:120]}")
        print(f"\n  Category breakdown ({len(sample)} messages):")
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
            print(f"    {cat:<30} {count:>3}  ({count/len(sample)*100:.0f}%)")

    _print_samples("FALSE POSITIVES — ham flagged as spam", fp_idx)
    _print_samples("FALSE NEGATIVES — spam that evaded detection", fn_idx)
    print()


# ---------------------------------------------------------------------------
# Discord test set evaluation
# ---------------------------------------------------------------------------

def evaluate_run(run: str) -> dict:
    trained_on = {
        "sms_only":       "SMS only",
        "naive":          "SMS + Discord",
        "dann":           "SMS + Discord + GRL",
        "behavioral":     "SMS + Discord + behavioral",
    }[run]

    print(f"\nLoading checkpoint: {run}")
    load_spam_model(run)

    df = load_discord_test()
    y_true = df["label"].values
    behavioral = _load_behavioral_features(df, run)
    y_pred = _predict(df["text"].tolist(), behavioral=behavioral)

    metrics = _scores(y_true, y_pred)
    _print_results(run, trained_on, metrics, y_true, y_pred)
    return {"run": run, "trained_on": trained_on, **metrics}


def results_table(rows: list[dict]):
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        required=True,
        choices=["sms_only", "naive", "dann", "behavioral", "all", "holdout", "errors"],
    )
    parser.add_argument(
        "--checkpoint",
        default="naive",
        choices=["sms_only", "naive", "dann", "behavioral"],
        help="Checkpoint to use for holdout/error evaluation (default: naive)",
    )
    args = parser.parse_args()

    if args.run == "errors":
        error_analysis(args.checkpoint)
    elif args.run == "holdout":
        evaluate_holdout(args.checkpoint)
    elif args.run == "all":
        rows = []
        for run in ["sms_only", "naive", "dann", "behavioral"]:
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
