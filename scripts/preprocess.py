"""
preprocess.py — text cleaning and train/test/holdout split pipeline

Input:  data/processed/super_dataset_clean.csv
        data/processed/discord_clean.csv
Output: data/processed/sms_text_cleaned.csv
        data/processed/discord_text_cleaned.csv
        data/holdout.csv            ← commit immediately, never regenerate
        data/splits/sms_train.csv
        data/splits/sms_test.csv
        data/splits/discord_train.csv
        data/splits/discord_test.csv

Run: python scripts/preprocess.py
"""

import re
import logging
from pathlib import Path

import nltk
import pandas as pd
from langdetect import detect, LangDetectException
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

nltk.download("stopwords", quiet=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PROCESSED = Path("data/processed")
SPLITS = Path("data/splits")
HOLDOUT_PATH = Path("data/holdout.csv")
RANDOM_STATE = 42
HOLDOUT_SIZE = 225

_STOP_WORDS = set(stopwords.words("english"))
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")
_PUNCT_RE = re.compile(r"[^\w\s$%@#]")
_MULTI_SPACE_RE = re.compile(r"\s+")
_REPEATED_CHARS_RE = re.compile(r"(.)\1{2,}")
_ALL_CAPS_RE = re.compile(r"\b[A-Z]{2,}\b")


def _googletrans_is_english(text: str) -> bool:
    """Fallback language check via deep-translator / Google (pass 2)."""
    try:
        from deep_translator import single_detection
        return single_detection(text, api_key=None) == "en"
    except Exception:
        return True  # keep on network/API failure


def is_english(text: str) -> bool:
    """Return True if text is English. langdetect pass 1, googletrans pass 2."""
    if not isinstance(text, str) or len(text.strip()) < 3:
        return True  # too short to detect reliably — keep
    try:
        return detect(text) == "en"
    except LangDetectException:
        # Ambiguous text (too short, mixed script) — use googletrans
        return _googletrans_is_english(text)


def clean_text(text: str) -> str:
    """Apply full text cleaning pipeline to a single message."""
    if not isinstance(text, str):
        return ""
    import emoji
    # Replace URLs before lowercasing so the regex matches cleanly
    text = _URL_RE.sub("URLTOKEN", text)
    # Convert emojis to text descriptions (e.g. 💰 → money_bag) before stripping non-ASCII
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = text.lower()
    # Remove remaining non-ASCII characters (non-emoji special Unicode)
    text = _NON_ASCII_RE.sub(" ", text)
    # Remove punctuation and special characters
    text = _PUNCT_RE.sub(" ", text)
    # Remove stop words
    tokens = [t for t in text.split() if t not in _STOP_WORDS]
    return _MULTI_SPACE_RE.sub(" ", " ".join(tokens)).strip()


def clean_df(df: pd.DataFrame, domain_name: str) -> pd.DataFrame:
    """Apply full cleaning pipeline to a DataFrame with a 'text' column."""
    before = len(df)

    df = df.dropna(subset=["text", "label"])
    df = df.drop_duplicates(subset=["text"])

    english_mask = df["text"].apply(is_english)
    dropped_lang = (~english_mask).sum()
    df = df[english_mask].copy()

    df["has_repeated_chars"] = df["text"].apply(
        lambda t: int(bool(_REPEATED_CHARS_RE.search(t))) if isinstance(t, str) else 0
    )
    df["has_all_caps"] = df["text"].apply(
        lambda t: int(bool(_ALL_CAPS_RE.search(t))) if isinstance(t, str) else 0
    )
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.strip().ne("")]

    after = len(df)
    log.info(
        f"[{domain_name}] {before:,} → {after:,} rows "
        f"(removed {before - after:,} total; {dropped_lang:,} non-English)"
    )
    return df.reset_index(drop=True)


def clean_sms() -> pd.DataFrame:
    src = PROCESSED / "super_dataset_clean.csv"
    df = pd.read_csv(src)
    df = clean_df(df, "sms")
    out = PROCESSED / "sms_text_cleaned.csv"
    df[["text", "label"]].to_csv(out, index=False)
    log.info(f"  → {out}")
    return df


def clean_discord() -> pd.DataFrame:
    src = PROCESSED / "discord_clean.csv"
    df = pd.read_csv(src)

    # capture every column that isn't text/label so none are silently dropped
    extra_cols = [c for c in df.columns if c not in ("text", "label")]

    df = clean_df(df, "discord")

    out = PROCESSED / "discord_text_cleaned.csv"
    save_cols = ["text", "label"] + [c for c in extra_cols if c in df.columns]
    df[save_cols].to_csv(out, index=False)
    log.info(f"  → {out}")
    return df


def make_holdout(sms_df: pd.DataFrame) -> pd.DataFrame:
    """Sample 225 spam messages from SMS for holdout. Generated once — never regenerated."""
    if HOLDOUT_PATH.exists():
        log.info(f"[holdout] already exists — skipping (delete manually to regenerate)")
        return pd.read_csv(HOLDOUT_PATH)

    spam = sms_df[sms_df["label"] == 1]
    holdout = spam.sample(n=HOLDOUT_SIZE, random_state=RANDOM_STATE)
    HOLDOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    holdout[["text", "label"]].to_csv(HOLDOUT_PATH, index=False)
    log.info(f"[holdout] {len(holdout)} spam messages → {HOLDOUT_PATH}  ← commit this now")
    return holdout


def split_sms(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 stratified split. Holdout messages excluded before splitting."""
    SPLITS.mkdir(parents=True, exist_ok=True)

    if HOLDOUT_PATH.exists():
        holdout_texts = set(pd.read_csv(HOLDOUT_PATH)["text"])
        df = df[~df["text"].isin(holdout_texts)]

    train, test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
    )
    train.to_csv(SPLITS / "sms_train.csv", index=False)
    test.to_csv(SPLITS / "sms_test.csv", index=False)
    log.info(f"[sms splits] train={len(train):,}  test={len(test):,}")
    return train, test


def split_discord(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 stratified split — kept separate from SMS for Phase 1/2 boundary."""
    SPLITS.mkdir(parents=True, exist_ok=True)

    train, test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
    )
    train.to_csv(SPLITS / "discord_train.csv", index=False)
    test.to_csv(SPLITS / "discord_test.csv", index=False)
    log.info(f"[discord splits] train={len(train):,}  test={len(test):,}")
    return train, test


if __name__ == "__main__":
    log.info("=== Preprocessing pipeline ===\n")

    sms = clean_sms()
    print()
    discord = clean_discord()
    print()

    make_holdout(sms)
    print()

    split_sms(sms)
    split_discord(discord)

    log.info("\nDone. Commit data/holdout.csv before anyone starts training.")
