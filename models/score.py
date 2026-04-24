"""
score.py — spam scoring pipeline

Usage:
  from score import load_spam_model, score

  load_spam_model("sms_only")   # or "naive" / "dann"
  result = score("Win a free iPhone! Claim now: https://bit.ly/scam")
"""

import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import torch

sys.path.insert(0, str(Path(__file__).parent))
from features import _get_tokenizer, tokenize

ROOT = Path(__file__).parent.parent
CHECKPOINTS = ROOT / "checkpoints"

THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# URL extraction
# ---------------------------------------------------------------------------

_URL_PATTERN = re.compile(
    r"https?://[^\s<>\"')\]]+|www\.[^\s<>\"')\]]+",
    re.IGNORECASE,
)

URL_SHORTENERS = {
    "bit.ly",
    "goo.gl",
    "tinyurl.com",
    "t.co",
    "ow.ly",
    "buff.ly",
    "is.gd",
    "rb.gy",
    "short.link",
    "tiny.cc",
    "shorte.st",
    "adf.ly",
}

# Must match the substitution applied during training (preprocess.py)
_URL_SUB_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def extract_urls(text: str) -> list[str]:
    """Return all URLs found in text."""
    return _URL_PATTERN.findall(text)


def is_shortened(url: str) -> bool:
    """Return True if the URL uses a known shortener domain."""
    try:
        host = urlparse(url).netloc.lower().removeprefix("www.")
        return host in URL_SHORTENERS
    except Exception:
        return False


def flag_shortened_urls(text: str) -> list[str]:
    """Return any URLs in text that use a known shortener domain."""
    return [url for url in extract_urls(text) if is_shortened(url)]


# ---------------------------------------------------------------------------
# Model state (populated by load_spam_model)
# ---------------------------------------------------------------------------

_spam_model = None
_spam_device = None
_spam_is_dann = False


def load_spam_model(checkpoint: str) -> None:
    """Load a spam classifier checkpoint by name or path.

    Args:
        checkpoint: run name ("sms_only", "naive", "dann") or an absolute path.
    """
    global _spam_model, _spam_device, _spam_is_dann

    ckpt_dir = Path(checkpoint) if Path(checkpoint).is_absolute() else CHECKPOINTS / checkpoint
    _spam_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (ckpt_dir / "model.pt").exists():
        from dann import DANNSpamClassifier
        model = DANNSpamClassifier()
        model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=_spam_device))
        _spam_is_dann = True
    else:
        from transformers import RobertaForSequenceClassification
        model = RobertaForSequenceClassification.from_pretrained(str(ckpt_dir))
        _spam_is_dann = False

    model.to(_spam_device)
    model.eval()
    _spam_model = model


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def get_spam_score(text: str) -> float:
    """Return probability (0–1) that text is spam."""
    if _spam_model is None:
        raise RuntimeError("Call load_spam_model(checkpoint) before scoring.")

    enc = tokenize(text)
    input_ids = enc["input_ids"].to(_spam_device)
    attention_mask = enc["attention_mask"].to(_spam_device)

    with torch.no_grad():
        if _spam_is_dann:
            logits, _ = _spam_model(input_ids, attention_mask)
        else:
            logits = _spam_model(input_ids=input_ids, attention_mask=attention_mask).logits

    probs = torch.softmax(logits, dim=-1)
    return probs[0, 1].item()


def score(text: str, threshold: float = THRESHOLD) -> dict:
    """Score a message for spam.

    URLs are replaced with URLTOKEN before scoring to match training distribution.

    Returns:
        spam_score  (float) — probability the text is spam (0–1)
        flagged     (bool)  — True if spam_score >= threshold
    """
    cleaned = _URL_SUB_RE.sub("URLTOKEN", text)
    spam = get_spam_score(cleaned)
    return {
        "spam_score": round(spam, 4),
        "flagged": spam >= threshold,
    }
