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
_spam_is_behavioral = False
_spam_scaler = None


def load_spam_model(checkpoint: str) -> None:
    """Load a spam classifier checkpoint by name or path.

    Args:
        checkpoint: run name ("sms_only", "naive", "dann", "behavioral") or absolute path.
    """
    global _spam_model, _spam_device, _spam_is_dann, _spam_is_behavioral, _spam_scaler

    ckpt_dir = Path(checkpoint) if Path(checkpoint).is_absolute() else CHECKPOINTS / checkpoint
    _spam_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (ckpt_dir / "scaler.pkl").exists():
        import joblib
        from behavioral import BehavioralSpamClassifier, N_BEHAVIORAL
        model = BehavioralSpamClassifier(N_BEHAVIORAL)
        model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=_spam_device))
        _spam_scaler = joblib.load(ckpt_dir / "scaler.pkl")
        _spam_is_behavioral = True
        _spam_is_dann = False
    elif (ckpt_dir / "model.pt").exists():
        from dann import DANNSpamClassifier
        model = DANNSpamClassifier()
        model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=_spam_device))
        _spam_is_dann = True
        _spam_is_behavioral = False
        _spam_scaler = None
    else:
        from transformers import RobertaForSequenceClassification
        model = RobertaForSequenceClassification.from_pretrained(str(ckpt_dir))
        _spam_is_dann = False
        _spam_is_behavioral = False
        _spam_scaler = None

    model.to(_spam_device)
    model.eval()
    _spam_model = model


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def get_spam_score(text: str, behavioral: "np.ndarray | None" = None) -> float:
    """Return probability (0–1) that text is spam.

    Args:
        text:       message text (URLs already substituted with URLTOKEN)
        behavioral: optional array of shape (N_BEHAVIORAL,) for the behavioral
                    checkpoint. Ignored for sms_only/naive/dann. Uses zeros if None.
    """
    if _spam_model is None:
        raise RuntimeError("Call load_spam_model(checkpoint) before scoring.")

    enc = tokenize(text)
    input_ids = enc["input_ids"].to(_spam_device)
    attention_mask = enc["attention_mask"].to(_spam_device)

    with torch.no_grad():
        if _spam_is_behavioral:
            import numpy as np
            from behavioral import N_BEHAVIORAL
            if behavioral is None:
                bvec = np.zeros(N_BEHAVIORAL, dtype=np.float32)
            else:
                bvec = np.asarray(behavioral, dtype=np.float32)
            bvec_t = torch.tensor(bvec, device=_spam_device).unsqueeze(0)
            logits = _spam_model(input_ids, attention_mask, bvec_t)
        elif _spam_is_dann:
            logits, _ = _spam_model(input_ids, attention_mask)
        else:
            logits = _spam_model(input_ids=input_ids, attention_mask=attention_mask).logits

    probs = torch.softmax(logits, dim=-1)
    return probs[0, 1].item()


def score(text: str, behavioral: "dict | None" = None, threshold: float = THRESHOLD) -> dict:
    """Score a message for spam.

    URLs are replaced with URLTOKEN before scoring to match training distribution.

    Args:
        text:       raw message text
        behavioral: optional dict of behavioral features for the behavioral checkpoint,
                    e.g. {"time_since_join": 5e6, "num_roles": 2, ...}.
                    Ignored for sms_only/naive/dann checkpoints.

    Returns:
        spam_score  (float) — probability the text is spam (0–1)
        flagged     (bool)  — True if spam_score >= threshold
    """
    import numpy as np
    cleaned = _URL_SUB_RE.sub("URLTOKEN", text)

    bvec = None
    if _spam_is_behavioral:
        from behavioral import BEHAVIORAL_COLS
        if behavioral is not None:
            # Use feature mean for any missing keys (maps to 0 in scaled space)
            raw = np.array(
                [behavioral.get(c, _spam_scaler.mean_[i]) for i, c in enumerate(BEHAVIORAL_COLS)],
                dtype=np.float32,
            )
            scaled = _spam_scaler.transform(raw.reshape(1, -1))[0].astype(np.float32)
            has_flag = np.array([1.0], dtype=np.float32)
        else:
            # No behavioral context (SMS or unknown) — all zeros + flag = 0
            scaled = np.zeros(len(BEHAVIORAL_COLS), dtype=np.float32)
            has_flag = np.array([0.0], dtype=np.float32)
        bvec = np.concatenate([scaled, has_flag])

    spam = get_spam_score(cleaned, behavioral=bvec)
    return {
        "spam_score": round(spam, 4),
        "flagged": spam >= threshold,
    }
