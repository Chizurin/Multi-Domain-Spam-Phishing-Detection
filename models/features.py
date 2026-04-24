"""
features.py — RoBERTa tokenizer

Single entry point for all tokenization. Import and call tokenize(text) from
any script that needs tokenized input — never tokenize independently elsewhere.
"""

from functools import lru_cache

from transformers import RobertaTokenizer

MODEL_NAME = "roberta-base"
MAX_LENGTH = 512


@lru_cache(maxsize=1)
def _get_tokenizer() -> RobertaTokenizer:
    return RobertaTokenizer.from_pretrained(MODEL_NAME)


def tokenize(text: str) -> dict:
    """Tokenize a single message for RoBERTa.

    Returns a dict with input_ids and attention_mask as PyTorch tensors,
    ready to pass directly to the model.
    """
    return _get_tokenizer()(
        text,
        max_length=MAX_LENGTH,
        padding=False,
        truncation=True,
        return_tensors="pt",
    )
