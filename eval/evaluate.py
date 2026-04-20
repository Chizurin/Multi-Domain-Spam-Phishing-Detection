"""
evaluate.py — evaluation and results for all runs

Holdout set is loaded directly from data/holdout.csv.
It is committed to the repo and must never be regenerated.
"""

from pathlib import Path

import pandas as pd

HOLDOUT_PATH = Path("data/holdout.csv")
SPLITS_PATH = Path("data/splits")


def load_holdout() -> pd.DataFrame:
    if not HOLDOUT_PATH.exists():
        raise FileNotFoundError(
            f"{HOLDOUT_PATH} not found.\n"
            "The holdout set is committed to the repo and must never be regenerated.\n"
            "Restore it with: git checkout data/holdout.csv"
        )
    return pd.read_csv(HOLDOUT_PATH)


def load_discord_test() -> pd.DataFrame:
    path = SPLITS_PATH / "discord_test.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/preprocess.py first.")
    return pd.read_csv(path)
