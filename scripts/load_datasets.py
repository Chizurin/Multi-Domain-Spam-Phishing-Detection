"""
load_datasets.py — load and normalize raw datasets into data/processed/

Outputs
-------
data/processed/super_dataset_clean.csv   columns: text, label (0=ham, 1=spam)
data/processed/discord_clean.csv         columns: text, label, time_since_join, has_link, message_length

Run: python scripts/load_datasets.py
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)


# ── 1. Super Dataset (SMS) ────────────────────────────────────────────────────
def load_super_dataset() -> pd.DataFrame:
    raw_path = Path("data/raw/super_sms_dataset.csv")
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found.\n"
            "Download from https://github.com/smspamresearch/spstudy and copy the CSV there."
        )

    df = pd.read_csv(raw_path, encoding="latin-1", engine="python")
    df = df.rename(columns={"SMSes": "text", "Labels": "label"})
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    out = PROCESSED / "super_dataset_clean.csv"
    df[["text", "label"]].to_csv(out, index=False)
    print(f"[super]   {len(df):,} rows → {out}  (spam={df['label'].mean():.1%})")
    return df


# ── 2. Discord phishing-scam ──────────────────────────────────────────────────
def load_discord_dataset() -> pd.DataFrame:
    hf = load_dataset("wangyuancheng/discord-phishing-scam", split="train")
    df = hf.to_pandas()

    df = df.rename(columns={"lable": "label", "msg_content": "text"})
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].str.replace(r"[\r\n]+", " ", regex=True)

    keep = ["text", "label", "msg_timestamp", "usr_joined_at", "time_since_join",
            "message_length", "word_count", "has_link", "has_mention", "num_roles"]
    keep = [c for c in keep if c in df.columns]

    out = PROCESSED / "discord_clean.csv"
    df[keep].to_csv(out, index=False)
    print(f"[discord] {len(df):,} rows → {out}  (spam={df['label'].mean():.1%})")
    return df[keep]


if __name__ == "__main__":
    print("Loading datasets...\n")
    load_super_dataset()
    load_discord_dataset()
    print("\nDone. Processed files written to data/processed/")
