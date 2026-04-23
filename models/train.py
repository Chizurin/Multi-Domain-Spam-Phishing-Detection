"""
train.py — model training pipeline

Commands:
  python models/train.py --mode sms_only   → Run A: RoBERTa on SMS only
  python models/train.py --mode naive      → Run B: RoBERTa on SMS + Discord
  python models/train.py --mode dann       → Run C: domain-adversarial training
  python models/train.py --mode phishing   → URL phishing classifier (Random Forest)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).parent))
from features import _get_tokenizer, tokenize

ROOT = Path(__file__).parent.parent
SPLITS = ROOT / "data" / "splits"
CHECKPOINTS = ROOT / "checkpoints"
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


class SpamDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        self.texts = df["text"].fillna("").tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = tokenize(self.texts[idx])
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "f1": f1_score(labels, preds, average="binary", zero_division=0),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall": recall_score(labels, preds, average="binary", zero_division=0),
    }


def train_sms_only():
    out_dir = CHECKPOINTS / "sms_only"

    train_df = pd.read_csv(SPLITS / "sms_train.csv")[["text", "label"]].dropna()
    test_df = pd.read_csv(SPLITS / "sms_test.csv")[["text", "label"]].dropna()
    log.info(f"SMS train: {len(train_df):,}  test: {len(test_df):,}")

    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        optim="adamw_torch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=RANDOM_STATE,
        logging_steps=100,
        dataloader_num_workers=0,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=SpamDataset(train_df),
        eval_dataset=SpamDataset(test_df),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    _get_tokenizer().save_pretrained(str(out_dir))

    results = trainer.evaluate()
    log.info(
        f"\nSMS test — F1: {results['eval_f1']:.4f}  "
        f"P: {results['eval_precision']:.4f}  "
        f"R: {results['eval_recall']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", required=True, choices=["sms_only", "naive", "dann", "phishing"]
    )
    args = parser.parse_args()

    if args.mode == "sms_only":
        train_sms_only()
    else:
        raise NotImplementedError(f"--mode {args.mode} not yet implemented")


if __name__ == "__main__":
    main()
