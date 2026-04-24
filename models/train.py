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

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import DataCollatorWithPadding, RobertaForSequenceClassification, Trainer, TrainingArguments

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


class WeightedTrainer(Trainer):
    """Trainer subclass that applies per-class loss weights to CrossEntropyLoss."""

    def __init__(self, class_weights: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(outputs.logits.device)
        )
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


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

    data_collator = DataCollatorWithPadding(tokenizer=_get_tokenizer())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=SpamDataset(train_df),
        eval_dataset=SpamDataset(test_df),
        compute_metrics=compute_metrics,
        data_collator=data_collator,
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


def train_naive():
    out_dir = CHECKPOINTS / "naive"

    sms_train = pd.read_csv(SPLITS / "sms_train.csv")[["text", "label"]].dropna()
    discord_train = pd.read_csv(SPLITS / "discord_train.csv")[["text", "label"]].dropna()
    train_df = pd.concat([sms_train, discord_train], ignore_index=True)

    sms_test = pd.read_csv(SPLITS / "sms_test.csv")[["text", "label"]].dropna()
    discord_test = pd.read_csv(SPLITS / "discord_test.csv")[["text", "label"]].dropna()
    test_df = pd.concat([sms_test, discord_test], ignore_index=True)

    log.info(
        f"Train: {len(train_df):,} (SMS {len(sms_train):,} + Discord {len(discord_train):,})  "
        f"Test: {len(test_df):,}"
    )
    log.info(f"Train label dist: {train_df['label'].value_counts().to_dict()}")

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_df["label"].values,
    )
    class_weights = torch.tensor(weights, dtype=torch.float)
    log.info(f"Class weights — ham: {weights[0]:.4f}  spam: {weights[1]:.4f}")

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

    data_collator = DataCollatorWithPadding(tokenizer=_get_tokenizer())

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=SpamDataset(train_df),
        eval_dataset=SpamDataset(test_df),
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    _get_tokenizer().save_pretrained(str(out_dir))

    results = trainer.evaluate()
    log.info(
        f"\nCombined test — F1: {results['eval_f1']:.4f}  "
        f"P: {results['eval_precision']:.4f}  "
        f"R: {results['eval_recall']:.4f}"
    )


def train_phishing():
    from ucimlrepo import fetch_ucirepo

    out_path = CHECKPOINTS / "phishing_classifier.pkl"

    log.info("Loading PhiUSIIL dataset...")
    ds = fetch_ucirepo(id=967)
    X = ds.data.features.select_dtypes(include="number")
    y = ds.data.targets.iloc[:, 0].values

    log.info(f"Features: {X.shape[1]}  Samples: {len(y):,}")
    log.info(f"Label dist — ham: {(y == 0).sum():,}  phishing: {(y == 1).sum():,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    log.info(f"Train: {len(y_train):,}  Test: {len(y_test):,}")

    rf = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    log.info("Training Random Forest + calibrating (3-fold, isotonic)...")
    calibrated = CalibratedClassifierCV(rf, cv=3, method="isotonic")
    calibrated.fit(X_train, y_train)

    probs = calibrated.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    log.info(
        f"\nPhishing test — F1: {f1_score(y_test, preds):.4f}  "
        f"P: {precision_score(y_test, preds):.4f}  "
        f"R: {recall_score(y_test, preds):.4f}  "
        f"AUC: {roc_auc_score(y_test, probs):.4f}"
    )

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, out_path)
    log.info(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", required=True, choices=["sms_only", "naive", "dann", "phishing"]
    )
    args = parser.parse_args()

    if args.mode == "sms_only":
        train_sms_only()
    elif args.mode == "naive":
        train_naive()
    elif args.mode == "phishing":
        train_phishing()
    else:
        raise NotImplementedError(f"--mode {args.mode} not yet implemented")


if __name__ == "__main__":
    main()
