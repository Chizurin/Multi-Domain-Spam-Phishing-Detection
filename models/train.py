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
import math
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
        labels = inputs["labels"]
        outputs = model(**inputs)
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(outputs.logits.device)
        )
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


class DANNDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        self.texts = df["text"].fillna("").tolist()
        self.labels = df["label"].tolist()
        self.domains = df["domain"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = tokenize(self.texts[idx])
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "domain_labels": torch.tensor(self.domains[idx], dtype=torch.long),
        }


def _dann_collate(batch):
    """Dynamic padding collate that preserves domain_labels alongside span labels."""
    pad_id = _get_tokenizer().pad_token_id
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = torch.stack([
        torch.nn.functional.pad(item["input_ids"], (0, max_len - item["input_ids"].size(0)), value=pad_id)
        for item in batch
    ])
    attention_mask = torch.stack([
        torch.nn.functional.pad(item["attention_mask"], (0, max_len - item["attention_mask"].size(0)), value=0)
        for item in batch
    ])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.stack([item["labels"] for item in batch]),
        "domain_labels": torch.stack([item["domain_labels"] for item in batch]),
    }


def _evaluate_dann(model, test_df: pd.DataFrame, device: torch.device):
    loader = torch.utils.data.DataLoader(
        DANNDataset(test_df), batch_size=64, shuffle=False, collate_fn=_dann_collate
    )
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            spam_logits, _ = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            all_preds.extend(spam_logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(batch["labels"].tolist())
    return (
        f1_score(all_labels, all_preds, average="binary", zero_division=0),
        precision_score(all_labels, all_preds, average="binary", zero_division=0),
        recall_score(all_labels, all_preds, average="binary", zero_division=0),
    )


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

    # UCI Phishing Websites (id=327) — 10 features computable from URL string alone.
    # All encoded as -1 (phishing) / 0 (suspicious) / 1 (legitimate).
    # Features requiring WHOIS, page fetch, or external APIs are excluded.
    url_features = [
        "having_ip_address",       # -1 if IP address used as domain
        "url_length",              # 1 if <54, 0 if 54-75, -1 if >75
        "shortining_service",      # -1 if known URL shortener
        "having_at_symbol",        # -1 if @ present in URL
        "double_slash_redirecting",# -1 if // appears after position 7
        "prefix_suffix",           # -1 if - present in domain
        "having_sub_domain",       # 1=no subdomain, 0=one, -1=many
        "sslfinal_state",          # 1=HTTPS, 0=suspicious, -1=HTTP
        "port",                    # -1 if non-standard port used
        "https_token",             # -1 if 'https' appears in domain name
    ]

    log.info("Loading UCI Phishing Websites dataset (id=327)...")
    ds = fetch_ucirepo(id=327)
    X = ds.data.features[url_features]
    # result: 1=legitimate, -1=phishing → remap to 0=legitimate, 1=phishing
    y = (ds.data.targets.iloc[:, 0].values == -1).astype(int)

    log.info(f"Features: {X.shape[1]}  Samples: {len(y):,}")
    log.info(f"Label dist — legitimate: {(y == 0).sum():,}  phishing: {(y == 1).sum():,}")

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


def train_dann():
    from dann import DANNSpamClassifier

    out_dir = CHECKPOINTS / "dann"
    out_dir.mkdir(parents=True, exist_ok=True)

    sms_train = pd.read_csv(SPLITS / "sms_train.csv")[["text", "label", "domain"]].dropna()
    discord_train = pd.read_csv(SPLITS / "discord_train.csv")[["text", "label", "domain"]].dropna()
    train_df = pd.concat([sms_train, discord_train], ignore_index=True)

    sms_test = pd.read_csv(SPLITS / "sms_test.csv")[["text", "label", "domain"]].dropna()
    discord_test = pd.read_csv(SPLITS / "discord_test.csv")[["text", "label", "domain"]].dropna()
    test_df = pd.concat([sms_test, discord_test], ignore_index=True)

    log.info(f"Train: {len(train_df):,}  Test: {len(test_df):,}")
    log.info(f"Train label dist:  {train_df['label'].value_counts().to_dict()}")
    log.info(f"Train domain dist: {train_df['domain'].value_counts().to_dict()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    spam_w = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_df["label"].values)
    domain_w = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_df["domain"].values)
    log.info(f"Spam weights   — ham: {spam_w[0]:.4f}  spam: {spam_w[1]:.4f}")
    log.info(f"Domain weights — SMS: {domain_w[0]:.4f}  Discord: {domain_w[1]:.4f}")

    spam_criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(spam_w, dtype=torch.float).to(device)
    )
    domain_criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(domain_w, dtype=torch.float).to(device)
    )

    NUM_EPOCHS = 3
    BATCH_SIZE = 16
    LR = 2e-5
    LOG_EVERY = 100

    train_loader = torch.utils.data.DataLoader(
        DANNDataset(train_df),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=_dann_collate,
    )

    model = DANNSpamClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    total_steps = NUM_EPOCHS * len(train_loader)
    global_step = 0
    best_f1 = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch in train_loader:
            progress = global_step / total_steps
            lambda_ = 2 / (1 + math.exp(-10 * progress)) - 1
            model.grl.lambda_ = lambda_

            spam_logits, domain_logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )

            spam_loss = spam_criterion(spam_logits, batch["labels"].to(device))
            domain_loss = domain_criterion(domain_logits, batch["domain_labels"].to(device))
            total_loss = spam_loss + lambda_ * domain_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if global_step % LOG_EVERY == 0:
                log.info(
                    f"step {global_step:>6}/{total_steps} | epoch {epoch + 1}/{NUM_EPOCHS} | "
                    f"λ={lambda_:.4f} | spam_loss={spam_loss.item():.4f} | "
                    f"domain_loss={domain_loss.item():.4f}"
                )

            global_step += 1

        f1, precision, recall = _evaluate_dann(model, test_df, device)
        log.info(f"\nEpoch {epoch + 1} eval — F1: {f1:.4f}  P: {precision:.4f}  R: {recall:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), out_dir / "model.pt")
            log.info(f"  → Saved best model (F1={best_f1:.4f})")

    _get_tokenizer().save_pretrained(str(out_dir))
    log.info(f"\nTraining complete — best F1: {best_f1:.4f}  Checkpoint → {out_dir}")


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
    elif args.mode == "dann":
        train_dann()
    else:
        raise NotImplementedError(f"--mode {args.mode} not yet implemented")


if __name__ == "__main__":
    main()
