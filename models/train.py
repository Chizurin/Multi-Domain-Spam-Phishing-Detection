"""
train.py — model training pipeline

Commands:
  python models/train.py --mode sms_only    → Run A: RoBERTa on SMS only
  python models/train.py --mode naive       → Run B: RoBERTa on SMS + Discord
  python models/train.py --mode dann        → Run C: domain-adversarial training
  python models/train.py --mode behavioral  → Run D: RoBERTa + behavioral features
"""

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
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


def train_behavioral(encoder_path: str = "roberta-base", out_name: str = "behavioral"):
    import joblib
    from sklearn.preprocessing import StandardScaler

    from behavioral import BEHAVIORAL_COLS, N_BEHAVIORAL, BehavioralSpamClassifier

    out_dir = CHECKPOINTS / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    sms_train = pd.read_csv(SPLITS / "sms_train.csv")[["text", "label"]].dropna()
    sms_test  = pd.read_csv(SPLITS / "sms_test.csv")[["text", "label"]].dropna()
    discord_train = pd.read_csv(SPLITS / "discord_train.csv")
    discord_test  = pd.read_csv(SPLITS / "discord_test.csv")

    # Fit scaler on complete Discord training rows only — no NaN contamination
    complete_mask = discord_train[BEHAVIORAL_COLS].notna().all(axis=1)
    scaler = StandardScaler()
    scaler.fit(discord_train.loc[complete_mask, BEHAVIORAL_COLS])
    joblib.dump(scaler, out_dir / "scaler.pkl")
    log.info(f"Scaler fit on {complete_mask.sum()} complete Discord rows → {out_dir / 'scaler.pkl'}")

    def _prepare_behavioral(df, has_data: bool) -> np.ndarray:
        """Return (N, N_BEHAVIORAL) array with scaling and has_behavioral flag.

        Missing features → filled with feature mean → maps to 0.0 in scaled space.
        has_data=False (SMS) → all feature dims 0.0, has_behavioral flag = 0.
        """
        raw_cols = BEHAVIORAL_COLS  # the 8 raw feature columns (flag appended below)
        n = len(df)
        features = np.zeros((n, len(raw_cols)), dtype=np.float32)
        if has_data:
            for i, col in enumerate(raw_cols):
                if col in df.columns:
                    # fill NaN with feature mean so they map to 0 in scaled space
                    filled = df[col].fillna(scaler.mean_[i]).values.astype(np.float32)
                    features[:, i] = filled
            features = scaler.transform(features).astype(np.float32)
            # has_behavioral flag: 1 if all raw features were present, else 0
            has_flag = df[raw_cols].notna().all(axis=1).values.astype(np.float32).reshape(-1, 1)
        else:
            # SMS — no behavioral data at all, flag = 0
            has_flag = np.zeros((n, 1), dtype=np.float32)
        return np.hstack([features, has_flag])

    discord_train_b = _prepare_behavioral(discord_train, has_data=True)
    discord_test_b  = _prepare_behavioral(discord_test,  has_data=True)
    sms_train_b     = _prepare_behavioral(sms_train,     has_data=False)
    sms_test_b      = _prepare_behavioral(sms_test,      has_data=False)

    keep = ["text", "label"]
    train_df = pd.concat([sms_train[keep], discord_train[keep]], ignore_index=True)
    test_df  = pd.concat([sms_test[keep],  discord_test[keep]],  ignore_index=True)
    train_behavioral_arr = np.vstack([sms_train_b, discord_train_b])
    test_behavioral_arr  = np.vstack([sms_test_b,  discord_test_b])

    log.info(f"Train: {len(train_df):,}  Test: {len(test_df):,}")
    log.info(f"Train label dist: {train_df['label'].value_counts().to_dict()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_df["label"].values)
    log.info(f"Class weights — ham: {class_weights[0]:.4f}  spam: {class_weights[1]:.4f}")
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float).to(device)
    )

    NUM_EPOCHS = 3
    BATCH_SIZE = 16
    LR = 2e-5
    LOG_EVERY = 100

    class BehavioralDataset(torch.utils.data.Dataset):
        def __init__(self, df, behavioral_arr):
            self.texts      = df["text"].tolist()
            self.labels     = df["label"].tolist()
            self.behavioral = behavioral_arr

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return {
                "text":       self.texts[idx],
                "label":      self.labels[idx],
                "behavioral": self.behavioral[idx],
            }

    def _behavioral_collate(batch):
        tokenizer   = _get_tokenizer()
        texts       = [b["text"] for b in batch]
        labels      = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        behavioral  = torch.tensor(np.array([b["behavioral"] for b in batch]), dtype=torch.float32)
        enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels":         labels,
            "behavioral":     behavioral,
        }

    def _evaluate_behavioral(model, df, behavioral_arr, device):
        model.eval()
        all_preds, all_labels = [], []
        loader = torch.utils.data.DataLoader(
            BehavioralDataset(df, behavioral_arr), batch_size=32, collate_fn=_behavioral_collate
        )
        with torch.no_grad():
            for batch in loader:
                logits = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["behavioral"].to(device),
                )
                preds = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].tolist())
        return (
            f1_score(all_labels, all_preds, zero_division=0),
            precision_score(all_labels, all_preds, zero_division=0),
            recall_score(all_labels, all_preds, zero_division=0),
        )

    train_loader = torch.utils.data.DataLoader(
        BehavioralDataset(train_df, train_behavioral_arr),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=_behavioral_collate,
    )

    model = BehavioralSpamClassifier(N_BEHAVIORAL, encoder_path=encoder_path).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    total_steps = NUM_EPOCHS * len(train_loader)
    global_step = 0
    best_f1 = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch in train_loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["behavioral"].to(device),
            )
            loss = criterion(logits, batch["labels"].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % LOG_EVERY == 0:
                log.info(
                    f"step {global_step:>6}/{total_steps} | epoch {epoch + 1}/{NUM_EPOCHS} | "
                    f"loss={loss.item():.4f}"
                )
            global_step += 1

        f1, precision, recall = _evaluate_behavioral(model, test_df, test_behavioral_arr, device)
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
        "--mode", required=True, choices=["sms_only", "naive", "dann", "behavioral"]
    )
    args = parser.parse_args()

    if args.mode == "sms_only":
        train_sms_only()
    elif args.mode == "naive":
        train_naive()
    elif args.mode == "dann":
        train_dann()
    elif args.mode == "behavioral":
        train_behavioral()
    else:
        raise NotImplementedError(f"--mode {args.mode} not yet implemented")


if __name__ == "__main__":
    main()
