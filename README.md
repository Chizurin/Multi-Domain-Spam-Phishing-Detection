# Multi-Domain Spam & Phishing Detection

COMP5415/COMP4415 Social Computing, UMass Lowell
Group: Andrew Belyea, Brandon Bui, Shane Cummings

**Core research question:** does a model trained on SMS data generalize to Discord without retraining?
The F1 difference between zero-shot (Phase 1) and retrained (Phase 2) Discord performance is the primary result.

---

## Quick start

```bash
uv venv && source .venv/bin/activate
uv pip sync requirements.lock

# 1. Download Super Dataset CSV from https://github.com/smspamresearch/spstudy
#    and copy it to data/raw/super_sms_dataset.csv

# 2. Load all datasets
python scripts/load_datasets.py

# 3. Clean text, generate splits, and create holdout set
python scripts/preprocess.py
# Then immediately: git add data/holdout.csv && git commit
```

> To update dependencies: edit `requirements.txt`, run `uv pip compile requirements.txt -o requirements.lock`, and commit the updated lockfile.

---

## Checkpoints

Trained checkpoints are hosted on HuggingFace Hub. Use the scripts below to download them (skip training) or upload your own after training.

Valid run names for both scripts: `sms_only`, `naive`, `dann`.

### Downloading checkpoints (skip training)

```bash
# Download all checkpoints
python scripts/download_checkpoints.py --username <hf-username>

# Download specific runs only
python scripts/download_checkpoints.py --username <hf-username> --runs sms_only naive
```

No login required. Files are saved directly to the paths the rest of the project expects:

| Run | Saved to |
|---|---|
| `sms_only` | `checkpoints/sms_only/` |
| `naive` | `checkpoints/naive/` |
| `dann` | `checkpoints/dann/` |

### Uploading checkpoints

```bash
# One-time login
huggingface-cli login

# Upload all trained runs
python scripts/upload_checkpoints.py --username your-hf-username

# Upload specific runs only
python scripts/upload_checkpoints.py --username your-hf-username --runs sms_only naive
```

The script skips any run whose checkpoint directory doesn't exist yet.

---

## Datasets

| Dataset | Rows | Spam % | Use |
|---|---|---|---|
| Super SMS Dataset | 67,018 | 39.1% | Train |
| Discord phishing-scam | 2,000 | 13.8% | Phase 1 test / Phase 2 train+test |

`load_datasets.py` writes raw-normalized CSVs to `data/processed/`.
`preprocess.py` writes cleaned CSVs (`sms_text_cleaned.csv`, `discord_text_cleaned.csv`) to the same directory.

---
