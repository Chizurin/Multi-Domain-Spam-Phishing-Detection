# Multi-Domain Spam & Phishing Detection

COMP5415/COMP4415 Social Computing, UMass Lowell
Group: Andrew Belyea, Brandon Bui, Shane Cummings

**Core research question:** does a model trained on SMS data generalize to Discord without retraining?
The F1 difference between zero-shot (Phase 1) and retrained (Phase 2) Discord performance is the primary result.

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# 1. Download Super Dataset CSV from https://github.com/smspamresearch/spstudy
#    and copy it to data/raw/super_sms_dataset.csv

# 2. Load all datasets
python scripts/load_datasets.py

# 3. Clean text (language filtering, dedup, stop words, punctuation)
python scripts/preprocess.py
```

---

## Datasets

| Dataset | Rows | Spam % | Use |
|---|---|---|---|
| Super SMS Dataset | 67,018 | 39.1% | Train |
| Discord phishing-scam | 2,000 | 13.8% | Phase 1 test / Phase 2 train+test |
| PhiUSIIL (UCI #967) | 235,795 URLs | ~50% | Phishing detector (separate pipeline) |

`load_datasets.py` writes raw-normalized CSVs to `data/processed/`.
`preprocess.py` writes cleaned CSVs (`sms_text_cleaned.csv`, `discord_text_cleaned.csv`) to the same directory.

---
