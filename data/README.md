# Datasets

Only the Super SMS dataset needs to be downloaded manually and placed in `data/raw/`

## Super Dataset (SMS)
- Source: https://github.com/smspamresearch/spstudy
- File: `data/raw/super_sms_dataset.csv`
- 67,018 messages, 39.1% spam. Columns: `SMSes`, `Labels`
- Paper: Salman et al. IEEE Access 2024, https://doi.org/10.1109/ACCESS.2024.3364671

## Discord Phishing-Scam
- Source: HuggingFace `wangyuancheng/discord-phishing-scam`
- Downloaded automatically by `scripts/load_datasets.py`
- 2,000 messages, 13.8% spam. Note: label column is `lable` (typo in original).

## PhiUSIIL URL Features (UCI #967)
- Source: UCI ML Repository, fetched automatically via `ucimlrepo`
- 235,795 URLs, 87 features, ~50% phishing
- Paper: Prasad & Chandra 2024, https://doi.org/10.1016/j.cose.2023.103545
- Used for the phishing detector pipeline (separate from SMS spam classifier).

