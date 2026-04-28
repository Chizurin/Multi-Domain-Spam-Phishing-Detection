"""
behavioral.py — BehavioralSpamClassifier

Extends RoBERTa with behavioral features (time_since_join, num_roles, etc.)
by concatenating them onto the CLS token before the classification head.

Missing value strategy:
  - Scaler is fit on complete Discord rows only (no NaN contamination)
  - NaN values are filled with the feature mean before scaling → maps to 0 in scaled space
  - SMS rows have no behavioral data → all feature dims set to 0 in scaled space
  - A `has_behavioral` flag (1 = full data present, 0 = missing/SMS) is appended
    so the model can distinguish "Discord with data" from "Discord missing" from "SMS"

Features used (text-redundant features excluded):
  - time_since_join  — account age in server, not inferable from text
  - num_roles        — server membership depth, not inferable from text
  - has_mention      — whether message mentions another user

Total input to classifier: 768 (CLS) + 3 (behavioral) + 1 (has_behavioral) = 772
"""

import torch
import torch.nn as nn
from transformers import RobertaModel

BEHAVIORAL_COLS = [
    "time_since_join",
    "num_roles",
    "has_mention",
]
N_BEHAVIORAL = len(BEHAVIORAL_COLS) + 1  # +1 for has_behavioral flag


class BehavioralSpamClassifier(nn.Module):
    """RoBERTa encoder with behavioral features concatenated onto the CLS token.

    Architecture:
      RoBERTa CLS (768) + behavioral (3) + has_behavioral (1)
        → Linear(772, 256) → ReLU → Dropout → Linear(256, 2)
    """

    def __init__(self, n_behavioral: int = N_BEHAVIORAL, encoder_path: str = "roberta-base"):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(encoder_path)
        self.classifier = nn.Sequential(
            nn.Linear(768 + n_behavioral, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        behavioral: torch.Tensor,
    ) -> torch.Tensor:
        cls = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[:, 0]
        combined = torch.cat([cls, behavioral.float()], dim=-1)
        return self.classifier(combined)
