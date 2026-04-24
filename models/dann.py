"""
dann.py — Domain-Adversarial Neural Network components

DANNSpamClassifier:
  - Shared RoBERTa encoder (CLS token, 768-dim)
  - Spam head: linear → ReLU → dropout → linear (2 classes)
  - Domain head: GRL → linear → ReLU → dropout → linear (2 classes)

Reference: Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from transformers import RobertaModel


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(torch.tensor(lambda_))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (lambda_,) = ctx.saved_tensors
        return -lambda_ * grad_output, None


class GRL(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


class ClassificationHead(nn.Module):
    """Two linear layers with ReLU and dropout between them."""

    def __init__(self, hidden_size: int = 768, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class DANNSpamClassifier(nn.Module):
    def __init__(self, encoder_name: str = "roberta-base", dropout: float = 0.1):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for roberta-base

        self.spam_head = ClassificationHead(hidden_size, num_labels=2, dropout=dropout)
        self.grl = GRL(lambda_=1.0)
        self.domain_head = ClassificationHead(hidden_size, num_labels=2, dropout=dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]

        spam_logits = self.spam_head(cls)
        domain_logits = self.domain_head(self.grl(cls))

        return spam_logits, domain_logits
