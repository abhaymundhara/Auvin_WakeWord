from __future__ import annotations

import torch
import torch.nn as nn


class ConvWakeHead(nn.Module):
    def __init__(self, input_dim: int = 96, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 16, 96] -> conv expects [B, C, T]
        x = x.transpose(1, 2)
        x = self.net(x).squeeze(-1)
        return self.classifier(x).squeeze(-1)


def spec_augment(x: torch.Tensor, time_mask: int = 4, feat_mask: int = 8) -> torch.Tensor:
    out = x.clone()
    b, t, f = out.shape
    for i in range(b):
        if t > time_mask:
            t0 = torch.randint(0, t - time_mask, (1,)).item()
            out[i, t0 : t0 + time_mask, :] = 0
        if f > feat_mask:
            f0 = torch.randint(0, f - feat_mask, (1,)).item()
            out[i, :, f0 : f0 + feat_mask] = 0
    return out
