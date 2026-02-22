import torch
import torch.nn as nn
from math import sqrt

from ml_collections import ConfigDict

import torch
import torch.nn as nn


class LearnedLengthAwarePE(nn.Module):
    def __init__(self, ar_config: ConfigDict):
        super().__init__()
        hidden_dim = ar_config.d_model
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, ar_config.d_model * 2)
        )

    def forward(self, seq_len, device="cuda"):
        pos = torch.arange(start=1, end=seq_len + 1, device=device).float()
        p = pos / seq_len
        features = torch.stack([p, torch.sin(p * torch.pi), torch.cos(p * torch.pi)], dim=-1)
        return self.mlp(features)

class IncrementalDecoderLayer(nn.Module):
    def __init__(self, ar_config: ConfigDict):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            ar_config.d_model,
            ar_config.nhead,
            dropout=ar_config.dropout,
        )

        self.ffn = nn.Sequential(
            nn.Linear(ar_config.d_model, ar_config.dim_ff),
            nn.ReLU(),
            nn.Dropout(ar_config.dropout),
            nn.Linear(ar_config.dim_ff, ar_config.d_model),
        )

        self.norm1 = nn.LayerNorm(ar_config.d_model)
        self.norm2 = nn.LayerNorm(ar_config.d_model, elementwise_affine=False)
        self.dropout = nn.Dropout(ar_config.dropout)

    def forward(self, embeds):
        q = embeds[-1:, :]
        attn_out, _ = self.self_attn(q, embeds, embeds)
        x = self.norm1(q + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x  # (B,1,D)

class ARModel(nn.Module):
    def __init__(self, ar_config: ConfigDict):
        super().__init__()

        self.config = ar_config
        self.init_embedding = nn.Parameter(torch.randn(1, ar_config.d_model) * sqrt(0.02), requires_grad=True)
        self.pe = LearnedLengthAwarePE(self.config)
        self.decoder = IncrementalDecoderLayer(self.config)
        self.mlp = nn.Linear(ar_config.d_model, 1)

    def forward(self, num_steps: int):
        pos_encodings = self.pe(num_steps)
        pos_encodings_scale, pos_encodings_bias = torch.chunk(pos_encodings, 2, dim=-1)
        embeds = self.init_embedding * pos_encodings_scale[:1] + pos_encodings_bias[:1]
        for i in range(1, num_steps):
            new_embed = self.decoder(embeds) * pos_encodings_scale[i:i+1] + pos_encodings_bias[i:i+1]
            embeds = torch.cat([embeds, new_embed], dim=0)
        # embeds [num_steps, d_model]
        return self.mlp(embeds).squeeze(-1) # [num_steps,]