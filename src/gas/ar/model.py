import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

from ml_collections import ConfigDict


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
        d_model = ar_config.d_model
        nhead = ar_config.nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.attn_dropout_p = ar_config.dropout

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ar_config.dim_ff),
            nn.ReLU(),
            nn.Dropout(ar_config.dropout),
            nn.Linear(ar_config.dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.dropout = nn.Dropout(ar_config.dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), self.nhead, self.head_dim).permute(1, 0, 2)

    def forward(
        self,
        last_embed: torch.Tensor,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        last_embed : [1, d_model]   — the most recently appended embed (query source)
        k_cache    : [i, d_model]   — K projections for all previous embeds, or None
        v_cache    : [i, d_model]   — V projections for all previous embeds, or None

        Returns (output [1, d_model], new_k_cache [i+1, d_model], new_v_cache [i+1, d_model]).
        K/V for last_embed are computed here once and appended to the cache.
        """
        q, new_k, new_v = self.qkv_proj(last_embed).chunk(3, dim=-1)

        k = torch.cat([k_cache, new_k], dim=0) if k_cache is not None else new_k
        v = torch.cat([v_cache, new_v], dim=0) if v_cache is not None else new_v 

        q_mh = self._split_heads(q)
        k_mh = self._split_heads(k)
        v_mh = self._split_heads(v)

        dropout_p = self.attn_dropout_p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q_mh, k_mh, v_mh, dropout_p=dropout_p)

        attn_out = attn_out.permute(1, 0, 2).reshape(1, self.d_model)
        attn_out = self.out_proj(attn_out)

        x = self.norm1(last_embed + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))

        return x, k, v


class ARModel(nn.Module):
    def __init__(self, ar_config: ConfigDict):
        super().__init__()

        self.config = ar_config
        self.learn_correctors = getattr(ar_config, 'learn_correctors', False)
        self.corrector_order = getattr(ar_config, 'corrector_order', 1)

        out_dim = 1 + 2 * self.corrector_order if self.learn_correctors else 1

        self.init_embedding = nn.Parameter(torch.randn(1, ar_config.d_model) * sqrt(0.02), requires_grad=True)
        self.pe = LearnedLengthAwarePE(self.config)
        self.decoder = IncrementalDecoderLayer(self.config)
        self.mlp = nn.Linear(ar_config.d_model, out_dim)

        if self.learn_correctors:
            with torch.no_grad():
                self.mlp.weight[1:].zero_()
                self.mlp.bias[1:].zero_()

    def forward(self, num_steps: int):
        device = next(self.parameters()).device
        pos_encodings = self.pe(num_steps, device=device)
        pe_scale, pe_bias = torch.chunk(pos_encodings, 2, dim=-1)

        last_embed = self.init_embedding * pe_scale[:1] + pe_bias[:1]
        all_embeds = [last_embed]
        k_cache: torch.Tensor | None = None
        v_cache: torch.Tensor | None = None

        for i in range(1, num_steps):
            new_embed, k_cache, v_cache = self.decoder(last_embed, k_cache, v_cache)
            last_embed = new_embed * pe_scale[i:i+1] + pe_bias[i:i+1]
            all_embeds.append(last_embed)

        embeds = torch.cat(all_embeds, dim=0)
        out = self.mlp(embeds)
        if not self.learn_correctors:
            return out.squeeze(-1)
        return out