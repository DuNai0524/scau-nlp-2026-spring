"""Tiny DeBERTa model for 5-shot SLU intent detection.

Simplified DeBERTa architecture: standard multi-head self-attention with learnable
relative position bias, no disentangled attention. 2 layers, 128 hidden dim.
"""

import math

import torch
import torch.nn as nn

from .data_loader import PAD_ID


class RelativePositionBias(nn.Module):
    """Learnable relative position bias for self-attention."""

    def __init__(self, max_seq_len: int, num_heads: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        # Positions range from -(max_seq_len-1) to +(max_seq_len-1)
        self.relative_bias = nn.Embedding(2 * max_seq_len - 1, num_heads)

    def forward(self, qlen: int, klen: int, device: torch.device) -> torch.Tensor:
        # position indices: query_pos - key_pos
        q_pos = torch.arange(qlen, device=device).unsqueeze(1)
        k_pos = torch.arange(klen, device=device).unsqueeze(0)
        rel_pos = q_pos - k_pos  # (qlen, klen), range [-(klen-1), qlen-1]
        # Shift to [0, 2*max_seq_len-2]
        rel_pos = rel_pos + self.max_seq_len - 1
        rel_pos = rel_pos.clamp(0, 2 * self.max_seq_len - 2)
        # (qlen, klen, num_heads) -> (num_heads, qlen, klen)
        bias = self.relative_bias(rel_pos).permute(2, 0, 1)
        return bias


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention with relative position bias."""

    def __init__(self, hidden_dim: int, num_heads: int, max_seq_len: int, dropout: float):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.rel_pos_bias = RelativePositionBias(max_seq_len, num_heads)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # (B, heads, S, head_dim) @ (B, heads, head_dim, S) -> (B, heads, S, S)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        scores = scores + self.rel_pos_bias(S, S, x.device)

        if mask is not None:
            # mask: (B, S), True = real token, False = pad
            # expand to (B, 1, 1, S) so it broadcasts over heads and query positions
            pad_mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~pad_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)  # (B, heads, S, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-LN transformer block: LN -> Attention -> residual -> LN -> FFN -> residual."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadSelfAttention(hidden_dim, num_heads, max_seq_len, dropout)
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Self-attention with residual
        h = self.attn_norm(x)
        x = x + self.attn_dropout(self.attn(h, mask))
        # FFN with residual
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TinyDeBERTa(nn.Module):
    """Tiny DeBERTa for text classification.

    Pipeline: Embedding(200d) -> Linear(200->128) -> TransformerBlocks x2
              -> MeanPool -> Dropout -> Linear(128->num_classes)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        max_seq_len: int,
        max_length: int,
        dropout: float,
        pretrained_embeddings: torch.Tensor | None = None,
    ):
        super().__init__()
        self.max_length = max_length

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        self.proj = nn.Linear(embed_dim, hidden_dim)
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ffn_dim, max_seq_len, dropout)
            for _ in range(num_layers)
        ])

        self.pool_norm = nn.LayerNorm(hidden_dim)
        self.classifier_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, S), padding with PAD_ID
        mask = input_ids != PAD_ID  # (B, S), True = real token

        x = self.embedding(input_ids)  # (B, S, embed_dim)
        x = self.proj(x)  # (B, S, hidden_dim)
        x = self.embed_dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.pool_norm(x)

        # Mean pooling over non-pad tokens
        mask_expanded = mask.unsqueeze(-1).float()  # (B, S, 1)
        x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)  # (B, hidden_dim)

        x = self.classifier_dropout(x)
        logits = self.classifier(x)  # (B, num_classes)
        return logits
