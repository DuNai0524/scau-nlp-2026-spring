"""TextCNN model for SLU intent detection (Kim 2014)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        filter_sizes: list[int] | None = None,
        num_filters: int = 64,
        dropout: float = 0.5,
        pretrained_embeddings: torch.Tensor | None = None,
        freeze_embeddings: bool = True,
    ):
        super().__init__()
        if filter_sizes is None:
            filter_sizes = [2, 3, 4]

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in filter_sizes
        ])

        total_filters = num_filters * len(filter_sizes)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(total_filters, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, seq_len)
        x = self.embedding(input_ids)          # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)                # (batch, embed_dim, seq_len)

        conv_outs = []
        for conv in self.convs:
            h = F.relu(conv(x))                # (batch, num_filters, seq_len - k + 1)
            h = h.max(dim=2).values            # (batch, num_filters)
            conv_outs.append(h)

        x = torch.cat(conv_outs, dim=1)        # (batch, num_filters * len(filter_sizes))
        x = self.dropout(x)
        logits = self.fc(x)                    # (batch, num_classes)
        return logits
