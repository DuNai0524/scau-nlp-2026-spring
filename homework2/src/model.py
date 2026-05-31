"""DADGNN model: Attention Diffusion GNN for text classification."""

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax

from .graph_utils import batch_graphs


class SingleHeadGATLayer(nn.Module):
    """Single-head GAT layer with k-step attention diffusion (PPR-style)."""

    def __init__(self, in_dim: int, out_dim: int, k: int = 5, alpha: float = 0.5):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.W = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.a_l = nn.Parameter(torch.FloatTensor(out_dim, 1))
        self.a_r = nn.Parameter(torch.FloatTensor(out_dim, 1))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_l)
        nn.init.xavier_uniform_(self.a_r)

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        feat = h @ self.W

        with g.local_scope():
            g.ndata["el"] = feat @ self.a_l
            g.ndata["er"] = feat @ self.a_r
            g.apply_edges(fn.u_add_v("el", "er", "e"))
            attention = edge_softmax(g, g.edata["e"])

            z = feat.clone()
            for _ in range(self.k):
                g.ndata["z"] = z
                g.edata["a"] = attention
                g.update_all(fn.u_mul_e("z", "a", "m"), fn.sum("m", "z_new"))
                z = self.alpha * feat + (1 - self.alpha) * g.ndata["z_new"]

        return z


class GATLayer(nn.Module):
    """Multi-head GAT layer with mean merge."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 2, k: int = 5, alpha: float = 0.5):
        super().__init__()
        self.heads = nn.ModuleList([
            SingleHeadGATLayer(in_dim, out_dim, k, alpha)
            for _ in range(num_heads)
        ])

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(g, h) for head in self.heads], dim=0).mean(dim=0)


class GATNet(nn.Module):
    """Stacked GAT layers with attention diffusion."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 5,
        num_heads: int = 2,
        k: int = 5,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATLayer(in_dim, hidden_dim, num_heads, k, alpha))
        for _ in range(num_layers - 2):
            self.layers.append(GATLayer(hidden_dim, hidden_dim, num_heads, k, alpha))
        self.layers.append(GATLayer(hidden_dim, out_dim, num_heads=1, k=k, alpha=alpha))

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            h = F.elu(layer(g, h))
        h = self.layers[-1](g, h)
        return h


class WeightAndSum(nn.Module):
    """Batch-aware graph-level readout: weighted sum of node features per graph."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.weight_gate = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, bg: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        with bg.local_scope():
            weights = self.weight_gate(h)
            bg.ndata["wh"] = h * weights
            return dgl.readout_nodes(bg, "wh", op="sum")


class DADGNNModel(nn.Module):
    """DADGNN: Document graph classification with Attention Diffusion GNN.

    Pipeline: Embedding → Linear(300→64)+ReLU → GATNet(64→64) → WeightAndSum → Linear(64→34) → logits
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        num_hidden: int = 64,
        num_classes: int = 34,
        num_layers: int = 5,
        num_heads: int = 2,
        k: int = 5,
        alpha: float = 0.5,
        ngram: int = 4,
        max_length: int = 350,
        dropout: float = 0.5,
        pretrained_embeddings: torch.Tensor | None = None,
    ):
        super().__init__()
        self.ngram = ngram
        self.max_length = max_length

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, num_hidden),
            nn.ReLU(),
        )
        self.gat = GATNet(num_hidden, num_hidden, num_hidden, num_layers, num_heads, k, alpha)
        self.readout = WeightAndSum(num_hidden)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_hidden, num_classes),
        )

    def forward(self, doc_ids_list: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        # Build node features: embedding → input_proj → hidden dim
        node_hidden = self.input_proj(self.embedding.weight)  # (vocab_size, num_hidden)

        # Build batched graph using hidden-dim features
        bg = batch_graphs(doc_ids_list, node_hidden, self.ngram, self.max_length, device)
        bg = bg.to(device)

        # GNN propagation in hidden space
        h = bg.ndata["h"]  # (total_nodes, num_hidden)
        out = self.gat(bg, h)  # (total_nodes, num_hidden)

        # Graph-level readout → (batch_size, num_hidden)
        graph_repr = self.readout(bg, out)

        # Classify → (batch_size, num_classes)
        logits = self.classifier(graph_repr)
        return logits
