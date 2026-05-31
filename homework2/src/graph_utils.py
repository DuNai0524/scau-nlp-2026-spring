"""Document-to-DGL-graph conversion utilities for DADGNN."""

import dgl
import torch


def doc_to_graph(
    doc_ids: list[int] | torch.Tensor,
    node_hidden: torch.Tensor,
    ngram: int = 4,
    max_length: int = 350,
    device: torch.device | str = "cpu",
) -> dgl.DGLGraph:
    """Convert a single document's token IDs into a DGL graph.

    Steps: truncate → deduplicate tokens as nodes → n-gram sliding window edges
    → look up node features from embedding matrix.
    """
    if isinstance(doc_ids, torch.Tensor):
        doc_ids = doc_ids.tolist()

    doc_ids = doc_ids[:max_length]

    # Deduplicate while preserving order
    seen: set[int] = set()
    unique_ids: list[int] = []
    for tid in doc_ids:
        if tid not in seen:
            seen.add(tid)
            unique_ids.append(tid)

    num_nodes = len(unique_ids)
    if num_nodes == 0:
        g = dgl.graph((torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
        g.ndata["h"] = torch.zeros(0, node_hidden.shape[1], device=device)
        return g

    # Build node id -> graph node index mapping
    id_to_idx = {tid: i for i, tid in enumerate(unique_ids)}

    # N-gram sliding window edges
    src_list: list[int] = []
    dst_list: list[int] = []
    for i in range(len(doc_ids)):
        for j in range(i + 1, min(i + ngram, len(doc_ids))):
            src_idx = id_to_idx[doc_ids[i]]
            dst_idx = id_to_idx[doc_ids[j]]
            if src_idx != dst_idx:
                src_list.append(src_idx)
                dst_list.append(dst_idx)

    src = torch.tensor(src_list, dtype=torch.long)
    dst = torch.tensor(dst_list, dtype=torch.long)
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g = dgl.add_self_loop(g)

    # Node features from embedding lookup
    node_indices = torch.tensor(unique_ids, dtype=torch.long)
    g.ndata["h"] = node_hidden[node_indices].to(device)

    return g


def batch_graphs(
    doc_ids_list: list[list[int] | torch.Tensor],
    node_hidden: torch.Tensor,
    ngram: int = 4,
    max_length: int = 350,
    device: torch.device | str = "cpu",
) -> dgl.DGLGraph:
    """Build a batched DGL graph from multiple documents."""
    graphs = [
        doc_to_graph(ids, node_hidden, ngram, max_length, device)
        for ids in doc_ids_list
    ]
    return dgl.batch(graphs)
