"""Vocabulary building and Word2Vec embedding loading for Homework 2-b."""

import os

import jieba
import numpy as np

from .data_loader import PAD_ID, PAD_TOKEN, UNK_ID, UNK_TOKEN, tokenize_chinese


def build_vocab(documents: list[str], min_freq: int = 1) -> tuple[dict[str, int], dict[int, str]]:
    freq: dict[str, int] = {}
    for doc in documents:
        for token in tokenize_chinese(doc):
            freq[token] = freq.get(token, 0) + 1

    word2id = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    idx = len(word2id)
    for word, count in sorted(freq.items()):
        if count >= min_freq:
            word2id[word] = idx
            idx += 1

    id2word = {v: k for k, v in word2id.items()}
    return word2id, id2word


def encode_document(tokens: list[str], word2id: dict[str, int], max_length: int) -> list[int]:
    ids = [word2id.get(t, UNK_ID) for t in tokens]
    return ids[:max_length]


def load_chinese_embeddings(path: str, word2id: dict[str, int], embed_dim: int = 200) -> np.ndarray:
    """Load Chinese Word2Vec embeddings. Uncovered words are randomly initialized."""
    vocab_size = len(word2id)
    embeddings = np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.1
    embeddings[PAD_ID] = np.zeros(embed_dim, dtype=np.float32)

    if not os.path.exists(path):
        print(f"  Embedding file not found: {path}. Using random initialization.")
        return embeddings

    print(f"  Loading embeddings from: {path}")
    found = 0

    try:
        from gensim.models import KeyedVectors

        binary = path.endswith(".bin")
        if path.endswith(".kv"):
            kv = KeyedVectors.load(path, mmap="r")
        else:
            kv = KeyedVectors.load_word2vec_format(path, binary=binary)

        actual_dim = kv.vector_size
        if actual_dim != embed_dim:
            print(f"  Note: embedding dim ({actual_dim}) != requested dim ({embed_dim}). Using {actual_dim}.")
            embeddings = np.random.randn(vocab_size, actual_dim).astype(np.float32) * 0.1
            embeddings[PAD_ID] = np.zeros(actual_dim, dtype=np.float32)

        for word, idx in word2id.items():
            if word in kv:
                embeddings[idx] = kv[word]
                found += 1
    except (ValueError, UnicodeDecodeError):
        print("  Gensim load failed, trying raw text parse...")
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) < embed_dim + 1:
                    continue
                word = parts[0]
                if word in word2id:
                    try:
                        vec = np.array([float(x) for x in parts[1 : embed_dim + 1]], dtype=np.float32)
                        embeddings[word2id[word]] = vec
                        found += 1
                    except ValueError:
                        continue

    print(f"  Covered {found}/{vocab_size} words from pre-trained embeddings.")
    return embeddings
