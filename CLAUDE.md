# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SCAU NLP course (Spring 2026) homework assignments. Python 3.12, managed by **uv** (not pip).

## Common Commands

```bash
uv sync                          # Install dependencies
cd homework1 && python main.py search           # Grid search hyperparameters
cd homework1 && python main.py train            # Train with best/default config
cd homework1 && python main.py predict          # Generate submission.csv
cd homework1 && python main.py compare_features # Compare BoW vs N-gram vs TF-IDF
cd homework2 && python main.py train            # Train DADGNN model
cd homework2 && python main.py predict          # Generate submission.csv
```

Homework1 commands run from `homework1/`, homework2 from `homework2/`. Data CSVs are gitignored.

## Architecture (homework1)

SLU intent detection on Chinese text using a **from-scratch softmax regression model** (numpy only, no deep learning frameworks).

Pipeline flow: `main.py` → `data_loader` → `features` → `model` → `train` → `predict`

- **`src/data_loader.py`** — CSV loading, Chinese text cleaning (regex + jieba tokenization), stop word removal, `LabelEncoder` for category↔index mapping.
- **`src/features.py`** — `FeatureExtractor` wrapping sklearn's `CountVectorizer`/`TfidfVectorizer`. Three methods: `bow` (unigram), `ngram` (configurable range, default bigram), `tfidf`.
- **`src/model.py`** — `SoftmaxRegression` class with manual forward/backward passes. Supports `relu`, `leaky_relu`, `tanh` activations and `cross_entropy` / `mse` losses. Includes dropout and Xavier initialization.
- **`src/optimizer.py`** — `SGD` (with momentum) and `Adam` implementations.
- **`src/train.py`** — Training loop with early stopping, mini-batch support, and grid search over activation × loss × optimizer × lr × dropout.
- **`src/predict.py`** — Generates `submission.csv` from test predictions.

## Architecture (homework2)

SLU intent detection on Chinese text using **DADGNN** (Attention Diffusion GNN, EMNLP 2021). Replaces BERT fine-tuning with a graph neural network approach.

Pipeline flow: `main.py` → `data_loader` → `graph_utils` → `model` → `train` → `predict`

- **`src/data_loader.py`** — CSV loading, jieba tokenization, vocabulary building, Chinese Word2Vec embedding loading (gensim), `GraphTextDataset`.
- **`src/graph_utils.py`** — Document-to-DGL-graph conversion: token deduplication as nodes, n-gram sliding window edges, batch graph construction.
- **`src/model.py`** — `SingleHeadGATLayer` (k-step attention diffusion), `GATLayer` (multi-head), `GATNet` (stacked layers), `WeightAndSum` (graph readout), `DADGNNModel` (full pipeline).
- **`src/train.py`** — Native PyTorch training loop with Adam, CrossEntropyLoss, early stopping. Builds vocab + loads embeddings + trains DADGNN.
- **`src/predict.py`** — Loads saved model + vocab, runs batch inference, generates `submission.csv`.

## Key Design Decisions

- Homework1: All ML is numpy-only (no PyTorch/TensorFlow) — intentional for the course.
- Homework2: Uses DGL + PyTorch for graph neural network. Graphs built dynamically in forward pass.
- DGL 2.2.0 requires PyTorch <2.4 (graphbolt C++ lib compatibility). Patched `graphbolt/__init__.py` to gracefully skip missing C++ libs.
- Chinese Word2Vec embeddings at `homework2/data/sgns.merge.word` (gitignored). Falls back to random init if absent.
- Python version pinned to 3.12 for DGL compatibility (DGL has no cp313 wheels).
