# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SCAU NLP course (Spring 2026) homework assignments. Python 3.13, managed by **uv** (not pip).

## Common Commands

```bash
uv sync                          # Install dependencies
cd homework1 && python main.py search           # Grid search hyperparameters
cd homework1 && python main.py train            # Train with best/default config
cd homework1 && python main.py predict          # Generate submission.csv
cd homework1 && python main.py compare_features # Compare BoW vs N-gram vs TF-IDF
```

All commands run from the `homework1/` directory. Data CSVs (`train.csv`, `dev.csv`, `test.csv`) go in `homework1/data/` (gitignored). Results go in `homework1/results/` (also gitignored).

## Architecture (homework1)

SLU intent detection on Chinese text using a **from-scratch softmax regression model** (numpy only, no deep learning frameworks).

Pipeline flow: `main.py` → `data_loader` → `features` → `model` → `train` → `predict`

- **`src/data_loader.py`** — CSV loading, Chinese text cleaning (regex + jieba tokenization), stop word removal, `LabelEncoder` for category↔index mapping.
- **`src/features.py`** — `FeatureExtractor` wrapping sklearn's `CountVectorizer`/`TfidfVectorizer`. Three methods: `bow` (unigram), `ngram` (configurable range, default bigram), `tfidf`.
- **`src/model.py`** — `SoftmaxRegression` class with manual forward/backward passes. Supports `relu`, `leaky_relu`, `tanh` activations and `cross_entropy` / `mse` losses. Includes dropout and Xavier initialization.
- **`src/optimizer.py`** — `SGD` (with momentum) and `Adam` implementations.
- **`src/train.py`** — Training loop with early stopping, mini-batch support, and grid search over activation × loss × optimizer × lr × dropout.
- **`src/predict.py`** — Generates `submission.csv` from test predictions.

## Key Design Decisions

- All ML is numpy-only (no PyTorch/TensorFlow) — this is intentional for the course.
- Grid search space: 3 activations × 2 losses × 2 optimizers × 3 learning rates × 5 dropout values = 180 combinations.
- Default config when no `best_config.json` exists: relu + cross_entropy + adam, lr=0.01, dropout=0.1.
