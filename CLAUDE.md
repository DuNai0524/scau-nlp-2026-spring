# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SCAU NLP course (Spring 2026) homework assignments. Python 3.12, managed by **uv** (not pip). Each homework solves the same Chinese SLU intent-detection task (34 classes) with a different modeling paradigm, sharing the dataset under `nlp-text-classification-experiments/`.

Shared dataset (gitignored, used by all homeworks):
- `train_new_5shot.csv` — 162 samples, 5 per class
- `dev_new.csv` — 3200 samples (validation / early stopping)
- `kaggle_test.csv` — 4000 samples (submission targets)

## Common Commands

```bash
uv sync                          # Install dependencies for the workspace

# homework1 — run from homework1/
python main.py search            # Grid search hyperparameters
python main.py train             # Train with best/default config
python main.py predict           # Generate submission.csv
python main.py compare_features  # Compare BoW vs N-gram vs TF-IDF

# homework2 — run from homework2/
python main.py train             # Train DADGNN model
python main.py predict           # Generate submission.csv

# homework2-b — run from homework2-b/
python main.py train             # Train Tiny DeBERTa
python main.py predict           # Generate submission.csv

# homework2-c — run from homework2-c/
python main.py train             # Train TextCNN
python main.py predict           # Generate submission.csv

# homework3 — run from homework3/
python main.py eval --mode zero-shot     # Evaluate zero-shot on dev set
python main.py eval --mode few-shot      # Evaluate few-shot on dev set
python main.py compare                   # Compare zero-shot vs few-shot
python main.py predict --mode few-shot   # Generate submission.csv

# homework4 — run from homework4/ (intended for an RTX 3090 box, not local)
bash setup.sh                    # One-time env install
./run.sh                         # Full pipeline: prepare → train → predict
./run.sh prepare | train | eval | predict   # Individual stages
```

Each homework's commands run from its own directory. All `submission.csv`, `data/`, `results/`, and `*.pt` outputs are gitignored.

## Architecture (homework1) — numpy-only softmax regression

From-scratch SLU intent detection (no PyTorch/TensorFlow — intentional course constraint).

Pipeline: `main.py` → `data_loader` → `features` → `model` → `train` → `predict`

- **`src/data_loader.py`** — CSV loading, regex + jieba cleaning, stop word removal, `LabelEncoder`.
- **`src/features.py`** — `FeatureExtractor` over sklearn `CountVectorizer`/`TfidfVectorizer`. Methods: `bow`, `ngram` (configurable range), `tfidf`.
- **`src/model.py`** — `SoftmaxRegression` with manual forward/backward. Activations: `relu`/`leaky_relu`/`tanh`. Losses: `cross_entropy`/`mse`. Dropout + Xavier init.
- **`src/optimizer.py`** — `SGD` (with momentum) and `Adam`.
- **`src/train.py`** — Mini-batch training, early stopping, grid search over activation × loss × optimizer × lr × dropout.
- **`src/predict.py`** — Generates `submission.csv`.

## Architecture (homework2) — DADGNN (graph neural network)

Attention-Diffusion GNN (EMNLP 2021) on document-as-graph representation.

Pipeline: `main.py` → `data_loader` → `graph_utils` → `model` → `train` → `predict`

- **`src/data_loader.py`** — CSV loading, jieba tokenization, vocab building, Chinese Word2Vec loading (gensim), `GraphTextDataset`.
- **`src/graph_utils.py`** — Document → DGL graph: deduped tokens become nodes, n-gram sliding window edges, batched graph construction.
- **`src/model.py`** — `SingleHeadGATLayer` (k-step attention diffusion), `GATLayer` (multi-head), `GATNet` (stack), `WeightAndSum` (readout), `DADGNNModel` (full pipeline).
- **`src/train.py`** — PyTorch loop with Adam, CrossEntropyLoss, early stopping. Builds vocab + loads embeddings + trains.
- **`src/predict.py`** — Loads model + vocab, batch inference, generates `submission.csv`.

## Architecture (homework2-b) — Tiny DeBERTa from scratch

Transformer encoder trained from scratch on 162 samples to compare against DADGNN. Severe-overfitting regime → tiny capacity + heavy regularization.

Configuration: 2 layers, 128 hidden, 2 heads, FFN 512 (~1M params). Standard self-attention + **learnable relative position bias** (no disentangled attention — too many params for 162 samples). Input = jieba + Word2Vec 200d → linear projection to 128d. Mean-pool head → dropout → linear. Dropout 0.5, weight_decay 1e-3, label_smoothing 0.1, patience 10.

- **`src/data_loader.py`** — Reuses jieba pipeline; produces `Dataset` with PAD/UNK ids.
- **`src/vocab.py`** — Vocab building + Word2Vec loading (reuses `../homework2/data/sgns.merge.word`).
- **`src/model.py`** — `RelativePositionBias`, encoder layers, `TinyDeBERTa` classifier.
- **`src/train.py`** / **`src/predict.py`** — Standard PyTorch loop / batch inference.

## Architecture (homework2-c) — TextCNN

Kim 2014 TextCNN baseline targeting better-than-DADGNN dev accuracy in the 5-shot regime.

Configuration: filter sizes `[2,3,4]`, 64 filters each (192-dim concat), Word2Vec 200d **frozen**, Dropout 0.5 → Linear(192, 34). AdamW, lr 5e-4, batch 16, max_length 128, label_smoothing 0.1, weight_decay 1e-3, patience 30. ~108K trainable params.

- **`src/data_loader.py`**, **`src/vocab.py`**, **`src/model.py`** (`TextCNN`), **`src/train.py`**, **`src/predict.py`** — same shape as homework2-b.

## Architecture (homework3) — LLM + prompt engineering

No training/fine-tuning. OpenAI-compatible API call (ChatGLM/DeepSeek/Qwen/Moonshot).

Pipeline: `main.py` → `data_loader` → `prompt_builder` → `llm_client` → `evaluator` / `predict`

- **`src/data_loader.py`** — CSV loading, label mapping (34 categories, 0–33), few-shot example selection (shortest per category).
- **`src/prompt_builder.py`** — Zero-shot and few-shot prompts with category list + dialogue + output constraints.
- **`src/llm_client.py`** — OpenAI-compatible client with provider presets (`zhipu`/`deepseek`/`qwen`/`moonshot`). Env-var or constructor config.
- **`src/evaluator.py`** — Regex output parsing, accuracy + per-class report.
- **`src/eval_runner.py`** — Dev-set eval loop with retry logic.
- **`src/predict.py`** — Test-set prediction → `submission.csv`.

Env vars: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` (or pick a provider preset).

## Architecture (homework4) — LLaMA-Factory LoRA SFT on Qwen2.5-7B

Full LoRA supervised fine-tune of `Qwen/Qwen2.5-7B-Instruct` for the same 34-way intent task. Designed to run on a remote RTX 3090 (24 GB) box, not the local machine — paths in `run.sh` (`/root/nlp_homework_04`, `/root/LlamaFactory/...`) are remote-host paths.

- **`prepare_data.py`** — Converts CSVs to LLaMA-Factory Alpaca-format JSON under `data/intent_classification/`.
- **`configs/qwen2.5_lora_sft.yaml`** — LoRA config (rank 64, alpha 128, dropout 0.05, target=all; lr 2e-4, batch 4, grad-accum 4, 10 epochs, max_length 1024). See `configs/qlora_memory_guide.md` for memory tuning.
- **`train.sh`** — Invokes `llamafactory-cli train` with the YAML.
- **`predict.py`** — Loads base model + LoRA adapter, batched inference → `submission.csv`. `debug_predict.py` is a small reproducer.
- **`run.sh`** — Orchestrator: `prepare | train | eval | predict | all`. Eval mode predicts on `dev_new.csv` and reports accuracy.

`homework4/` has its own `pyproject.toml` and `uv.lock` because the LLaMA-Factory stack diverges from the workspace deps.

## Key Design Decisions

- **homework1** is numpy-only by design — do not introduce PyTorch/TensorFlow there.
- **homework2** uses DGL + PyTorch; graphs are built dynamically inside `forward`.
- **DGL 2.2.0 requires `torch<2.4`** (graphbolt C++ ABI). The workspace pins `torch>=2.3,<2.4`. `graphbolt/__init__.py` was patched in `.venv` to skip the missing C++ libs gracefully — if you recreate the venv and DGL imports fail, that patch needs to be reapplied.
- **Python is pinned to 3.12** (`.python-version`) — DGL has no cp313 wheels.
- **Chinese Word2Vec** lives at `homework2/data/sgns.merge.word` (gitignored). homework2-b and homework2-c reuse the same file via `../homework2/data/`. Missing file → random init fallback.
- **homework3** few-shot examples = shortest sample per category (selected in `data_loader`), not random.
- **homework4** is the only homework that expects a GPU host with LLaMA-Factory installed; do not try to run `train.sh` locally. Use it as reference / for editing configs and pre/post-processing.
- All four homeworks emit `submission.csv` at their own root with columns `Id,Category` for Kaggle submission.
