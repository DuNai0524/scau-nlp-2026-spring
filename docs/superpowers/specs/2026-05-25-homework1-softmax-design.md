# Homework 1 Softmax Regression Improvement Design

## Goal

Improve validation and submission performance for `homework1` with the minimum necessary code changes while preserving the current project structure.

## Scope

- Keep the existing `main.py` entry structure.
- Keep the homework centered on a NumPy implementation.
- Fix training bugs that currently suppress valid configurations.
- Restore the classifier to standard softmax regression behavior.
- Search feature and optimization settings that matter for linear text classification.
- Update the prediction flow so it can optionally retrain on `train + dev` before generating the final submission.

## Non-Goals

- Do not introduce new dependencies.
- Do not replace the NumPy training pipeline with a different framework or estimator.
- Do not perform unrelated refactoring.

## Design

### Model

- Change the classifier to standard multiclass softmax regression:
  `X -> linear logits -> softmax -> cross-entropy`.
- Remove activation and dropout on class logits because they distort the linear decision boundary and hurt optimization.
- Fix gradient scaling so batch normalization happens exactly once.

### Training

- Fix the `cross_entropy_loss` call path so cross-entropy candidates are actually trainable during grid search.
- Replace the current search emphasis on `activation`, `loss`, and `dropout` with a smaller search space around:
  - feature method
  - n-gram range
  - max features
  - optimizer
  - learning rate
- Keep early stopping and best-checkpoint restoration.

### Data and Prediction

- Extend data preparation to accept configurable feature settings.
- Preserve the current train/dev workflow for model selection.
- Add an explicit option to retrain on `train + dev` after selecting the best config, then predict on test.

### Documentation

- Update `homework1/README.md` so the run commands, search space, and data description match the implementation.

## Validation

- Run the homework training/search commands needed to confirm:
  - cross-entropy training no longer crashes
  - grid search returns meaningful non-zero scores
  - training and prediction still complete successfully
  - README instructions match the updated CLI behavior
