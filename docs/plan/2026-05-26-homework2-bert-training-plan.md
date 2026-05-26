# Homework2 BERT Training Plan

## Summary

Create a new `homework2/` that matches `homework1` at the outer shell level: same style of directory layout, README sections, `main.py` entry, `results/` artifacts, and final `submission.csv` format. Internally, replace the NumPy softmax pipeline with a minimal BERT fine-tuning pipeline based on `bert-base-chinese`, `torch`, and the existing `transformers` package.

The implementation will keep `train/dev` separated for the full workflow, including final prediction. No extra hyperparameter search command will be added unless it is already necessary during implementation.

## Key Changes

### Project shape and CLI

- Add `homework2/` with `main.py`, `README.md`, and a small `src/` package mirroring the homework1 organization.
- Expose only the commands that are necessary for this homework:
  - `python main.py train`
  - `python main.py predict`
- Keep output locations homework1-style:
  - `homework2/results/`
  - `homework2/submission.csv`

### Dependencies and reproducibility

- Update `pyproject.toml` to declare the runtime dependencies required by homework2:
  - `torch`
  - `transformers`
- Do not add `datasets`, `evaluate`, or `accelerate`.
- Keep the rest of the project dependency footprint unchanged.

### Data and preprocessing

- Read the same CSV files from `nlp-text-classification-experiments/`.
- Use `sentence_sep` as the model input text.
- Preserve turn boundaries by converting literal `[SEP]` markers into the tokenizer separator token form before tokenization.
- Tokenize with:
  - `truncation=True`
  - `max_length=512`
  - dynamic padding through `DataCollatorWithPadding`
- Build label indices from labeled data and keep a stable `label_raw -> c_numerical` mapping so the submission stays exactly `ID,c_numerical`.

### Training and saved artifacts

- Use `AutoTokenizer` + `AutoModelForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=...)`.
- Use Hugging Face `Trainer` with a minimal, explicit config:
  - evaluation once per epoch
  - save once per epoch
  - `load_best_model_at_end=True`
  - best model chosen by validation accuracy
- Use a small default config suitable for the current machine and dataset size:
  - learning rate `2e-5`
  - batch size `8`
  - epochs `5`
  - weight decay `0.01`
- Save enough artifacts for reuse by `predict`:
  - best model checkpoint/tokenizer under `homework2/results/`
  - effective training config as JSON
  - label mapping JSON
  - training/eval metrics JSON
- Make `predict` load the saved trained artifacts and fail fast with a clear message if training has not been run yet. It should not silently retrain.

### Documentation

- Write `homework2/README.md` in the same style as `homework1/README.md`.
- Briefly document that homework2 uses BERT fine-tuning and that long samples are truncated to the model limit.

## Test Plan

- Data smoke test:
  - CSV loading works for train/dev/test
  - label mapping covers every predicted label
- Training smoke test:
  - `python main.py train` completes on the real dataset
  - best checkpoint and metrics files are created in `homework2/results/`
- Prediction smoke test:
  - `python main.py predict` creates `homework2/submission.csv`
  - output column names are exactly `ID,c_numerical`
  - row count matches `kaggle_test.csv`
- Failure-path check:
  - `predict` without trained artifacts exits with a clear error
- Docs consistency check:
  - README commands and artifact names match actual implementation

## Assumptions

- “对齐 homework1” means aligning the outer shell and outputs, not forcing homework2 to keep full command parity.
- Final prediction continues to respect `train/dev` separation.
- The chosen backbone is `bert-base-chinese`.
- Only the minimum necessary runtime dependencies are added.
