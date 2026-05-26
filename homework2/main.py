"""BERT-based SLU intent detection entry point.

Usage:
    python main.py train
    python main.py predict
"""

import argparse
import os

from src.data_loader import load_csv
from src.predict import predict_and_save
from src.train import train_and_save

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "nlp-text-classification-experiments")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_splits():
    """Load the standard train/dev/test CSV splits."""
    print("Loading data...")
    train_df = load_csv(os.path.join(DATA_DIR, "train_new_5shot.csv"))
    dev_df = load_csv(os.path.join(DATA_DIR, "dev_new.csv"))
    test_df = load_csv(os.path.join(DATA_DIR, "kaggle_test.csv"))
    print(f"  Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
    return train_df, dev_df, test_df


def run_train():
    """Fine-tune BERT on the train split and evaluate on dev."""
    train_df, dev_df, _ = load_splits()
    artifacts = train_and_save(train_df, dev_df, RESULTS_DIR)
    print("\nTraining finished.")
    print(f"Best checkpoint: {artifacts['best_model_checkpoint']}")
    print(f"Saved model dir: {artifacts['model_dir']}")
    print(f"Validation accuracy: {artifacts['eval_metrics']['eval_accuracy']:.4f}")


def run_predict():
    """Load the saved model artifacts and generate a submission CSV."""
    _, _, test_df = load_splits()
    output_path = os.path.join(os.path.dirname(__file__), "submission.csv")
    predict_and_save(
        test_df=test_df,
        results_dir=RESULTS_DIR,
        output_path=output_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Homework 2 BERT intent classification")
    parser.add_argument("command", choices=["train", "predict"])
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_train()
    elif args.command == "predict":
        run_predict()


if __name__ == "__main__":
    main()
