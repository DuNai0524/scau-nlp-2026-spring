"""Homework 3: LLM-based intent classification with prompt engineering.

Usage:
    python main.py eval --mode zero-shot [--max-samples N]
    python main.py eval --mode few-shot  [--max-samples N]
    python main.py predict --mode few-shot [--output submission.csv]
    python main.py compare [--max-samples N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import get_label_mapping, load_train, select_fewshot_examples
from src.eval_runner import run_eval
from src.llm_client import LLMClient
from src.predict import run_predict


def cmd_eval(args: argparse.Namespace) -> None:
    """Run evaluation on dev set."""
    client = LLMClient()
    train_df = load_train()
    label_mapping = get_label_mapping(train_df)
    examples = select_fewshot_examples(train_df) if args.mode == "few-shot" else None

    run_eval(
        mode=args.mode,
        label_mapping=label_mapping,
        examples=examples,
        client=client,
        max_samples=args.max_samples,
    )


def cmd_predict(args: argparse.Namespace) -> None:
    """Generate predictions for test set."""
    client = LLMClient()
    train_df = load_train()
    label_mapping = get_label_mapping(train_df)
    examples = select_fewshot_examples(train_df) if args.mode == "few-shot" else None

    run_predict(
        mode=args.mode,
        label_mapping=label_mapping,
        examples=examples,
        client=client,
        output_path=args.output,
    )


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare zero-shot vs few-shot on dev set."""
    client = LLMClient()
    train_df = load_train()
    label_mapping = get_label_mapping(train_df)
    examples = select_fewshot_examples(train_df)

    print("\n" + "=" * 60)
    print("Zero-shot 评估")
    print("=" * 60)
    run_eval(
        mode="zero-shot",
        label_mapping=label_mapping,
        client=client,
        max_samples=args.max_samples,
    )

    print("\n" + "=" * 60)
    print("Few-shot 评估")
    print("=" * 60)
    run_eval(
        mode="few-shot",
        label_mapping=label_mapping,
        examples=examples,
        client=client,
        max_samples=args.max_samples,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Homework 3: LLM Intent Classification")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # eval
    p_eval = subparsers.add_parser("eval", help="Evaluate on dev set")
    p_eval.add_argument("--mode", choices=["zero-shot", "few-shot"], required=True)
    p_eval.add_argument("--max-samples", type=int, default=None, help="Limit samples for quick testing")
    p_eval.set_defaults(func=cmd_eval)

    # predict
    p_predict = subparsers.add_parser("predict", help="Generate test predictions")
    p_predict.add_argument("--mode", choices=["zero-shot", "few-shot"], required=True)
    p_predict.add_argument("--output", default="submission.csv", help="Output CSV path")
    p_predict.set_defaults(func=cmd_predict)

    # compare
    p_compare = subparsers.add_parser("compare", help="Compare zero-shot vs few-shot")
    p_compare.add_argument("--max-samples", type=int, default=None, help="Limit samples for quick testing")
    p_compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
