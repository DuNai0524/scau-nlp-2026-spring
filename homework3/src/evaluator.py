"""Output parsing and evaluation metrics."""

from __future__ import annotations

import re


def parse_prediction(raw_output: str, max_label: int = 33) -> int | None:
    """Extract the first integer from LLM output and validate range.

    Args:
        raw_output: Raw string returned by the LLM.
        max_label: Maximum valid label (inclusive).

    Returns:
        The predicted label as int, or None if parsing fails.
    """
    match = re.search(r"(?<![-\d])\d+", raw_output)
    if match is None:
        return None
    label = int(match.group())
    if 0 <= label <= max_label:
        return label
    return None


def compute_accuracy(predictions: list[int | None], ground_truth: list[int]) -> float:
    """Compute accuracy, skipping failed predictions.

    Args:
        predictions: List of predicted labels (None for parse failures).
        ground_truth: List of true labels.

    Returns:
        Accuracy as a float in [0, 1].
    """
    correct = 0
    total = 0
    for pred, true in zip(predictions, ground_truth):
        if pred is None:
            continue
        total += 1
        if pred == true:
            correct += 1
    return correct / total if total > 0 else 0.0


def compute_per_class_accuracy(
    predictions: list[int | None],
    ground_truth: list[int],
) -> dict[int, dict]:
    """Compute per-class accuracy and counts.

    Returns:
        Dict mapping label -> {correct, total, accuracy}.
    """
    stats: dict[int, dict] = {}
    for pred, true in zip(predictions, ground_truth):
        if true not in stats:
            stats[true] = {"correct": 0, "total": 0}
        stats[true]["total"] += 1
        if pred == true:
            stats[true]["correct"] += 1
    for label in stats:
        s = stats[label]
        s["accuracy"] = s["correct"] / s["total"] if s["total"] > 0 else 0.0
    return stats


def print_eval_report(
    predictions: list[int | None],
    ground_truth: list[int],
    label_mapping: dict[int, str],
) -> None:
    """Print a formatted evaluation report."""
    acc = compute_accuracy(predictions, ground_truth)
    per_class = compute_per_class_accuracy(predictions, ground_truth)

    failed = sum(1 for p in predictions if p is None)
    print(f"\n{'='*60}")
    print(f"总体准确率: {acc:.4f} ({int(acc * (len(ground_truth) - failed))}/{len(ground_truth) - failed})")
    print(f"解析失败数: {failed}/{len(ground_truth)}")
    print(f"{'='*60}")

    print(f"\n{'类别':<40} {'正确':>4} {'总数':>4} {'准确率':>8}")
    print("-" * 60)
    for label in sorted(per_class):
        s = per_class[label]
        name = label_mapping.get(label, f"未知({label})")
        if len(name) > 36:
            name = name[:33] + "..."
        print(f"{name:<40} {s['correct']:>4} {s['total']:>4} {s['accuracy']:>8.4f}")
