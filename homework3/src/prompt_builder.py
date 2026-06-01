"""Prompt construction for zero-shot and few-shot intent classification."""

from __future__ import annotations

import pandas as pd


def _format_category_list(label_mapping: dict[int, str]) -> str:
    """Format the category list as numbered lines."""
    lines = []
    for idx in sorted(label_mapping):
        lines.append(f"{idx}: {label_mapping[idx]}")
    return "\n".join(lines)


def build_system_prompt() -> str:
    """System message shared across all modes."""
    return "你是一个中文客服对话意图分类助手。你需要根据客服对话内容，判断用户的意图类别。"


def build_zero_shot_prompt(
    text: str,
    label_mapping: dict[int, str],
) -> str:
    """Build a zero-shot classification prompt.

    Args:
        text: The dialogue text (sentence_sep format).
        label_mapping: {c_numerical: label_raw} mapping.

    Returns:
        The full user message string.
    """
    categories = _format_category_list(label_mapping)
    return f"""请将以下客服对话分类到对应的意图类别中。

## 意图类别列表
{categories}

## 对话内容
{text}

## 要求
请只输出对应的意图类别编号（0-{max(label_mapping)}），不要输出任何解释或多余内容。
输出格式：直接输出一个数字。"""


def build_few_shot_prompt(
    text: str,
    label_mapping: dict[int, str],
    examples: pd.DataFrame,
) -> str:
    """Build a few-shot classification prompt.

    Args:
        text: The dialogue text (sentence_sep format).
        label_mapping: {c_numerical: label_raw} mapping.
        examples: DataFrame with columns [c_numerical, label_raw, sentence_sep].

    Returns:
        The full user message string.
    """
    categories = _format_category_list(label_mapping)

    example_lines = []
    for _, row in examples.iterrows():
        example_lines.append(
            f"### 对话\n{row['sentence_sep']}\n### 意图类别\n{row['c_numerical']}"
        )
    examples_block = "\n\n".join(example_lines)

    return f"""请将以下客服对话分类到对应的意图类别中。

## 意图类别列表
{categories}

## 分类示例
{examples_block}

## 待分类对话
{text}

## 要求
请只输出对应的意图类别编号（0-{max(label_mapping)}），不要输出任何解释或多余内容。
输出格式：直接输出一个数字。"""


def build_prompt(
    text: str,
    label_mapping: dict[int, str],
    mode: str,
    examples: pd.DataFrame | None = None,
) -> str:
    """Dispatch to zero-shot or few-shot prompt builder."""
    if mode == "zero-shot":
        return build_zero_shot_prompt(text, label_mapping)
    elif mode == "few-shot":
        if examples is None:
            raise ValueError("examples DataFrame is required for few-shot mode")
        return build_few_shot_prompt(text, label_mapping, examples)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'zero-shot' or 'few-shot'.")
