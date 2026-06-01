"""OpenAI-compatible LLM client for intent classification.

Supports any provider with an OpenAI-compatible API endpoint.
Configure via environment variables or constructor parameters.

Usage:
    export LLM_API_KEY="your-key"
    export LLM_BASE_URL="https://open.bigmodel.cn/api/paas/v4/"  # 智谱
    export LLM_MODEL="glm-4-flash"

    client = LLMClient()
    response = client.chat("你好")
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from homework3/ directory regardless of cwd
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# Provider presets for convenience
PROVIDERS = {
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "model": "glm-4-flash",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
    },
}


class LLMClient:
    """Thin wrapper around OpenAI client for chat completions."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        provider: str | None = None,
    ):
        # Resolve from provider preset, then explicit args, then env vars
        preset = PROVIDERS.get(provider, {}) if provider else {}

        self.model = (
            model
            or preset.get("model")
            or os.environ.get("LLM_MODEL", "glm-4-flash")
        )
        self._client = OpenAI(
            api_key=(
                api_key
                or os.environ.get("LLM_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            ),
            base_url=(
                base_url
                or preset.get("base_url")
                or os.environ.get("LLM_BASE_URL")
            ),
        )

    def chat(
        self,
        message: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        """Send a single user message and return the assistant's reply."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
