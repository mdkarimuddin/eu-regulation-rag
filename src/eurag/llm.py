"""LLM access behind a tiny protocol, so tests and offline runs can use a scripted model."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Protocol

DEFAULT_MODEL = os.environ.get("EURAG_MODEL", "claude-sonnet-5-5")


class LLMError(RuntimeError):
    pass


@dataclass
class LLMReply:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str, max_tokens: int = 600) -> LLMReply: ...


def supports_temperature(create) -> bool:
    """Newer SDK versions and models do not accept ``temperature``; only send it when the signature has it."""
    try:
        return "temperature" in inspect.signature(create).parameters
    except (TypeError, ValueError):
        return False


class AnthropicLLM:
    def __init__(self, model: str = DEFAULT_MODEL, client=None) -> None:
        self.model, self.name = model, f"anthropic:{model}"
        if client is not None:  # injected client (tests)
            self._client = client
            self.temperature_zero = supports_temperature(client.messages.create)
            return
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key or key.strip() in {"...", "sk-ant-..."} or key.endswith("..."):
            raise LLMError(
                "ANTHROPIC_API_KEY is missing or a placeholder. Set a real key in your own shell "
                "(export ANTHROPIC_API_KEY=...); never commit it."
            )
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError("install the extra: pip install 'eu-regulation-rag[anthropic]'") from exc
        self._client = anthropic.Anthropic()
        self.temperature_zero = supports_temperature(self._client.messages.create)

    def complete(self, system: str, user: str, max_tokens: int = 600) -> LLMReply:
        kwargs = {"temperature": 0} if self.temperature_zero else {}
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            **kwargs,
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return LLMReply(text, resp.usage.input_tokens, resp.usage.output_tokens)
