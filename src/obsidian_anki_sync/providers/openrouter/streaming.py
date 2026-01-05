"""Streaming result handler for OpenRouter completions."""

from __future__ import annotations

import json
from typing import Any

from obsidian_anki_sync.utils.llm_logging import log_llm_error, log_llm_success
from obsidian_anki_sync.utils.logging import get_logger

from .models import DEFAULT_CONTEXT_WINDOW, MODEL_CONTEXT_WINDOWS

logger = get_logger(__name__)


class OpenRouterStreamResult:
    """Iterator that streams OpenRouter completions as they arrive."""

    def __init__(
        self,
        client: Any,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        model: str,
        request_start_time: float,
        operation: str,
        had_json_schema: bool,
    ):
        self._client = client
        self._url = url
        self._payload = payload
        self._headers = headers
        self.model = model
        self._request_start_time = request_start_time
        self._operation = operation
        self._had_json_schema = had_json_schema
        self._consumed = False
        self._response_parts: list[str] = []
        self.response_text: str = ""
        self.finish_reason: str = "stop"
        self.usage: dict[str, Any] = {}
        self._context_window = MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)

    def __iter__(self):
        if self._consumed:
            msg = "Streaming result already consumed"
            raise RuntimeError(msg)

        self._consumed = True
        try:
            with self._client.stream(
                "POST",
                self._url,
                json=self._payload,
                headers=self._headers,
            ) as response:
                response.raise_for_status()

                for raw_line in response.iter_lines():
                    if not raw_line:
                        continue

                    line = raw_line.strip()
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue

                    data_str = line[5:].strip()
                    if not data_str:
                        continue
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.debug(
                            "openrouter_stream_chunk_parse_failed",
                            model=self.model,
                            data_preview=data_str[:200],
                        )
                        continue

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue

                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        self._response_parts.append(piece)
                        yield piece

                    finish_reason = choices[0].get("finish_reason")
                    if finish_reason:
                        self.finish_reason = finish_reason

                    usage_data = chunk.get("usage")
                    if usage_data:
                        self.usage = usage_data

        except Exception as exc:
            log_llm_error(
                model=self.model,
                operation=self._operation,
                start_time=self._request_start_time,
                error=exc,
                retryable=False,
            )
            raise
        finally:
            self.response_text = "".join(self._response_parts)

            prompt_tokens = self.usage.get("prompt_tokens", 0)
            completion_tokens = self.usage.get(
                "completion_tokens", max(len(self.response_text) // 4, 0)
            )
            total_tokens = self.usage.get(
                "total_tokens", prompt_tokens + completion_tokens
            )

            log_llm_success(
                model=self.model,
                operation=self._operation,
                start_time=self._request_start_time,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                response_length=len(self.response_text),
                finish_reason=self.finish_reason,
                context_window=self._context_window,
                estimate_cost_flag=True,
                had_json_schema=self._had_json_schema,
            )

    def collect(self) -> str:
        """Consume the stream (if needed) and return the final response."""
        if not self._consumed:
            for _ in self:
                pass
        return self.response_text


__all__ = ["OpenRouterStreamResult"]
