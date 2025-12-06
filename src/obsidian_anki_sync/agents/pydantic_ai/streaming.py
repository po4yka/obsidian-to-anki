"""Streaming utilities and logging helpers for PydanticAI agents."""

from __future__ import annotations

import asyncio
import contextlib
import html
import time
from typing import TypeVar

from obsidian_anki_sync.utils.llm_logging import log_llm_stream_chunk
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)

OutputT = TypeVar("OutputT")
DepsT = TypeVar("DepsT")


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token on average)."""
    return len(text) // 4


def _get_model_name(agent: object) -> str:
    """Extract model name from agent if available."""
    try:
        if hasattr(agent, "model") and getattr(agent, "model"):
            model = getattr(agent, "model")
            if hasattr(model, "model_name"):
                return str(model.model_name)
            if hasattr(model, "name"):
                return str(model.name)
            return str(model)
    except Exception:
        pass
    return "unknown"


def _truncate_for_log(text: str, max_length: int = 200) -> str:
    """Truncate text for logging with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def _decode_html_encoded_apf(apf_html: str) -> str:
    """Decode HTML entities in APF content if the LLM returned encoded HTML."""
    if "&lt;!--" in apf_html or "&lt;pre&gt;" in apf_html or "&lt;ul&gt;" in apf_html:
        decoded = html.unescape(apf_html)
        logger.debug(
            "apf_html_decoded",
            original_length=len(apf_html),
            decoded_length=len(decoded),
            was_encoded=True,
        )
        return decoded
    return apf_html


async def run_agent_with_streaming(
    agent: object,
    prompt: str,
    deps: DepsT,
    operation_name: str = "agent",
    log_interval: float = 30.0,
) -> OutputT:
    """Run an agent with streaming to prevent connection timeouts."""
    start_time = time.time()
    last_log_time = start_time
    chunk_count = 0
    total_content_length = 0

    model_name = _get_model_name(agent)
    prompt_tokens_est = _estimate_tokens(prompt)

    logger.info(
        "llm_call_start",
        operation=operation_name,
        model=model_name,
        prompt_length=len(prompt),
        prompt_tokens_est=prompt_tokens_est,
        prompt_preview=_truncate_for_log(prompt, 150),
    )

    try:
        async with agent.run_stream(prompt, deps=deps) as result:
            async for partial in result.stream():
                chunk_count += 1
                current_time = time.time()

                if partial and hasattr(partial, "__len__"):
                    with contextlib.suppress(Exception):
                        total_content_length = len(str(partial))

                log_llm_stream_chunk(
                    model=model_name,
                    operation=operation_name,
                    chunk_index=chunk_count,
                    elapsed_seconds=current_time - start_time,
                    chunk_chars=len(str(partial)) if partial else 0,
                    chunk_preview=_truncate_for_log(str(partial), 120)
                    if partial
                    else "",
                )

                if current_time - last_log_time >= log_interval:
                    elapsed = current_time - start_time
                    tokens_per_sec = (
                        total_content_length / 4 / elapsed if elapsed > 0 else 0
                    )
                    logger.info(
                        "llm_streaming_progress",
                        operation=operation_name,
                        model=model_name,
                        elapsed_seconds=round(elapsed, 1),
                        chunks_received=chunk_count,
                        content_length=total_content_length,
                        tokens_per_sec=round(tokens_per_sec, 1),
                    )
                    last_log_time = current_time

            output = result.data

        elapsed = time.time() - start_time

        usage_info = {}
        if hasattr(result, "usage") and result.usage:
            usage = result.usage
            if hasattr(usage, "request_tokens"):
                usage_info["request_tokens"] = usage.request_tokens
            if hasattr(usage, "response_tokens"):
                usage_info["response_tokens"] = usage.response_tokens
            if hasattr(usage, "total_tokens"):
                usage_info["total_tokens"] = usage.total_tokens
            if not usage_info and hasattr(usage, "__dict__"):
                usage_info["usage"] = str(usage)

        try:
            output_preview = _truncate_for_log(str(output), 200)
        except Exception:
            output_preview = "<unable to serialize>"

        logger.info(
            "llm_call_complete",
            operation=operation_name,
            model=model_name,
            elapsed_seconds=round(elapsed, 2),
            total_chunks=chunk_count,
            output_preview=output_preview,
            **usage_info,
        )

        return output

    except asyncio.CancelledError:
        elapsed = time.time() - start_time
        logger.warning(
            "llm_call_cancelled",
            operation=operation_name,
            model=model_name,
            elapsed_seconds=round(elapsed, 2),
            chunks_received=chunk_count,
        )
        raise
    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error(
            "llm_call_failed",
            operation=operation_name,
            model=model_name,
            elapsed_seconds=round(elapsed, 2),
            chunks_received=chunk_count,
            error_type=type(exc).__name__,
            error_message=str(exc)[:500],
        )
        raise

