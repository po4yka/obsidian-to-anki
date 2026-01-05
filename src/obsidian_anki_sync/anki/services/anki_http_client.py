"""HTTP client for AnkiConnect API communication."""

import contextlib
import random
import time
from types import TracebackType
from typing import Any, Literal

import httpx

from obsidian_anki_sync.domain.interfaces.anki_http_client import IAnkiHttpClient
from obsidian_anki_sync.exceptions import AnkiConnectError
from obsidian_anki_sync.utils.async_runner import AsyncioRunner
from obsidian_anki_sync.utils.logging import get_logger
from obsidian_anki_sync.utils.retry import retry

logger = get_logger(__name__)


class AnkiHttpClient(IAnkiHttpClient):
    """HTTP client for communicating with AnkiConnect API.

    Handles low-level HTTP communication, connection management,
    health checks, and retry logic for AnkiConnect operations.
    """

    def __init__(
        self,
        url: str,
        timeout: float = 180.0,
        enable_health_checks: bool = True,
        async_runner: AsyncioRunner | None = None,
        max_keepalive_connections: int = 10,
        max_connections: int = 20,
        keepalive_expiry: float = 60.0,
        verify_connectivity: bool = True,
    ):
        """
        Initialize HTTP client.

        Args:
            url: AnkiConnect URL
            timeout: Request timeout in seconds
            enable_health_checks: Whether to perform periodic health checks
            async_runner: Optional async runner for sync/async bridging
            max_keepalive_connections: Max idle connections to keep alive
            max_connections: Max total connections in pool
            keepalive_expiry: Seconds before idle connections expire
            verify_connectivity: If True, verify AnkiConnect is reachable at startup
        """
        self.url = url
        self.enable_health_checks = enable_health_checks
        self._last_health_check = 0.0
        self._health_check_interval = 60.0  # Check health every 60 seconds when healthy
        self._health_check_recovery_interval = 5.0  # Fast retry when unhealthy
        self._max_health_backoff = 300.0  # Max 5 minutes between checks
        self._consecutive_health_failures = 0
        self._is_healthy = True
        self._async_runner = async_runner or AsyncioRunner.get_global()

        # Configure connection pooling for better performance
        pool_limits = httpx.Limits(
            max_keepalive_connections=max_keepalive_connections,
            max_connections=max_connections,
            keepalive_expiry=keepalive_expiry,
        )

        # Using sync client for compatibility with existing sync code paths
        self.session = httpx.Client(timeout=timeout, limits=pool_limits)
        self._async_client = httpx.AsyncClient(timeout=timeout, limits=pool_limits)

        logger.info(
            "anki_http_client_initialized",
            url=url,
            health_checks=enable_health_checks,
            max_connections=max_connections,
            max_keepalive=max_keepalive_connections,
            verify_connectivity=verify_connectivity,
        )

        # Verify connectivity at startup if requested (fail-fast behavior)
        if verify_connectivity:
            self._verify_connectivity()

    def _verify_connectivity(self) -> None:
        """Verify AnkiConnect connectivity at startup."""
        try:
            # Use direct HTTP call to version action as a lightweight connectivity check
            # (avoiding invoke() which triggers _check_health and async runner)
            payload = {"action": "version", "version": 6, "params": {}}
            response = self.session.post(self.url, json=payload)
            response.raise_for_status()
            result = response.json()
            if result.get("error"):
                msg = f"AnkiConnect error: {result['error']}"
                raise AnkiConnectError(msg)
            version = result.get("result")
            logger.info(
                "anki_connectivity_verified",
                url=self.url,
                anki_connect_version=version,
            )
        except (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.NetworkError,
        ) as e:
            logger.error(
                "anki_connectivity_failed_network",
                url=self.url,
                error=str(e),
                error_type=type(e).__name__,
            )
            msg = f"Cannot connect to AnkiConnect at {self.url}"
            suggestion = (
                "Ensure Anki is running. "
                "Check AnkiConnect addon is installed and enabled. "
                f"Verify URL is correct: {self.url}. "
                "Check firewall settings allow the connection."
            )
            raise AnkiConnectError(msg, suggestion=suggestion) from e
        except AnkiConnectError:
            # Re-raise AnkiConnectError as-is
            raise
        except Exception as e:
            logger.error(
                "anki_connectivity_failed_unexpected",
                url=self.url,
                error=str(e),
                error_type=type(e).__name__,
            )
            msg = f"AnkiConnect connectivity check failed: {e}"
            suggestion = (
                "Ensure Anki is running with AnkiConnect addon. "
                f"Check AnkiConnect is listening at {self.url}."
            )
            raise AnkiConnectError(msg, suggestion=suggestion) from e

    def _get_health_check_interval(self) -> float:
        """Calculate health check interval with exponential backoff and jitter."""
        if self._is_healthy:
            # Use normal interval when healthy
            base_interval = self._health_check_interval
        else:
            # Use shorter recovery interval with exponential backoff
            backoff_multiplier = 2 ** min(self._consecutive_health_failures, 6)
            base_interval = min(
                self._health_check_recovery_interval * backoff_multiplier,
                self._max_health_backoff,
            )

        # Add jitter (0.8x to 1.2x) to prevent thundering herd
        jitter = random.uniform(0.8, 1.2)
        return base_interval * jitter

    def _check_health(self) -> bool:
        """
        Check if AnkiConnect is healthy and responding.

        Uses exponential backoff with jitter when unhealthy to balance
        quick recovery with avoiding overwhelming a struggling service.

        Returns:
            True if healthy, False otherwise
        """
        if not self.enable_health_checks:
            return True

        current_time = time.time()
        check_interval = self._get_health_check_interval()
        if current_time - self._last_health_check < check_interval:
            return self._is_healthy

        try:
            # Simple health check using version action (direct call to avoid recursion)
            payload = {"action": "version", "version": 6, "params": {}}
            response = self.session.post(self.url, json=payload)
            response.raise_for_status()
            result = response.json()
            if result.get("error"):
                msg = f"AnkiConnect error: {result['error']}"
                raise AnkiConnectError(msg)
            self._is_healthy = True
            self._consecutive_health_failures = 0  # Reset on success
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
            self._is_healthy = False
            self._consecutive_health_failures += 1
            logger.warning(
                "anki_health_check_network_error",
                url=self.url,
                error=str(e),
                error_type=type(e).__name__,
                consecutive_failures=self._consecutive_health_failures,
            )
        except (httpx.HTTPStatusError, AnkiConnectError) as e:
            self._is_healthy = False
            self._consecutive_health_failures += 1
            logger.warning(
                "anki_health_check_api_error",
                url=self.url,
                error=str(e),
                error_type=type(e).__name__,
                consecutive_failures=self._consecutive_health_failures,
            )
        except (ValueError, TypeError) as e:
            # Invalid JSON response
            self._is_healthy = False
            self._consecutive_health_failures += 1
            logger.warning(
                "anki_health_check_parse_error",
                url=self.url,
                error=str(e),
                consecutive_failures=self._consecutive_health_failures,
            )

        self._last_health_check = current_time
        return self._is_healthy

    @retry(
        max_attempts=5,
        initial_delay=2.0,
        backoff_factor=2.0,
        exceptions=(httpx.HTTPError, httpx.TimeoutException, AnkiConnectError),
    )
    def invoke(self, action: str, params: dict[str, Any] | None = None) -> Any:
        """
        Invoke AnkiConnect action synchronously.

        Args:
            action: Action name
            params: Action parameters

        Returns:
            Action result

        Raises:
            AnkiConnectError: If the action fails
        """
        # Perform health check before making requests
        if not self._check_health():
            logger.warning(
                "anki_unhealthy_skipping_request", action=action, url=self.url
            )
            msg = "AnkiConnect is not responding - check if Anki is running"
            raise AnkiConnectError(msg)

        return self._async_runner.run(self.invoke_async(action, params))

    async def invoke_async(
        self, action: str, params: dict[str, Any] | None = None
    ) -> Any:
        """Invoke AnkiConnect action asynchronously."""
        payload = {"action": action, "version": 6, "params": params or {}}

        logger.debug("anki_invoke", action=action)

        try:
            response = await self._async_client.post(self.url, json=payload)
            response.raise_for_status()
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            self._is_healthy = False  # Mark as unhealthy on connection errors
            msg = f"Connection error to AnkiConnect: {e}"
            raise AnkiConnectError(msg)
        except httpx.HTTPStatusError as e:
            msg = f"HTTP {e.response.status_code} from AnkiConnect: {e}"
            raise AnkiConnectError(msg)
        except httpx.HTTPError as e:
            msg = f"HTTP error calling AnkiConnect: {e}"
            raise AnkiConnectError(msg)

        try:
            result = response.json()
        except (ValueError, TypeError) as e:
            msg = f"Invalid JSON response: {e}"
            raise AnkiConnectError(msg)

        # Validate response structure
        if not isinstance(result, dict):
            msg = f"Invalid response type: expected dict, got {type(result).__name__}"
            raise AnkiConnectError(msg)

        if "error" not in result and "result" not in result:
            msg = f"Malformed response: missing error/result fields in {result}"
            raise AnkiConnectError(msg)

        if result.get("error") is not None:
            error_msg = result["error"]
            if not isinstance(error_msg, str):
                error_msg = str(error_msg)
            msg = f"AnkiConnect error: {error_msg}"
            raise AnkiConnectError(msg)

        return result.get("result")

    def check_connection(self) -> bool:
        """Check if AnkiConnect is accessible and healthy."""
        return self._check_health()

    @property
    def async_client(self) -> httpx.AsyncClient:
        """Get the async HTTP client (for backward compatibility)."""
        return self._async_client

    def close(self) -> None:
        """Close HTTP sessions and cleanup resources."""
        if hasattr(self, "session") and self.session:
            try:
                self.session.close()
                logger.debug("anki_sync_client_closed", url=self.url)
            except Exception as e:
                logger.warning(
                    "anki_sync_client_cleanup_failed", url=self.url, error=str(e)
                )

        if hasattr(self, "_async_client") and self._async_client:
            try:
                # Run async cleanup in event loop
                self._async_runner.run(self._async_client.aclose())
                logger.debug("anki_async_client_closed", url=self.url)
            except Exception as e:
                logger.warning(
                    "anki_async_client_cleanup_failed", url=self.url, error=str(e)
                )

    async def aclose(self) -> None:
        """Async cleanup for async contexts."""
        if hasattr(self, "_async_client") and self._async_client:
            await self._async_client.aclose()
        if hasattr(self, "session") and self.session:
            self.session.close()

    def __enter__(self) -> "AnkiHttpClient":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Context manager exit with cleanup."""
        self.close()
        return False

    async def __aenter__(self) -> "AnkiHttpClient":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Async context manager exit with cleanup."""
        await self.aclose()
        return False

    def __del__(self) -> None:
        """Cleanup on deletion."""
        with contextlib.suppress(Exception):
            # Only close sync client in __del__ to avoid issues with event loop
            if hasattr(self, "session") and self.session:
                self.session.close()
