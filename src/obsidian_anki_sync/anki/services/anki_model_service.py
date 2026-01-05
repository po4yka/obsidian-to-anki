"""Service for Anki model (note type) operations."""

from typing import cast

from obsidian_anki_sync.anki.services.anki_cache import AnkiCache
from obsidian_anki_sync.anki.services.anki_http_client import AnkiHttpClient
from obsidian_anki_sync.domain.interfaces.anki_model_service import IAnkiModelService
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class AnkiModelService(IAnkiModelService):
    """Service for Anki model (note type) operations.

    Handles model-related operations including retrieving model names,
    field definitions, and model metadata, with caching support.
    """

    def __init__(self, http_client: AnkiHttpClient, cache: AnkiCache):
        """
        Initialize model service.

        Args:
            http_client: HTTP client for AnkiConnect communication
            cache: Cache for metadata
        """
        self._http_client = http_client
        self._cache = cache
        logger.debug("anki_model_service_initialized")

    def get_model_names(self, use_cache: bool = True) -> list[str]:
        """Get list of available note model names."""
        cache_key = "model_names"
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cast("list[str]", cached)

        result = cast("list[str]", self._http_client.invoke("modelNames"))
        self._cache.set(cache_key, result)
        return result

    async def get_model_names_async(self, use_cache: bool = True) -> list[str]:
        """Get list of available note model names (async)."""
        cache_key = "model_names"
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cast("list[str]", cached)

        result = cast("list[str]", await self._http_client.invoke_async("modelNames"))
        self._cache.set(cache_key, result)
        return result

    def get_model_field_names(
        self, model_name: str, use_cache: bool = True
    ) -> list[str]:
        """Get field names for a specific model."""
        cache_key = f"model_field_names:{model_name}"
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cast("list[str]", cached)

        result = cast(
            "list[str]",
            self._http_client.invoke("modelFieldNames", {"modelName": model_name}),
        )
        self._cache.set(cache_key, result)
        return result

    async def get_model_field_names_async(
        self, model_name: str, use_cache: bool = True
    ) -> list[str]:
        """Get field names for a specific model (async)."""
        cache_key = f"model_field_names:{model_name}"
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cast("list[str]", cached)

        result = cast(
            "list[str]",
            await self._http_client.invoke_async(
                "modelFieldNames", {"modelName": model_name}
            ),
        )
        self._cache.set(cache_key, result)
        return result

    def get_model_names_and_ids(self) -> dict[str, int]:
        """Get note type names and their IDs."""
        return cast("dict[str, int]", self._http_client.invoke("modelNamesAndIds"))
