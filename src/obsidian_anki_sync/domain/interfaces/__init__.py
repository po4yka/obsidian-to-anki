"""Domain interfaces package."""

from .anki_client import IAnkiClient
from .anki_config import IAnkiConfig
from .card_generator import ICardGenerator
from .connection_checker import IConnectionChecker
from .llm_config import ILLMConfig
from .llm_generator import IGenerator
from .llm_provider import ILLMProvider
from .model_provider import IModelProvider
from .note_parser import INoteParser
from .state_repository import IStateRepository
from .vault_config import IVaultConfig

__all__ = [
    "IAnkiClient",
    "IAnkiConfig",
    "ICardGenerator",
    "IConnectionChecker",
    "IGenerator",
    "ILLMConfig",
    "ILLMProvider",
    "IModelProvider",
    "INoteParser",
    "IStateRepository",
    "IVaultConfig",
]
