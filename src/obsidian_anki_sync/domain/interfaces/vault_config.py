"""Interface for Obsidian vault configuration."""

from abc import ABC, abstractmethod
from pathlib import Path


class IVaultConfig(ABC):
    """Interface for Obsidian vault configuration.

    This interface defines the contract for accessing vault-related
    configuration following the Interface Segregation Principle.
    """

    @property
    @abstractmethod
    def vault_path(self) -> Path:
        """Get the path to the Obsidian vault.

        Returns:
            Path to the vault directory
        """
        pass

    @property
    @abstractmethod
    def source_dir(self) -> Path:
        """Get the source directory within the vault.

        Returns:
            Relative path from vault root to source directory
        """
        pass

    @property
    @abstractmethod
    def source_subdirs(self) -> list[Path | None]:
        """Get optional list of source subdirectories.

        Returns:
            List of relative paths to search for notes, or None
        """
        pass

    @abstractmethod
    def get_source_paths(self) -> list[Path]:
        """Get all source paths to search for notes.

        Returns:
            List of absolute paths to search directories
        """
        pass

    @abstractmethod
    def validate_vault_access(self) -> bool:
        """Validate that the vault is accessible.

        Returns:
            True if vault is accessible, False otherwise
        """
        pass
