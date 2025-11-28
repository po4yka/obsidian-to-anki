"""Pre-flight checks for validating environment before sync operations."""

import shutil
from pathlib import Path

import psutil
from pydantic import BaseModel, ConfigDict, Field

from ..config import Config
from ..utils.logging import get_logger

logger = get_logger(__name__)


class CheckResult(BaseModel):
    """Result of a pre-flight check."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1, description="Check name")
    passed: bool = Field(description="Whether the check passed")
    message: str = Field(min_length=1, description="Result message")
    severity: str = Field(description="Severity level: 'error', 'warning', 'info'")
    fixable: bool = Field(default=False, description="Whether the issue is fixable")
    fix_suggestion: str | None = Field(
        default=None, description="Suggestion for fixing the issue"
    )


class PreflightChecker:
    """Performs pre-flight checks before running sync operations."""

    def __init__(self, config: Config):
        """Initialize pre-flight checker.

        Args:
            config: Service configuration
        """
        self.config = config
        self.results: list[CheckResult] = []

    def run_all_checks(
        self, check_anki: bool = True, check_llm: bool = True
    ) -> tuple[bool, list[CheckResult]]:
        """Run all pre-flight checks.

        Args:
            check_anki: Whether to check Anki connectivity
            check_llm: Whether to check LLM provider connectivity

        Returns:
            Tuple of (all_passed, results)
        """
        logger.info(
            "preflight_checks_started", check_anki=check_anki, check_llm=check_llm
        )

        self.results = []

        # Critical checks (must pass)
        self._check_vault_path()
        self._check_source_dir()
        self._check_db_path()

        # Provider checks
        if check_llm:
            self._check_llm_provider()

        # Anki checks (skip for dry-run and export)
        if check_anki:
            self._check_anki_connectivity()

        # Additional checks
        self._check_note_type()
        self._check_deck_name()

        # Extended checks
        self._check_git_repo()
        self._check_vault_structure()
        self._check_disk_space()
        self._check_memory()
        if check_anki or check_llm:
            self._check_network_latency(check_anki, check_llm)

        # Count errors
        errors = [r for r in self.results if not r.passed and r.severity == "error"]
        warnings = [r for r in self.results if not r.passed and r.severity == "warning"]

        all_passed = len(errors) == 0

        logger.info(
            "preflight_checks_completed",
            passed=all_passed,
            total_checks=len(self.results),
            errors=len(errors),
            warnings=len(warnings),
        )

        return all_passed, self.results

    def _check_vault_path(self) -> None:
        """Check if vault path exists and is accessible."""
        vault_path = self.config.vault_path

        if not vault_path:
            self.results.append(
                CheckResult(
                    name="Vault Path",
                    passed=False,
                    message="VAULT_PATH is not configured",
                    severity="error",
                    fixable=True,
                    fix_suggestion="Set VAULT_PATH in your .env file or environment variables",
                )
            )
            return

        vault_path_obj = Path(vault_path)

        if not vault_path_obj.exists():
            self.results.append(
                CheckResult(
                    name="Vault Path",
                    passed=False,
                    message=f"Vault path does not exist: {vault_path}",
                    severity="error",
                    fixable=False,
                    fix_suggestion="Create the directory or update VAULT_PATH to a valid location",
                )
            )
            return

        if not vault_path_obj.is_dir():
            self.results.append(
                CheckResult(
                    name="Vault Path",
                    passed=False,
                    message=f"Vault path is not a directory: {vault_path}",
                    severity="error",
                    fixable=False,
                )
            )
            return

        # Check read access
        try:
            list(vault_path_obj.iterdir())
            self.results.append(
                CheckResult(
                    name="Vault Path",
                    passed=True,
                    message=f"Vault path is accessible: {vault_path}",
                    severity="info",
                )
            )
        except PermissionError:
            self.results.append(
                CheckResult(
                    name="Vault Path",
                    passed=False,
                    message=f"No read permission for vault path: {vault_path}",
                    severity="error",
                    fixable=True,
                    fix_suggestion=f"Grant read permissions: chmod +r {vault_path}",
                )
            )

    def _check_source_dir(self) -> None:
        """Check if source directory/directories exist within vault."""
        vault_path = self.config.vault_path

        if not vault_path:
            # Already handled by vault path check
            return

        # Use source_subdirs if configured, otherwise use source_dir
        if self.config.source_subdirs:
            source_dirs = self.config.source_subdirs
            dir_description = f"{len(source_dirs)} source directories"
        else:
            source_dirs = [self.config.source_dir]
            dir_description = "source directory"

        # Check each directory
        total_notes = 0
        missing_dirs = []
        invalid_dirs = []

        for source_dir in source_dirs:
            if not source_dir:
                continue

            full_path = Path(vault_path) / source_dir

            if not full_path.exists():
                missing_dirs.append(str(source_dir))
                continue

            if not full_path.is_dir():
                invalid_dirs.append(str(source_dir))
                continue

            # Count notes in directory
            note_files = list(full_path.glob("**/*.md"))
            total_notes += len(note_files)

        # Report results
        if missing_dirs:
            self.results.append(
                CheckResult(
                    name="Source Directory",
                    passed=False,
                    message=f"Source directories not found: {', '.join(missing_dirs)}",
                    severity="error",
                    fixable=True,
                    fix_suggestion="Create the directories or update source_subdirs in config.yaml",
                )
            )
            return

        if invalid_dirs:
            self.results.append(
                CheckResult(
                    name="Source Directory",
                    passed=False,
                    message=f"Source paths are not directories: {', '.join(invalid_dirs)}",
                    severity="error",
                    fixable=False,
                )
            )
            return

        # Success
        self.results.append(
            CheckResult(
                name="Source Directory",
                passed=True,
                message=f"Found {total_notes} markdown files in {dir_description}",
                severity="info",
            )
        )

    def _check_db_path(self) -> None:
        """Check if database path is writable."""
        db_path = Path(self.config.db_path)

        # Check if parent directory exists and is writable
        parent_dir = db_path.parent
        if not parent_dir.exists():
            self.results.append(
                CheckResult(
                    name="Database Path",
                    passed=False,
                    message=f"Database parent directory does not exist: {parent_dir}",
                    severity="error",
                    fixable=True,
                    fix_suggestion=f"Create directory: mkdir -p {parent_dir}",
                )
            )
            return

        # Check if we can write to the directory
        try:
            test_file = parent_dir / ".write_test"
            test_file.touch()
            test_file.unlink()

            if db_path.exists():
                msg = f"Database file exists and is accessible: {db_path}"
            else:
                msg = f"Database will be created at: {db_path}"

            self.results.append(
                CheckResult(
                    name="Database Path",
                    passed=True,
                    message=msg,
                    severity="info",
                )
            )
        except PermissionError:
            self.results.append(
                CheckResult(
                    name="Database Path",
                    passed=False,
                    message=f"No write permission for database directory: {parent_dir}",
                    severity="error",
                    fixable=True,
                    fix_suggestion=f"Grant write permissions: chmod +w {parent_dir}",
                )
            )

    def _check_llm_provider(self) -> None:
        """Check LLM provider configuration and connectivity."""
        provider_name = self.config.llm_provider

        # Skip check if using agent system (will be checked separately)
        if (
            self.config.use_agent_system
            or self.config.use_langgraph
            or self.config.use_pydantic_ai
        ):
            provider_type = (
                "LangGraph" if self.config.use_langgraph else "legacy agent system"
            )
            self.results.append(
                CheckResult(
                    name="LLM Provider",
                    passed=True,
                    message=f"Using {provider_type} (OpenRouter/PydanticAI will be checked during sync)",
                    severity="info",
                )
            )
            return

        # Check provider-specific configuration
        if provider_name == "openrouter":
            api_key = self.config.openrouter_api_key
            if not api_key or api_key == "your_api_key_here":
                self.results.append(
                    CheckResult(
                        name="LLM Provider (OpenRouter)",
                        passed=False,
                        message="OPENROUTER_API_KEY is not set or invalid",
                        severity="error",
                        fixable=True,
                        fix_suggestion="Set OPENROUTER_API_KEY in your .env file with a valid API key from https://openrouter.ai/",
                    )
                )
                return

            # Try to connect to OpenRouter
            try:
                from ..providers.factory import ProviderFactory

                provider = ProviderFactory.create_from_config(self.config)

                # Try to list models as a connectivity check
                models = provider.list_models()

                if models:
                    self.results.append(
                        CheckResult(
                            name="LLM Provider (OpenRouter)",
                            passed=True,
                            message=f"Connected to OpenRouter ({len(models)} models available)",
                            severity="info",
                        )
                    )
                else:
                    self.results.append(
                        CheckResult(
                            name="LLM Provider (OpenRouter)",
                            passed=False,
                            message="OpenRouter API key may be invalid (no models returned)",
                            severity="warning",
                            fixable=True,
                            fix_suggestion="Verify your OPENROUTER_API_KEY is valid at https://openrouter.ai/",
                        )
                    )
            except Exception as e:
                self.results.append(
                    CheckResult(
                        name="LLM Provider (OpenRouter)",
                        passed=False,
                        message=f"Failed to connect to OpenRouter: {str(e)}",
                        severity="error",
                        fixable=True,
                        fix_suggestion="Check your internet connection and API key",
                    )
                )

        elif provider_name == "ollama":
            # Check Ollama connectivity
            try:
                from ..providers.factory import ProviderFactory

                provider = ProviderFactory.create_from_config(self.config)

                if provider.check_connection():
                    # List available models
                    models = provider.list_models()
                    self.results.append(
                        CheckResult(
                            name="LLM Provider (Ollama)",
                            passed=True,
                            message=f"Connected to Ollama ({len(models)} models available)",
                            severity="info",
                        )
                    )
                else:
                    self.results.append(
                        CheckResult(
                            name="LLM Provider (Ollama)",
                            passed=False,
                            message=f"Cannot connect to Ollama at {self.config.ollama_base_url}",
                            severity="error",
                            fixable=True,
                            fix_suggestion="Start Ollama: ollama serve",
                        )
                    )
            except Exception as e:
                self.results.append(
                    CheckResult(
                        name="LLM Provider (Ollama)",
                        passed=False,
                        message=f"Failed to connect to Ollama: {str(e)}",
                        severity="error",
                        fixable=True,
                        fix_suggestion="Start Ollama: ollama serve",
                    )
                )

        elif provider_name == "lm_studio":
            # Check LM Studio connectivity
            try:
                from ..providers.factory import ProviderFactory

                provider = ProviderFactory.create_from_config(self.config)

                if provider.check_connection():
                    models = provider.list_models()
                    self.results.append(
                        CheckResult(
                            name="LLM Provider (LM Studio)",
                            passed=True,
                            message=f"Connected to LM Studio ({len(models)} models available)",
                            severity="info",
                        )
                    )
                else:
                    self.results.append(
                        CheckResult(
                            name="LLM Provider (LM Studio)",
                            passed=False,
                            message=f"Cannot connect to LM Studio at {self.config.lm_studio_base_url}",
                            severity="error",
                            fixable=True,
                            fix_suggestion="Start LM Studio and ensure the server is running",
                        )
                    )
            except Exception as e:
                self.results.append(
                    CheckResult(
                        name="LLM Provider (LM Studio)",
                        passed=False,
                        message=f"Failed to connect to LM Studio: {str(e)}",
                        severity="error",
                        fixable=True,
                        fix_suggestion="Start LM Studio and ensure the server is running",
                    )
                )

    def _check_anki_connectivity(self) -> None:
        """Check Anki/AnkiConnect connectivity."""
        try:
            from ..anki.client import AnkiClient

            with AnkiClient(self.config.anki_connect_url) as anki:
                # Try to get deck names
                decks = anki.get_deck_names()
                self.results.append(
                    CheckResult(
                        name="Anki Connectivity",
                        passed=True,
                        message=f"Connected to Anki ({len(decks)} decks available)",
                        severity="info",
                    )
                )
        except ConnectionError as e:
            self.results.append(
                CheckResult(
                    name="Anki Connectivity",
                    passed=False,
                    message=f"Cannot connect to AnkiConnect: {str(e)}",
                    severity="error",
                    fixable=True,
                    fix_suggestion="1. Start Anki\n2. Install AnkiConnect add-on (2055492159)\n3. Restart Anki",
                )
            )
        except Exception as e:
            self.results.append(
                CheckResult(
                    name="Anki Connectivity",
                    passed=False,
                    message=f"Anki connectivity check failed: {str(e)}",
                    severity="error",
                    fixable=True,
                    fix_suggestion="Ensure Anki is running and AnkiConnect is installed",
                )
            )

    def _check_note_type(self) -> None:
        """Check if configured note type exists in Anki."""
        note_type = self.config.anki_note_type

        if not note_type:
            self.results.append(
                CheckResult(
                    name="Anki Note Type",
                    passed=False,
                    message="ANKI_NOTE_TYPE is not configured",
                    severity="warning",
                    fixable=True,
                    fix_suggestion="Set ANKI_NOTE_TYPE in your .env file (e.g., 'APF::Simple')",
                )
            )
            return

        # Only check if Anki is accessible
        if not any(r.name == "Anki Connectivity" and r.passed for r in self.results):
            return

        try:
            from ..anki.client import AnkiClient

            with AnkiClient(self.config.anki_connect_url) as anki:
                model_names = anki.get_model_names()

                if note_type in model_names:
                    self.results.append(
                        CheckResult(
                            name="Anki Note Type",
                            passed=True,
                            message=f"Note type exists in Anki: {note_type}",
                            severity="info",
                        )
                    )
                else:
                    self.results.append(
                        CheckResult(
                            name="Anki Note Type",
                            passed=False,
                            message=f"Note type not found in Anki: {note_type}",
                            severity="warning",
                            fixable=True,
                            fix_suggestion=f"Available types: {', '.join(model_names[:5])}...",
                        )
                    )
        except Exception:
            # Anki connectivity already failed, skip this check
            pass

    def _check_deck_name(self) -> None:
        """Check if configured deck exists in Anki."""
        deck_name = self.config.anki_deck_name

        if not deck_name:
            self.results.append(
                CheckResult(
                    name="Anki Deck",
                    passed=False,
                    message="ANKI_DECK_NAME is not configured",
                    severity="warning",
                    fixable=True,
                    fix_suggestion="Set ANKI_DECK_NAME in your .env file",
                )
            )
            return

        # Only check if Anki is accessible
        if not any(r.name == "Anki Connectivity" and r.passed for r in self.results):
            return

        try:
            from ..anki.client import AnkiClient

            with AnkiClient(self.config.anki_connect_url) as anki:
                deck_names = anki.get_deck_names()

                if deck_name in deck_names:
                    self.results.append(
                        CheckResult(
                            name="Anki Deck",
                            passed=True,
                            message=f"Deck exists in Anki: {deck_name}",
                            severity="info",
                        )
                    )
                else:
                    self.results.append(
                        CheckResult(
                            name="Anki Deck",
                            passed=False,
                            message=f"Deck not found in Anki: {deck_name} (will be created)",
                            severity="warning",
                            fixable=False,
                        )
                    )
        except Exception:
            # Anki connectivity already failed, skip this check
            pass

    def _check_git_repo(self) -> None:
        """Check if vault is a git repository."""
        vault_path = self.config.vault_path
        if not vault_path:
            return

        git_dir = Path(vault_path) / ".git"
        if not git_dir.exists():
            self.results.append(
                CheckResult(
                    name="Git Repository",
                    passed=False,
                    message="Vault is not a git repository",
                    severity="warning",
                    fixable=True,
                    fix_suggestion="Initialize git repository: git init",
                )
            )
        else:
            self.results.append(
                CheckResult(
                    name="Git Repository",
                    passed=True,
                    message="Vault is a git repository",
                    severity="info",
                )
            )

    def _check_vault_structure(self) -> None:
        """Check for standard Obsidian vault structure."""
        vault_path = self.config.vault_path
        if not vault_path:
            return

        obsidian_dir = Path(vault_path) / ".obsidian"
        if not obsidian_dir.exists():
            self.results.append(
                CheckResult(
                    name="Vault Structure",
                    passed=False,
                    message=".obsidian directory not found",
                    severity="warning",
                    fixable=False,
                    fix_suggestion="Ensure this is a valid Obsidian vault",
                )
            )
        else:
            self.results.append(
                CheckResult(
                    name="Vault Structure",
                    passed=True,
                    message="Valid Obsidian vault structure found",
                    severity="info",
                )
            )

    def _check_disk_space(self) -> None:
        """Check available disk space."""
        paths_to_check = [
            ("Database", self.config.db_path.parent),
            ("Logs", self.config.project_log_dir),
        ]

        # Deduplicate paths
        checked_paths = set()

        for name, path in paths_to_check:
            if not path.exists():
                continue

            resolved = path.resolve()
            if resolved in checked_paths:
                continue
            checked_paths.add(resolved)

            try:
                total, used, free = shutil.disk_usage(path)
                free_mb = free / (1024 * 1024)

                if free_mb < 100:
                    self.results.append(
                        CheckResult(
                            name=f"Disk Space ({name})",
                            passed=False,
                            message=f"Critical low disk space: {free_mb:.1f}MB free",
                            severity="error",
                            fixable=True,
                            fix_suggestion="Free up disk space",
                        )
                    )
                elif free_mb < 500:
                    self.results.append(
                        CheckResult(
                            name=f"Disk Space ({name})",
                            passed=False,
                            message=f"Low disk space: {free_mb:.1f}MB free",
                            severity="warning",
                            fixable=True,
                            fix_suggestion="Free up disk space",
                        )
                    )
                else:
                    self.results.append(
                        CheckResult(
                            name=f"Disk Space ({name})",
                            passed=True,
                            message=f"Sufficient disk space: {free_mb:.1f}MB free",
                            severity="info",
                        )
                    )
            except Exception as e:
                logger.warning("disk_space_check_failed", path=str(path), error=str(e))

    def _check_memory(self) -> None:
        """Check system memory."""
        # Only relevant for local LLMs
        if self.config.llm_provider not in ("ollama", "lm_studio"):
            return

        try:
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024 * 1024 * 1024)

            if available_gb < 4.0:
                self.results.append(
                    CheckResult(
                        name="System Memory",
                        passed=False,
                        message=f"Low available memory for local LLM: {available_gb:.1f}GB",
                        severity="warning",
                        fixable=True,
                        fix_suggestion="Close other applications or use a cloud provider",
                    )
                )
            else:
                self.results.append(
                    CheckResult(
                        name="System Memory",
                        passed=True,
                        message=f"Available memory: {available_gb:.1f}GB",
                        severity="info",
                    )
                )
        except Exception as e:
            logger.warning("memory_check_failed", error=str(e))

    def _check_network_latency(self, check_anki: bool, check_llm: bool) -> None:
        """Check network latency to services."""
        import time

        if check_anki:
            try:
                start = time.time()
                from ..anki.client import AnkiClient

                with AnkiClient(self.config.anki_connect_url) as anki:
                    anki.invoke("version")
                latency = (time.time() - start) * 1000

                if latency > 200:
                    self.results.append(
                        CheckResult(
                            name="Anki Latency",
                            passed=False,
                            message=f"High latency to AnkiConnect: {latency:.0f}ms",
                            severity="warning",
                            fixable=False,
                        )
                    )
                else:
                    self.results.append(
                        CheckResult(
                            name="Anki Latency",
                            passed=True,
                            message=f"AnkiConnect latency: {latency:.0f}ms",
                            severity="info",
                        )
                    )
            except Exception:
                pass  # Already handled by connectivity check


def run_preflight_checks(
    config: Config, check_anki: bool = True, check_llm: bool = True
) -> tuple[bool, list[CheckResult]]:
    """Run pre-flight checks and return results.

    Args:
        config: Service configuration
        check_anki: Whether to check Anki connectivity
        check_llm: Whether to check LLM provider connectivity

    Returns:
        Tuple of (all_passed, results)
    """
    checker = PreflightChecker(config)
    return checker.run_all_checks(check_anki=check_anki, check_llm=check_llm)
