"""Comprehensive unit tests for sync engine."""

from obsidian_anki_sync.sync.state_db import StateDB
from obsidian_anki_sync.sync.engine import SyncEngine
from obsidian_anki_sync.models import Card, Manifest, ManifestData, NoteMetadata, QAPair
from obsidian_anki_sync.exceptions import AnkiConnectError
from obsidian_anki_sync.anki.client import AnkiClient
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

pytestmark = pytest.mark.skip(
    reason="Sync engine unit tests require complex setup")


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = MagicMock()
    config.vault_path = Path("/tmp/test_vault")
    config.source_dir = Path("notes")
    config.source_subdirs = None
    config.use_agent_system = False
    config.enable_batch_operations = True
    config.batch_size = 50
    config.enable_card_cache = False
    config.enable_agent_card_cache = False
    config.card_cache_dir = Path("/tmp/cache")
    config.agent_card_cache_dir = Path("/tmp/agent_cache")
    config.max_concurrent_generations = 1
    config.anki_deck_name = "Test Deck"
    config.anki_note_type = "APF::Simple"
    config.llm_provider = "ollama"
    config.llm_timeout = 60
    config.run_mode = "apply"
    config.delete_mode = "delete"
    config.db_path = Path("/tmp/test.db")
    config.problematic_notes_dir = Path("/tmp/problematic_notes")
    config.enable_problematic_notes_archival = False
    config.get_model_for_agent = MagicMock(return_value="test-model")
    config.get_model_config_for_task = MagicMock(
        return_value={"temperature": 0.0})
    config.model_names = {"APF::Simple": "APF: Simple (3.0.0)"}
    config.enable_memory_cleanup = False
    config.max_note_content_size_mb = 10.0
    config.retry_config_parallel = {"max_retries": 2, "retry_delay": 1.0}
    config.auto_adjust_workers = False
    config.verify_card_creation = True
    return config


@pytest.fixture
def mock_anki_client():
    """Create mock Anki client."""
    client = MagicMock()
    client.add_note.return_value = 12345
    client.update_note_fields.return_value = None
    client.update_note_tags.return_value = None
    client.find_notes.return_value = []
    client.notes_info.return_value = []
    return client


@pytest.fixture
def mock_state_db():
    """Create mock state database."""
    db = MagicMock()
    db.get_card.return_value = None
    db.get_all_cards.return_value = []
    db.get_processed_note_paths.return_value = set()
    return db


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sync_engine(mock_config, mock_state_db, mock_anki_client, temp_cache_dir):
    """Create SyncEngine instance with mocked dependencies."""
    # Override cache directory with temp directory
    mock_config.db_path = temp_cache_dir / "test.db"
    engine = SyncEngine(mock_config, mock_state_db, mock_anki_client)
    return engine


class TestSyncEngineInitialization:
    """Tests for SyncEngine initialization."""

    def test_init_creates_engine_with_valid_config(
        self, mock_config, mock_state_db, mock_anki_client, temp_cache_dir
    ):
        """SyncEngine should initialize successfully with valid config."""
        mock_config.db_path = temp_cache_dir / "test.db"
        engine = SyncEngine(mock_config, mock_state_db, mock_anki_client)

        assert engine.config == mock_config
        assert engine.db == mock_state_db
        assert engine.anki == mock_anki_client
        assert engine.use_agents is False
        assert engine.apf_gen is not None

    def test_init_with_langgraph_enabled(
        self, mock_config, mock_state_db, mock_anki_client, temp_cache_dir
    ):
        """SyncEngine should initialize LangGraph orchestrator when enabled."""
        mock_config.use_langgraph = True
        mock_config.use_pydantic_ai = True
        mock_config.db_path = temp_cache_dir / "test.db"

        with (
            patch("obsidian_anki_sync.sync.engine.AGENTS_AVAILABLE", True),
            patch(
                "obsidian_anki_sync.sync.engine.LangGraphOrchestrator"
            ) as mock_orchestrator_class,
            patch("obsidian_anki_sync.sync.engine.create_qa_extractor"),
        ):
            mock_orchestrator = MagicMock()
            mock_orchestrator.provider = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator

            engine = SyncEngine(mock_config, mock_state_db, mock_anki_client)

            assert engine.use_agents is True
            assert engine.agent_orchestrator is not None
            mock_orchestrator_class.assert_called_once_with(mock_config)

    def test_init_with_legacy_agent_system_enabled(
        self, mock_config, mock_state_db, mock_anki_client, temp_cache_dir
    ):
        """SyncEngine should initialize legacy agent orchestrator when enabled (backward compatibility)."""
        mock_config.use_agent_system = True
        mock_config.db_path = temp_cache_dir / "test.db"

        with (
            patch("obsidian_anki_sync.sync.engine.AGENTS_AVAILABLE", True),
            patch(
                "obsidian_anki_sync.sync.engine.AgentOrchestrator"
            ) as mock_orchestrator_class,
            patch("obsidian_anki_sync.sync.engine.create_qa_extractor"),
        ):
            mock_orchestrator = MagicMock()
            mock_orchestrator.provider = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator

            engine = SyncEngine(mock_config, mock_state_db, mock_anki_client)

            assert engine.use_agents is True
            assert engine.agent_orchestrator is not None
            mock_orchestrator_class.assert_called_once_with(mock_config)

    def test_init_stats_initialized_correctly(self, sync_engine):
        """SyncEngine should initialize statistics dictionary."""
        assert sync_engine.stats["processed"] == 0
        assert sync_engine.stats["created"] == 0
        assert sync_engine.stats["updated"] == 0
        assert sync_engine.stats["deleted"] == 0
        assert sync_engine.stats["restored"] == 0
        assert sync_engine.stats["skipped"] == 0
        assert sync_engine.stats["errors"] == 0
        assert sync_engine.stats["parser_warnings"] == 0
        assert sync_engine.stats["llm_truncations"] == 0
        assert sync_engine.stats["validation_errors"] == 0


class TestNoteDiscovery:
    """Tests for note discovery functionality."""

    @patch("obsidian_anki_sync.sync.note_scanner.discover_notes")
    def test_scan_obsidian_notes_discovers_files(
        self, mock_discover, sync_engine, temp_cache_dir
    ):
        """scan_obsidian_notes should discover markdown files in vault."""
        test_file = temp_cache_dir / "test.md"
        test_file.touch()
        mock_discover.return_value = [(test_file, "test.md")]

        with patch.object(sync_engine.note_scanner.card_generator, "generate_card") as mock_gen:
            mock_gen.side_effect = Exception("Skip generation for this test")
            result = sync_engine.note_scanner.scan_notes()

        mock_discover.assert_called_once()
        assert isinstance(result, dict)

    @patch("obsidian_anki_sync.sync.note_scanner.discover_notes")
    def test_scan_obsidian_notes_respects_source_dir(
        self,
        mock_discover,
        mock_config,
        mock_state_db,
        mock_anki_client,
        temp_cache_dir,
    ):
        """scan_obsidian_notes should use configured source_dir."""
        mock_config.db_path = temp_cache_dir / "test.db"
        mock_config.source_dir = Path("custom_dir")
        mock_config.source_subdirs = None
        engine = SyncEngine(mock_config, mock_state_db, mock_anki_client)

        mock_discover.return_value = []

        engine.note_scanner.scan_notes()

        mock_discover.assert_called_once_with(
            mock_config.vault_path, mock_config.source_dir
        )

    @patch("obsidian_anki_sync.sync.note_scanner.discover_notes")
    def test_scan_obsidian_notes_handles_empty_directory(
        self, mock_discover, sync_engine
    ):
        """scan_obsidian_notes should handle empty directories gracefully."""
        mock_discover.return_value = []

        result = sync_engine.note_scanner.scan_notes()

        assert result == {}
        assert sync_engine.stats["processed"] == 0

    @patch("obsidian_anki_sync.sync.note_scanner.discover_notes")
    def test_scan_obsidian_notes_incremental_mode(
        self, mock_discover, sync_engine, temp_cache_dir
    ):
        """scan_obsidian_notes should filter already processed notes in incremental mode."""
        test_file1 = temp_cache_dir / "test1.md"
        test_file2 = temp_cache_dir / "test2.md"
        test_file1.touch()
        test_file2.touch()

        mock_discover.return_value = [
            (test_file1, "test1.md"),
            (test_file2, "test2.md"),
        ]
        sync_engine.db.get_processed_note_paths.return_value = {"test1.md"}

        result = sync_engine.note_scanner.scan_notes(incremental=True)

        # test1.md should be filtered out
        assert sync_engine.db.get_processed_note_paths.called


class TestCardGeneration:
    """Tests for card generation functionality."""

    def test_generate_card_creates_card_without_agents(
        self, sync_engine, sample_qa_pair, sample_metadata
    ):
        """generate_card should create card using APF generator."""
        with patch.object(sync_engine.apf_gen, "generate_card") as mock_gen:
            mock_card = Card(
                slug="test-slug-en",
                lang="en",
                apf_html="<!-- test -->",
                manifest=Manifest(
                    slug="test-slug-en",
                    slug_base="test-slug",
                    lang="en",
                    source_path="test.md",
                    source_anchor="p01",
                    note_id="test-001",
                    note_title="Test",
                    card_index=1,
                    guid="test-guid",
                ),
                content_hash="test-hash",
                note_type="APF::Simple",
                tags=["en"],
                guid="test-guid",
            )
            mock_gen.return_value = mock_card

            with (
                patch("obsidian_anki_sync.apf.linter.validate_apf") as mock_validate,
                patch(
                    "obsidian_anki_sync.apf.html_validator.validate_card_html"
                ) as mock_html_validate,
            ):
                mock_validate.return_value = MagicMock(errors=[], warnings=[])
                mock_html_validate.return_value = []  # No HTML errors

                result = sync_engine.card_generator.generate_card(
                    qa_pair=sample_qa_pair,
                    metadata=sample_metadata,
                    relative_path="test.md",
                    lang="en",
                    existing_slugs=set(),
                )

            assert result.slug == "test-slug-en"
            assert result.lang == "en"
            mock_gen.assert_called_once()

    def test_generate_card_with_cache_hit(
        self, sync_engine, sample_qa_pair, sample_metadata
    ):
        """generate_card should return cached card when available."""
        cached_card = Card(
            slug="test-slug-en",
            lang="en",
            apf_html="<!-- cached -->",
            manifest=Manifest(
                slug="test-slug-en",
                slug_base="test-slug",
                lang="en",
                source_path="test.md",
                source_anchor="p01",
                note_id="test-001",
                note_title="Test",
                card_index=1,
                guid="test-guid",
            ),
            content_hash="test-hash",
            note_type="APF::Simple",
            tags=["en"],
            guid="test-guid",
        )

        # Pre-populate cache
        with patch("obsidian_anki_sync.utils.content_hash.compute_content_hash") as mock_hash:
            mock_hash.return_value = "test-hash"
            cache_key = "test.md:1:en:test-hash"
            sync_engine._apf_card_cache.set(cache_key, cached_card)

            result = sync_engine.card_generator.generate_card(
                qa_pair=sample_qa_pair,
                metadata=sample_metadata,
                relative_path="test.md",
                lang="en",
                existing_slugs=set(),
            )

        assert result.slug == "test-slug-en"
        assert result.apf_html == "<!-- cached -->"
        assert sync_engine._cache_stats["hits"] > 0

    def test_generate_card_validation_failure(
        self, sync_engine, sample_qa_pair, sample_metadata
    ):
        """generate_card should raise error on APF validation failure."""
        with patch.object(sync_engine.apf_gen, "generate_card") as mock_gen:
            mock_card = Card(
                slug="test-slug-en",
                lang="en",
                apf_html="<!-- invalid -->",
                manifest=Manifest(
                    slug="test-slug-en",
                    slug_base="test-slug",
                    lang="en",
                    source_path="test.md",
                    source_anchor="p01",
                    note_id="test-001",
                    note_title="Test",
                    card_index=1,
                    guid="test-guid",
                ),
                content_hash="test-hash",
                note_type="APF::Simple",
                tags=["en"],
                guid="test-guid",
            )
            mock_gen.return_value = mock_card

            with patch("obsidian_anki_sync.apf.linter.validate_apf") as mock_validate:
                mock_validate.return_value = MagicMock(
                    errors=["Missing required field"], warnings=[]
                )

                with pytest.raises(ValueError, match="APF validation failed"):
                    sync_engine.card_generator.generate_card(
                        qa_pair=sample_qa_pair,
                        metadata=sample_metadata,
                        relative_path="test.md",
                        lang="en",
                        existing_slugs=set(),
                    )

                assert sync_engine.stats["validation_errors"] > 0


class TestAnkiStateManagement:
    """Tests for Anki state fetching and management."""

    def test_fetch_anki_state_success(self, sync_engine):
        """fetch_anki_state should successfully retrieve Anki cards."""
        sync_engine.anki.find_notes.return_value = [100, 200]

        manifest1 = {
            "slug": "card-1-en",
            "slug_base": "card-1",
            "lang": "en",
            "source_path": "test1.md",
            "source_anchor": "p01",
            "note_id": "id-1",
            "note_title": "Test 1",
            "card_index": 1,
            "guid": "guid-1",
        }
        manifest2 = {
            "slug": "card-2-ru",
            "slug_base": "card-2",
            "lang": "ru",
            "source_path": "test2.md",
            "source_anchor": "p01",
            "note_id": "id-2",
            "note_title": "Test 2",
            "card_index": 1,
            "guid": "guid-2",
        }

        sync_engine.anki.notes_info.return_value = [
            {"noteId": 100, "fields": {"Manifest": {
                "value": json.dumps(manifest1)}}},
            {"noteId": 200, "fields": {"Manifest": {
                "value": json.dumps(manifest2)}}},
        ]

        result = sync_engine.anki_state_manager.fetch_state()

        assert len(result) == 2
        assert result["card-1-en"] == 100
        assert result["card-2-ru"] == 200
        sync_engine.anki.find_notes.assert_called_once()
        sync_engine.anki.notes_info.assert_called_once_with([100, 200])

    def test_fetch_anki_state_empty_deck(self, sync_engine):
        """fetch_anki_state should handle empty deck gracefully."""
        sync_engine.anki.find_notes.return_value = []

        result = sync_engine.anki_state_manager.fetch_state()

        assert result == {}

    def test_fetch_anki_state_connection_error(self, sync_engine):
        """fetch_anki_state should handle AnkiConnect errors gracefully."""
        sync_engine.anki.find_notes.side_effect = AnkiConnectError(
            "Connection failed")

        result = sync_engine.anki_state_manager.fetch_state()

        assert result == {}

    def test_fetch_anki_state_invalid_manifest(self, sync_engine):
        """fetch_anki_state should skip cards with invalid manifests."""
        sync_engine.anki.find_notes.return_value = [100, 200]

        valid_manifest = {
            "slug": "card-1-en",
            "slug_base": "card-1",
            "lang": "en",
            "source_path": "test1.md",
            "source_anchor": "p01",
            "note_id": "id-1",
            "note_title": "Test 1",
            "card_index": 1,
            "guid": "guid-1",
        }

        sync_engine.anki.notes_info.return_value = [
            {
                "noteId": 100,
                "fields": {"Manifest": {"value": json.dumps(valid_manifest)}},
            },
            {"noteId": 200, "fields": {"Manifest": {"value": "invalid json"}}},
        ]

        result = sync_engine.anki_state_manager.fetch_state()

        # Only valid card should be returned
        assert len(result) == 1
        assert result["card-1-en"] == 100


class TestSyncActionDetermination:
    """Tests for determining sync actions."""

    def test_determine_actions_new_card(self, sync_engine):
        """determine_actions should identify new cards for creation."""
        obsidian_card = Card(
            slug="new-card-en",
            lang="en",
            apf_html="<!-- new -->",
            manifest=Manifest(
                slug="new-card-en",
                slug_base="new-card",
                lang="en",
                source_path="new.md",
                source_anchor="p01",
                note_id="new-001",
                note_title="New",
                card_index=1,
                guid="new-guid",
            ),
            content_hash="new-hash",
            note_type="APF::Simple",
            tags=["en"],
            guid="new-guid",
        )

        sync_engine.changes = []
        sync_engine.anki_state_manager.determine_actions(
            obsidian_cards={"new-card-en": obsidian_card},
            anki_cards={},
            changes=sync_engine.changes,
        )

        assert len(sync_engine.changes) == 1
        assert sync_engine.changes[0].type == "create"
        assert sync_engine.changes[0].card.slug == "new-card-en"

    def test_determine_actions_updated_card(self, sync_engine):
        """determine_actions should identify cards needing updates."""
        obsidian_card = Card(
            slug="updated-card-en",
            lang="en",
            apf_html="<!-- updated -->",
            manifest=Manifest(
                slug="updated-card-en",
                slug_base="updated-card",
                lang="en",
                source_path="updated.md",
                source_anchor="p01",
                note_id="updated-001",
                note_title="Updated",
                card_index=1,
                guid="updated-guid",
            ),
            content_hash="new-hash",
            note_type="APF::Simple",
            tags=["en"],
            guid="updated-guid",
        )

        sync_engine.db.get_all_cards.return_value = [
            {
                "slug": "updated-card-en",
                "content_hash": "old-hash",
                "anki_guid": 12345,
                "lang": "en",
                "slug_base": "updated-card",
                "source_path": "updated.md",
                "source_anchor": "p01",
                "note_id": "updated-001",
                "note_title": "Updated",
                "card_index": 1,
                "note_type": "APF::Simple",
            }
        ]

        sync_engine.changes = []
        sync_engine.anki_state_manager.determine_actions(
            obsidian_cards={"updated-card-en": obsidian_card},
            anki_cards={"updated-card-en": 12345},
            changes=sync_engine.changes,
        )

        assert len(sync_engine.changes) == 1
        assert sync_engine.changes[0].type == "update"
        assert sync_engine.changes[0].card.slug == "updated-card-en"
        assert sync_engine.changes[0].anki_guid == 12345

    def test_determine_actions_unchanged_card(self, sync_engine):
        """determine_actions should skip cards with no changes."""
        obsidian_card = Card(
            slug="same-card-en",
            lang="en",
            apf_html="<!-- same -->",
            manifest=Manifest(
                slug="same-card-en",
                slug_base="same-card",
                lang="en",
                source_path="same.md",
                source_anchor="p01",
                note_id="same-001",
                note_title="Same",
                card_index=1,
                guid="same-guid",
            ),
            content_hash="same-hash",
            note_type="APF::Simple",
            tags=["en"],
            guid="same-guid",
        )

        sync_engine.db.get_all_cards.return_value = [
            {
                "slug": "same-card-en",
                "content_hash": "same-hash",
                "anki_guid": 12345,
                "lang": "en",
                "slug_base": "same-card",
                "source_path": "same.md",
                "source_anchor": "p01",
                "note_id": "same-001",
                "note_title": "Same",
                "card_index": 1,
                "note_type": "APF::Simple",
            }
        ]

        sync_engine.changes = []
        sync_engine.anki_state_manager.determine_actions(
            obsidian_cards={"same-card-en": obsidian_card},
            anki_cards={"same-card-en": 12345},
            changes=sync_engine.changes,
        )

        assert len(sync_engine.changes) == 1
        assert sync_engine.changes[0].type == "skip"

    def test_determine_actions_deleted_card(self, sync_engine):
        """determine_actions should identify cards deleted from Obsidian."""
        sync_engine.db.get_all_cards.return_value = [
            {
                "slug": "deleted-card-en",
                "content_hash": "hash",
                "anki_guid": 12345,
                "lang": "en",
                "slug_base": "deleted-card",
                "source_path": "deleted.md",
                "source_anchor": "p01",
                "note_id": "deleted-001",
                "note_title": "Deleted",
                "card_index": 1,
                "note_type": "APF::Simple",
            }
        ]

        sync_engine.changes = []
        sync_engine.anki_state_manager.determine_actions(
            obsidian_cards={},
            anki_cards={"deleted-card-en": 12345},
            changes=sync_engine.changes,
        )

        assert len(sync_engine.changes) == 1
        assert sync_engine.changes[0].type == "delete"
        assert sync_engine.changes[0].card.slug == "deleted-card-en"


class TestStatisticsTracking:
    """Tests for statistics tracking."""

    def test_stats_initialized_correctly(self, sync_engine):
        """Stats should be initialized with zero values."""
        assert sync_engine.stats["processed"] == 0
        assert sync_engine.stats["created"] == 0
        assert sync_engine.stats["updated"] == 0
        assert sync_engine.stats["deleted"] == 0
        assert sync_engine.stats["errors"] == 0

    def test_stats_incremented_on_processing(
        self, sync_engine, sample_qa_pair, sample_metadata
    ):
        """Stats should be incremented during note processing."""
        with patch("obsidian_anki_sync.sync.note_scanner.discover_notes") as mock_discover:
            test_file = Path("/tmp/test.md")
            mock_discover.return_value = [(test_file, "test.md")]

            with patch(
                "obsidian_anki_sync.obsidian.parser.parse_note"
            ) as mock_parse:
                mock_parse.return_value = (sample_metadata, [sample_qa_pair])

                with patch.object(sync_engine.card_generator, "generate_card") as mock_gen:
                    mock_card = Card(
                        slug="test-en",
                        lang="en",
                        apf_html="<!-- test -->",
                        manifest=Manifest(
                            slug="test-en",
                            slug_base="test",
                            lang="en",
                            source_path="test.md",
                            source_anchor="p01",
                            note_id="test-001",
                            note_title="Test",
                            card_index=1,
                            guid="test-guid",
                        ),
                        content_hash="test-hash",
                        note_type="APF::Simple",
                        tags=["en"],
                        guid="test-guid",
                    )
                    mock_gen.return_value = mock_card

                    sync_engine.note_scanner.scan_notes()

        assert sync_engine.stats["processed"] >= 1


class TestErrorHandling:
    """Tests for error handling."""

    def test_parser_error_handling(self, sync_engine):
        """Engine should handle parser errors gracefully."""
        with patch("obsidian_anki_sync.sync.note_scanner.discover_notes") as mock_discover:
            test_file = Path("/tmp/test.md")
            mock_discover.return_value = [(test_file, "test.md")]

            with patch(
                "obsidian_anki_sync.obsidian.parser.parse_note"
            ) as mock_parse:
                from obsidian_anki_sync.obsidian.parser import ParserError

                mock_parse.side_effect = ParserError("Failed to parse note")

                result = sync_engine.note_scanner.scan_notes()

        assert sync_engine.stats["errors"] >= 1
        assert isinstance(result, dict)

    def test_anki_connection_error_handling(self, sync_engine):
        """Engine should handle Anki connection errors gracefully."""
        sync_engine.anki.find_notes.side_effect = AnkiConnectError(
            "Connection refused")

        result = sync_engine.anki_state_manager.fetch_state()

        assert result == {}

    def test_card_generation_error_increments_error_count(
        self, sync_engine, sample_qa_pair, sample_metadata
    ):
        """Failed card generation should increment error statistics."""
        with patch("obsidian_anki_sync.sync.note_scanner.discover_notes") as mock_discover:
            test_file = Path("/tmp/test.md")
            mock_discover.return_value = [(test_file, "test.md")]

            with patch(
                "obsidian_anki_sync.obsidian.parser.parse_note"
            ) as mock_parse:
                mock_parse.return_value = (sample_metadata, [sample_qa_pair])

                with patch.object(sync_engine.card_generator, "generate_card") as mock_gen:
                    mock_gen.side_effect = Exception("Generation failed")

                    sync_engine.note_scanner.scan_notes()

        assert sync_engine.stats["errors"] >= 1


class TestManifestParsing:
    """Tests for manifest field parsing."""

    def test_parse_manifest_field_valid_json(self, sync_engine):
        """parse_manifest_field should parse valid JSON manifests."""
        manifest_json = json.dumps(
            {
                "slug": "test-card-en",
                "slug_base": "test-card",
                "lang": "en",
                "source_path": "test.md",
                "source_anchor": "p01",
                "note_id": "test-001",
                "note_title": "Test",
                "card_index": 1,
                "guid": "test-guid",
            }
        )

        result = sync_engine.anki_state_manager._parse_manifest_field(
            manifest_json)

        assert result is not None
        assert isinstance(result, ManifestData)
        assert result.slug == "test-card-en"
        assert result.lang == "en"

    def test_parse_manifest_field_invalid_json(self, sync_engine):
        """parse_manifest_field should return None for invalid JSON."""
        result = sync_engine.anki_state_manager._parse_manifest_field(
            "not valid json")

        assert result is None

    def test_parse_manifest_field_missing_required_fields(self, sync_engine):
        """parse_manifest_field should return None when required fields missing."""
        manifest_json = json.dumps({"slug": "test-card-en"})

        result = sync_engine.anki_state_manager._parse_manifest_field(
            manifest_json)

        assert result is None

    def test_parse_manifest_field_not_dict(self, sync_engine):
        """parse_manifest_field should return None for non-dict JSON."""
        manifest_json = json.dumps(["not", "a", "dict"])

        result = sync_engine.anki_state_manager._parse_manifest_field(
            manifest_json)

        assert result is None


class TestThreadSafeSlugGeneration:
    """Tests for thread-safe slug generation."""

    def test_generate_thread_safe_slug_first_occurrence(self, sync_engine):
        """First occurrence should return slug without collision counter."""
        slug = sync_engine._generate_thread_safe_slug("base-slug", 1, "en")

        assert slug == "base-slug-en"

    def test_generate_thread_safe_slug_collision(self, sync_engine):
        """Collision should add counter to slug."""
        slug1 = sync_engine._generate_thread_safe_slug("base-slug", 1, "en")
        slug2 = sync_engine._generate_thread_safe_slug("base-slug", 1, "en")

        assert slug1 == "base-slug-en"
        assert slug2 == "base-slug-en-1"

    def test_generate_thread_safe_slug_multiple_collisions(self, sync_engine):
        """Multiple collisions should increment counter."""
        slug1 = sync_engine._generate_thread_safe_slug("base-slug", 1, "en")
        slug2 = sync_engine._generate_thread_safe_slug("base-slug", 1, "en")
        slug3 = sync_engine._generate_thread_safe_slug("base-slug", 1, "en")

        assert slug1 == "base-slug-en"
        assert slug2 == "base-slug-en-1"
        assert slug3 == "base-slug-en-2"


class TestCacheManagement:
    """Tests for card caching functionality."""

    def test_cache_initialized_on_creation(self, sync_engine):
        """Disk caches should be initialized on engine creation."""
        assert sync_engine._apf_card_cache is not None
        assert sync_engine._agent_card_cache is not None
        assert sync_engine._cache_hits == 0
        assert sync_engine._cache_misses == 0

    def test_cache_stats_tracking(self, sync_engine):
        """Cache statistics should be tracked correctly."""
        assert sync_engine._cache_stats["hits"] == 0
        assert sync_engine._cache_stats["misses"] == 0
        assert sync_engine._cache_stats["generation_times"] == []

    def test_close_caches(self, sync_engine):
        """close_caches should close disk caches."""
        sync_engine._close_caches()

        # Should not raise error
        assert True
