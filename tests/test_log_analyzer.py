"""Tests for log analyzer utilities."""

from pathlib import Path

import pytest

from obsidian_anki_sync.utils.log_analyzer import LogAnalyzer


class TestLogAnalyzer:
    """Test log analyzer functionality."""

    def test_find_log_files(self, temp_dir):
        """Test finding log files."""
        log_dir = temp_dir / "logs"
        log_dir.mkdir()

        # Create some log files
        (log_dir / "obsidian-anki-sync_2025-01-15.log").write_text("test")
        (log_dir / "errors_2025-01-15.log").write_text("error")
        (log_dir / "other.log").write_text("other")

        analyzer = LogAnalyzer(log_dir=log_dir)

        # Find all log files
        all_logs = analyzer.find_log_files()
        assert len(all_logs) >= 3

        # Find only error logs
        error_logs = analyzer.find_log_files(error_logs_only=True)
        assert len(error_logs) >= 1
        assert all("errors_" in str(f) for f in error_logs)

    def test_parse_log_line(self, temp_dir):
        """Test parsing a log line."""
        analyzer = LogAnalyzer(log_dir=temp_dir)

        # Valid log line
        line = "2025-01-15 10:30:45.123 | ERROR    | module:function:42 - error message file=test.md error=test"
        entry = analyzer._parse_log_line(line, 1)

        assert entry is not None
        assert entry["level"] == "ERROR"
        assert entry["name"] == "module"
        assert entry["function"] == "function"
        assert entry["line"] == 42
        assert entry["file"] == "test.md"
        assert entry["error"] == "test"

        # Invalid log line
        invalid_line = "not a log line"
        entry = analyzer._parse_log_line(invalid_line, 1)
        assert entry is None

    def test_parse_log_file(self, temp_dir):
        """Test parsing a log file."""
        log_dir = temp_dir / "logs"
        log_dir.mkdir()

        log_file = log_dir / "test.log"
        log_content = """2025-01-15 10:30:45.123 | ERROR    | module:function:42 - error message file=test.md
2025-01-15 10:30:46.456 | INFO     | module:function:43 - info message
2025-01-15 10:30:47.789 | WARNING  | module:function:44 - warning message
"""
        log_file.write_text(log_content, encoding="utf-8")

        analyzer = LogAnalyzer(log_dir=log_dir)
        entries = analyzer.parse_log_file(log_file)

        assert len(entries) == 3
        assert entries[0]["level"] == "ERROR"
        assert entries[1]["level"] == "INFO"
        assert entries[2]["level"] == "WARNING"

    def test_analyze_errors(self, temp_dir):
        """Test error analysis."""
        log_dir = temp_dir / "logs"
        log_dir.mkdir()

        # Create error log file
        error_log = log_dir / "errors_2025-01-15.log"
        error_log_content = """2025-01-15 10:30:45.123 | ERROR    | sync.engine:_process_single_note:42 - card_generation_failed file=test.md error_type=ParserError error_msg=Missing field
2025-01-15 10:30:46.456 | ERROR    | sync.engine:_process_single_note:43 - card_generation_failed file=test2.md error_type=ValidationError error_msg=Invalid format
2025-01-15 10:30:47.789 | ERROR    | sync.indexer:index_vault:44 - indexing_failed file=test3.md error_type=ParserError error_msg=Parse error
"""
        error_log.write_text(error_log_content, encoding="utf-8")

        analyzer = LogAnalyzer(log_dir=log_dir)
        analysis = analyzer.analyze_errors()

        assert analysis["total_errors"] == 3
        assert "ParserError" in analysis["errors_by_type"]
        assert analysis["errors_by_type"]["ParserError"] == 2
        assert analysis["errors_by_type"]["ValidationError"] == 1

    def test_generate_summary_report(self, temp_dir):
        """Test summary report generation."""
        log_dir = temp_dir / "logs"
        log_dir.mkdir()

        # Create log file with various entries
        log_file = log_dir / "obsidian-anki-sync_2025-01-15.log"
        log_content = """2025-01-15 10:30:45.123 | ERROR    | module:function:42 - error message
2025-01-15 10:30:46.456 | INFO     | module:function:43 - llm_request_start operation=qa_extraction
2025-01-15 10:30:47.789 | INFO     | module:function:44 - sync_started
2025-01-15 10:30:48.012 | WARNING  | module:function:45 - warning message
"""
        log_file.write_text(log_content, encoding="utf-8")

        analyzer = LogAnalyzer(log_dir=log_dir)
        summary = analyzer.generate_summary_report(days=7)

        assert summary["total_log_entries"] >= 4
        assert "ERROR" in summary["levels"]
        assert "INFO" in summary["levels"]
        assert "WARNING" in summary["levels"]
        assert summary["errors"]["total_errors"] >= 1
