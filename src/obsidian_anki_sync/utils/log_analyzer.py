"""Utilities for analyzing log files and identifying error patterns.

This module provides functionality to parse log files, extract error patterns,
and generate summary reports for diagnosing issues.
"""

import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from ..utils.logging import get_logger

logger = get_logger(__name__)


class LogAnalyzer:
    """Analyze log files for error patterns and trends."""

    def __init__(self, log_dir: Path | None = None):
        """Initialize log analyzer.

        Args:
            log_dir: Directory containing log files (default: ./logs)
        """
        if log_dir is None:
            log_dir = Path("./logs")
        self.log_dir = Path(log_dir)

    def find_log_files(
        self, pattern: str = "*.log", error_logs_only: bool = False
    ) -> list[Path]:
        """Find log files in the log directory.

        Args:
            pattern: Glob pattern for log files
            error_logs_only: If True, only return error log files

        Returns:
            List of log file paths, sorted by modification time (newest first)
        """
        if not self.log_dir.exists():
            return []

        if error_logs_only:
            pattern = "errors_*.log"

        log_files = sorted(
            self.log_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )

        return log_files

    def parse_log_file(self, log_file: Path) -> list[dict[str, Any]]:
        """Parse a log file and extract log entries.

        Args:
            log_file: Path to log file

        Returns:
            List of parsed log entries
        """
        entries = []

        if not log_file.exists():
            return entries

        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    entry = self._parse_log_line(line, line_num)
                    if entry:
                        entries.append(entry)
        except (IOError, UnicodeDecodeError) as e:
            logger.warning(
                "failed_to_parse_log_file",
                log_file=str(log_file),
                error=str(e),
            )

        return entries

    def _parse_log_line(self, line: str, line_num: int) -> dict[str, Any] | None:
        """Parse a single log line.

        Args:
            line: Log line content
            line_num: Line number

        Returns:
            Parsed log entry dict or None if line doesn't match expected format
        """
        # Loguru format: {time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}
        pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{3})?) \| (\w+)\s+\| ([^:]+):([^:]+):(\d+) - (.+)"
        match = re.match(pattern, line)

        if not match:
            return None

        timestamp_str, level, name, function, line_no, message = match.groups()

        try:
            timestamp = datetime.strptime(
                timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            try:
                timestamp = datetime.strptime(
                    timestamp_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                timestamp = None

        # Extract structured fields from message
        structured_fields = self._extract_structured_fields(message)

        return {
            "timestamp": timestamp,
            "level": level,
            "name": name,
            "function": function,
            "line": int(line_no) if line_no.isdigit() else None,
            "message": message,
            "line_num": line_num,
            **structured_fields,
        }

    def _extract_structured_fields(self, message: str) -> dict[str, Any]:
        """Extract structured fields from log message.

        Args:
            message: Log message

        Returns:
            Dictionary of extracted fields
        """
        fields = {}

        # Extract common patterns like file=, error=, etc.
        patterns = {
            "file": r"file=([^\s|]+)",
            "error": r"error=([^\s|]+)",
            "error_type": r"error_type=([^\s|]+)",
            "error_msg": r"error_msg=([^\s|]+)",
            "source_path": r"source_path=([^\s|]+)",
            "model": r"model=([^\s|]+)",
            "operation": r"operation=([^\s|]+)",
        }

        for field_name, pattern in patterns.items():
            match = re.search(pattern, message)
            if match:
                fields[field_name] = match.group(1)

        return fields

    def analyze_errors(
        self, log_files: list[Path] | None = None, days: int | None = None
    ) -> dict[str, Any]:
        """Analyze errors in log files.

        Args:
            log_files: List of log files to analyze (if None, finds all)
            days: Only analyze logs from last N days (if None, analyze all)

        Returns:
            Dictionary with error analysis results
        """
        if log_files is None:
            log_files = self.find_log_files(error_logs_only=True)

        if not log_files:
            return {
                "total_errors": 0,
                "errors_by_type": {},
                "errors_by_file": {},
                "errors_by_stage": {},
                "recent_errors": [],
            }

        all_entries = []
        for log_file in log_files:
            entries = self.parse_log_file(log_file)
            all_entries.extend(entries)

        # Filter by date if specified
        if days:
            cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
            all_entries = [
                e
                for e in all_entries
                if e.get("timestamp") and e["timestamp"].timestamp() > cutoff_date
            ]

        # Filter to ERROR and CRITICAL levels
        error_entries = [
            e for e in all_entries if e.get("level") in ("ERROR", "CRITICAL")
        ]

        # Analyze errors
        errors_by_type: Counter[str] = Counter()
        errors_by_file: Counter[str] = Counter()
        errors_by_stage: Counter[str] = Counter()

        for entry in error_entries:
            error_type = entry.get("error_type") or entry.get(
                "message", "Unknown")
            errors_by_type[error_type] += 1

            file_path = entry.get("file") or entry.get("source_path")
            if file_path:
                errors_by_file[file_path] += 1

            # Try to determine processing stage from message
            message = entry.get("message", "")
            if "parsing" in message.lower() or "parser" in message.lower():
                errors_by_stage["parsing"] += 1
            elif "validation" in message.lower() or "validator" in message.lower():
                errors_by_stage["validation"] += 1
            elif "llm" in message.lower() or "provider" in message.lower():
                errors_by_stage["llm"] += 1
            elif "generation" in message.lower() or "generator" in message.lower():
                errors_by_stage["generation"] += 1
            elif "indexing" in message.lower():
                errors_by_stage["indexing"] += 1
            else:
                errors_by_stage["other"] += 1

        # Get recent errors (last 20)
        recent_errors = sorted(
            error_entries,
            key=lambda e: e.get("timestamp").timestamp() if e.get(
                "timestamp") else 0,
            reverse=True,
        )[:20]

        return {
            "total_errors": len(error_entries),
            "errors_by_type": dict(errors_by_type.most_common(10)),
            "errors_by_file": dict(errors_by_file.most_common(10)),
            "errors_by_stage": dict(errors_by_stage),
            "recent_errors": [
                {
                    "timestamp": str(e.get("timestamp", "")),
                    "level": e.get("level"),
                    "error_type": e.get("error_type"),
                    "file": e.get("file") or e.get("source_path"),
                    # Truncate long messages
                    "message": e.get("message", "")[:200],
                }
                for e in recent_errors
            ],
        }

    def generate_summary_report(
        self, days: int = 7
    ) -> dict[str, Any]:
        """Generate a summary report of recent activity.

        Args:
            days: Number of days to analyze

        Returns:
            Summary report dictionary
        """
        log_files = self.find_log_files()
        all_entries = []

        for log_file in log_files:
            entries = self.parse_log_file(log_file)
            all_entries.extend(entries)

        # Filter by date
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        recent_entries = [
            e
            for e in all_entries
            if e.get("timestamp") and e["timestamp"].timestamp() > cutoff_date
        ]

        # Count by level
        levels = Counter(e.get("level", "UNKNOWN") for e in recent_entries)

        # Count by operation type
        operations = Counter()
        for entry in recent_entries:
            message = entry.get("message", "")
            if "llm_request" in message:
                operations["llm_requests"] += 1
            elif "sync" in message.lower():
                operations["sync_operations"] += 1
            elif "index" in message.lower():
                operations["index_operations"] += 1

        # Get error analysis
        error_analysis = self.analyze_errors(days=days)

        return {
            "period_days": days,
            "total_log_entries": len(recent_entries),
            "levels": dict(levels),
            "operations": dict(operations),
            "errors": error_analysis,
        }
