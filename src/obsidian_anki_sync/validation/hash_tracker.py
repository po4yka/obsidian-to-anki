"""File hash tracking for incremental validation."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

CACHE_FILE = ".validation_cache.json"
CACHE_VERSION = 1


class HashTracker:
    """Tracks file hashes to enable incremental validation.

    Stores file hashes and validation status in a JSON cache file.
    This allows skipping unchanged files on subsequent validation runs.
    """

    def __init__(self, vault_root: Path) -> None:
        """Initialize hash tracker.

        Args:
            vault_root: Root path of the vault (used for relative paths)
        """
        self.vault_root = vault_root
        self.cache_file = vault_root / CACHE_FILE
        self.cache = self._load_cache()

    def _load_cache(self) -> dict[str, Any]:
        """Load cache from file or return empty cache."""
        if not self.cache_file.exists():
            return {"version": CACHE_VERSION, "files": {}}

        try:
            with open(self.cache_file, encoding="utf-8") as f:
                data = json.load(f)

            # Check version compatibility
            if data.get("version") != CACHE_VERSION:
                return {"version": CACHE_VERSION, "files": {}}

            return data
        except (json.JSONDecodeError, OSError):
            return {"version": CACHE_VERSION, "files": {}}

    def save_cache(self) -> None:
        """Save cache to file."""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except OSError as e:
            # Log warning but don't fail validation
            print(f"Warning: Could not save validation cache: {e}")

    def compute_hash(self, filepath: Path) -> str:
        """Compute MD5 hash of file content.

        Args:
            filepath: Path to file

        Returns:
            Hex string of MD5 hash, or empty string on error
        """
        try:
            content = filepath.read_bytes()
            return hashlib.md5(content).hexdigest()
        except OSError:
            return ""

    def get_relative_path(self, filepath: Path) -> str:
        """Get path relative to vault root.

        Args:
            filepath: Absolute or relative path

        Returns:
            Path string relative to vault root
        """
        try:
            return str(filepath.relative_to(self.vault_root))
        except ValueError:
            return str(filepath)

    def is_changed(self, filepath: Path) -> bool:
        """Check if file changed since last validation.

        Args:
            filepath: Path to check

        Returns:
            True if file is new or changed, False if unchanged
        """
        rel_path = self.get_relative_path(filepath)
        cached = self.cache.get("files", {}).get(rel_path)

        if not cached:
            return True

        current_hash = self.compute_hash(filepath)
        return current_hash != cached.get("hash")

    def update(self, filepath: Path, passed: bool, issues_count: int) -> None:
        """Update cache entry after validation.

        Args:
            filepath: Path to validated file
            passed: Whether validation passed (no issues)
            issues_count: Number of issues found
        """
        rel_path = self.get_relative_path(filepath)
        file_hash = self.compute_hash(filepath)

        self.cache.setdefault("files", {})[rel_path] = {
            "hash": file_hash,
            "last_validated": datetime.now().isoformat(),
            "status": "passed" if passed else "failed",
            "issues_count": issues_count,
        }

    def get_changed_files(self, files: list[Path]) -> tuple[list[Path], int]:
        """Filter to only changed files.

        Args:
            files: List of file paths to check

        Returns:
            Tuple of (changed_files, skipped_count)
        """
        changed = []
        skipped = 0

        for filepath in files:
            if self.is_changed(filepath):
                changed.append(filepath)
            else:
                skipped += 1

        return changed, skipped

    def clear_cache(self) -> None:
        """Clear all cached entries."""
        self.cache = {"version": CACHE_VERSION, "files": {}}
        self.save_cache()

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dict with total_cached, passed, and failed counts
        """
        files = self.cache.get("files", {})
        passed = sum(1 for f in files.values() if f.get("status") == "passed")
        failed = sum(1 for f in files.values() if f.get("status") == "failed")

        return {
            "total_cached": len(files),
            "passed": passed,
            "failed": failed,
        }
