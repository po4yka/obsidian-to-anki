"""Quick test script for logging."""
from obsidian_anki_sync.utils.logging import configure_logging, get_logger
import tempfile
import pathlib

with tempfile.TemporaryDirectory() as tmpdir:
    configure_logging(log_level="INFO", project_log_dir=pathlib.Path(tmpdir))
    logger = get_logger("test")
    logger.info("test_message", key="value")
    logger.error("test_error", error_type="TestError")
    print("Logging test passed!")
