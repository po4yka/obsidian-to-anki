"""Clipboard integration for copy mode."""


from ..utils.logging import get_logger

logger = get_logger(__name__)

try:
    import pyperclip

    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False
    logger.warning("pyperclip not available, clipboard features disabled")


def copy_to_clipboard(text: str) -> bool:
    """
    Copy text to system clipboard.

    Args:
        text: Text to copy

    Returns:
        True if successful, False otherwise
    """
    if not CLIPBOARD_AVAILABLE:
        logger.warning("clipboard_not_available")
        return False

    try:
        pyperclip.copy(text)
        logger.debug("text_copied_to_clipboard", length=len(text))
        return True
    except Exception as e:
        logger.error("clipboard_copy_failed", error=str(e))
        return False


def get_from_clipboard() -> str | None:
    """
    Get text from system clipboard.

    Returns:
        Clipboard text or None if unavailable
    """
    if not CLIPBOARD_AVAILABLE:
        logger.warning("clipboard_not_available")
        return None

    try:
        text = pyperclip.paste()
        logger.debug("text_retrieved_from_clipboard", length=len(text))
        return text
    except Exception as e:
        logger.error("clipboard_paste_failed", error=str(e))
        return None
