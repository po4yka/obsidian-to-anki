"""Deterministic GUID generation utilities."""

from __future__ import annotations

import base64
import hashlib
from typing import Iterable


def deterministic_guid(parts: Iterable[str], length: int = 16) -> str:
    """Generate a deterministic GUID from the given string parts.

    Args:
        parts: Iterable of components to hash.
        length: Desired length of resulting GUID (default 16).

    Returns:
        Deterministic GUID string consisting of url-safe characters.
    """
    joined = "::".join(parts).encode("utf-8")
    digest = hashlib.sha1(joined).digest()
    guid = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return guid[:length]
