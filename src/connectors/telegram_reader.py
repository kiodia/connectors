"""
Telegram connector – read full message content from the shared cache.

The Telegram Bot API ``getUpdates`` is a consume-once queue, so MatchMake cannot
re-fetch a past message from Telegram the way the Gmail connector re-fetches an
email. Instead, the keopy producer writes every fetched message to a shared JSON
cache keyed by ``"chat_id:message_id"``; this connector reads the full message
back from that same cache when a Newsroom event is selected.

The cache path is shared via the ``TELEGRAM_CACHE_PATH`` env var, defaulting to
the same datasets location used by keopy.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# Default shared cache location (must match the keopy-side default).
DEFAULT_CACHE_PATH = r"C:\Users\vankomme\datasets\telegram_cache.json"


class Telegram:
    """Read-only accessor for cached Telegram messages."""

    def __init__(self, cache_path: Optional[str] = None):
        """Resolve the shared cache path (arg → env → default)."""
        self.cache_path = Path(
            cache_path or os.getenv("TELEGRAM_CACHE_PATH", DEFAULT_CACHE_PATH)
        )

    def read_message(self, message_key: str) -> Dict[str, Any]:
        """Return the full cached message dict for ``"chat_id:message_id"``.

        Parameters
        ----------
        message_key : str
            The stable key stored as the event ``id_string``.

        Returns
        -------
        dict
            The cached message dict (keys mirror the ``TelegramMessage`` model).

        Raises
        ------
        KeyError
            If the message is not present in the cache.
        """
        cache = self._read_cache()
        if message_key not in cache:
            raise KeyError(
                f"Telegram message '{message_key}' not found in cache {self.cache_path}"
            )
        return cache[message_key]

    def _read_cache(self) -> Dict[str, Any]:
        """Load the shared cache, returning an empty dict if missing/corrupt."""
        if not self.cache_path.is_file():
            log.warning(f"Telegram cache not found at {self.cache_path}")
            return {}
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Error reading Telegram cache {self.cache_path}: {e}")
            return {}
