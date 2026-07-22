"""
Telegram connector – read messages from YOUR Telegram account via Telethon.

Unlike a bot (Bot API), this logs in as your *user* account using the MTProto
Client API, so it can read your own chats — here, the conversation with the
StanHermesBot dialog. Authentication uses api_id / api_hash (from
https://my.telegram.org) plus your phone number; the first run is interactive
(Telegram sends a login code) and then a reusable session file is stored, so
subsequent runs are non-interactive.

To replicate the Gmail flow (where MatchMake retrieves the full message on
selection), every fetched message is also written to a shared JSON cache keyed
by ``"chat_id:message_id"``; the MatchMake side reads the full message back from
that same cache.

Requires: ``pip install telethon``
"""

import os
import re
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from telethon.sync import TelegramClient

from connectors.telegram_message import TelegramMessage

log = logging.getLogger(__name__)

# Default shared cache + session locations (same datasets dir used elsewhere).
DEFAULT_CACHE_PATH = r"C:\Users\vankomme\datasets\telegram_cache.json"
DEFAULT_SESSION_PATH = r"C:\Users\vankomme\datasets\telegram_user"
# The dialog to read from (the bot's @username; display name is "StanHermesBot").
DEFAULT_TARGET = "@StanAngelBot"

# URL extraction pattern (kept consistent with the Gmail/Newsroom link parsing).
_URL_RE = re.compile(r'https?://[^\s<>"\']+')


class Telegram:
    """Reads messages from a single Telegram dialog using a Telethon user session.

    Usage::

        telegram = Telegram(api_id=123, api_hash="abc", session="stan",
                            phone="+41...", target="@StanAngelBot")
        messages = telegram.read_as_messages(limit=50)
        full = telegram.read_message(messages[0].id)
    """

    def __init__(
        self,
        api_id: int,
        api_hash: str,
        session: Optional[str] = None,
        phone: Optional[str] = None,
        target: Optional[str] = None,
        cache_path: Optional[str] = None,
    ):
        """Initialize the connector.

        Args:
            api_id: Telegram API ID from my.telegram.org.
            api_hash: Telegram API hash from my.telegram.org.
            session: Path/name of the Telethon session file (``.session`` is
                appended). Falls back to ``TELEGRAM_SESSION`` env / default.
            phone: Phone number (E.164, e.g. ``+41...``) for the first login.
                Falls back to ``TELEGRAM_PHONE`` env.
            target: The dialog to read (a @username or chat id). Falls back to
                ``TELEGRAM_TARGET`` env / default (the StanHermesBot dialog).
            cache_path: Shared message cache JSON. Falls back to
                ``TELEGRAM_CACHE_PATH`` env / default.
        """
        if not api_id or not api_hash:
            raise ValueError(
                "Telegram api_id and api_hash are required "
                "(set TELEGRAM_API_ID / TELEGRAM_API_HASH from my.telegram.org)."
            )
        self.api_id = int(api_id)
        self.api_hash = api_hash
        self.session = session or os.getenv("TELEGRAM_SESSION", DEFAULT_SESSION_PATH)
        self.phone = phone or os.getenv("TELEGRAM_PHONE", "")
        self.target = target or os.getenv("TELEGRAM_TARGET", DEFAULT_TARGET)
        self.cache_path = Path(
            cache_path or os.getenv("TELEGRAM_CACHE_PATH", DEFAULT_CACHE_PATH)
        )
        log.info(
            f"Telegram connector initialized (session: {self.session}, "
            f"target: {self.target}, cache: {self.cache_path})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_as_messages(self, limit: int = 50, target: Optional[str] = None) -> List[TelegramMessage]:
        """Read recent messages from the target dialog as ``TelegramMessage`` objects.

        Returns messages in chronological order (oldest first) and writes each to
        the shared cache so it can be retrieved later by the MatchMake side.

        Args:
            limit: Maximum number of recent messages to read from the dialog.
            target: Optional override for the dialog to read (@username or id).

        Returns:
            List of ``TelegramMessage`` objects (may be empty).
        """
        target = target or self.target
        messages: List[TelegramMessage] = []

        client = TelegramClient(self.session, self.api_id, self.api_hash)
        try:
            # start() connects and, on first run, performs the login (phone + the
            # code Telegram sends). Once the session exists it is a quick no-op
            # reconnect. To allow non-interactive first logins (e.g. when stdin
            # is not available), the login code and 2FA password can be supplied
            # via the TELEGRAM_CODE / TELEGRAM_PASSWORD env vars; when unset,
            # Telethon falls back to its default interactive input() prompts.
            login_code = os.getenv("TELEGRAM_CODE", "").strip()
            login_password = os.getenv("TELEGRAM_PASSWORD", "") or None

            start_kwargs: Dict[str, Any] = {}
            if self.phone:
                start_kwargs["phone"] = self.phone
            if login_code:
                # code_callback must be a callable returning the code string.
                start_kwargs["code_callback"] = lambda: login_code
                log.info("Using login code from TELEGRAM_CODE env (non-interactive).")
            if login_password:
                start_kwargs["password"] = login_password

            client.start(**start_kwargs)

            entity = client.get_entity(target)
            chat_title = (
                getattr(entity, "title", None)
                or getattr(entity, "username", None)
                or getattr(entity, "first_name", None)
                or str(target)
            )

            tl_messages = client.get_messages(entity, limit=limit)
            # get_messages returns newest first; reverse for chronological order.
            for tl in reversed(tl_messages):
                telegram_message = self._to_message(tl, entity, chat_title)
                if telegram_message is not None:
                    messages.append(telegram_message)
        except Exception as e:
            log.error(f"Error reading Telegram messages: {e}")
        finally:
            client.disconnect()

        if messages:
            self._write_cache(messages)

        log.info(f"Fetched {len(messages)} Telegram message(s) from '{target}'")
        return messages

    def read_message(self, message_key: str) -> Optional[Dict[str, Any]]:
        """Return a single cached message dict by its ``"chat_id:message_id"`` key.

        Args:
            message_key: The stable key stored as the event ``id_string``.

        Returns:
            The cached message dict, or ``None`` if not present.
        """
        cache = self._read_cache()
        message = cache.get(message_key)
        if message is None:
            log.warning(f"Telegram message '{message_key}' not found in cache {self.cache_path}")
        return message

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_message(self, tl, entity, chat_title: str) -> Optional[TelegramMessage]:
        """Map a Telethon ``Message`` to a ``TelegramMessage``."""
        try:
            if tl is None or tl.id is None:
                return None

            text = tl.message or ""

            sender_name = ""
            sender_username = ""
            sender = getattr(tl, "sender", None)
            if sender is not None:
                sender_name = " ".join(
                    part for part in [getattr(sender, "first_name", None), getattr(sender, "last_name", None)] if part
                ).strip()
                sender_username = getattr(sender, "username", "") or ""

            chat_id = tl.chat_id if tl.chat_id is not None else getattr(entity, "id", 0)
            date_iso = (
                tl.date.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
                if tl.date
                else ""
            )
            links = list(dict.fromkeys(_URL_RE.findall(text))) if text else []

            return TelegramMessage(
                id=f"{chat_id}:{tl.id}",
                message_id=tl.id,
                chat_id=chat_id,
                chat_title=chat_title,
                sender=sender_name,
                sender_username=sender_username,
                text=text or None,
                date=date_iso,
                links=links,
            )
        except Exception as e:
            log.error(f"Error mapping Telegram message: {e}")
            return None

    def _read_cache(self) -> Dict[str, Any]:
        """Load the shared cache, returning an empty dict if missing/corrupt."""
        if not self.cache_path.is_file():
            return {}
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Error reading Telegram cache {self.cache_path}: {e}")
            return {}

    def _write_cache(self, messages: List[TelegramMessage]) -> None:
        """Merge messages into the shared cache JSON, keyed by message id."""
        cache = self._read_cache()
        for message in messages:
            cache[message.id] = message.model_dump(mode="json")
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
            log.info(f"Wrote {len(messages)} message(s) to Telegram cache {self.cache_path}")
        except Exception as e:
            log.error(f"Error writing Telegram cache {self.cache_path}: {e}")
