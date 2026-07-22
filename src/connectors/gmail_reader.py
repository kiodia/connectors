"""
Gmail connector – read message content/attachments and delete messages.

Uses OAuth 2.0 credentials (credentials.json) and a cached token (token.json)
located at the project root.
"""

import base64
import os
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# If modifying these scopes, delete the token.json file.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
]

# Resolve paths relative to the project root (parent of connectors/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CREDENTIALS_PATH = _PROJECT_ROOT / "credentials.json"
TOKEN_PATH = _PROJECT_ROOT / "token.json"


class Gmail:
    """Thin wrapper around the Gmail API for reading and deleting messages."""

    def __init__(self):
        """Authenticate with Google and build the Gmail API service.

        * If a valid *token.json* exists it is reused.
        * If the token is expired but a refresh-token is available it is
          refreshed automatically.
        * Otherwise the OAuth consent flow is launched via the browser.
        """
        creds = None

        if TOKEN_PATH.exists():
            creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(CREDENTIALS_PATH), SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Persist the (possibly refreshed) token for the next run.
            with open(TOKEN_PATH, "w") as token_file:
                token_file.write(creds.to_json())

        self.service = build("gmail", "v1", credentials=creds)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_message(self, message_id: str) -> dict:
        """Return the full content and attachments of a message.

        Parameters
        ----------
        message_id : str
            The Gmail message ID (e.g. obtained from ``users.messages.list``).

        Returns
        -------
        dict
            A dictionary with the following keys:

            * **id** – the message ID
            * **thread_id** – the thread ID
            * **subject** – the Subject header (or ``""``)
            * **from** – the From header (or ``""``)
            * **to** – the To header (or ``""``)
            * **date** – the Date header (or ``""``)
            * **body** – the plain-text body (or ``""``)
            * **html** – the HTML body (or ``""``)
            * **attachments** – a list of dicts, each containing:
              ``filename``, ``mime_type``, ``size``, and ``data``
              (raw bytes of the attachment).
        """
        msg = (
            self.service.users()
            .messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )

        headers = {
            h["name"].lower(): h["value"]
            for h in msg.get("payload", {}).get("headers", [])
        }

        result = {
            "id": msg["id"],
            "thread_id": msg.get("threadId", ""),
            "subject": headers.get("subject", ""),
            "from": headers.get("from", ""),
            "to": headers.get("to", ""),
            "date": headers.get("date", ""),
            "body": "",
            "html": "",
            "attachments": [],
        }

        # Walk the MIME tree to extract body parts and attachments.
        self._parse_parts(msg["payload"], result)

        return result

    def delete_message(self, message_id: str) -> None:
        """Permanently delete a message.

        .. note::
            This uses ``messages.trash`` (moves to Trash) rather than
            ``messages.delete`` (immediate permanent deletion) so the
            action can be undone from the Gmail UI.  Change to
            ``.delete()`` if permanent deletion is desired.

        Parameters
        ----------
        message_id : str
            The Gmail message ID to delete.
        """
        (
            self.service.users()
            .messages()
            .trash(userId="me", id=message_id)
            .execute()
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_parts(self, payload: dict, result: dict) -> None:
        """Recursively walk MIME parts to populate *result*."""
        mime_type = payload.get("mimeType", "")
        body = payload.get("body", {})
        parts = payload.get("parts", [])

        # Leaf node with inline content
        if not parts:
            if mime_type == "text/plain" and body.get("data"):
                result["body"] += self._decode_body(body["data"])
            elif mime_type == "text/html" and body.get("data"):
                result["html"] += self._decode_body(body["data"])

            # Attachment (identified by an attachmentId)
            attachment_id = body.get("attachmentId")
            if attachment_id:
                att = (
                    self.service.users()
                    .messages()
                    .attachments()
                    .get(
                        userId="me",
                        messageId=result["id"],
                        id=attachment_id,
                    )
                    .execute()
                )
                result["attachments"].append(
                    {
                        "filename": payload.get("filename", ""),
                        "mime_type": mime_type,
                        "size": att.get("size", 0),
                        "data": base64.urlsafe_b64decode(att["data"]),
                    }
                )
            return

        # Multipart – recurse into children
        for part in parts:
            self._parse_parts(part, result)

    @staticmethod
    def _decode_body(data: str) -> str:
        """Decode a base64url-encoded Gmail body payload to a UTF-8 string."""
        return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
