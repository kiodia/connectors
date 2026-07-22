from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class TelegramMessage(BaseModel):
    """Model for storing a complete Telegram message with links and matchmake metadata.

    Mirrors the structure of ``connectors.mail.Mail`` so that the Telegram flow
    can reuse the same Events API / Newsroom plumbing as the Gmail flow.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "id": "123456789:42",
                "message_id": 42,
                "chat_id": 123456789,
                "chat_title": "StanHermesBot",
                "sender": "Robert van Kommer",
                "sender_username": "rvankommer",
                "text": "Check this out https://arxiv.org/abs/2511.00402",
                "date": "2026-06-17T12:00:00Z",
                "links": ["https://arxiv.org/abs/2511.00402"],
                "persona": "tester",
                "liked": True,
                "disliked": False,
            }
        },
    )

    # --- Telegram core fields ---
    id: str = Field(..., description="Stable key 'chat_id:message_id' used as the event id_string")
    message_id: int = Field(..., description="Telegram message ID within the chat")
    chat_id: int = Field(..., description="Telegram chat ID the message belongs to")
    chat_title: str = Field("", description="Title or name of the chat / bot")
    sender: str = Field("", description="Display name of the message sender")
    sender_username: str = Field("", description="Telegram @username of the sender")
    text: Optional[str] = Field(None, description="Plain text body of the message")
    date: str = Field("", description="ISO timestamp when the message was sent")
    links: List[str] = Field(default_factory=list, description="All URLs extracted from the message text")

    # --- MatchMake metadata fields ---
    persona: str = Field(default="", description="Name of the persona associated with this message")
    liked: bool = Field(default=False, description="Whether the message is liked")
    disliked: bool = Field(default=False, description="Whether the message is disliked")
