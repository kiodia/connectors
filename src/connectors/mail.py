from typing import List, Optional, Any
from pydantic import BaseModel, HttpUrl, Field, ConfigDict


class Attachment(BaseModel):
    """Model for email attachments."""
    filename: str = Field(..., description="Name of the attached file")
    mime_type: Optional[str] = Field(None, description="MIME type of the attachment (e.g., 'application/pdf')")
    size_bytes: int = Field(..., description="Size of the attachment in bytes")
    download_url: Optional[HttpUrl] = Field(None, description="URL to download the attachment from Gmail")
    content_id: Optional[str] = Field(None, description="Content-ID for inline attachments")


class Mail(BaseModel):
    """Model for storing complete email data with links, attachments, and matchmake metadata."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "id": "msg-001-abc123",
                "thread_id": "thread-001",
                "sender": "digest@arxiv.org",
                "recipients": ["user@example.com"],
                "subject": "Weekly ML Research Digest",
                "body_plain": "A summary of the latest machine learning papers",
                "body_html": None,
                "links": ["https://arxiv.org/digest/2026-03"],
                "attachments": [],
                "date_received": "2026-03-20T12:00:00Z",
                "is_read": False,
                "labels": ["INBOX"],
                "persona": "tester",
                "embedding": [0.1, 0.2, 0.3],
                "type_of_embedding": "openai-text-embedding-3-small",
                "size_of_embedding": 1536,
                "liked": True,
                "disliked": False
            }
        }
    )

    # --- Gmail core fields ---
    id: str = Field(..., description="Unique Gmail message ID")
    thread_id: str = Field(..., description="Gmail thread ID for grouping conversation")
    sender: str = Field(..., description="Email address of the sender")
    recipients: List[str] = Field(..., description="List of recipient email addresses")
    subject: str = Field("", description="Email subject line")
    body_plain: Optional[str] = Field(None, description="Plain text version of the email body")
    body_html: Optional[str] = Field(None, description="HTML version of the email body")
    links: List[HttpUrl] = Field(default_factory=list, description="All URLs extracted from the email body")
    attachments: List[Attachment] = Field(default_factory=list, description="List of file attachments")
    date_received: str = Field(..., description="Timestamp when the email was received")
    is_read: bool = Field(False, description="Whether the email has been read")
    labels: List[str] = Field(default_factory=list, description="Gmail labels applied to this email")

    # --- MatchMake metadata fields ---
    persona: str = Field(default="", description="Name of the persona associated with this mail")
    embedding: List[float] = Field(
        default_factory=list,
        description="Embedding vector as a list of floats"
    )
    type_of_embedding: str = Field(
        default="",
        description="Type of embedding model used"
    )
    size_of_embedding: int = Field(
        default=0,
        description="Dimension size of the embedding vector"
    )
    liked: bool = Field(default=False, description="Whether the mail is liked")
    disliked: bool = Field(default=False, description="Whether the mail is disliked")

