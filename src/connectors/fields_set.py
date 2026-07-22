"""
Minimalist field set for watchlist data sources.

Every data source (arXiv papers, IMDB movies, web pages, emails, etc.)
has its own rich schema.  ``FieldsSet`` defines the **canonical subset**
that MatchMake cares about so that downstream consumers (rendering,
LLM context, newsroom cards …) can work uniformly regardless of the
original source.

The four canonical fields are:

* **title** – human-readable name (file name, paper title, web-page
  heading …).
* **description** – the main textual content (abstract, summary, body
  text …).
* **timestamp** – creation / publication date of the original data.
* **links** – zero or more URLs that let the user dive deeper
  (YouTube video, PDF, DOI …).

A ``FieldMapping`` helper describes how a *specific* data source maps
its native field names onto the canonical set.  This is persisted
alongside the watch so the mapping is done once and reused everywhere.

@author: vankomme
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

import logging

log = logging.getLogger(__name__)


# ── canonical (minimalist) record ────────────────────────────────────

class FieldsSet(BaseModel):
    """
    Minimalist, source-agnostic representation of a watchlist record.

    Every ingested data point – regardless of whether it originates
    from arXiv, IMDB, a local folder, Gmail, etc. – can be projected
    onto these four fields so that the rest of the application has a
    single, uniform interface.

    Attributes:
        title:       Human-readable name of the record (e.g. paper
                     title, movie name, file name, email subject).
        description: Main textual content (e.g. abstract, synopsis,
                     email body, web-page summary).
        timestamp:   When the original data was created / published.
                     ``None`` when the source does not expose a date.
        links:       Zero or more URLs that let the user explore the
                     original source (PDF link, YouTube URL, DOI …).
    """

    title: str = Field(
        default="",
        description="Human-readable name of the record.",
    )
    description: str = Field(
        default="",
        description="Main textual content (abstract, summary, body …).",
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Creation / publication date of the original data.",
    )
    links: List[str] = Field(
        default_factory=list,
        description="URLs for deeper exploration (PDF, video, DOI …).",
    )


# ── source -> canonical mapping ──────────────────────────────────────

class FieldMapping(BaseModel):
    """
    Describes how a specific data source maps its native payload field
    names onto the four canonical ``FieldsSet`` fields.

    Each attribute holds the **source field name** (a key that exists
    in the Qdrant payload) whose value should fill the corresponding
    canonical field.  When set to ``None`` (or omitted) the canonical
    field will keep its default (empty string / None / []).

    Example for arXiv::

        FieldMapping(
            title_field="title",
            description_field="abstract",
            timestamp_field="published",
            link_fields=["pdf_url"],
        )

    Example for IMDB::

        FieldMapping(
            title_field="movie_name",
            description_field="description",
            timestamp_field="year",
            link_fields=["img_link"],
        )

    Example for Gmail::

        FieldMapping(
            title_field="subject",
            description_field="body_plain",
            timestamp_field="date_received",
            link_fields=["links"],
        )
    """

    title_field: Optional[str] = Field(
        default=None,
        description="Source field name that maps to FieldsSet.title.",
    )
    description_field: Optional[str] = Field(
        default=None,
        description="Source field name that maps to FieldsSet.description.",
    )
    timestamp_field: Optional[str] = Field(
        default=None,
        description="Source field name that maps to FieldsSet.timestamp.",
    )
    link_fields: List[str] = Field(
        default_factory=list,
        description=(
            "One or more source field names whose values are collected "
            "into FieldsSet.links (e.g. ['pdf_url', 'doi'])."
        ),
    )

    # ── convenience ──────────────────────────────────────────────────

    def apply(self, payload: Dict) -> FieldsSet:
        """
        Project a raw source payload onto a ``FieldsSet`` instance.

        Args:
            payload: Dictionary of field-name -> value from the data
                     source (e.g. a Qdrant point payload).

        Returns:
            A ``FieldsSet`` populated with values drawn from *payload*
            according to this mapping.
        """
        title = str(payload.get(self.title_field, "")) if self.title_field else ""
        description = str(payload.get(self.description_field, "")) if self.description_field else ""

        # Timestamp: try to parse if present
        timestamp = None
        if self.timestamp_field and self.timestamp_field in payload:
            raw = payload[self.timestamp_field]
            if isinstance(raw, datetime):
                timestamp = raw
            elif isinstance(raw, str):
                try:
                    timestamp = datetime.fromisoformat(raw)
                except ValueError:
                    log.debug("Could not parse timestamp '%s'", raw)

        # Links: collect from one or more source fields
        links: List[str] = []
        for lf in self.link_fields:
            value = payload.get(lf)
            if value is None:
                continue
            if isinstance(value, list):
                links.extend(str(v) for v in value)
            else:
                links.append(str(value))

        return FieldsSet(
            title=title,
            description=description,
            timestamp=timestamp,
            links=links,
        )
