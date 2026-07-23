"""Grounding entities in a vector database.

:class:`~connectors.grounding.data_grounding.DataGrounding` manages one Qdrant
collection of entities of one kind — a video, a paper, a lab, an enterprise, a
startup, an artwork, a music clip, a concert, a photograph, a touristic
activity, a place: creating the collection, storing Pydantic payloads with
their id, timestamps, embedding and markdown rendering, removing them one by
one or by time-to-live, and deciding whether an incoming entity is new.

Which fields represent an entity is the caller's decision — this package reads
whatever model it is handed. Storage and embedding come from ``vectorize``, the
duplicate detection from ``matching_learning.matching.fuzzy``.

Each datapoint holds both of the things matching needs: a vector, for the fast
first pass, and a readable markdown rendering with its images, for the LLM pass
that follows and for display in MatchMake.
"""

from connectors.grounding.data_grounding import (
    CREATED_AT, CREATED_TS, EXTERNAL_ID, MARKDOWN, PAYLOAD_TYPE, UPDATED_AT,
    UPDATED_TS, DataGrounding,
)

__all__ = [
    "DataGrounding",
    "CREATED_AT",
    "UPDATED_AT",
    "CREATED_TS",
    "UPDATED_TS",
    "EXTERNAL_ID",
    "PAYLOAD_TYPE",
    "MARKDOWN",
]
