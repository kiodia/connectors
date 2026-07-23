"""Grounding entities of any kind in a Qdrant collection.

A YouTube video, a research paper, a lab, an enterprise, a startup, a work of
visual art, a music clip, a concert, a photograph, a touristic activity, a
place — whatever the entity, grounding it raises the same five questions:
does the collection exist, is this entity already in it, when did we first
see it, when should it expire, and how does it read to a human or an LLM.
:class:`DataGrounding` answers those five for one collection, so nothing
above it writes Qdrant plumbing.

What it deliberately does *not* decide is **what an entity is**. That is the
agent's job: for each information source it defines the Pydantic model whose
fields best represent that kind of entity — a video is a title, a channel, a
duration and a thumbnail; a startup is a name, a sector, a founding year and
a logo; a concert is a name, a venue, a date and a poster — and hands the
instances here. :class:`DataGrounding` reads whatever model it is given, so
adding a new kind of source means writing a model, not touching this class.

Since most sources are web content reached through a list of URLs, the URL
fields are treated as first-class: they identify an entity (the cheapest
duplicate check there is, before any fuzzy or vector comparison), and they
render as the images and links of the markdown view.

The pieces come from the sibling libraries rather than being reimplemented:

* **vectorize** — the :class:`qdrant_client.QdrantClient` (directly, or a
  :class:`vectorize.vector_db.Qdrant` wrapper, both accepted) for storage,
  and :class:`vectorize.embeddings.Embed` driven by the
  :class:`~vectorize.embeddings.EmbeddingVector` passed at construction, so
  the embedding *source* is a deployment choice, not a hard-coded model.
* **matching_learning** — :mod:`matching_learning.matching.fuzzy` for the
  novelty check: normalized name similarity and embedding cosine, the two
  complementary ways two datapoints turn out to be the same entity.

Every datapoint carries the model's fields flattened at the payload root (so
``show_fields`` and ``vector_embedded_fields`` keep working) plus the
grounding metadata :data:`CREATED_AT`, :data:`UPDATED_AT` and their epoch
twins — Qdrant range filters need numbers, which is what makes
:meth:`DataGrounding.purge` a single server-side delete instead of a scan.

It also carries a :data:`MARKDOWN` rendering of itself — heading, image,
body, links — which serves the two halves of matching at once. Embeddings
retrieve fast but say nothing a person or an LLM can read; a payload of
loose fields reads poorly and has to be re-assembled by every consumer.
Storing the rendered form next to the vector means the cheap embedding pass
narrows the candidates and the LLM pass then reasons over text that is
already whole, with its images, instead of a dict — and MatchMake's
CardViewer can display the datapoint with no per-collection config.

The same three calls ground any source; only the model changes::

    class Startup(BaseModel):           # the agent's view of this entity
        name: str
        pitch: str = ""
        sector: str = ""
        founded: int = 0
        logo_url: str = ""
        website: str = ""

    grounding = DataGrounding(client, "swiss_startups", embedding_vector,
                              vector_dimension=768, ttl=timedelta(days=90))
    for startup in scraped:             # from a list of URLs
        if grounding.is_new(startup):
            grounding.add(startup, point_id=startup.website)
    grounding.purge()

Nothing there names a field: the heading, the images, the links and the
duplicate checks are found in whatever model arrives (``name`` here,
``title`` for a paper or a video). Where a source needs it, every one of
those choices can be pinned explicitly — ``title_field``,
``identity_fields``, ``embedded_fields``, ``markdown_renderer``.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel
from qdrant_client.models import (
    Distance, FieldCondition, Filter, FilterSelector, MatchValue, PointIdsList,
    PointStruct, Range, VectorParams,
)

from matching_learning.matching.fuzzy import (
    DEFAULT_STRING_THRESHOLD, DEFAULT_VECTOR_THRESHOLD, best_string_match,
)
from vectorize.embeddings import AggregationModes, Embed, EmbeddingVector

import logging
log = logging.getLogger(__name__)

#: Payload keys holding the grounding metadata. The ISO strings are what a
#: human (or an LLM reading a payload) sees; the ``_ts`` epoch-second twins
#: are what Qdrant's numeric ``Range`` filter can select on, which is how
#: :meth:`DataGrounding.purge` deletes server-side.
CREATED_AT = "created_at"
UPDATED_AT = "updated_at"
CREATED_TS = "created_ts"
UPDATED_TS = "updated_ts"
EXTERNAL_ID = "external_id"
PAYLOAD_TYPE = "payload_type"
#: The datapoint rendered as markdown, images included — what an LLM reads
#: after the vector search has narrowed the field, and what MatchMake renders.
MARKDOWN = "markdown"

#: The metadata keys DataGrounding owns; a connector payload field with one of
#: these names would be overwritten, so they are excluded from the embedded
#: text and reported by :meth:`DataGrounding.metadata_keys`. ``markdown`` is
#: excluded from the embedding for a second reason: it restates every other
#: field, and embedding it would count each of them twice.
_METADATA_KEYS = frozenset(
    {CREATED_AT, UPDATED_AT, CREATED_TS, UPDATED_TS, EXTERNAL_ID, PAYLOAD_TYPE,
     MARKDOWN}
)

#: How many stored names :meth:`DataGrounding.is_new` scrolls through for the
#: fuzzy comparison. Fuzzy matching has no index to lean on, so the scan is
#: bounded; the vector check (which *is* indexed) covers the rest.
DEFAULT_FUZZY_SCAN_LIMIT = 2000

# ----------------------------------------------------------------------
# Field conventions
#
# An entity model is the agent's to design, so nothing below is required —
# these are the names such models tend to use, letting a plain model work
# with no configuration at all. Every one of them can be overridden per
# collection when a source calls its fields something else.
# ----------------------------------------------------------------------
#: Fields that carry an entity's name, most specific first. A paper or a video
#: has a ``title``, a lab or a startup a ``name``, an article a ``headline`` —
#: the first non-empty one becomes the markdown heading and the string the
#: fuzzy duplicate check compares.
_TITLE_HINTS = ("title", "name", "headline", "label", "subject", "caption")
#: Fields whose URL *identifies* the entity: two datapoints sharing one are the
#: same thing, whatever their names say. This is the cheapest duplicate check
#: there is, and the one that matters most when a source is a list of URLs.
_IDENTITY_HINTS = ("url", "source_url", "detail_url", "page_url", "link",
                   "permalink", "watch_url", "website", "homepage", "doi",
                   "isbn", "identifier", "external_id")
#: Field names whose value is an image URL — rendered as ``![alt](url)``.
#: Covers a poster, a video thumbnail, a company logo, an artwork photograph,
#: a venue picture, an album cover.
_IMAGE_HINTS = ("image", "img", "poster", "thumbnail", "thumb", "photo",
                "picture", "cover", "avatar", "banner", "logo", "artwork",
                "still", "screenshot", "headshot", "gallery")
#: Field names whose value is a link — rendered as ``[label](url)``.
_LINK_HINTS = ("url", "link", "href", "uri", "permalink", "website",
               "homepage", "watch", "stream", "download", "pdf", "doi",
               "tickets", "booking", "profile", "repository")
#: Field names whose value is prose — rendered as its own paragraph rather
#: than as a ``**Label:** value`` line. A synopsis, an abstract, a pitch, a
#: biography, a programme note: every kind of source has one.
_BODY_HINTS = ("description", "summary", "synopsis", "abstract", "content",
               "text", "body", "excerpt", "snippet", "overview", "pitch",
               "bio", "biography", "notes", "programme", "program", "review",
               "transcript", "lyrics")
#: A string this long is prose whatever it is called.
_BODY_MIN_LENGTH = 200
#: Values that look like a URL, whatever the field is called.
_URL_VALUE = re.compile(r"^(?:https?://|//|/|data:image/)", re.IGNORECASE)
#: Image URLs recognised by extension, for fields named neutrally ("media").
_IMAGE_EXTENSION = re.compile(r"\.(?:png|jpe?g|gif|webp|svg|avif)(?:[?#].*)?$",
                              re.IGNORECASE)


class DataGrounding:
    """Manages one Qdrant collection of entities of one kind.

    One instance owns exactly one collection: creating it, adding and updating
    datapoints with their timestamps, embedding and markdown, removing them
    individually or by age, and deciding whether an incoming entity is new.

    The entity model is the caller's. Whatever fields it declares, this class
    finds the name, the identifying URLs, the images and the prose in them
    (see :data:`_TITLE_HINTS` and its neighbours); ``title_field``,
    ``identity_fields`` and ``embedded_fields`` pin those choices down when a
    source names things its own way.

    Args:
        client:           A :class:`qdrant_client.QdrantClient`, or a
                          :class:`vectorize.vector_db.Qdrant` wrapper (its
                          ``.client`` is used).
        collection_name:  The Qdrant collection this instance manages.
        embedding:        The :class:`~vectorize.embeddings.EmbeddingVector`
                          describing the embedding source (local
                          SentenceTransformer or Google API). The vectors
                          added to every datapoint come from it.
        vector_dimension: Target (Matryoshka) dimension; defaults to the
                          embedding's full native dimension.
        embedded_fields:  Payload fields concatenated into the embedded text.
                          Defaults to every string field of the model,
                          grounding metadata and URLs excluded — a URL is an
                          address, not a description of the entity.
        title_field:      The field naming the entity (``"title"``,
                          ``"name"``, ...). Auto-detected per payload when
                          left ``None``.
        identity_fields:  Fields whose URL or code identifies the entity, for
                          the exact duplicate check that runs before the fuzzy
                          and vector ones. Auto-detected when left ``None``.
        ttl:              Default age after which :meth:`purge` drops a
                          datapoint. ``None`` means never purge by default.
        named_vector:     Store vectors under the embedding's reference name
                          (vectorize's convention, and what lets a collection
                          hold several embeddings later). Only consulted when
                          this instance creates the collection — an existing
                          collection's own layout always wins.
        markdown_field:   Payload field holding the markdown rendering, or
                          ``None``/``""`` to store none.
        markdown_renderer: ``payload -> markdown`` override, for a source
                          whose entities deserve a hand-written layout. By
                          default the model's own ``to_markdown()`` is used
                          when it has one, else :meth:`render_markdown`
                          builds it from the fields.
    """

    def __init__(self,
                 client: Any,
                 collection_name: str,
                 embedding: EmbeddingVector,
                 vector_dimension: Optional[int] = None,
                 embedded_fields: Optional[Sequence[str]] = None,
                 title_field: Optional[str] = None,
                 identity_fields: Optional[Sequence[str]] = None,
                 ttl: Optional[timedelta] = None,
                 named_vector: bool = True,
                 aggregation: AggregationModes = AggregationModes.MEAN_POOLING,
                 markdown_field: Optional[str] = MARKDOWN,
                 markdown_renderer: Optional[Callable[[BaseModel], str]] = None):
        # Accept either the raw client or vectorize's Qdrant wrapper, so a
        # caller that already holds one of them never has to unwrap it.
        self.client = getattr(client, "client", client)
        if self.client is None:
            raise ValueError(
                "DataGrounding needs an open Qdrant client; the one given is "
                "None (a vectorize.Qdrant whose database path was missing?)."
            )
        self.collection_name = collection_name
        self.embedding = embedding
        self.vector_dimension = vector_dimension or embedding.dimensions[0]
        self.embedded_fields = list(embedded_fields) if embedded_fields else None
        # A pinned field is the only candidate; otherwise every convention is,
        # and the payload decides which one it actually has.
        self.title_field = title_field
        self.title_fields = [title_field] if title_field else list(_TITLE_HINTS)
        self.identity_fields = (list(identity_fields) if identity_fields
                                else list(_IDENTITY_HINTS))
        self.ttl = ttl
        self.named_vector = named_vector
        self.aggregation = aggregation
        self.markdown_field = markdown_field or ""
        self.markdown_renderer = markdown_renderer
        self._embed: Optional[Embed] = None

        log.info(
            "DataGrounding - collection '%s' grounded with embedding '%s' "
            "(dim=%d, title=%s, identity=%s, ttl=%s)",
            collection_name, embedding.reference_name, self.vector_dimension,
            title_field or "auto", "auto" if identity_fields is None else
            ",".join(self.identity_fields), ttl,
        )

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    def exists(self) -> bool:
        """Whether the managed collection is present in Qdrant."""
        return bool(self.client.collection_exists(self.collection_name))

    def create(self) -> bool:
        """Create the collection when it does not exist yet.

        Idempotent: an existing collection is left exactly as it is (including
        a vector layout that differs from this instance's ``named_vector``
        preference — the stored data decides, not the caller).

        Returns:
            True when the collection exists afterwards, False on error.
        """
        if self.exists():
            log.info("DataGrounding - collection '%s' already exists.",
                     self.collection_name)
            return True

        params = VectorParams(size=self.vector_dimension, distance=Distance.COSINE)
        config = ({self.embedding.reference_name: params}
                  if self.named_vector else params)
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=config,
            )
            log.info(
                "DataGrounding - collection '%s' created (%s vector, dim=%d, cosine).",
                self.collection_name,
                f"named '{self.embedding.reference_name}'" if self.named_vector
                else "unnamed",
                self.vector_dimension,
            )
            return True
        except Exception as exc:  # noqa: BLE001 - report, never crash the ingest
            log.error("DataGrounding - creating collection '%s' failed: %s",
                      self.collection_name, exc)
            return False

    def count(self) -> int:
        """Number of datapoints currently stored (0 when the collection is absent)."""
        if not self.exists():
            return 0
        try:
            return int(self.client.get_collection(self.collection_name).points_count or 0)
        except Exception as exc:  # noqa: BLE001
            log.error("DataGrounding - counting '%s' failed: %s",
                      self.collection_name, exc)
            return 0

    @staticmethod
    def metadata_keys() -> frozenset:
        """The payload keys DataGrounding writes itself (see :data:`CREATED_AT`)."""
        return _METADATA_KEYS

    # ------------------------------------------------------------------
    # Datapoints
    # ------------------------------------------------------------------

    def add(self, payload: BaseModel,
            point_id: Optional[Union[str, int]] = None,
            created_at: Optional[datetime] = None,
            updated_at: Optional[datetime] = None,
            vector: Optional[Sequence[float]] = None) -> Optional[PointStruct]:
        """Store one Pydantic object as a datapoint, embedding included.

        The point carries its explicit id, its creation and last-update
        timestamps, the flattened model fields as payload, the markdown
        rendering (see :meth:`to_markdown`), and the vector encoded from
        :attr:`embedded_fields` with the configured embedding.

        Args:
            point_id:   The datapoint's id. Qdrant accepts only UUIDs and
                        unsigned integers, so a natural key like a detail-page
                        URL is hashed into a stable UUID5 and kept verbatim in
                        the payload's ``external_id`` — passing the same key
                        again therefore updates the same datapoint. Defaults
                        to a fresh UUID4.
            created_at: First-seen time (defaults to now, UTC). Ignored in
                        favour of the stored value when the id already exists,
                        so re-adding an item never rewrites its history.
            updated_at: Last-update time (defaults to now, UTC).
            vector:     A precomputed embedding; encoded from the payload when
                        omitted.

        Returns:
            The upserted :class:`~qdrant_client.models.PointStruct`, or
            ``None`` when the datapoint could not be stored.
        """
        if not self.create():
            return None

        now = datetime.now(timezone.utc)
        created = created_at or now
        updated = updated_at or now

        resolved_id, external_id = self._resolve_id(point_id)
        body = payload.model_dump()

        # Re-adding a known datapoint keeps its original creation time: the
        # first sighting is the fact worth preserving, the update time is what
        # moves.
        stored_created = self._stored_created_at(resolved_id)
        if stored_created is not None:
            created = stored_created

        body.update({
            CREATED_AT: created.isoformat(),
            UPDATED_AT: updated.isoformat(),
            CREATED_TS: created.timestamp(),
            UPDATED_TS: updated.timestamp(),
            PAYLOAD_TYPE: type(payload).__name__,
        })
        if external_id is not None:
            body[EXTERNAL_ID] = external_id
        if self.markdown_field:
            body[self.markdown_field] = self.to_markdown(payload)

        if vector is None:
            vector = self.encode(payload)
        if vector is None:
            log.warning(
                "DataGrounding - no embeddable text in %s (id=%s); "
                "datapoint not stored.", type(payload).__name__, resolved_id,
            )
            return None

        point = PointStruct(id=resolved_id,
                            vector=self._wrap_vector(list(vector)),
                            payload=body)
        try:
            self.client.upsert(collection_name=self.collection_name, points=[point])
            log.info("DataGrounding - datapoint %s stored in '%s' (%s).",
                     resolved_id, self.collection_name, type(payload).__name__)
            return point
        except Exception as exc:  # noqa: BLE001
            log.error("DataGrounding - storing datapoint %s in '%s' failed: %s",
                      resolved_id, self.collection_name, exc)
            return None

    def update(self, point_id: Union[str, int], payload: BaseModel,
               updated_at: Optional[datetime] = None) -> Optional[PointStruct]:
        """Replace a datapoint's content and stamp it as just updated.

        A thin alias of :meth:`add` on an existing id — kept as its own name
        because "update this item" and "add this item" read very differently
        at a call site even though Qdrant's upsert makes them one operation.
        The original ``created_at`` is preserved.
        """
        return self.add(payload, point_id=point_id, updated_at=updated_at)

    def get(self, point_id: Union[str, int]) -> Optional[Dict[str, Any]]:
        """The stored payload of one datapoint, or ``None`` when absent."""
        resolved_id, _ = self._resolve_id(point_id)
        if not self.exists():
            return None
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[resolved_id],
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("DataGrounding - retrieving %s from '%s' failed: %s",
                      resolved_id, self.collection_name, exc)
            return None
        return dict(points[0].payload or {}) if points else None

    def remove(self, point_id: Union[str, int]) -> bool:
        """Delete one datapoint by its id (natural key or UUID alike).

        Returns True when the delete was issued, False when the collection is
        missing or Qdrant refused it. Deleting an unknown id is not an error —
        the datapoint is gone either way.
        """
        if not self.exists():
            log.warning("DataGrounding - collection '%s' does not exist; "
                        "nothing to remove.", self.collection_name)
            return False
        resolved_id, _ = self._resolve_id(point_id)
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=[resolved_id]),
            )
            log.info("DataGrounding - datapoint %s removed from '%s'.",
                     resolved_id, self.collection_name)
            return True
        except Exception as exc:  # noqa: BLE001
            log.error("DataGrounding - removing %s from '%s' failed: %s",
                      resolved_id, self.collection_name, exc)
            return False

    def clear(self) -> bool:
        """Remove every datapoint, keeping the collection itself."""
        if not self.exists():
            return False
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(filter=Filter()),
            )
            log.info("DataGrounding - all datapoints removed from '%s'.",
                     self.collection_name)
            return True
        except Exception as exc:  # noqa: BLE001
            log.error("DataGrounding - clearing '%s' failed: %s",
                      self.collection_name, exc)
            return False

    # ------------------------------------------------------------------
    # Time-to-live
    # ------------------------------------------------------------------

    def purge(self, ttl: Optional[timedelta] = None,
              now: Optional[datetime] = None,
              by: str = CREATED_TS) -> int:
        """Delete every datapoint older than the time-to-live.

        Age is measured on ``created_ts`` by default — a datapoint expires a
        fixed time after it was *first seen*, so an item that keeps being
        re-scraped still ages out. Pass ``by=UPDATED_TS`` for the other policy:
        expire only what has stopped being refreshed.

        Args:
            ttl: Overrides the instance's :attr:`ttl` for this call. Purging
                 with neither set is a no-op rather than an error, so a
                 scheduled purge on a TTL-less collection stays harmless.
            now: The reference time (defaults to now, UTC) — mostly a test seam.
            by:  :data:`CREATED_TS` or :data:`UPDATED_TS`.

        Returns:
            The number of datapoints deleted; ``-1`` when the purge failed.
        """
        effective_ttl = ttl if ttl is not None else self.ttl
        if effective_ttl is None:
            log.info("DataGrounding - no TTL configured for '%s'; nothing purged.",
                     self.collection_name)
            return 0
        if by not in (CREATED_TS, UPDATED_TS):
            raise ValueError(f"purge(by=...) must be {CREATED_TS!r} or {UPDATED_TS!r}")
        if not self.exists():
            return 0

        cutoff = ((now or datetime.now(timezone.utc)) - effective_ttl).timestamp()
        expired = Filter(must=[FieldCondition(key=by, range=Range(lt=cutoff))])

        try:
            # Count first: Qdrant's delete does not report how much it removed,
            # and callers (schedulers, the keopy API) log what was dropped.
            doomed = self.client.count(
                collection_name=self.collection_name,
                count_filter=expired,
                exact=True,
            ).count
            if doomed:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=FilterSelector(filter=expired),
                )
            log.info(
                "DataGrounding - purged %d datapoint(s) from '%s' older than "
                "%s (%s < %s).",
                doomed, self.collection_name, effective_ttl, by,
                datetime.fromtimestamp(cutoff, timezone.utc).isoformat(),
            )
            return int(doomed)
        except Exception as exc:  # noqa: BLE001
            log.error("DataGrounding - purging '%s' failed: %s",
                      self.collection_name, exc)
            return -1

    # ------------------------------------------------------------------
    # Novelty
    # ------------------------------------------------------------------

    def is_new(self, payload: BaseModel,
               string_threshold: float = DEFAULT_STRING_THRESHOLD,
               vector_threshold: float = DEFAULT_VECTOR_THRESHOLD,
               scan_limit: int = DEFAULT_FUZZY_SCAN_LIMIT) -> bool:
        """Whether this entity is not already in the collection.

        Convenience wrapper over :meth:`find_duplicate` for the common
        ``if grounding.is_new(entity): grounding.add(entity)`` shape.
        """
        return self.find_duplicate(
            payload,
            string_threshold=string_threshold,
            vector_threshold=vector_threshold,
            scan_limit=scan_limit,
        ) is None

    def find_duplicate(self, payload: BaseModel,
                       string_threshold: float = DEFAULT_STRING_THRESHOLD,
                       vector_threshold: float = DEFAULT_VECTOR_THRESHOLD,
                       scan_limit: int = DEFAULT_FUZZY_SCAN_LIMIT
                       ) -> Optional[Tuple[Union[str, int], float, str]]:
        """The datapoint already representing this entity, if any.

        Three checks, run in increasing cost order, each catching duplicates
        the next one cannot:

        1. **identity** — an exact payload match on the entity's identifying
           URL or code (:attr:`identity_fields`). When a source is a list of
           URLs this settles most cases outright, for free: two records
           sharing a canonical URL, a DOI or a watch link are one entity
           however differently they are worded.
        2. **fuzzy name** — normalized similarity against the stored names
           (:mod:`matching_learning.matching.fuzzy`). Catches re-runs where
           the same entity comes back with different punctuation, casing or a
           year suffix, and needs no embedding.
        3. **vector similarity** — cosine against the collection's index.
           Catches the same entity described under a different name, which no
           string comparison can see.

        Returns:
            ``(point_id, score, "identity" | "fuzzy" | "vector")`` for the
            first check that fires, or ``None`` when the entity looks new.
        """
        if not self.exists() or self.count() == 0:
            return None

        body = payload.model_dump()

        identity = self._find_by_identity(body)
        if identity is not None:
            log.info("DataGrounding - identity match on %s in '%s'.",
                     identity, self.collection_name)
            return identity, 1.0, "identity"

        title = self._title_of(body)
        if title:
            match = best_string_match(
                title,
                self._stored_titles(limit=scan_limit),
                threshold=string_threshold,
                key=lambda pair: pair[1],
            )
            if match is not None:
                (point_id, stored_title), score = match
                log.info(
                    "DataGrounding - '%s' matches stored '%s' (id=%s, fuzzy=%.3f) "
                    "in '%s'.", title, stored_title, point_id, score,
                    self.collection_name,
                )
                return point_id, score, "fuzzy"

        vector = self.encode(payload)
        if vector is None:
            return None
        hits = self.search(vector, top_k=1)
        if hits and hits[0]["score"] >= vector_threshold:
            log.info(
                "DataGrounding - payload matches datapoint %s "
                "(cosine=%.3f) in '%s'.",
                hits[0]["id"], hits[0]["score"], self.collection_name,
            )
            return hits[0]["id"], hits[0]["score"], "vector"
        return None

    def search(self, vector: Sequence[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """The ``top_k`` datapoints closest to ``vector``.

        Returns dicts of ``id``, ``score`` and ``payload`` — the shape the
        keopy API hands to an agent.
        """
        if not self.exists():
            return []
        try:
            query = self._query_vector(list(vector))
            if hasattr(self.client, "query_points"):
                # ``search`` is deprecated in recent qdrant-client releases;
                # ``query_points`` takes the vector name as its own argument
                # instead of a ``(name, vector)`` tuple.
                name, values = query if isinstance(query, tuple) else (None, query)
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=values,
                    using=name,
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False,
                ).points
            else:
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query,
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False,
                )
        except Exception as exc:  # noqa: BLE001
            log.error("DataGrounding - searching '%s' failed: %s",
                      self.collection_name, exc)
            return []
        return [
            {"id": hit.id, "score": hit.score, "payload": dict(hit.payload or {})}
            for hit in results
        ]

    def search_text(self, text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """:meth:`search` from free text, embedded with this collection's model."""
        if not text:
            return []
        return self.search(self.embed().encode(text), top_k=top_k)

    def search_markdown(self, text: str, top_k: int = 10) -> List[str]:
        """The markdown of the ``top_k`` closest datapoints, best match first.

        The hand-off between the two matching stages: the vector search picks
        the candidates cheaply, and what comes back is readable documents —
        ready to drop into an LLM prompt for the slower, sharper judgement, or
        into a viewer for a person to read. Datapoints stored before a
        ``markdown_field`` was configured are skipped rather than returned
        empty.
        """
        return [
            hit["payload"][self.markdown_field]
            for hit in self.search_text(text, top_k=top_k)
            if self.markdown_field and hit["payload"].get(self.markdown_field)
        ]

    # ------------------------------------------------------------------
    # Markdown rendering
    # ------------------------------------------------------------------

    def to_markdown(self, payload: BaseModel) -> str:
        """The markdown stored alongside a datapoint.

        Three sources, in order of how much the caller has said:

        1. the ``markdown_renderer`` given at construction;
        2. the payload model's own ``to_markdown()``, when it defines one —
           a connector that already knows how its items should read stays in
           charge of it;
        3. :meth:`render_markdown`, the generic field-driven rendering.

        A payload that already carries a non-empty value in
        :attr:`markdown_field` is taken at its word and returned as-is.

        Never raises: a renderer that fails degrades to the generic rendering,
        and that in turn to ``""`` — a datapoint is still worth storing
        without its markdown.
        """
        existing = getattr(payload, self.markdown_field, None) if self.markdown_field else None
        if isinstance(existing, str) and existing.strip():
            return existing

        for source in (self.markdown_renderer, getattr(payload, "to_markdown", None)):
            if source is None:
                continue
            try:
                rendered = source(payload) if source is self.markdown_renderer else source()
                if isinstance(rendered, str) and rendered.strip():
                    return rendered
            except Exception as exc:  # noqa: BLE001 - fall through to the generic form
                log.warning(
                    "DataGrounding - markdown renderer failed for %s (%s); "
                    "falling back to the generic rendering.",
                    type(payload).__name__, exc,
                )
        try:
            return self.render_markdown(payload)
        except Exception as exc:  # noqa: BLE001
            log.error("DataGrounding - rendering markdown for %s failed: %s",
                      type(payload).__name__, exc)
            return ""

    def render_markdown(self, payload: BaseModel) -> str:
        """Render an entity's fields as markdown, images included.

        One layout serves every kind of source — a concert, a startup, a
        video, a paper all read the same way, which is what lets MatchMake's
        CardViewer display any grounded collection with no per-collection
        config::

            # Wolves in the Throne Room

            ![Wolves in the Throne Room](https://fri-son.ch/img/wittr.jpg)

            Cascadian black metal at Fri-Son, doors at 20:00.

            **Venue:** Fri-Son, Fribourg
            **Date:** 2026-09-12

            [Tickets](https://fri-son.ch/tickets/wittr)

        Fields are classified by name, then by value: the entity's name
        becomes the heading, image fields become inline images (so an LLM
        reading the payload sees the picture too), prose fields become
        paragraphs, remaining links become a trailing link list, and anything
        else becomes a ``**Label:** value`` line. Lists of image URLs render
        as a series of images, lists of links as a series of links. Empty
        fields and grounding metadata are left out.
        """
        body = payload.model_dump()
        title = self._title_of(body)
        title_field = next(
            (field for field in self.title_fields if body.get(field) == title), None
        )

        heading = [f"# {title}"] if title else []
        images, paragraphs, facts, links = [], [], [], []

        for name, value in body.items():
            if name in _METADATA_KEYS or name == title_field:
                continue
            if value is None or value == "" or value == [] or value == {}:
                continue

            urls = self._url_list(value)
            if urls and self._is_image_field(name, urls[0]):
                images.extend(f"![{title or self._label(name)}]({url})" for url in urls)
            elif urls:
                links.extend(f"[{self._label(name)}]({url})" for url in urls)
            elif self._is_image_field(name, value):
                images.append(f"![{title or self._label(name)}]({value})")
            elif self._is_link_field(name, value):
                links.append(f"[{self._label(name)}]({value})")
            elif self._is_body_field(name, value):
                paragraphs.append(str(value).strip())
            else:
                facts.append(f"**{self._label(name)}:** {self._format_value(value)}")

        blocks = []
        for group in (heading, images, paragraphs):
            blocks.extend(group)
        if facts:
            blocks.append("\n".join(facts))
        blocks.extend(links)
        return "\n\n".join(block for block in blocks if block).strip()

    # -- field classification ------------------------------------------

    @staticmethod
    def _label(name: str) -> str:
        """``"poster_url"`` -> ``"Poster"`` — a field name as a human label."""
        for hint in _LINK_HINTS:
            if name.endswith(f"_{hint}"):
                name = name[: -len(hint) - 1]
                break
        return name.replace("_", " ").strip().capitalize() or "Field"

    @staticmethod
    def _is_url(value: Any) -> bool:
        return isinstance(value, str) and bool(_URL_VALUE.match(value.strip()))

    @classmethod
    def _url_list(cls, value: Any) -> List[str]:
        """The URLs in a list field — a photo gallery, a set of source links.

        Empty unless *every* entry is a URL, so a list of names or tags still
        renders as one ``**Label:** a, b, c`` line.
        """
        if not isinstance(value, (list, tuple)) or not value:
            return []
        items = [item for item in value if item not in (None, "")]
        if items and all(cls._is_url(item) for item in items):
            return [str(item).strip() for item in items]
        return []

    @classmethod
    def _is_image_field(cls, name: str, value: Any) -> bool:
        """An image is a URL whose field name says so, or whose path does."""
        if not cls._is_url(value):
            return False
        lowered = name.lower()
        return (any(hint in lowered for hint in _IMAGE_HINTS)
                or bool(_IMAGE_EXTENSION.search(value.strip())))

    @classmethod
    def _is_link_field(cls, name: str, value: Any) -> bool:
        return cls._is_url(value)

    @staticmethod
    def _is_body_field(name: str, value: Any) -> bool:
        """Prose: a field named like prose, or a string long enough to be it."""
        if not isinstance(value, str):
            return False
        lowered = name.lower()
        return (any(hint in lowered for hint in _BODY_HINTS)
                or len(value) >= _BODY_MIN_LENGTH)

    @staticmethod
    def _format_value(value: Any) -> str:
        """A scalar or list as one readable line."""
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(item) for item in value if item not in (None, ""))
        if isinstance(value, dict):
            return ", ".join(f"{k}: {v}" for k, v in value.items() if v not in (None, ""))
        if isinstance(value, bool):
            return "yes" if value else "no"
        return str(value).strip()

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self) -> Embed:
        """The lazily built :class:`~vectorize.embeddings.Embed` for this collection.

        Deferred until first use so constructing a ``DataGrounding`` never
        loads a SentenceTransformer model or requires an API key — a purge or
        a remove needs neither.
        """
        if self._embed is None:
            self._embed = Embed(self.embedding, self.aggregation, self.vector_dimension)
        return self._embed

    def encode(self, payload: BaseModel) -> Optional[List[float]]:
        """The embedding of a payload's text fields, or ``None`` when it has none."""
        text = self.embeddable_text(payload)
        if not text:
            return None
        return self.embed().encode(text)

    def embeddable_text(self, payload: BaseModel) -> str:
        """The text embedded for a payload: :attr:`embedded_fields` joined.

        With no explicit ``embedded_fields``, every non-empty string field of
        the model is used except the grounding metadata and **the URLs**. A
        URL is where an entity lives, not what it is: embedding it adds no
        meaning and does add noise, since entities from one site share a
        domain prefix and would drift together for that reason alone. This is
        the one place the default deliberately differs from
        ``vectorize.vector_db.Qdrant._reencode_vector``, which embeds every
        string field — pass ``embedded_fields`` explicitly on a collection
        that must survive being re-encoded by vectorize.
        """
        body = payload.model_dump()
        fields = self.embedded_fields or [
            key for key, value in body.items()
            if isinstance(value, str) and key not in _METADATA_KEYS
            and not self._is_url(value)
        ]
        return " ".join(str(body[f]) for f in fields if body.get(f)).strip()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _title_of(self, body: Dict[str, Any]) -> str:
        """The entity's name: the first :attr:`title_fields` entry it fills in.

        A pinned ``title_field`` makes this exact; otherwise a model saying
        ``name`` (a lab, a startup, a place) works as readily as one saying
        ``title`` (a paper, a video), with no configuration either way.
        """
        for field in self.title_fields:
            value = body.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _identity_of(self, body: Dict[str, Any]) -> List[Tuple[str, str]]:
        """The ``(field, value)`` pairs that identify this entity."""
        found = []
        for field in self.identity_fields:
            value = body.get(field)
            if isinstance(value, str) and value.strip():
                found.append((field, value.strip()))
        return found

    def _find_by_identity(self, body: Dict[str, Any]) -> Optional[Union[str, int]]:
        """The id of a stored datapoint sharing an identifying URL or code.

        A single filtered lookup: any identity field matching any of this
        entity's values is the same entity. Cross-field on purpose — the URL
        one source calls ``detail_url`` another calls ``link``, and it is
        still the same page.
        """
        pairs = self._identity_of(body)
        if not pairs:
            return None
        conditions = [
            FieldCondition(key=field, match=MatchValue(value=value))
            for field in self.identity_fields
            for _, value in pairs
        ]
        try:
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(should=conditions),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
        except Exception as exc:  # noqa: BLE001 - unknown, let the next tier decide
            log.error("DataGrounding - identity lookup in '%s' failed: %s",
                      self.collection_name, exc)
            return None
        return points[0].id if points else None

    @staticmethod
    def _resolve_id(point_id: Optional[Union[str, int]]
                    ) -> Tuple[Union[str, int], Optional[str]]:
        """Map a caller's id onto one Qdrant accepts.

        Ints and UUID strings pass through untouched. Any other string is a
        natural key (a URL, a slug): it becomes a deterministic UUID5 so the
        same key always addresses the same datapoint, and the original is
        returned alongside to be kept in the payload's ``external_id``.
        """
        if point_id is None:
            return str(uuid.uuid4()), None
        if isinstance(point_id, int):
            return point_id, None
        text = str(point_id)
        try:
            return str(uuid.UUID(text)), None
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, text)), text

    def _vector_name(self) -> Optional[str]:
        """The collection's vector name, or ``None`` when it uses unnamed vectors."""
        try:
            config = self.client.get_collection(self.collection_name).config.params.vectors
        except Exception:  # noqa: BLE001 - fall back to this instance's preference
            return self.embedding.reference_name if self.named_vector else None
        if isinstance(config, dict):
            # Prefer this instance's embedding when the collection holds several.
            if self.embedding.reference_name in config:
                return self.embedding.reference_name
            return next(iter(config), None)
        return None

    def _wrap_vector(self, vector: List[float]):
        """Shape a vector for upsert: named when the collection uses named vectors."""
        name = self._vector_name()
        return {name: vector} if name else vector

    def _query_vector(self, vector: List[float]):
        """Shape a vector for search — Qdrant wants ``(name, vector)`` when named."""
        name = self._vector_name()
        return (name, vector) if name else vector

    def _stored_created_at(self, resolved_id: Union[str, int]) -> Optional[datetime]:
        """The ``created_at`` already stored for an id, when the datapoint exists."""
        payload = self.get(resolved_id)
        if not payload or not payload.get(CREATED_AT):
            return None
        try:
            return datetime.fromisoformat(payload[CREATED_AT])
        except (TypeError, ValueError):
            return None

    def _stored_titles(self, limit: int) -> List[Tuple[Union[str, int], str]]:
        """``(point_id, name)`` for up to *limit* stored datapoints.

        Scrolls the name fields only (never vectors), which keeps the fuzzy
        check cheap enough to run before every insert.
        """
        titles: List[Tuple[Union[str, int], str]] = []
        offset = None
        batch = 256
        try:
            while len(titles) < limit:
                points, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=min(batch, limit - len(titles)),
                    offset=offset,
                    with_payload=self.title_fields,
                    with_vectors=False,
                )
                if not points:
                    break
                for point in points:
                    title = self._title_of(point.payload or {})
                    if title:
                        titles.append((point.id, title))
                if offset is None:
                    break
        except Exception as exc:  # noqa: BLE001 - a failed scan means "unknown",
            # and the vector check in find_duplicate still gets its say.
            log.error("DataGrounding - reading titles from '%s' failed: %s",
                      self.collection_name, exc)
        return titles
