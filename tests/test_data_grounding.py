"""DataGrounding against a real in-memory Qdrant, with a stub embedder.

The embedding is stubbed (a deterministic hash of the text) so the suite never
downloads a SentenceTransformer model nor calls the Google API — everything
being tested here is the grounding logic around the embedding, not the
embedding itself.

``CinemaMovie`` is used where any entity would do, and the models defined near
the bottom (a video, a startup, a place, ...) where the point is precisely
that the class is not built for one kind of entity.
"""
import hashlib
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("qdrant_client")
pytest.importorskip("vectorize")

from qdrant_client import QdrantClient  # noqa: E402

from pydantic import BaseModel, Field  # noqa: E402

from connectors.cinemotion import CinemaMovie  # noqa: E402
from connectors.grounding import (  # noqa: E402
    CREATED_AT, EXTERNAL_ID, MARKDOWN, UPDATED_AT, UPDATED_TS, DataGrounding,
)
from vectorize.embeddings import EmbeddingSourceType, EmbeddingVector  # noqa: E402

DIM = 16


class StubEmbed:
    """Deterministic pseudo-embedding: the same text always yields the same vector."""

    def encode(self, text):
        digest = hashlib.sha256(text.encode()).digest()
        vector = [digest[i % len(digest)] / 255.0 for i in range(DIM)]
        norm = sum(v * v for v in vector) ** 0.5
        return [v / norm for v in vector]


@pytest.fixture
def embedding():
    return EmbeddingVector(
        reference_name="stub/model",
        source_type=EmbeddingSourceType.SENTENCE_TRANSFORMER,
        api_key_name="NONE",
        dimensions=[DIM],
    )


@pytest.fixture
def grounding(embedding):
    """No ``title_field`` / ``identity_fields``: the defaults must cope alone."""
    instance = DataGrounding(
        QdrantClient(":memory:"), "entities", embedding,
        vector_dimension=DIM, ttl=timedelta(days=30),
    )
    instance._embed = StubEmbed()
    return instance


@pytest.fixture
def movie():
    return CinemaMovie(
        title="BACKROOMS",
        poster_url="https://cinemotion.ch/image/FA0009872.jpg",
        detail_url="https://www.cinemotion.ch/cette-semaine/backrooms-9872",
        description="A horror movie about endless rooms.",
    )


# ----------------------------------------------------------------------
# Collection lifecycle
# ----------------------------------------------------------------------

def test_create_is_idempotent(grounding):
    assert not grounding.exists()
    assert grounding.create()
    assert grounding.exists()
    assert grounding.create(), "creating an existing collection must succeed quietly"
    assert grounding.count() == 0


def test_count_and_exists_on_a_missing_collection(grounding):
    assert grounding.count() == 0
    assert not grounding.exists()


# ----------------------------------------------------------------------
# Datapoints
# ----------------------------------------------------------------------

def test_add_stores_payload_timestamps_and_external_id(grounding, movie):
    point = grounding.add(movie, point_id=movie.detail_url)
    assert point is not None
    assert grounding.count() == 1

    stored = grounding.get(movie.detail_url)
    assert stored["title"] == "BACKROOMS"
    assert stored["description"] == movie.description
    assert stored[EXTERNAL_ID] == movie.detail_url
    assert stored["payload_type"] == "CinemaMovie"
    assert datetime.fromisoformat(stored[CREATED_AT])
    assert datetime.fromisoformat(stored[UPDATED_AT])


def test_a_natural_key_always_addresses_the_same_datapoint(grounding, movie):
    first = grounding.add(movie, point_id=movie.detail_url)
    second = grounding.add(movie, point_id=movie.detail_url)
    assert first.id == second.id
    assert grounding.count() == 1


@pytest.mark.parametrize("point_id", [42, "3f0b3a3e-2d64-4f9e-8a5b-9c6c1f0d2e11"])
def test_explicit_int_and_uuid_ids_are_kept_verbatim(grounding, movie, point_id):
    point = grounding.add(movie, point_id=point_id)
    assert str(point.id) == str(point_id)
    assert EXTERNAL_ID not in grounding.get(point_id)


def test_update_refreshes_content_but_keeps_created_at(grounding, movie):
    grounding.add(movie, point_id=movie.detail_url,
                  created_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
    revised = movie.model_copy(update={"description": "Updated synopsis."})
    grounding.update(movie.detail_url, revised)

    stored = grounding.get(movie.detail_url)
    assert stored["description"] == "Updated synopsis."
    assert stored[CREATED_AT] == datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat()
    assert stored[UPDATED_AT] > stored[CREATED_AT]
    assert grounding.count() == 1


def test_add_skips_a_payload_with_no_embeddable_text(grounding):
    empty = CinemaMovie(title="", poster_url="", detail_url="", description="")
    assert grounding.add(empty, point_id="empty") is None


def test_remove_and_clear(grounding, movie):
    grounding.add(movie, point_id=movie.detail_url)
    assert grounding.remove(movie.detail_url)
    assert grounding.count() == 0
    assert grounding.get(movie.detail_url) is None

    grounding.add(movie, point_id=movie.detail_url)
    assert grounding.clear()
    assert grounding.count() == 0
    assert grounding.exists(), "clear keeps the collection itself"


def test_remove_on_a_missing_collection_reports_false(grounding):
    assert grounding.remove("anything") is False


# ----------------------------------------------------------------------
# Time-to-live
# ----------------------------------------------------------------------

def test_purge_drops_only_what_is_older_than_the_ttl(grounding, movie):
    now = datetime.now(timezone.utc)
    grounding.add(movie, point_id="fresh")
    old = movie.model_copy(update={"title": "An Ancient Film"})
    grounding.add(old, point_id="old",
                  created_at=now - timedelta(days=90),
                  updated_at=now - timedelta(days=90))

    assert grounding.purge() == 1
    assert grounding.count() == 1
    assert grounding.get("old") is None
    assert grounding.purge() == 0


def test_purge_by_update_time(grounding, movie):
    now = datetime.now(timezone.utc)
    # First seen long ago, refreshed today: expired by creation, alive by update.
    grounding.add(movie, point_id=movie.detail_url,
                  created_at=now - timedelta(days=90), updated_at=now)
    assert grounding.purge(by=UPDATED_TS) == 0
    assert grounding.purge() == 1


def test_purge_without_a_ttl_is_a_no_op(embedding, movie):
    grounding = DataGrounding(QdrantClient(":memory:"), "no_ttl", embedding,
                              vector_dimension=DIM)
    grounding._embed = StubEmbed()
    grounding.add(movie, point_id="a")
    assert grounding.purge() == 0
    assert grounding.count() == 1


def test_purge_rejects_an_unknown_field(grounding):
    with pytest.raises(ValueError):
        grounding.purge(by="title")


# ----------------------------------------------------------------------
# Novelty
# ----------------------------------------------------------------------

def test_everything_is_new_in_an_empty_collection(grounding, movie):
    assert grounding.is_new(movie)


def test_a_stored_item_is_not_new(grounding, movie):
    grounding.add(movie, point_id=movie.detail_url)
    assert not grounding.is_new(movie)


def test_a_retitled_duplicate_is_caught_by_the_fuzzy_check(grounding, movie):
    grounding.add(movie, point_id=movie.detail_url)
    variant = CinemaMovie(title="Backrooms (2025)", detail_url="elsewhere",
                          description="Different words entirely.")
    found = grounding.find_duplicate(variant)
    assert found is not None
    point_id, score, method = found
    assert method == "fuzzy"
    assert score >= 0.85


def test_the_same_content_under_another_title_is_caught_by_the_vector_check(
        embedding):
    grounding = DataGrounding(QdrantClient(":memory:"), "vectors", embedding,
                              vector_dimension=DIM, embedded_fields=["description"])
    grounding._embed = StubEmbed()
    synopsis = "A man wanders an infinite office maze."
    grounding.add(CinemaMovie(title="Endless Rooms", description=synopsis),
                  point_id="en")

    found = grounding.find_duplicate(
        CinemaMovie(title="Der Endlose Flur", description=synopsis))
    assert found is not None
    assert found[2] == "vector"
    assert found[1] >= 0.95


def test_an_unrelated_item_is_new(grounding, movie):
    grounding.add(movie, point_id=movie.detail_url)
    assert grounding.is_new(CinemaMovie(title="Anora",
                                        description="A completely different story."))


# ----------------------------------------------------------------------
# Embedding and search
# ----------------------------------------------------------------------

def test_embeddable_text_defaults_to_every_string_field(grounding, movie):
    text = grounding.embeddable_text(movie)
    assert movie.title in text and movie.description in text


def test_embeddable_text_honours_explicit_fields(embedding, movie):
    grounding = DataGrounding(QdrantClient(":memory:"), "fields", embedding,
                              vector_dimension=DIM, embedded_fields=["title"])
    assert grounding.embeddable_text(movie) == movie.title


def test_search_text_finds_the_stored_datapoint(grounding, movie):
    grounding.add(movie, point_id=movie.detail_url)
    hits = grounding.search_text(grounding.embeddable_text(movie), top_k=1)
    assert hits and hits[0]["score"] > 0.99
    assert hits[0]["payload"]["title"] == "BACKROOMS"


def test_search_on_a_missing_collection_returns_nothing(grounding):
    assert grounding.search_text("anything") == []


def test_unnamed_vector_collections_are_supported(embedding, movie):
    grounding = DataGrounding(QdrantClient(":memory:"), "unnamed", embedding,
                              vector_dimension=DIM, named_vector=False)
    grounding._embed = StubEmbed()
    assert grounding.add(movie, point_id="a") is not None
    assert grounding.search_text(movie.description, top_k=1)


def test_a_closed_client_is_rejected_at_construction(embedding):
    with pytest.raises(ValueError):
        DataGrounding(None, "nope", embedding)


# ----------------------------------------------------------------------
# Markdown rendering
# ----------------------------------------------------------------------

class Paper(BaseModel):
    """A payload exercising every field class the renderer distinguishes."""
    title: str
    abstract: str = ""
    authors: list = Field(default_factory=list)
    year: int = 0
    peer_reviewed: bool = False
    media: str = ""          # image recognised by extension, not by name
    pdf_url: str = ""
    nothing: str = ""


def test_rendering_follows_the_cinemotion_card_layout(grounding, movie):
    """Heading, poster, synopsis, link — the shape ``cinemotion.to_item`` writes
    by hand, so a grounded datapoint renders like a live one."""
    rendered = grounding.to_markdown(movie)
    assert rendered.startswith("# BACKROOMS")
    assert f"![BACKROOMS]({movie.poster_url})" in rendered
    assert movie.description in rendered
    assert f"[Detail]({movie.detail_url})" in rendered


def test_markdown_is_stored_on_the_datapoint(grounding, movie):
    grounding.add(movie, point_id=movie.detail_url)
    stored = grounding.get(movie.detail_url)
    assert stored[MARKDOWN] == grounding.to_markdown(movie)


def test_markdown_is_not_embedded(grounding, movie):
    """It restates every other field; embedding it would count each one twice."""
    assert MARKDOWN not in grounding.embeddable_text(movie)


def test_rendering_classifies_images_prose_facts_and_links(grounding):
    paper = Paper(title="Attention Is All You Need",
                  abstract="We propose the Transformer.",
                  authors=["Vaswani", "Shazeer"], year=2017,
                  peer_reviewed=True,
                  media="https://arxiv.org/figures/fig1.png",
                  pdf_url="https://arxiv.org/pdf/1706.03762")
    rendered = grounding.to_markdown(paper)

    assert rendered.startswith("# Attention Is All You Need")
    assert "![Attention Is All You Need](https://arxiv.org/figures/fig1.png)" in rendered
    assert "We propose the Transformer." in rendered
    assert "**Authors:** Vaswani, Shazeer" in rendered
    assert "**Year:** 2017" in rendered
    assert "**Peer reviewed:** yes" in rendered
    assert "[Pdf](https://arxiv.org/pdf/1706.03762)" in rendered
    assert "nothing" not in rendered.lower(), "empty fields are left out"


def test_a_payload_model_may_render_itself(grounding):
    class Custom(BaseModel):
        title: str

        def to_markdown(self):
            return "# custom rendering"

    assert grounding.to_markdown(Custom(title="ignored")) == "# custom rendering"


def test_a_failing_renderer_degrades_to_the_generic_rendering(grounding):
    class Broken(BaseModel):
        title: str

        def to_markdown(self):
            raise RuntimeError("boom")

    assert grounding.to_markdown(Broken(title="Still Rendered")) == "# Still Rendered"


def test_an_explicit_renderer_wins(embedding, movie):
    grounding = DataGrounding(QdrantClient(":memory:"), "custom", embedding,
                              vector_dimension=DIM,
                              markdown_renderer=lambda payload: f"> {payload.title}")
    assert grounding.to_markdown(movie) == "> BACKROOMS"


def test_a_payload_that_already_has_markdown_is_trusted(grounding):
    class Doc(BaseModel):
        title: str
        markdown: str

    assert grounding.to_markdown(
        Doc(title="t", markdown="# already written")) == "# already written"


def test_markdown_can_be_switched_off(embedding, movie):
    grounding = DataGrounding(QdrantClient(":memory:"), "plain", embedding,
                              vector_dimension=DIM, markdown_field=None)
    grounding._embed = StubEmbed()
    grounding.add(movie, point_id="a")
    assert MARKDOWN not in grounding.get("a")
    assert grounding.search_markdown("rooms") == []


def test_search_markdown_returns_readable_documents(grounding, movie):
    grounding.add(movie, point_id=movie.detail_url)
    docs = grounding.search_markdown(movie.description, top_k=3)
    assert docs and docs[0].startswith("# BACKROOMS")


# ----------------------------------------------------------------------
# Any kind of entity
#
# These models share no field names with each other or with CinemaMovie:
# what is being checked is that an agent can define whatever shape an
# information source calls for and ground it with no configuration.
# ----------------------------------------------------------------------

class Video(BaseModel):
    title: str
    channel: str = ""
    description: str = ""
    views: int = 0
    thumbnail_url: str = ""
    watch_url: str = ""


class Startup(BaseModel):
    name: str                       # not "title"
    pitch: str = ""
    sector: str = ""
    founded: int = 0
    hiring: bool = False
    logo_url: str = ""
    website: str = ""


class Concert(BaseModel):
    name: str
    lineup: list = Field(default_factory=list)
    venue: str = ""
    date: str = ""
    poster: str = ""
    tickets: str = ""


class Place(BaseModel):
    name: str
    overview: str = ""
    region: str = ""
    photos: list = Field(default_factory=list)   # a list of image URLs
    source_url: str = ""


VIDEO = Video(title="How Qdrant Works", channel="Vector DB Weekly",
              description="A walkthrough of HNSW indexing.", views=13400,
              thumbnail_url="https://i.ytimg.com/vi/abc123/hq.jpg",
              watch_url="https://www.youtube.com/watch?v=abc123")
STARTUP = Startup(name="Nagra Quantum", pitch="Post-quantum key management.",
                  sector="Security", founded=2021, hiring=True,
                  logo_url="https://nagraquantum.ch/logo.svg",
                  website="https://nagraquantum.ch")
CONCERT = Concert(name="Wolves in the Throne Room",
                  lineup=["Wolves in the Throne Room", "Blackbraid"],
                  venue="Fri-Son", date="2026-09-12",
                  poster="https://fri-son.ch/img/wittr.jpg",
                  tickets="https://fri-son.ch/tickets/wittr")
PLACE = Place(name="Gorges de l'Areuse",
              overview="A limestone gorge walk between Noiraigue and Boudry.",
              region="Neuchatel",
              photos=["https://ne.ch/areuse1.jpg", "https://ne.ch/areuse2.jpg"],
              source_url="https://www.neuchateltourisme.ch/areuse")


@pytest.mark.parametrize("entity, heading", [
    (VIDEO, "# How Qdrant Works"),
    (STARTUP, "# Nagra Quantum"),       # named by "name", not "title"
    (CONCERT, "# Wolves in the Throne Room"),
    (PLACE, "# Gorges de l'Areuse"),
])
def test_the_entity_name_is_found_whatever_the_field_is_called(
        grounding, entity, heading):
    assert grounding.to_markdown(entity).startswith(heading)


@pytest.mark.parametrize("entity", [VIDEO, STARTUP, CONCERT, PLACE])
def test_any_entity_grounds_with_no_configuration(grounding, entity):
    assert grounding.add(entity) is not None
    assert not grounding.is_new(entity)


def test_images_are_recognised_by_name_and_by_extension(grounding):
    assert f"![How Qdrant Works]({VIDEO.thumbnail_url})" in grounding.to_markdown(VIDEO)
    assert f"![Nagra Quantum]({STARTUP.logo_url})" in grounding.to_markdown(STARTUP)
    # "poster" and "tickets" are named neither *_url nor *_image
    rendered = grounding.to_markdown(CONCERT)
    assert f"![Wolves in the Throne Room]({CONCERT.poster})" in rendered
    assert f"[Tickets]({CONCERT.tickets})" in rendered


def test_a_list_of_image_urls_renders_as_images(grounding):
    rendered = grounding.to_markdown(PLACE)
    assert rendered.count("![Gorges de l'Areuse](") == 2


def test_a_list_of_plain_values_stays_one_fact_line(grounding):
    rendered = grounding.to_markdown(CONCERT)
    assert "**Lineup:** Wolves in the Throne Room, Blackbraid" in rendered
    assert "**Venue:** Fri-Son" in rendered
    assert "**Date:** 2026-09-12" in rendered


def test_urls_are_left_out_of_the_embedded_text(grounding):
    """A URL is where an entity lives, not what it is."""
    text = grounding.embeddable_text(VIDEO)
    assert "How Qdrant Works" in text and "HNSW" in text
    assert "youtube.com" not in text and "ytimg" not in text


def test_a_shared_url_identifies_the_entity_however_it_is_worded(grounding):
    grounding.add(VIDEO)
    renamed = Video(title="Qdrant Internals Explained",
                    description="Entirely different words.",
                    watch_url=VIDEO.watch_url)
    found = grounding.find_duplicate(renamed)
    assert found is not None and found[2] == "identity"


def test_identity_matching_crosses_field_names(grounding):
    """One source's ``detail_url`` is another's ``link`` — still one page."""
    class SourceA(BaseModel):
        name: str
        detail_url: str = ""

    class SourceB(BaseModel):
        name: str
        link: str = ""

    grounding.add(SourceA(name="Idiap Research Institute", detail_url="https://idiap.ch"))
    found = grounding.find_duplicate(SourceB(name="Idiap", link="https://idiap.ch"))
    assert found is not None and found[2] == "identity"
    assert grounding.is_new(SourceB(name="CSEM", link="https://csem.ch"))


def test_fuzzy_matching_works_on_a_name_field(grounding):
    grounding.add(STARTUP)
    found = grounding.find_duplicate(
        Startup(name="NAGRA QUANTUM (2021)", pitch="Different words entirely."))
    assert found is not None and found[2] == "fuzzy"


def test_fields_can_be_pinned_when_a_source_names_things_its_own_way(embedding):
    class Sample(BaseModel):
        designation: str            # none of the conventional names
        ref: str = ""

    grounding = DataGrounding(QdrantClient(":memory:"), "pinned", embedding,
                              vector_dimension=DIM, title_field="designation",
                              identity_fields=["ref"])
    grounding._embed = StubEmbed()
    grounding.add(Sample(designation="Sample 42", ref="LAB-42"))

    assert grounding.to_markdown(
        Sample(designation="Sample 42", ref="LAB-42")).startswith("# Sample 42")
    found = grounding.find_duplicate(Sample(designation="Unrelated", ref="LAB-42"))
    assert found is not None and found[2] == "identity"


def test_an_entity_with_no_recognisable_name_still_grounds(grounding):
    class Anonymous(BaseModel):
        body: str

    entity = Anonymous(body="Just prose, no name anywhere.")
    assert grounding.to_markdown(entity) == "Just prose, no name anywhere."
    assert grounding.add(entity) is not None
    assert grounding.is_new(Anonymous(body="Completely unrelated content."))
