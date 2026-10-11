"""What a scraped page grounds AS: the entities it is about.

:class:`~connectors.grounding.data_grounding.DataGrounding` stores entities,
not pages — "a video, a paper, a lab, an enterprise, a startup, an artwork, a
music clip, a concert". This module is the step before it: turning one
scraped page into the entities it actually lists, so a datapoint is *one
movie* (title, poster, synopsis) rather than *one whole cinema homepage*.

Why it matters for what the user sees: the datapoint's markdown is what the
Newsroom renders in a ShowCard when an Angel event is clicked. Grounding a
whole page stores the page's markdown — navigation, cookie banners, every
movie at once. Grounding the movies stores each movie's own card — heading,
poster image, synopsis — identical to what the live-scrape path
(:func:`connectors.cinemotion.to_item`) shows, because both call the same
``to_markdown()``.

The registry is **open and ordered**: each entry is a ``(matches, extract)``
pair, consulted in order, first non-empty result wins. A domain nobody wrote
an extractor for still grounds — as a single :class:`WebDocument` for the
whole page, exactly as before — so any JSON list of URLs from any infomedia
domain keeps working. Adding a source means adding one extractor here, not
touching the grounding, the Angels or the Newsroom.

Both hosts use this one implementation: MatchMake through
``api.web_grounding`` (in-app runs) and keopy through ``angel_grounding.py``
(the scheduled grounding), so a movie grounded on either side is the same
datapoint.
"""
from __future__ import annotations

from typing import Callable, List, Optional

from pydantic import BaseModel, Field

import logging
log = logging.getLogger(__name__)


#: Fields whose value identifies an entity, most specific first — used to
#: derive a stable Qdrant point id (see :func:`entity_id`), so re-grounding
#: the same movie updates its datapoint instead of duplicating it.
_ID_FIELDS = ("detail_url", "url", "source_url", "page_url", "link",
              "permalink", "website", "doi", "identifier", "external_id")


class WebDocument(BaseModel):
    """A whole scraped page as a groundable entity — the generic fallback.

    Used when no extractor claims the page. DataGrounding's field
    conventions read it with no configuration: ``title`` names the entity,
    ``url`` identifies it, ``description`` is the prose that gets embedded.
    """
    title: str = Field(default="", description="The page title")
    description: str = Field(default="", description="The page content as markdown")
    url: str = Field(default="", description="The page URL — the entity's identity")


def page_document(page) -> WebDocument:
    """A scraped page (anything with ``url`` / ``title`` / ``markdown``, so
    both :class:`connectors.scraper.ScrapedPage` and the Angel bundle's
    stdlib twin) as a :class:`WebDocument`."""
    return WebDocument(
        title=getattr(page, "title", "") or "",
        description=getattr(page, "markdown", "") or "",
        url=getattr(page, "url", "") or "",
    )


def entity_id(entity: BaseModel) -> str:
    """A stable natural key for one entity: its first identifying URL, else
    its title. DataGrounding hashes it into a UUID5, so the same entity
    always addresses the same datapoint.
    """
    body = entity.model_dump() if hasattr(entity, "model_dump") else dict(entity)
    for field in _ID_FIELDS:
        value = body.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for field in ("title", "name", "headline"):
        value = body.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


# ─────────────────────────────────────────────────────────────────────────
#  Extractors
# ─────────────────────────────────────────────────────────────────────────

def _is_cinemotion(page) -> bool:
    return "cinemotion.ch" in (getattr(page, "url", "") or "").lower()


def _cinemotion_movies(page, fetch: Optional[Callable] = None) -> List[BaseModel]:
    """The movies of a cinemotion.ch program page, synopses included.

    :func:`connectors.cinemotion.parse_listing` reads the listing markdown
    into movies (title, poster, detail page); each movie's synopsis lives on
    its own detail page, so when a ``fetch`` callback is given (the host's
    scraper) that page is retrieved and
    :func:`connectors.cinemotion.extract_description` pulls the synopsis
    out — the same two-step the Fribourg cinema demo performs. Without
    ``fetch`` the movies still ground, with an empty description.
    """
    from connectors.cinemotion import extract_description, parse_listing

    movies = parse_listing(getattr(page, "markdown", "") or "")
    if not movies:
        return []
    if fetch is not None:
        for movie in movies:
            if not movie.detail_url:
                continue
            try:
                detail = fetch(movie.detail_url)
            except Exception as exc:  # noqa: BLE001 - a movie without its synopsis still grounds
                log.warning("entities - could not fetch %s: %s",
                            movie.detail_url, exc)
                continue
            if detail is not None and getattr(detail, "success", False):
                movie.description = extract_description(
                    getattr(detail, "markdown", "") or "", movie.title)
    log.info("entities - %d movie(s) extracted from %s",
             len(movies), getattr(page, "url", "?"))
    return movies


def _host_is(*hosts: str) -> Callable:
    def matches(page) -> bool:
        url = (getattr(page, "url", "") or "").lower()
        return any(host in url for host in hosts)
    return matches


def _fill_details(movies: List[BaseModel], fetch: Optional[Callable], fill: Callable) -> None:
    """Open each movie's own page with ``fetch`` and let ``fill(movie,
    markdown)`` complete it; a page that cannot be fetched leaves the movie
    as the listing described it."""
    if fetch is None:
        return
    for movie in movies:
        if not movie.detail_url:
            continue
        try:
            detail = fetch(movie.detail_url)
        except Exception as exc:  # noqa: BLE001 - the listing's movie still grounds
            log.warning("entities - could not fetch %s: %s", movie.detail_url, exc)
            continue
        if detail is not None and getattr(detail, "success", False):
            fill(movie, getattr(detail, "markdown", "") or "")


def _arena_movies(page, fetch: Optional[Callable] = None) -> List[BaseModel]:
    """The movies of an Arena Cinemas program page; the listing carries the
    synopses, so nothing is fetched."""
    from connectors.cinema_programs import parse_arena
    movies = parse_arena(getattr(page, "markdown", "") or "")
    log.info("entities - %d movie(s) extracted from %s", len(movies), getattr(page, "url", "?"))
    return movies


def _korso_movies(page, fetch: Optional[Callable] = None) -> List[BaseModel]:
    """The movies of Kino Korso's program, credits and synopsis from each
    movie's own page."""
    from connectors.cinema_programs import korso_description, korso_poster, parse_korso
    movies = parse_korso(getattr(page, "markdown", "") or "")

    def fill(movie, markdown):
        movie.description = korso_description(markdown, movie.title) or movie.description
        movie.poster_url = movie.poster_url or korso_poster(markdown, movie.title)
    _fill_details(movies, fetch, fill)
    log.info("entities - %d movie(s) extracted from %s", len(movies), getattr(page, "url", "?"))
    return movies


def _outnow_movies(page, fetch: Optional[Callable] = None) -> List[BaseModel]:
    """The movies of an OutNow city program, synopsis, direction and cast
    from each movie's own page."""
    from connectors.cinema_programs import outnow_description, parse_outnow
    movies = parse_outnow(getattr(page, "markdown", "") or "")

    def fill(movie, markdown):
        movie.description = outnow_description(markdown) or movie.description
    _fill_details(movies, fetch, fill)
    log.info("entities - %d movie(s) extracted from %s", len(movies), getattr(page, "url", "?"))
    return movies


#: ``(matches, extract)`` pairs, consulted in order; the first that returns
#: entities wins. Append a pair to ground a new source as its own entities.
EXTRACTORS = [
    (_is_cinemotion, _cinemotion_movies),
    (_host_is("arena.ch"), _arena_movies),
    (_host_is("korso.ch"), _korso_movies),
    (_host_is("outnow.ch"), _outnow_movies),
]


def _title_key(entity: BaseModel) -> str:
    """A title as a merge key: unaccented, lower-case, letters and digits only."""
    import unicodedata
    title = str(getattr(entity, "title", "") or "")
    plain = "".join(c for c in unicodedata.normalize("NFKD", title)
                    if not unicodedata.combining(c))
    return "".join(c for c in plain.lower() if c.isalnum())


def merge_duplicates(entities: List[BaseModel]) -> List[BaseModel]:
    """One entity per title and kind, across every page of a grounding run.

    The same movie is listed by several programs (Cinemotion's Rex, Arena,
    OutNow's city listing), each knowing part of it: one has the genres, one
    the synopsis. Grounding them apart would keep whichever came first and
    drop the rest as a duplicate, so they are merged before: the first
    entity stays (its URL is the identity), list fields gain the others'
    values, and an empty or shorter text field takes the longest one found.
    Entities without a title are left as they are."""
    merged: List[BaseModel] = []
    first: dict = {}
    for entity in entities:
        key = (type(entity).__name__, _title_key(entity))
        if not key[1]:
            merged.append(entity)
            continue
        kept = first.get(key)
        if kept is None:
            first[key] = entity
            merged.append(entity)
            continue
        for name in type(kept).model_fields:
            mine, theirs = getattr(kept, name), getattr(entity, name)
            if isinstance(mine, list) and isinstance(theirs, list):
                setattr(kept, name, mine + [v for v in theirs if v not in mine])
            elif isinstance(mine, str) and isinstance(theirs, str):
                if name == "description" and len(theirs) > len(mine):
                    setattr(kept, name, theirs)
                elif not mine and theirs:
                    setattr(kept, name, theirs)
    if len(merged) < len(entities):
        log.info("entities - %d listing(s) merged into %d entit(ies)",
                 len(entities), len(merged))
    return merged


def extract_entities(page, fetch: Optional[Callable] = None) -> List[BaseModel]:
    """The entities one scraped page is about.

    Args:
        page:  A scraped page (``url`` / ``title`` / ``markdown``).
        fetch: Optional ``url -> page`` callback the extractors use to open
               an entity's own detail page (a movie's synopsis, a paper's
               abstract). The host's scraper, in practice.

    Returns:
        The extracted entity models, or ``[page_document(page)]`` when no
        extractor claims the page — never empty for a page with content, so
        grounding an unknown domain still stores something useful.
    """
    for matches, extract in EXTRACTORS:
        try:
            if not matches(page):
                continue
            entities = extract(page, fetch)
        except Exception as exc:  # noqa: BLE001 - a failed extractor degrades to the page
            log.error("entities - extractor failed on %s: %s",
                      getattr(page, "url", "?"), exc)
            continue
        if entities:
            return entities
    return [page_document(page)]
