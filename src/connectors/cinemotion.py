"""Cinemotion.ch connector: the Fribourg (Rex), Bulle (Prado) and Payerne
(Apollo) cinema program.

Cinemotion's "Séances" listing page renders its markdown (via the Crawl4AI
scraper, see :mod:`connectors.scraper`) in a very regular shape — one block
per movie, each starting with a linked poster image and title, e.g.::

    [ ![BACKROOMS](https://cinemotion.ch/image/FA0009872.jpg) ](https://www.cinemotion.ch/cette-semaine/backrooms-9872)
    BACKROOMS
    [Infos et séances](https://www.cinemotion.ch/cette-semaine/9872)
    ...

so :func:`parse_listing` extracts (title, poster, detail page) with a single
regex — no HTML/DOM parsing needed. Each movie's own detail page carries its
synopsis (director, cast, plot) between its ``# <TITLE>`` heading and the
``Année :`` metadata block; :func:`extract_description` pulls that out the
same way.

Used by :meth:`api.angel_api.AngelAPI.retrieve_cinema_movies`, which the
Newsroom's Fribourg cinema event click handler
(:class:`ui.newsroom.Newsroom`) calls to retrieve the current lineup, then
narrows it down to the one movie the clicked event's title is actually
about (:func:`movie_from_event_title` + :func:`find_movie`) and shows just
that movie in a :class:`~ui.viewers.showers.show_card.ShowCard` dialog —
the same full-markdown popup a clicked CardView card opens.
:func:`to_item` builds the CARD item backing that markdown and
:func:`rank_movies` orders the lineup by best match to the active persona,
with :class:`matching_learning.matching.ranker.PersonaRanker` — the same ranking Personal
match applies to imdb results — so both the Newsroom UI and the headless
``retrieve_cinema_movies`` flow (``cli_anything``) share one implementation.
"""
import re
import uuid
from typing import List, Optional

from pydantic import BaseModel, Field

import logging
log = logging.getLogger(__name__)

#: Cinemotion's live "Séances" listing for all three cinemas (Bulle, Fribourg,
#: Payerne) — the same URL the Fribourg cinema Guardian Angel demo scrapes
#: (see ``tests/test_angel_demo_cinema.py``).
FRIBOURG_CINEMA_URL = "https://www.cinemotion.ch/seances"

_LISTING_ENTRY = re.compile(
    r"\[ !\[(?P<title>[^\]]*)\]\((?P<poster>https://cinemotion\.ch/image/[^\)]+)\) \]"
    r"\((?P<detail_url>https://www\.cinemotion\.ch/cette-semaine/[^\)]+)\)"
)


class CinemaMovie(BaseModel):
    """One movie on cinemotion.ch's current program."""
    title: str = Field(..., description="The movie's title")
    poster_url: str = Field(default="", description="Poster image URL")
    detail_url: str = Field(default="", description="The movie's own page on cinemotion.ch")
    description: str = Field(
        default="",
        description="Synopsis (director, cast, plot) as markdown, from the "
                    "movie's detail page; \"\" until filled in"
    )


def parse_listing(markdown: str) -> List[CinemaMovie]:
    """Parse cinemotion.ch's Séances listing markdown into movies, in the
    order they appear on the page (deduplicated by detail page URL, since the
    same movie can be linked more than once per showtime)."""
    seen = set()
    movies = []
    for match in _LISTING_ENTRY.finditer(markdown or ""):
        detail_url = match.group("detail_url")
        if detail_url in seen:
            continue
        seen.add(detail_url)
        movies.append(CinemaMovie(
            title=match.group("title").strip(),
            poster_url=match.group("poster"),
            detail_url=detail_url,
        ))
    return movies


def extract_description(markdown: str, title: str) -> str:
    """Best-effort synopsis (director, cast, plot) from a movie's detail page.

    Cinemotion repeats the title as a ``# <TITLE>`` heading right before the
    synopsis block, which runs until the ``Année :`` metadata line; this
    grabs everything in between and drops the empty-alt Facebook share link
    line. Returns "" when the page does not match this shape (a redesigned
    page, a movie removed from the program, ...) rather than guessing.
    """
    if not markdown or not title:
        return ""
    pattern = re.compile(
        r"^# " + re.escape(title) + r"\s*$(.*?)(?:^Ann[ée]e\s*:|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(markdown)
    if not match:
        return ""
    body = re.sub(r"^\[ \]\(https://www\.facebook\.com/sharer[^\)]*\)\s*$", "",
                  match.group(1), flags=re.MULTILINE)
    return body.strip()


_EVENT_TITLE = re.compile(r"^Fribourg cinema: (?P<movie>.+?) matches .+$")


def movie_from_event_title(title: str) -> str:
    """Extract the movie name from a Newsroom event title of the shape
    ``"Fribourg cinema: {movie} matches {persona}"`` — the Guardian Angel
    demo's convention for naming the specific movie a program match is
    about (:meth:`api.angel_api.AngelAPI.best_match_title`, see
    ``tests/test_angel_demo_cinema.py``'s ``DEMO_SCRIPT``). Returns "" when
    ``title`` doesn't match this shape.
    """
    if not title:
        return ""
    match = _EVENT_TITLE.match(title.strip())
    return match.group("movie").strip() if match else ""


def find_movie(movies: List[CinemaMovie], movie_title: str) -> Optional[CinemaMovie]:
    """Return the movie in ``movies`` whose title matches ``movie_title``:
    case-insensitive exact match first, then a substring match either way
    (titles get truncated or reworded between an event and the live
    listing). None when ``movie_title`` is empty or nothing matches.
    """
    if not movie_title:
        return None
    needle = movie_title.strip().lower()
    for movie in movies:
        if movie.title.strip().lower() == needle:
            return movie
    for movie in movies:
        haystack = movie.title.lower()
        if needle in haystack or haystack in needle:
            return movie
    return None


def to_item(movie: CinemaMovie):
    """Build a CARD :class:`~state.item.Item` for one movie.

    There is no Watch/desktop_markdown config backing this live listing (it
    is not a Qdrant collection), so ``item_text`` is a self-contained
    markdown block — title heading, poster image, then the synopsis — which
    is exactly what CardViewer falls back to rendering when it finds no
    Watch config for the turn (see ``CardViewer._build_rendered_markdown``).
    """
    from state.item import Item
    from state.watch import ViewerType

    lines = [f"# {movie.title}", ""]
    if movie.poster_url:
        lines.append(f"![{movie.title}]({movie.poster_url})")
        lines.append("")
    if movie.description:
        lines.append(movie.description)
    return Item(
        viewer_type=ViewerType.CARD,
        item_id=f"{uuid.uuid4()}",
        item_text="\n".join(lines),
        item_image_path=movie.poster_url,
        item_media_path=movie.detail_url,
        payload={
            "title": movie.title,
            "poster_url": movie.poster_url,
            "detail_url": movie.detail_url,
            "description": movie.description,
        },
    )


def rank_movies(session_flow, movies: List[CinemaMovie]) -> List[CinemaMovie]:
    """Reorder ``movies`` by best match to the active persona's ``SKILL.md``
    — the same :class:`matching_learning.matching.ranker.PersonaRanker` Personal match uses to
    rank imdb results (``ui.personal_match_dialogue.dialogue_manager.
    DialgueManager.rank``).

    Best-effort: falls back to the original (scrape) order when there is no
    persona SKILL.md, no LLM, or ranking otherwise fails, so a live listing
    is never lost over a ranking hiccup.
    """
    if not movies:
        return movies
    from matching_learning.matching.ranker import PersonaRanker

    items = [to_item(movie) for movie in movies]
    movie_by_item_id = {item.item_id: movie for item, movie in zip(items, movies)}
    try:
        ranked_items = PersonaRanker(session_flow).rank(items)
    except Exception as exc:  # noqa: BLE001 - ranking is best-effort
        log.warning("cinemotion - persona ranking failed, keeping scrape order: %s", exc)
        return movies
    return [movie_by_item_id[item.item_id] for item in ranked_items]
