"""The movies of the Fribourg cinema programs other than cinemotion.ch.

Each parser reads one site's program page — as the scrapers render it to
markdown — into :class:`connectors.cinemotion.CinemaMovie`, the single movie
model, so a movie grounded from any cinema is the same kind of datapoint and
renders as the same Newsroom card (``CinemaMovie.to_markdown``):

* :func:`parse_arena` — Arena Cinemas (``arena.ch/<lang>/<city>/programme/
  horaires-des-films``): poster, title and synopsis are inline in the listing.
* :func:`parse_korso` — Kino Korso (``korso.ch``): the day's screenings as
  ``###`` titles, posters in the carousel above; the synopsis is on the
  movie's own page (:func:`korso_description`).
* :func:`parse_outnow` — OutNow's city program (``outnow.ch/<lang>/
  Programmes-cinema/<City>/``), every cinema of the city at once: poster,
  title, year and genres; the synopsis is on the movie's own page
  (:func:`outnow_description`).

The same movie playing in two cinemas is two datapoints with two detail
pages; :class:`connectors.grounding.DataGrounding` collapses them on the
fuzzy title. Every parser returns [] for a page it does not recognise, so
:func:`connectors.grounding.extract_entities` then grounds the page whole.
"""
import re
import unicodedata
from typing import Dict, List
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from connectors.cinemotion import CinemaMovie

import logging
log = logging.getLogger(__name__)

_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")

#: The genre words the Fribourg programs use (French, German, English),
#: unaccented and lower-case, by the English genre a movie is grounded with —
#: one vocabulary whatever the site, so a persona's "action" reads onto the
#: same ``genre:Action`` node for every cinema.
GENRES = {
    "action": "Action", "aventure": "Adventure", "abenteuer": "Adventure",
    "adventure": "Adventure", "animation": "Animation", "anime": "Animation",
    "zeichentrick": "Animation", "biographie": "Biography", "biopic": "Biography",
    "biography": "Biography", "comedie": "Comedy", "komodie": "Comedy",
    "comedy": "Comedy", "policier": "Crime", "krimi": "Crime", "crime": "Crime",
    "documentaire": "Documentary", "dokumentarfilm": "Documentary",
    "documentary": "Documentary", "drame": "Drama", "drama": "Drama",
    "famille": "Family", "familie": "Family", "family": "Family",
    "fantastique": "Fantasy", "fantasy": "Fantasy", "historique": "History",
    "histoire": "History", "historie": "History", "history": "History",
    "horreur": "Horror", "epouvante": "Horror", "horror": "Horror",
    "musique": "Music", "musical": "Music", "comedie musicale": "Music",
    "musik": "Music", "music": "Music", "mystere": "Mystery", "mystery": "Mystery",
    "romance": "Romance", "romantique": "Romance", "liebesfilm": "Romance",
    "science-fiction": "Science Fiction", "science fiction": "Science Fiction",
    "sci-fi": "Science Fiction", "thriller": "Thriller", "suspense": "Thriller",
    "guerre": "War", "kriegsfilm": "War", "war": "War", "western": "Western",
}


def _unaccented(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c)).lower().strip()


def genres_of(text: str) -> List[str]:
    """The English genres a listing's genre segment names, in its order —
    ``"Comédie, Famille"`` → ``["Comedy", "Family"]``. Pieces that are no
    genre (an age, a duration, a year) are skipped."""
    found: List[str] = []
    for piece in re.split(r"[,|/·•]", text or ""):
        genre = GENRES.get(_unaccented(piece.strip(" *_")))
        if genre and genre not in found:
            found.append(genre)
    return found


def _plain(markdown: str) -> str:
    """Markdown as text: images dropped, links reduced to their text,
    blank runs collapsed."""
    text = _LINK.sub(r"\1", _IMAGE.sub("", markdown or ""))
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


# ─────────────────────────────────────────────────────────────────────────
#  Arena Cinemas
# ─────────────────────────────────────────────────────────────────────────

ARENA_NAME = "Arena Cinemas"

_ARENA_ENTRY = re.compile(
    r"\[!\[(?P<original>[^\]]*)\]\((?P<poster>https://schedule\.arena\.ch/[^)]+)\)\]"
    r"\((?P<detail>https://www\.arena\.ch/[^)]*/detail\?fid=\d+[^)]*)\)\s*\n"
    r"## \[(?P<title>[^\]]+)\]\([^)]*\)\s*\n"
    r"(?P<body>.*?)(?=^\[D[ée]tails du film\]|^\[!\[|\Z)",
    re.MULTILINE | re.DOTALL,
)


def _arena_detail_url(url: str) -> str:
    """The detail page without its ``StartDate`` — the day the listing was
    read changes daily, the movie does not."""
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query) if k != "StartDate"]
    return urlunsplit(parts._replace(query=urlencode(query)))


def parse_arena(markdown: str, cinema: str = ARENA_NAME) -> List[CinemaMovie]:
    """The movies of an Arena "Horaires des films" page, in program order.

    The heading is the title the cinema shows (``Mochy``); the listing's own
    lines — original title, age, genres, duration, synopsis — are the
    description, so no detail page needs fetching."""
    movies, seen = [], set()
    for match in _ARENA_ENTRY.finditer(markdown or ""):
        detail = _arena_detail_url(match.group("detail"))
        if detail in seen:
            continue
        seen.add(detail)
        body = match.group("body")
        facts = next((line for line in body.splitlines() if "|" in line), "")
        movies.append(CinemaMovie(
            title=match.group("title").strip(),
            poster_url=match.group("poster"),
            detail_url=detail,
            description=_plain(body),
            cinema=cinema,
            genres=genres_of(facts),
        ))
    return movies


# ─────────────────────────────────────────────────────────────────────────
#  Kino Korso
# ─────────────────────────────────────────────────────────────────────────

KORSO_NAME = "Kino Korso"

_KORSO_SCREENING = re.compile(
    r"^###\s+\[\s*(?P<title>[^\]]+?)\s*\]\((?P<detail>https://www\.korso\.ch/[^)\s]+/)\)",
    re.MULTILINE)
_KORSO_POSTER = re.compile(
    r"\[!\[[^\]]*\]\((?P<poster>https://www\.korso\.ch/wp-content/[^)]+)\)\]"
    r"\((?P<detail>https://www\.korso\.ch/[^)\s]+/)\)")


def parse_korso(markdown: str, cinema: str = KORSO_NAME) -> List[CinemaMovie]:
    """The movies screened on Korso's program page, each once, posters from
    the carousel when it shows the movie."""
    posters: Dict[str, str] = {}
    for match in _KORSO_POSTER.finditer(markdown or ""):
        posters.setdefault(match.group("detail"), match.group("poster"))
    movies, seen = [], set()
    for match in _KORSO_SCREENING.finditer(markdown or ""):
        detail = match.group("detail")
        if detail in seen:
            continue
        seen.add(detail)
        movies.append(CinemaMovie(title=match.group("title").strip(),
                                  poster_url=posters.get(detail, ""),
                                  detail_url=detail, cinema=cinema))
    return movies


def _korso_block(markdown: str, title: str) -> str:
    """A Korso movie page from its ``# <title>`` heading to its
    ``PROJECTIONS`` section — the movie itself, without the site's header.
    The page's own first heading stands in when it spells the title
    differently from the program."""
    for heading in (re.escape(title), r"(?!\s*PROJECTIONS)[^\n]+"):
        match = re.search(r"^# " + heading + r"\s*$(.*?)(?:^#\s+PROJECTIONS|\Z)",
                          markdown or "", re.MULTILINE | re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def korso_description(markdown: str, title: str) -> str:
    """The credits and synopsis of a Korso movie page, "" when the page does
    not have that shape."""
    return _plain(_korso_block(markdown, title))


def korso_poster(markdown: str, title: str) -> str:
    """The poster a Korso movie page shows with the movie, "" when none."""
    match = re.search(r"!\[[^\]]*\]\((https://www\.korso\.ch/wp-content/[^)]+)\)",
                      _korso_block(markdown, title))
    return match.group(1) if match else ""


# ─────────────────────────────────────────────────────────────────────────
#  OutNow
# ─────────────────────────────────────────────────────────────────────────

OUTNOW_NAME = "OutNow"

_OUTNOW_ENTRY = re.compile(
    r"\[!\[(?P<title>[^\]]*)\]\((?P<poster>https://media\.outnow\.ch/[^)]+)\)\]"
    r"\((?P<detail>https://outnow\.ch/\w+/Movies/\d{4}/[^)/]+/)\)\s*\n"
    r"\[[^\]]*\]\([^)]*\)\s*\n"
    r"(?P<facts>\d{4}[^\n]*)")


def parse_outnow(markdown: str, cinema: str = OUTNOW_NAME) -> List[CinemaMovie]:
    """The movies of an OutNow city program — every cinema of the city —
    with their year and genres as a first description."""
    movies, seen = [], set()
    for match in _OUTNOW_ENTRY.finditer(markdown or ""):
        detail = match.group("detail")
        if detail in seen:
            continue
        seen.add(detail)
        facts = match.group("facts").strip()
        movies.append(CinemaMovie(title=match.group("title").strip(),
                                  poster_url=match.group("poster"),
                                  detail_url=detail,
                                  description=facts,
                                  cinema=cinema,
                                  genres=genres_of(facts.split("·", 1)[-1])))
    return movies


def outnow_description(markdown: str) -> str:
    """The synopsis, genres, direction and cast of an OutNow movie page:
    what follows its tab bar (``[Critique](…)``), up to the release dates.
    "" when the page does not have that shape."""
    match = re.search(r"^\[Critique\]\([^\n]*\n(.*?)(?:^Sortie cin[ée]ma|^## |\Z)",
                      markdown or "", re.MULTILINE | re.DOTALL)
    return _plain(match.group(1)) if match else ""
