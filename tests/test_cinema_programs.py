"""The Fribourg cinema program parsers (connectors.cinema_programs) and their
place in the entity extractors — on small snippets shaped like the markdown
the scrapers render for each site."""
from types import SimpleNamespace

from connectors.cinema_programs import (
    genres_of, korso_description, korso_poster, outnow_description, parse_arena,
    parse_korso, parse_outnow)
from connectors.cinemotion import CinemaMovie
from connectors.grounding import extract_entities, merge_duplicates

ARENA = """[ Ven 16 Oct](https://www.arena.ch/fr/fribourg/programme/horaires-des-films?StartDate=20261016)
[![Yugly](https://schedule.arena.ch/__images/artwork/1023934.ch_fr.jpg)](https://www.arena.ch/fr/fribourg/programme/horaires-des-films/detail?fid=1023934&StartDate=20261011)
## [Mochy](https://www.arena.ch/fr/fribourg/programme/overview/detail?fid=1023934&StartDate=20261011)
Titre original: _Yugly_
Age légal:  | Animation |  85 minutes
The ugliest dog in the world dreams of a beauty contest.
[Détails du film](https://www.arena.ch/fr/fribourg/programme/horaires-des-films/detail?fid=1023934&StartDate=20261011)
![](https://www.arena.ch/images/progi/fr.png)FR
[ Salle 2 13:45](https://www.arena.ch/fr/fribourg/programme/horaires-des-films/tickets?oid=1&fid=1023934#arnwebshop)
[![Fast Cars](https://schedule.arena.ch/__images/artwork/1000001.ov.jpg)](https://www.arena.ch/fr/fribourg/programme/horaires-des-films/detail?fid=1000001&StartDate=20261011)
## [Fast Cars](https://www.arena.ch/fr/fribourg/programme/overview/detail?fid=1000001&StartDate=20261011)
Age légal: **14(14)** | Action |  120 minutes
A chase across the Alps.
[Détails du film](https://www.arena.ch/fr/fribourg/programme/horaires-des-films/detail?fid=1000001&StartDate=20261011)
"""

KORSO = """[![](https://www.korso.ch/wp-content/uploads/2026/09/north-768x432.webp)](https://www.korso.ch/the-north/)
[Info & Billets](https://www.korso.ch/)
###  dimanche 11 octobre
[ 18:45  ](https://www.korso.ch/the-north/)
###  [ The North ](https://www.korso.ch/the-north/)
EN/de,fr
[ 20:15  ](https://www.korso.ch/la-frappe/)
###  [ La Frappe ](https://www.korso.ch/la-frappe/)
FR
###  lundi 12 octobre
[ 18:45  ](https://www.korso.ch/the-north/)
###  [ The North ](https://www.korso.ch/the-north/)
"""

KORSO_DETAIL = """[ ![Korso](https://www.korso.ch/wp-content/uploads/2024/02/logo.png) ](https://www.korso.ch/)
# The North
[](https://www.korso.ch/the-north/)
Un film de Bart Schrijver
Two old friends walk 600 kilometres across the Highlands.
![](https://www.korso.ch/wp-content/uploads/2026/09/north_poster.jpg)
#  PROJECTIONS
###  Aujourd'hui
"""

OUTNOW = """[![Digger](https://media.outnow.ch/Movies/Bilder/2026/Digger/artwork_small.jpg)](https://outnow.ch/fr/Movies/2026/Digger/)
[Digger](https://outnow.ch/fr/Movies/2026/Digger/)
2026 · Comédie, Drame
[](https://outnow.ch/fr/Account/Login/?gate=heart "Ajouter à la liste de suivi")
  1. Salle : 4
"""

OUTNOW_DETAIL = """# Digger
2026
[Critique](https://outnow.ch/fr/Movies/2026/Digger/Critique/)[Cinéma](https://outnow.ch/fr/Movies/2026/Digger/Programmes-cinema/)
An industrialist sets off a catastrophe and races to save humanity.
[Comédie](https://outnow.ch/fr/Cinema/Parcourir/?genre=comedy)

Interprètes
[Tom Cruise](https://outnow.ch/fr/Personne/Tom-Cruise/)

Sortie cinéma
     01.10.2026
"""


def _page(url, markdown):
    return SimpleNamespace(url=url, title="", markdown=markdown, success=True)


def test_arena_reads_the_shown_title_and_inline_synopsis():
    movies = parse_arena(ARENA)
    assert [m.title for m in movies] == ["Mochy", "Fast Cars"]
    mochy = movies[0]
    assert mochy.poster_url.endswith("1023934.ch_fr.jpg")
    # The listing day is not part of the movie's identity.
    assert mochy.detail_url == ("https://www.arena.ch/fr/fribourg/programme/"
                                "horaires-des-films/detail?fid=1023934")
    assert "Animation" in mochy.description and "beauty contest" in mochy.description
    assert "Détails du film" not in mochy.description
    assert mochy.cinema == "Arena Cinemas"
    assert mochy.genres == ["Animation"]
    assert movies[1].genres == ["Action"]


def test_korso_lists_each_movie_once_with_the_carousel_poster():
    movies = parse_korso(KORSO)
    assert [m.title for m in movies] == ["The North", "La Frappe"]
    assert movies[0].poster_url.endswith("north-768x432.webp")
    assert movies[1].poster_url == ""
    assert "Highlands" in korso_description(KORSO_DETAIL, "The North")
    assert "PROJECTIONS" not in korso_description(KORSO_DETAIL, "The North")
    # The movie's poster, not the site logo above the heading.
    assert korso_poster(KORSO_DETAIL, "The North").endswith("north_poster.jpg")
    # A heading spelled differently from the program still finds the movie.
    assert "Highlands" in korso_description(KORSO_DETAIL, "THE NORTH (VO)")


def test_outnow_reads_year_genres_then_the_detail_synopsis():
    [digger] = parse_outnow(OUTNOW)
    assert digger.title == "Digger"
    assert digger.detail_url == "https://outnow.ch/fr/Movies/2026/Digger/"
    assert digger.description == "2026 · Comédie, Drame"
    assert digger.genres == ["Comedy", "Drama"]
    synopsis = outnow_description(OUTNOW_DETAIL)
    assert "catastrophe" in synopsis and "Tom Cruise" in synopsis
    assert "Sortie" not in synopsis and "](" not in synopsis


def test_extractors_ground_movies_and_fill_details_with_fetch():
    details = {"https://www.korso.ch/the-north/": KORSO_DETAIL,
               "https://outnow.ch/fr/Movies/2026/Digger/": OUTNOW_DETAIL}
    fetch = lambda url: _page(url, details.get(url, ""))  # noqa: E731

    arena = extract_entities(_page("https://www.arena.ch/fr/fribourg/programme/"
                                   "horaires-des-films", ARENA), fetch=fetch)
    assert [type(m).__name__ for m in arena] == ["CinemaMovie", "CinemaMovie"]

    north = extract_entities(_page("https://www.korso.ch/", KORSO), fetch=fetch)[0]
    assert "Highlands" in north.description
    assert "*Kino Korso*" in north.to_markdown()

    digger = extract_entities(_page("https://outnow.ch/fr/Programmes-cinema/Fribourg/",
                                    OUTNOW), fetch=fetch)[0]
    assert "catastrophe" in digger.description


def test_an_unrecognised_page_of_those_sites_grounds_whole():
    [page] = extract_entities(_page("https://www.korso.ch/contact/", "# Contact\nRue 1"))
    assert type(page).__name__ == "WebDocument"


def test_genres_are_read_in_english_whatever_the_language():
    assert genres_of("Age légal: **8(8)** | Comédie, Famille |  89 minutes") == ["Comedy", "Family"]
    assert genres_of("Science-fiction / Horreur") == ["Science Fiction", "Horror"]
    assert genres_of("Komödie, Abenteuer") == ["Comedy", "Adventure"]
    assert genres_of("2026") == [] and genres_of("") == []


def test_a_movie_listed_by_several_programs_merges_into_one():
    """Cinemotion knows Digger's synopsis, OutNow its genres: grounded apart,
    the first would win and the genres be lost."""
    rex = CinemaMovie(title="DIGGER", detail_url="https://www.cinemotion.ch/cette-semaine/digger-9925",
                      description="A long synopsis about a catastrophe.")
    outnow = CinemaMovie(title="Digger", detail_url="https://outnow.ch/fr/Movies/2026/Digger/",
                         poster_url="https://media.outnow.ch/digger.jpg",
                         description="2026 · Comédie", genres=["Comedy", "Drama"], cinema="OutNow")
    arena = CinemaMovie(title="Digger", genres=["Comedy"], cinema="Arena Cinemas")
    other = CinemaMovie(title="Ducobu et le fantôme de Saint-Potache")
    other_again = CinemaMovie(title="Ducobu et le fantôme de Saint Potache", genres=["Comedy"])
    merged = merge_duplicates([rex, outnow, other, arena, other_again])
    assert merged == [rex, other]
    assert rex.detail_url.startswith("https://www.cinemotion.ch")     # the identity stays
    assert rex.genres == ["Comedy", "Drama"]
    assert rex.description == "A long synopsis about a catastrophe."
    assert rex.poster_url == "https://media.outnow.ch/digger.jpg" and rex.cinema == "OutNow"
    assert other.genres == ["Comedy"]
    assert "*OutNow · Comedy, Drama*" in rex.to_markdown()
