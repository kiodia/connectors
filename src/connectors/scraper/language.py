"""English-first language preference for scraped URLs.

Many sites publish the same page in several languages — the Swiss sites
MatchMake scrapes especially (fr / de / it / en). MatchMake always wants the
English rendition when the site has one, so a relational match or an angel
never works on a French page it cannot compare with the rest.

Sites offer their languages in one of two ways, and each gets its own
mechanism:

* **Server-side negotiation** — one URL, the language chosen from the
  request's ``Accept-Language``. Handled in the scrapers themselves
  (:data:`ACCEPT_LANGUAGE` / :data:`LOCALE`), so it costs no extra fetch and
  applies to every scrape MatchMake makes.
* **One URL per language** — ``/fr/page`` vs ``/en/page``,
  ``fr.example.com``, ``?lang=fr``. That is what this module resolves:
  :func:`english_variant` spots the language slot in a URL and returns the
  English counterpart *only when the page itself links to it*, which is what
  :meth:`connectors.scraper.scraper_base.Scraper.scrape_english_first` then
  fetches.

Requiring the candidate to appear among the page's own links is what keeps
this safe: no language table is needed and no URL is invented. A ``/ch/``
segment is only rewritten to ``/en/`` if the site genuinely offers that page.
"""
import re
from typing import List
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import logging
log = logging.getLogger(__name__)

#: Sent by the scrapers so a negotiating server answers in English.
ACCEPT_LANGUAGE = "en-US,en;q=0.9"

#: Browser locale requested alongside :data:`ACCEPT_LANGUAGE`.
LOCALE = "en-US"

#: A language slot in a URL: "en", "fr", "en-US", "de_CH".
_LANG_TOKEN = re.compile(r"^[a-z]{2}([-_][a-z]{2})?$", re.IGNORECASE)

#: Query parameters a site uses to carry the language.
_QUERY_KEYS = ("lang", "language", "locale", "hl")

#: The spellings of "English" a site may use in that slot.
_ENGLISH_TOKENS = ("en", "en-us", "en_us", "en-gb", "en_gb")


def _canon(url: str) -> str:
    """Canonical form used to compare two URLs: lowercased, no fragment and
    no trailing slash, so ``https://Example.com/EN/`` and
    ``http://example.com/en`` compare equal apart from the scheme."""
    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    return urlunsplit(
        (parts.scheme, parts.netloc, path, parts.query, "")
    ).lower()


def _is_english_token(token: str) -> bool:
    return bool(_LANG_TOKEN.match(token)) and token.lower().replace("_", "-").startswith("en")


def is_english(url: str) -> bool:
    """True when the URL already designates English in its language slot.

    Only a real language token counts, so a path like ``/enterprise/`` is not
    mistaken for English.
    """
    parts = urlsplit(url)

    segments = [s for s in parts.path.split("/") if s]
    if segments and _is_english_token(segments[0]):
        return True

    host = parts.netloc.split(":")[0]
    labels = host.split(".")
    if len(labels) > 1 and _is_english_token(labels[0]):
        return True

    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key.lower() in _QUERY_KEYS and _is_english_token(value):
            return True
    return False


def english_candidates(url: str) -> List[str]:
    """Every URL ``url`` would have if its language slot said English.

    Covers the three conventions: a leading path segment, a subdomain, and a
    language query parameter. Returns [] when the URL has no language slot to
    swap — nothing is invented.
    """
    parts = urlsplit(url)
    candidates = []

    # 1. leading path segment: /fr/page -> /en/page
    segments = parts.path.split("/")
    first = next((i for i, s in enumerate(segments) if s), None)
    if first is not None and _LANG_TOKEN.match(segments[first]):
        for token in _ENGLISH_TOKENS:
            swapped = list(segments)
            swapped[first] = token
            candidates.append(urlunsplit(
                (parts.scheme, parts.netloc, "/".join(swapped), parts.query, "")))

    # 2. subdomain: fr.example.com -> en.example.com
    labels = parts.netloc.split(".")
    if len(labels) > 2 and _LANG_TOKEN.match(labels[0]):
        for token in _ENGLISH_TOKENS:
            host = ".".join([token] + labels[1:])
            candidates.append(urlunsplit(
                (parts.scheme, host, parts.path, parts.query, "")))

    # 3. query parameter: ?lang=fr -> ?lang=en
    query = parse_qsl(parts.query, keep_blank_values=True)
    if any(key.lower() in _QUERY_KEYS for key, _ in query):
        for token in _ENGLISH_TOKENS:
            swapped = [(k, token if k.lower() in _QUERY_KEYS else v) for k, v in query]
            candidates.append(urlunsplit(
                (parts.scheme, parts.netloc, parts.path, urlencode(swapped), "")))

    return candidates


def english_variant(url: str, links) -> str:
    """The English rendition of ``url`` among ``links``, or "" when there is
    none to prefer.

    Returns "" when ``url`` is already English, when it has no language slot,
    or when the page does not actually link to its English counterpart — the
    caller then simply keeps the page it fetched.
    """
    if is_english(url):
        return ""
    wanted = {_canon(candidate) for candidate in english_candidates(url)}
    if not wanted:
        return ""
    for link in links or []:
        if not link:
            continue
        absolute = urljoin(url, str(link))
        if _canon(absolute) in wanted:
            return absolute
    return ""
