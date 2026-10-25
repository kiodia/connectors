"""Image search APIs as a Dataspace source: discover images for a persona.

The cinema programs of Fribourg are a JSON file listing URLs to scrape; an
**image search Dataspace** is a JSON file listing image search APIs to query
(``image-search-apis.json``, ``"kind": "image_search"``). Each API is a URL
template — ``{query}`` the search text, ``{limit}`` the number of hits,
``${ENV_VAR}`` a key read from the environment — with the path to its list of
hits and the mapping of a hit onto an :class:`ImageHit`::

    {"name": "Openverse", "enabled": true, "auth": {"type": "none"},
     "endpoint": "https://api.openverse.org/v1/images/?q={query}&page_size={limit}",
     "results": "results",
     "fields": {"image_url": "url", "page_url": "foreign_landing_url",
                "title": "title", "author": "creator", "license": "license"}}

A field path is dotted (``imageinfo.0.url``; ``-1`` is the last element,
``*`` every element of a list); ``const:<text>`` is a fixed value;
``template:<text>`` builds one from ``{path}`` placeholders, read off the hit
first and off the answer's root then (``{config.iiif_url}``), with
``https://`` put in front of a result that has no scheme. ``detail`` names an
endpoint (``{id}``) a hit is read through when the search answers ids only
(The Met).

Both hosts use this one module, as they do :mod:`connectors.grounding`:
MatchMake (``api.image_search``) discovers images while a persona matches the
Dataspace in its Canvas AX, keopy (``angel_grounding.py``) grounds them for
the image Angel that notifies the Newsroom — so an image grounded on either
side is the same datapoint.

**What a datapoint's vector is.** :class:`ImageGrounding` embeds the picture
itself with its caption — ``models/gemini-embedding-2`` maps text and images
into one space — so the persona's words, its likes and the session query are
compared with what the image shows, not only with its title. An image that
cannot be fetched falls back to its text.
"""
from __future__ import annotations

import html
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from pydantic import BaseModel, Field

from connectors.grounding import DataGrounding

import logging
log = logging.getLogger(__name__)

#: The ``kind`` an image search Dataspace file declares.
KIND = "image_search"

#: Hits asked of each API per query when the caller says nothing.
DEFAULT_LIMIT = 8

#: Seconds one API request may take when its entry says nothing.
DEFAULT_TIMEOUT = 20.0

#: The User-Agent of every request an entry does not give its own.
USER_AGENT = "MatchMake/1.0"

#: The image types the multimodal embedding takes.
EMBEDDABLE_MIME = ("image/jpeg", "image/png", "image/webp", "image/heic", "image/heif")

#: Bytes an image may weigh to be embedded (a thumbnail is asked for first).
MAX_IMAGE_BYTES = 8 * 1024 * 1024

_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_.*\-]+)\}")
_TAG_RE = re.compile(r"<[^>]+>")
_YEAR_RE = re.compile(r"(?<!\d)(\d{3,4})(?!\d)")


# ═══════════════════════════════════════════════════════════════════════════
#  The entity
# ═══════════════════════════════════════════════════════════════════════════

class ImageHit(BaseModel):
    """One image an image search API found — the datapoint it grounds as.

    The fields are named so the rest reads them without configuration:
    ``title`` names it, ``page_url`` identifies it, ``image_url`` is rendered
    as its picture, ``author`` is its people family, ``genre`` its list
    family, and ``provider``, ``source_type``, ``license`` and ``year`` are
    the small categoricals and the band a relation agent's scorecard gets.
    """
    title: str = Field(default="", description="What the image is called")
    description: str = Field(default="", description="Its caption, alt text and subjects")
    image_url: str = Field(default="", description="The full image")
    thumbnail_url: str = Field(default="", description="A small version of it")
    page_url: str = Field(default="", description="Its page at the provider: its identity")
    author: str = Field(default="", description="The artist or photographer")
    license: str = Field(default="", description="Its license or rights statement")
    provider: str = Field(default="", description="The API that found it")
    source_type: str = Field(default="", description="open_license, museum, archive, stock, web")
    genre: List[str] = Field(default_factory=list, description="Classification: portrait, landscape, photograph")
    medium: str = Field(default="", description="Oil on canvas, gelatin silver print, digital")
    style: str = Field(default="", description="Movement or style")
    year: Optional[int] = Field(default=None, description="When it was made")
    tags: List[str] = Field(default_factory=list, description="Subjects the provider tagged")
    query: str = Field(default="", description="The search that found it")
    media_file: str = Field(default="", description="The display copy MatchMake serves "
                                                    "(``/<assets link>/<file>``), '' elsewhere")

    def caption(self) -> str:
        """The words embedded with the picture."""
        parts = [self.title, self.description, ", ".join(self.genre), self.medium,
                 self.style, ", ".join(self.tags[:12]), self.author]
        return ". ".join(p for p in (s.strip() for s in parts) if p)[:2000]

    def to_markdown(self) -> str:
        """The card the Canvas AX opens and the Newsroom shows."""
        lines = [f"## {self.title or 'Untitled image'}"]
        picture = self.media_file or self.thumbnail_url or self.image_url
        if picture:
            lines.append(f"![{_alt(self.title)}]({picture})")
        if self.description:
            lines.append(self.description)
        facts = []
        if self.author:
            facts.append(f"**Author:** {self.author}")
        if self.year:
            facts.append(f"**Year:** {self.year}")
        if self.genre:
            facts.append(f"**Genre:** {', '.join(self.genre)}")
        if self.medium:
            facts.append(f"**Medium:** {self.medium}")
        if self.style:
            facts.append(f"**Style:** {self.style}")
        if self.license:
            facts.append(f"**License:** {self.license}")
        if facts:
            lines.append("  \n".join(facts))
        if self.page_url:
            lines.append(f"[{self.provider or 'Source'}]({self.page_url})")
        return "\n\n".join(lines)


def _alt(text: str) -> str:
    return re.sub(r"[\[\]]", "", text or "image")[:80]


def hit_id(hit: ImageHit) -> str:
    """The natural key an image is grounded under: its page, else its picture."""
    return (hit.page_url or hit.image_url or hit.title).strip()


# ═══════════════════════════════════════════════════════════════════════════
#  The Dataspace file
# ═══════════════════════════════════════════════════════════════════════════

def load_spec(source: Any) -> Optional[Dict[str, Any]]:
    """The image search spec ``source`` holds — a path to the JSON file, its
    text, or the parsed dict — or ``None`` when it is not one (no ``"kind":
    "image_search"``, or unreadable)."""
    data = source
    if isinstance(source, (str, Path)):
        text = str(source).strip()
        if text.startswith("{"):
            try:
                data = json.loads(text)
            except ValueError:
                return None
        else:
            path = Path(text)
            try:
                if path.suffix.lower() != ".json" or not path.is_file():
                    return None
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                log.warning("image_search - could not read %s: %s", path, exc)
                return None
    if not isinstance(data, dict) or str(data.get("kind") or "") != KIND:
        return None
    return data


def is_image_search(source: Any) -> bool:
    """Whether ``source`` (a path, JSON text or dict) is an image search Dataspace."""
    return load_spec(source) is not None


def missing_keys(api: Mapping[str, Any], env: Mapping[str, str]) -> List[str]:
    """The environment variables an API entry needs and ``env`` lacks."""
    text = json.dumps({k: api.get(k) for k in ("endpoint", "auth", "headers", "detail")})
    return sorted({name for name in _ENV_RE.findall(text) if not env.get(name)})


def available_apis(spec: Mapping[str, Any], env: Optional[Mapping[str, str]] = None
                   ) -> List[Dict[str, Any]]:
    """The entries that can be queried now: enabled, and every key they need set."""
    env = os.environ if env is None else env
    out = []
    for api in spec.get("apis") or []:
        if not isinstance(api, dict) or not api.get("enabled", False) or not api.get("endpoint"):
            continue
        missing = missing_keys(api, env)
        if missing:
            log.info("image_search - '%s' skipped: %s not set", api.get("name"), ", ".join(missing))
            continue
        out.append(api)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Reading an answer
# ═══════════════════════════════════════════════════════════════════════════

def get_path(node: Any, path: str) -> Any:
    """The value at dotted ``path`` in ``node``; ``None`` when absent.
    ``-1`` indexes from the end, ``*`` maps the rest of the path over a list."""
    parts = [p for p in str(path or "").split(".") if p != ""]
    for i, part in enumerate(parts):
        if node is None:
            return None
        if part == "*":
            if not isinstance(node, list):
                return None
            rest = ".".join(parts[i + 1:])
            values = [get_path(v, rest) if rest else v for v in node]
            return [v for v in values if v not in (None, "", [])]
        if isinstance(node, list):
            try:
                index = int(part)
            except ValueError:
                return None
            if not -len(node) <= index < len(node):
                return None
            node = node[index]
        elif isinstance(node, dict):
            node = node.get(part)
        else:
            return None
    return node


def _expand(template: str, hit: Any, root: Any) -> str:
    """``{path}`` placeholders off the hit, else off the answer's root."""
    missing = False

    def value(match):
        nonlocal missing
        path = match.group(1)
        found = get_path(hit, path)
        if found in (None, ""):
            found = get_path(root, path)
        if found in (None, "") or isinstance(found, (dict, list)):
            missing = True
            return ""
        return str(found)

    text = _PLACEHOLDER_RE.sub(value, template)
    if missing:
        return ""
    if text and not re.match(r"^[a-z][a-z0-9+.-]*:", text, re.IGNORECASE):
        text = "https://" + text.lstrip("/")
    return text


def field_value(rule: str, hit: Any, root: Any = None) -> Any:
    """One ``fields`` rule applied to a hit: a path, ``const:`` or ``template:``."""
    rule = str(rule or "")
    if rule.startswith("const:"):
        return rule[len("const:"):]
    if rule.startswith("template:"):
        return _expand(rule[len("template:"):], hit, root)
    return get_path(hit, rule)


def _text(value: Any) -> str:
    """A field's value as one clean line: tags stripped, entities decoded."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "public domain" if value else ""
    if isinstance(value, list):
        return ", ".join(t for t in (_text(v) for v in value) if t)
    if isinstance(value, dict):
        return _text(value.get("name") or value.get("title") or value.get("value"))
    text = html.unescape(_TAG_RE.sub(" ", str(value)))
    return re.sub(r"\s+", " ", text).strip()


def _texts(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(_texts(item))
        return out
    if isinstance(value, str) and re.search(r"[,|]", value) and len(value) < 2000:
        # "a|b" (Wikimedia's categories, whose names hold commas) and "a, b"
        # (Pixabay's tags) are lists.
        separator = "|" if "|" in value else ","
        return [t for t in (_text(v) for v in value.split(separator)) if t]
    text = _text(value)
    return [text] if text else []


def _title(value: Any) -> str:
    """A hit's title without a file's trappings ("File:Sunflowers.jpg")."""
    text = re.sub(r"^File:", "", _text(value))
    text = re.sub(r"\.(?:jpe?g|png|tiff?|gif|webp|svg)$", "", text, flags=re.IGNORECASE)
    return text.replace("_", " ").strip()[:300]


def _year(value: Any) -> Optional[int]:
    if isinstance(value, int) and 0 < value < 3000:
        return value
    match = _YEAR_RE.search(_text(value))
    return int(match.group(1)) if match else None


def hit_from(api: Mapping[str, Any], hit: Any, root: Any = None, query: str = "") -> Optional[ImageHit]:
    """One hit of an answer as an :class:`ImageHit`, ``None`` when it has no picture."""
    fields = api.get("fields") or {}

    def get(name: str) -> Any:
        rule = fields.get(name)
        return field_value(rule, hit, root) if rule else None

    image_url = _text(get("image_url"))
    thumbnail_url = _text(get("thumbnail_url"))
    if not image_url and not thumbnail_url:
        return None
    mime = _text(get("mime")).lower()
    if mime and not mime.startswith("image/"):
        return None                         # a PDF, a video, a sound file
    unique: List[str] = []
    for tag in _texts(get("tags")):
        if tag.lower() not in {t.lower() for t in unique}:
            unique.append(tag)
    genre: List[str] = []
    for value in _texts(get("genre")):
        if value.lower() not in {g.lower() for g in genre}:
            genre.append(value)
    return ImageHit(
        title=_title(get("title")),
        description=_text(get("description"))[:1500],
        image_url=image_url or thumbnail_url,
        thumbnail_url=thumbnail_url or image_url,
        page_url=_text(get("page_url")),
        author=_text(get("author"))[:200],
        license=_text(get("license"))[:120],
        provider=str(api.get("name") or ""),
        source_type=str(api.get("category") or ""),
        genre=genre[:6],
        medium=_text(get("medium"))[:200],
        style=_text(get("style"))[:120],
        year=_year(get("year")),
        tags=unique[:20],
        query=query,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Querying
# ═══════════════════════════════════════════════════════════════════════════

def _fill(text: str, env: Mapping[str, str], **values) -> str:
    text = _ENV_RE.sub(lambda m: urllib.parse.quote(env.get(m.group(1), ""), safe=""), text)
    for name, value in values.items():
        text = text.replace("{" + name + "}", value)
    return text


def _headers(api: Mapping[str, Any], env: Mapping[str, str]) -> Dict[str, str]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    for name, value in (api.get("headers") or {}).items():
        headers[str(name)] = _ENV_RE.sub(lambda m: env.get(m.group(1), ""), str(value))
    auth = api.get("auth") or {}
    if auth.get("type") == "header" and auth.get("header"):
        value = auth.get("value") or "${" + str(auth.get("env") or "") + "}"
        headers[str(auth["header"])] = _ENV_RE.sub(lambda m: env.get(m.group(1), ""), str(value))
    return headers


def http_get_json(url: str, headers: Mapping[str, str], timeout: float) -> Any:
    """GET ``url`` and parse its JSON (the default transport)."""
    request = urllib.request.Request(url, headers=dict(headers))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def search_api(api: Mapping[str, Any], query: str, limit: int = DEFAULT_LIMIT,
               env: Optional[Mapping[str, str]] = None,
               fetch_json: Callable[[str, Mapping[str, str], float], Any] = http_get_json
               ) -> List[ImageHit]:
    """The images one API finds for ``query``; ``[]`` on any failure (logged)."""
    env = os.environ if env is None else env
    name = api.get("name") or "?"
    timeout = float(api.get("timeout_s") or DEFAULT_TIMEOUT)
    headers = _headers(api, env)
    url = _fill(str(api["endpoint"]), env, query=urllib.parse.quote(query), limit=str(int(limit)))
    try:
        root = fetch_json(url, headers, timeout)
    except Exception as exc:  # noqa: BLE001 - one API down leaves the others
        log.warning("image_search - '%s' failed for %r: %s", name, query, exc)
        return []
    raw = get_path(root, str(api.get("results") or ""))
    if api.get("results_is_map") and isinstance(raw, dict):
        raw = list(raw.values())
    if not isinstance(raw, list):
        log.info("image_search - '%s' answered no list at '%s' for %r", name, api.get("results"), query)
        return []
    detail = (api.get("detail") or {}).get("endpoint")
    hits: List[ImageHit] = []
    for entry in raw:
        if len(hits) >= limit:
            break
        hit, hit_root = entry, root
        if detail:
            try:
                hit = hit_root = fetch_json(_fill(str(detail), env, id=urllib.parse.quote(str(entry))),
                                            headers, timeout)
            except Exception as exc:  # noqa: BLE001 - one object fewer
                log.info("image_search - '%s' detail of %s failed: %s", name, entry, exc)
                continue
        image = hit_from(api, hit, hit_root, query=query)
        if image is not None:
            hits.append(image)
    log.info("image_search - '%s': %d image(s) for %r", name, len(hits), query)
    return hits


def search(spec: Mapping[str, Any], queries: Sequence[str], limit: int = DEFAULT_LIMIT,
           env: Optional[Mapping[str, str]] = None, deadline_s: float = 60.0,
           fetch_json: Callable[[str, Mapping[str, str], float], Any] = http_get_json,
           max_workers: int = 8) -> List[ImageHit]:
    """Every available API asked every query at once; the hits interleaved
    API by API (so no single provider fills the page) and deduplicated by
    picture and page. Answers arriving after ``deadline_s`` are left out."""
    env = os.environ if env is None else env
    queries = [q for q in (str(q or "").strip() for q in queries) if q]
    apis = available_apis(spec, env)
    if not queries or not apis:
        log.info("image_search - nothing to search (%d API(s), %d query(ies))",
                 len(apis), len(queries))
        return []
    jobs = [(api, query) for query in queries for api in apis]
    found: Dict[int, List[ImageHit]] = {}
    pool = ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(jobs))),
                              thread_name_prefix="image-search")
    try:
        futures = {pool.submit(search_api, api, query, limit, env, fetch_json): i
                   for i, (api, query) in enumerate(jobs)}
        try:
            for future in as_completed(futures, timeout=deadline_s):
                try:
                    found[futures[future]] = future.result()
                except Exception as exc:  # noqa: BLE001
                    log.warning("image_search - a search failed: %s", exc)
        except Exception:  # noqa: BLE001 - the deadline: keep what arrived
            log.warning("image_search - %d of %d search(es) still running after %.0f s; "
                        "left out", len(jobs) - len(found), len(jobs), deadline_s)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    lists = [found.get(i, []) for i in range(len(jobs))]
    merged: List[ImageHit] = []
    seen = set()
    for rank in range(max((len(l) for l in lists), default=0)):
        for hits in lists:
            if rank >= len(hits):
                continue
            hit = hits[rank]
            keys = {k for k in (hit.image_url, hit.page_url) if k}
            if keys & seen:
                continue
            seen |= keys
            merged.append(hit)
    log.info("image_search - %d image(s) from %d API(s) for %d query(ies)",
             len(merged), len(apis), len(queries))
    return merged


# ═══════════════════════════════════════════════════════════════════════════
#  What to search for
# ═══════════════════════════════════════════════════════════════════════════

#: Words a search engine does better without.
_STOP = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are",
         "i", "me", "my", "we", "our", "you", "your", "who", "that", "this", "it", "be",
         "like", "likes", "love", "loves", "want", "wants", "looking", "find", "about",
         "some", "very", "really", "also", "from", "at", "as", "by", "their", "they"}


def short_query(text: str, words: int = 6) -> str:
    """``text`` cut to the few content words an image search takes well."""
    tokens = re.findall(r"[^\W_][\w'’-]*", str(text or ""), re.UNICODE)
    kept = [t for t in tokens if t.lower() not in _STOP]
    return " ".join(kept[:words])


def _strings(value: Any) -> List[str]:
    if isinstance(value, str):
        return [v.strip() for v in re.split(r"[;,\n]", value) if v.strip()]
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out.extend(_strings(v) if not isinstance(v, dict) else
                       _strings(v.get("title") or v.get("name") or v.get("text") or ""))
        return out
    return []


def queries_for(profile: Optional[Mapping[str, Any]] = None, prompt: str = "",
                liked: Iterable[str] = (), n: int = 3) -> List[str]:
    """The searches that find images for a persona: what was typed first;
    then what its relation agent learned it likes (the tags, genres and
    titles of the images it liked); then its own words — keywords, likes,
    intention, prompt. At most ``n``, short, no repeat."""
    profile = profile or {}
    candidates: List[str] = []
    if str(prompt or "").strip():
        candidates.append(short_query(prompt, 8))
    liked = [t for t in liked if str(t or "").strip()]
    if liked:
        candidates.append(short_query(" ".join(liked), 5))
    keywords = _strings(profile.get("keywords"))
    if keywords:
        candidates.append(short_query(" ".join(keywords[:4]), 5))
    for like in _strings(profile.get("likes"))[:2]:
        candidates.append(short_query(like, 5))
    for key in ("intention", "prompt", "profile"):
        if profile.get(key):
            candidates.append(short_query(profile.get(key), 6))
    out: List[str] = []
    for query in candidates:
        if query and query.lower() not in {q.lower() for q in out}:
            out.append(query)
    return out[:max(1, n)]


# ═══════════════════════════════════════════════════════════════════════════
#  Grounding: the picture is what is embedded
# ═══════════════════════════════════════════════════════════════════════════

def fetch_image(url: str, timeout: float = 20.0,
                headers: Optional[Mapping[str, str]] = None) -> Optional[tuple]:
    """``(bytes, mime)`` of an image the embedding takes, else ``None``.
    ``headers`` are the image host's (an entry's ``image_headers``)."""
    try:
        request = urllib.request.Request(url, headers=dict(headers or {"User-Agent": USER_AGENT}))
        with urllib.request.urlopen(request, timeout=timeout) as response:
            mime = (response.headers.get_content_type() or "").lower()
            data = response.read(MAX_IMAGE_BYTES + 1)
    except Exception as exc:  # noqa: BLE001 - the text stands in
        log.info("image_search - could not fetch %s: %s", url, exc)
        return None
    if len(data) > MAX_IMAGE_BYTES:
        log.info("image_search - %s is over %d bytes", url, MAX_IMAGE_BYTES)
        return None
    if mime == "image/jpg":
        mime = "image/jpeg"
    if mime not in EMBEDDABLE_MIME:
        sniffed = _sniff(data)
        if sniffed is None:
            log.info("image_search - %s is %s, not an embeddable image", url, mime or "untyped")
            return None
        mime = sniffed
    return data, mime


def image_fetcher(spec: Optional[Mapping[str, Any]] = None,
                  env: Optional[Mapping[str, str]] = None,
                  timeout: float = 20.0) -> Callable[[str], Optional[tuple]]:
    """``url -> (bytes, mime)`` sending each image host the headers its entry
    names (``image_hosts`` + ``image_headers``: the Art Institute of Chicago
    serves its pictures only with its ``AIC-User-Agent``)."""
    env = os.environ if env is None else env
    by_host: List[tuple] = []
    for api in (spec or {}).get("apis") or []:
        if isinstance(api, dict) and api.get("image_headers"):
            headers = {"User-Agent": USER_AGENT}
            for name, value in api["image_headers"].items():
                headers[str(name)] = _ENV_RE.sub(lambda m: env.get(m.group(1), ""), str(value))
            for host in api.get("image_hosts") or []:
                by_host.append((str(host).lower(), headers))

    def fetch(url: str) -> Optional[tuple]:
        host = urllib.parse.urlparse(url).netloc.lower()
        headers = next((h for name, h in by_host if host == name or host.endswith("." + name)),
                       None)
        return fetch_image(url, timeout=timeout, headers=headers)

    return fetch


#: The longest side of a display copy, in pixels.
DISPLAY_SIDE = 1024


def display_copy(data: bytes, mime: str, side: int = DISPLAY_SIDE) -> Optional[tuple]:
    """``(bytes, extension)`` of a picture fit for a Canvas tile and a card:
    at most ``side`` pixels, JPEG (PNG when it has transparency). ``None``
    when the bytes are no picture Pillow can read."""
    import io
    try:
        from PIL import Image
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:  # noqa: BLE001 - not a picture
        log.info("image_search - not a readable picture (%s): %s", mime, exc)
        return None
    if max(image.size) > side:
        image.thumbnail((side, side))
    out = io.BytesIO()
    if image.mode in ("RGBA", "LA", "P") and "transparency" in image.info or image.mode == "RGBA":
        image.save(out, format="PNG", optimize=True)
        return out.getvalue(), ".png"
    image.convert("RGB").save(out, format="JPEG", quality=85, optimize=True)
    return out.getvalue(), ".jpg"


def _sniff(data: bytes) -> Optional[str]:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


class GeminiImageEmbedder:
    """``(image bytes, mime, caption) -> vector`` with a Gemini embedding model:
    the picture and its caption as one content, so one vector, in the space
    the model maps text into too."""

    _client = None
    _lock = threading.Lock()

    def __init__(self, reference_name: str = "models/gemini-embedding-2", dimension: int = 3072):
        self.model = str(reference_name).split("/")[-1]
        self.dimension = int(dimension)

    @classmethod
    def client(cls):
        with cls._lock:
            if cls._client is None:
                key = os.getenv("GEMINI_API_KEY")
                if not key:
                    raise RuntimeError("GEMINI_API_KEY is not set")
                from google import genai
                cls._client = genai.Client(api_key=key)
            return cls._client

    def __call__(self, data: bytes, mime: str, caption: str = "") -> List[float]:
        from google.genai import types
        parts = [types.Part.from_bytes(data=data, mime_type=mime)]
        if caption:
            parts.append(types.Part(text=caption))
        call = lambda: self.client().models.embed_content(  # noqa: E731
            model=self.model, contents=[types.Content(parts=parts)],
            config=types.EmbedContentConfig(output_dimensionality=self.dimension))
        try:
            from vectorize.embeddings.backoff import call_with_backoff
            response = call_with_backoff(call, what=f"embed_content('{self.model}', image)")
        except ImportError:
            response = call()
        values = [float(v) for v in response.embeddings[0].values]
        norm = sum(v * v for v in values) ** 0.5
        return [v / norm for v in values] if norm > 0 else values


class ImageGrounding(DataGrounding):
    """A :class:`~connectors.grounding.DataGrounding` of images whose vector
    is the picture with its caption (:class:`GeminiImageEmbedder`), its text
    when the picture cannot be fetched or the embedding is not a Gemini one.

    ``image_embedder`` and ``fetch`` replace the embedding call and the
    download (tests); ``workers`` is how many images are embedded at once.
    """

    def __init__(self, *args, image_embedder: Optional[Callable] = None,
                 fetch: Optional[Callable[[str], Optional[tuple]]] = None,
                 workers: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        if image_embedder is None and _is_gemini(self.embedding):
            image_embedder = GeminiImageEmbedder(self.embedding.reference_name,
                                                 self.vector_dimension)
        self.image_embedder = image_embedder
        self.fetch = fetch or (lambda url: fetch_image(url))
        self.workers = max(1, int(workers))

    def _picture_vector(self, payload: BaseModel) -> Optional[List[float]]:
        if self.image_embedder is None:
            return None
        body = payload.model_dump() if hasattr(payload, "model_dump") else dict(payload)
        caption = payload.caption() if hasattr(payload, "caption") else str(body.get("title") or "")
        for url in dict.fromkeys(u for u in (body.get("thumbnail_url"), body.get("image_url")) if u):
            fetched = self.fetch(url)
            if fetched is None:
                continue
            try:
                return list(self.image_embedder(fetched[0], fetched[1], caption))
            except Exception as exc:  # noqa: BLE001 - the text stands in
                log.warning("image_search - embedding the picture %s failed: %s", url, exc)
                return None
        return None

    def encode(self, payload: BaseModel) -> Optional[List[float]]:
        return self._picture_vector(payload) or super().encode(payload)

    def encode_many(self, payloads: Sequence[BaseModel]) -> List[Optional[List[float]]]:
        payloads = list(payloads or [])
        vectors: List[Optional[List[float]]] = [None] * len(payloads)
        if self.image_embedder is not None and payloads:
            with ThreadPoolExecutor(max_workers=min(self.workers, len(payloads)),
                                    thread_name_prefix="image-embed") as pool:
                for i, vector in enumerate(pool.map(self._picture_vector, payloads)):
                    vectors[i] = vector
        missing = [i for i, v in enumerate(vectors) if v is None]
        if missing:
            for i, vector in zip(missing, super().encode_many([payloads[i] for i in missing])):
                vectors[i] = vector
        pictures = len(payloads) - len(missing)
        log.info("image_search - %d of %d image(s) embedded from the picture", pictures, len(payloads))
        return vectors


def _is_gemini(embedding: Any) -> bool:
    name = str(getattr(embedding, "reference_name", "") or "")
    return name.startswith("models/gemini-embedding")


#: Above 1, no title is "the same" as another: two paintings both called
#: "Sunflowers" are two images. Only the page, the picture and the vector
#: tell a duplicate.
NO_FUZZY_TITLES = 1.01


def ground_images(grounding: Any, hits: Sequence[ImageHit]) -> List[ImageHit]:
    """Ground ``hits`` in ``grounding``; the images new to the collection."""
    if grounding is None or not hits:
        return []
    started = time.monotonic()
    new = grounding.ground_entities(list(hits), point_id_of=hit_id,
                                    string_threshold=NO_FUZZY_TITLES)
    log.info("image_search - %d new image(s) grounded in '%s' in %.1f s", len(new),
             getattr(grounding, "collection_name", "?"), time.monotonic() - started)
    return list(new)


# ═══════════════════════════════════════════════════════════════════════════
#  How MatchMake shows an image Dataspace
# ═══════════════════════════════════════════════════════════════════════════

#: The Watch configuration an image search Dataspace gets: the picture — its
#: display copy, which MatchMake serves itself — fills the Canvas AX tile,
#: the title and the author are its caption, the rest is the card the click
#: opens.
DESKTOP_MARKDOWN: Dict[str, Dict[str, str]] = {
    "media_file": {"markdown": "![Image]({{ value }})"},
    "image_url": {"markdown": "not used"},
    "title": {"markdown": "**{{ value }}**"},
    "author": {"markdown": "{{ value }}"},
    "year": {"markdown": "{{ value }}"},
    "description": {"markdown": "not used"},
    "genre": {"markdown": "not used"},
    "medium": {"markdown": "not used"},
    "style": {"markdown": "not used"},
    "license": {"markdown": "not used"},
    "provider": {"markdown": "not used"},
    "source_type": {"markdown": "not used"},
    "tags": {"markdown": "not used"},
    "thumbnail_url": {"markdown": "not used"},
    "page_url": {"markdown": "not used"},
    "query": {"markdown": "not used"},
}
FIELD_ORDER: List[str] = ["media_file", "image_url", "title", "author", "year", "description", "genre",
                          "medium", "style", "license", "provider", "source_type", "tags",
                          "thumbnail_url", "page_url", "query"]
