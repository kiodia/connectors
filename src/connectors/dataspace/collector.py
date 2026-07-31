"""Collect the URL list of a Dataspace, from every source at once.

A natural-language request — "all the cinemas in Fribourg", "the robotics labs
of EPFL", "the italian restaurants of Bulle", "this season's concerts" — is
turned into a :class:`~connectors.dataspace.blueprint.DataspaceBlueprint`: named
entities, each carrying the URLs to watch. Nothing here is specific to one
domain or to one host; the entity is whatever the request names, and the only
requirement is that each one ends up with a URL of its own.

What the file is for is grounding: :class:`connectors.grounding.DataGrounding`
stores **one datapoint per URL**, so the list this module produces is the input
of the scraping both MatchMake and keopy run. That is also why the checks here
are what they are — a dead URL grounds nothing, and two entities sharing a URL
ground as one.

**Three sources, merged** — :meth:`DataspaceCollector.collect` is the entry
point and runs them together, because each is strong where the others are weak:

* :meth:`~DataspaceCollector.research` — **live web search** (Gemini's
  ``google_search`` tool, ``GEMINI_API_KEY``). The model quotes URLs out of real
  results instead of reconstructing them, which is what stops the
  invented-address failure below. Best precision.
* :meth:`~DataspaceCollector.generate` — **from model knowledge**, through the
  host's own LLM. Instant, free, and it proposes entities a search page ranks
  too low to show. Best coverage of the well-known, worst precision.
* :meth:`~DataspaceCollector.discover` — **by crawling an index page** with
  :class:`connectors.scraper.Crawl4AIScraper`. The entities are chosen among
  the links the page really contains, so a URL can only be one it published.
  Slower, and the only way to be exhaustive about a set nobody summarizes.

Recall is the union of the three (:meth:`~DataspaceCollector.collect` merges
entities across sources on name or shared URL, so an entity found twice keeps
the best URLs of both). Precision is what happens next: every URL is fetched,
and an entity that loses its URLs is **repaired** — searched for by name —
before it is given up on. Asked for the cinemas of Fribourg from memory alone,
the model returned ``fribourg.arena.ch``, a plausible subdomain that does not
resolve; the real page, ``arena.ch/fr/fribourg``, is what search and repair
bring back.

**URLs are verified, not trusted**, whatever the source
(:meth:`~DataspaceCollector.verify`), and verification checks two different
things. A list whose links 404 is worthless; so is one whose entities share a
URL, since they collapse into a single datapoint — asked for the cinemas of
Fribourg the model once returned three all pointing at ``cinemotion.ch``, every
URL alive, and the collection held one page.
:meth:`~DataspaceCollector.duplicate_entities` is what catches that.

Hosts: MatchMake's ``api.dataspace_codegen.DataspaceCodeGenerator`` binds this
to the session LLM and its "New dataspace" dialogue; keopy can collect the same
lists for the Angels it schedules.
"""
import json
import os
import re
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
import urllib3

from connectors.dataspace.blueprint import (
    URL_KEYS, CollectionReport, DataspaceBlueprint,
)

import logging
log = logging.getLogger(__name__)

# Verification re-probes a rejected certificate chain with verify=False (see
# _is_reachable), which makes urllib3 warn on every such URL. Nothing else
# disables TLS verification, so silencing this one warning hides nothing else.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


#: Browser-ish header: a bare python-requests User-Agent is refused (403) by
#: enough sites that verification would drop URLs which are perfectly alive.
_VERIFY_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
}


class DataspaceCollector:
    """Generates a Dataspace URL list on-the-fly, from three merged sources.

    The text LLM of the recalled and crawled sources comes from the host (see
    :meth:`_llm`); the searched source brings its own model, Gemini with the
    ``google_search`` tool, so a caller with no LLM at all still collects.

    :meth:`collect` is what callers use; it runs the sources in parallel and
    then does the work that turns candidates into a Dataspace:

    1. :meth:`research` / :meth:`generate` / :meth:`discover` — the three
       sources, each returning its own blueprint (recall).
    2. merge — entities are matched across sources on their name or on a shared
       URL, and an entity found twice keeps the union of its URLs.
    3. :meth:`verify` — every URL is fetched; the dead ones are dropped
       (precision).
    4. :meth:`repair` — an entity that lost every URL is searched for by name
       rather than dropped, which is what saves the real address of an entity
       the model only half-remembered.
    5. :meth:`duplicate_entities` — entities left without a URL of their own
       are reported, so the list is never saved as one collapsed datapoint.
    """

    #: Seconds to wait for a URL to answer during verification.
    VERIFY_TIMEOUT = 8.0

    #: Parallel verification requests. The URLs of one dataspace usually span
    #: many hosts, so this is politeness-neutral and keeps a 50-entity list
    #: down to a couple of seconds.
    VERIFY_WORKERS = 8

    #: How many crawled links are offered to the selection call. Sized to fit
    #: a whole institutional directory in one pass — the EPFL labs index alone
    #: yields 524 usable links, and truncating it would silently drop the tail
    #: of an alphabetical listing. Past this the prompt is cut rather than
    #: split, which is visible in the log as a candidate count at the cap.
    MAX_CANDIDATES = 800

    #: Crawled links that can never be the entity a Dataspace watches:
    #: non-http schemes, static assets, and the usual utility pages.
    _SKIP_PATTERN = re.compile(
        r"^(?:mailto|tel|javascript|data):"
        r"|\.(?:png|jpe?g|gif|svg|webp|ico|css|js|pdf|docx?|pptx?|xlsx?|zip|mp4|mp3)(?:$|\?)"
        r"|/(?:login|signin|sign-in|logout|register|account|cart|checkout|search"
        r"|privacy|cookies?|imprint|impressum|disclaimer|sitemap|rss|feed)(?:$|/|\?)",
        re.IGNORECASE)

    #: Social networks: every institutional page links to them, none is ever
    #: the source being collected.
    _SOCIAL_HOSTS = ("facebook.", "twitter.", "x.com", "instagram.", "linkedin.",
                     "youtube.", "youtu.be", "tiktok.", "pinterest.", "reddit.",
                     "mastodon.", "bsky.")

    #: Search-grounded model for :meth:`research` and :meth:`repair`. The
    #: environment overrides it the way :mod:`api.identity_research` does, and
    #: falls back on the same default so both web-grounded calls of the app
    #: stay on one model.
    SEARCH_MODEL_ENV = ("GEMINI_DATASPACE_MODEL", "GEMINI_RESEARCH_MODEL")
    DEFAULT_SEARCH_MODEL = "gemini-3.5-flash"

    #: Host of Gemini's grounding redirects. The model sometimes quotes those
    #: instead of the page's own address; they answer (so verification keeps
    #: them), they expire, and they are not the source being watched.
    _REDIRECT_HOST = "vertexaisearch.cloud.google.com"

    def __init__(self, llm=None):
        #: The text LLM the recalled and crawled sources talk to: anything with
        #: a ``respond_json(prompt) -> str`` method, or a plain callable taking
        #: the prompt. None disables those two sources — the searched one has
        #: its own model and still runs. A host that resolves its LLM lazily
        #: (MatchMake's session can change model mid-run) overrides
        #: :meth:`_llm` instead of passing one here.
        self.llm = llm
        #: Which sources proposed each merged entity — see :meth:`_merge`.
        self._provenance: Dict[int, set] = {}
        #: Why the last :meth:`discover` returned nothing, in one sentence the
        #: UI can show. Crawling fails in ways the user can act on — the
        #: proposed index page was a 404, the page had no usable link — and
        #: "nothing was found" alone would send them looking in the wrong
        #: place. Reset at the start of every call.
        self.last_failure = ""

    def _llm(self):
        """The text LLM of the recalled and crawled sources, None when absent.

        The one host seam of this module: MatchMake overrides it to resolve the
        session's current model at call time, keopy passes its own to
        :meth:`__init__`, and a caller with neither still gets the searched
        source.
        """
        return self.llm

    @staticmethod
    def _respond_json(llm, prompt: str) -> str:
        """Ask ``llm`` for JSON, accepting either shape of LLM object.

        MatchMake and keopy both hand over ``matching_learning``'s ``LLM``,
        which answers to ``respond_json``; a caller wiring in something else
        only needs a callable taking the prompt.
        """
        if hasattr(llm, "respond_json"):
            return llm.respond_json(prompt) or ""
        return llm(prompt) or ""

    # ─────────────────────────────────────────────────────────────────────
    #  0. The pipeline
    # ─────────────────────────────────────────────────────────────────────

    def collect(self, specification: str, limit: int = 25,
                index_urls: List[str] = None, verify: bool = True,
                progress: Callable[[str], None] = None
                ) -> Tuple[Optional[DataspaceBlueprint], CollectionReport]:
        """Collect the Dataspace ``specification`` asks for, from every source.

        The three sources run **in parallel** (they are independent given the
        request, and the crawl is slow enough that doing it after the others
        would double the wait), their entities are merged, every URL is fetched,
        and the entities that lose theirs are searched for by name before being
        given up on.

        ``index_urls`` are index pages the user pasted; left empty the crawl
        source proposes its own. ``verify=False`` skips the fetching — and with
        it the repair, which has nothing to repair — but never the distinctness
        check, which is not a question about the live web. ``progress`` is
        called with a short status line at each stage, for the UI.

        Returns ``(blueprint, report)``, the blueprint being None when no source
        produced a single entity — :attr:`last_failure` then says why.
        """
        self.last_failure = ""
        report = CollectionReport()
        say = progress or (lambda message: None)

        # ── 1. The sources, in parallel ───────────────────────────────────
        say("Searching, recalling and crawling...")
        runners = {
            "search": lambda: self.research(specification, limit),
            "memory": lambda: self.generate(specification, limit),
            "crawl": lambda: self.discover(specification, index_urls, limit),
        }
        if not self.search_available():
            # Without the key the searched source cannot run at all; saying so
            # in the report is what explains a thinner list than usual.
            del runners["search"]
            report.failures["search"] = ("GEMINI_API_KEY is not set — the web "
                                         "search source did not run.")
        blueprints: Dict[str, DataspaceBlueprint] = {}
        with ThreadPoolExecutor(max_workers=len(runners)) as pool:
            futures = {pool.submit(runner): key for key, runner in runners.items()}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    blueprint = future.result()
                except Exception as exc:  # noqa: BLE001 - one dead source is survivable
                    log.error("DataspaceCollector - the %s source failed: %s", key, exc)
                    report.failures[key] = str(exc)
                    continue
                if blueprint is None or not blueprint.entities:
                    report.proposed[key] = 0
                    report.failures.setdefault(
                        key, self.last_failure if key == "crawl" else "nothing proposed")
                    continue
                blueprints[key] = blueprint
                report.proposed[key] = len(blueprint.entities)
        log.info("DataspaceCollector - sources proposed %s", report.proposed)

        if not blueprints:
            self.last_failure = (self.last_failure
                                 or "No source produced a single entity.")
            return None, report

        # ── 2. Merge ──────────────────────────────────────────────────────
        merged = self._merge(blueprints, limit)
        report.merged = len(merged.entities)
        report.corroborated = sum(1 for entity in merged.entities
                                  if len(self._provenance.get(id(entity), ())) > 1)

        if not verify:
            # Without verification nothing addresses the leads, so they are
            # dropped here rather than saved as entities pointing nowhere.
            merged.entities = [entity for entity in merged.entities
                               if any(entity.get(key) for key in URL_KEYS)]
            report.merged = len(merged.entities)
            report.duplicates = self.duplicate_entities(merged)
            return (merged if merged.entities else None), report

        # ── 3. Verify ─────────────────────────────────────────────────────
        say(f"Verifying {len(merged.urls)} URL(s)...")
        before = {self._entity_name(entity): entity for entity in merged.entities}
        merged, report.dropped, _ = self.verify(merged)
        survivors = {self._entity_name(entity) for entity in merged.entities}
        lost = [name for name in before if name not in survivors]

        # ── 4. Repair what verification killed ────────────────────────────
        if lost and self.search_available():
            say(f"Searching the real address of {len(lost)} source(s)...")
            repaired = self.repair(specification, lost)
            alive = []
            for entity in repaired:
                urls = [entity.get(key) for key in URL_KEYS if entity.get(key)]
                if any(self._is_reachable(url) for url in urls):
                    alive.append(entity)
            if alive:
                merged.entities.extend(alive)
                report.repaired = [self._entity_name(entity) for entity in alive]
                log.info("DataspaceCollector - repaired %d entit(ies) by search: %s",
                         len(alive), report.repaired)
            repaired_names = {self._entity_name(entity) for entity in alive}
            lost = [name for name in lost if name not in repaired_names]
        report.lost = lost

        # ── 5. Distinctness ───────────────────────────────────────────────
        # Verification and repair both change what an entity holds — one loses
        # the URL it was matched on, the other gains a page a neighbour already
        # had — so identities are settled once more before they are judged.
        self._consolidate(merged)
        report.merged = len(merged.entities)
        report.duplicates = self.duplicate_entities(merged)
        merged.entities = merged.entities[:limit]
        report.merged = len(merged.entities)
        log.info("DataspaceCollector - collected %d entit(ies) from %s",
                 report.merged, report.method())
        return (merged if merged.entities else None), report

    # ─────────────────────────────────────────────────────────────────────
    #  0b. Merging the sources
    # ─────────────────────────────────────────────────────────────────────

    def _merge(self, blueprints: Dict[str, DataspaceBlueprint],
               limit: int) -> DataspaceBlueprint:
        """One blueprint out of the sources', matched on name or shared URL.

        The same entity comes back differently from each source — "Arena
        Cinemas Fribourg Centre" with an invented subdomain from memory, "Arena
        Cinemas Fribourg" with the real page from search — so entities are
        matched on their **normalized name** (accents, case and punctuation
        removed) or on **any URL they share**, and a match keeps the union of
        their URLs: the searched page fills the field the recalled entity got
        wrong, and the recalled ``program_url`` survives where search gave only
        a home page.

        The descriptive fields (name, description, ``collection_key``,
        ``context``) are taken from the most grounded source that produced
        them, in the order search → crawl → memory.
        """
        order = [key for key in ("search", "crawl", "memory") if key in blueprints]
        head = blueprints[order[0]]
        merged = DataspaceBlueprint(
            name=head.name, description=head.description,
            collection_key=head.collection_key, context=dict(head.context),
        )
        #: entity object id -> the sources that proposed it. Keyed by identity
        #: because an entity dict is mutated in place as sources are merged in.
        self._provenance: Dict[int, set] = {}

        by_name: Dict[str, dict] = {}
        by_url: Dict[str, dict] = {}
        for key in order:
            for entity in blueprints[key].entities:
                name = self._normalize_name(self._entity_name(entity))
                urls = self._entity_urls(entity)
                target = by_name.get(name) if name else None
                if target is None and name:
                    # One source's fuller official name ("Arena Cinemas Fribourg
                    # Centre") against another's short one ("Arena Cinemas
                    # Fribourg") — see :meth:`_same_entity`.
                    for known, candidate in by_name.items():
                        if self._same_entity(name, known):
                            target = candidate
                            break
                if target is None:
                    for url in urls:
                        if url in by_url:
                            target = by_url[url]
                            break
                if target is None:
                    target = dict(entity)
                    merged.entities.append(target)
                    self._provenance[id(target)] = {key}
                else:
                    # Fill what the other source did not have; never overwrite a
                    # value that is already there (the first source is the more
                    # grounded one by construction of `order`).
                    for field, value in entity.items():
                        if value and not target.get(field):
                            target[field] = value
                    self._provenance[id(target)].add(key)
                if name:
                    by_name.setdefault(name, target)
                for url in self._entity_urls(target):
                    by_url.setdefault(url, target)

        log.info("DataspaceCollector - merged %s into %d entit(ies)",
                 {key: len(blueprints[key].entities) for key in order},
                 len(merged.entities))
        self._strip_echoed_urls(merged)
        merged.entities = merged.entities[:limit]
        return merged

    @classmethod
    def _strip_echoed_urls(cls, blueprint: DataspaceBlueprint):
        """Drop a ``program_url`` / ``listing_url`` that merely repeats ``url``.

        A source with nothing more specific to give fills every URL field with
        the same address. It changes nothing for the grounding (the URLs are
        deduplicated before they are fetched) but it is a claim the file should
        not make: an entity whose programme page is its home page has no
        programme page, and saying so is what tells the next run there is still
        something to look for.
        """
        for entity in blueprint.entities:
            home = (entity.get("url") or "").strip().rstrip("/").lower()
            if not home:
                continue
            for key in URL_KEYS[1:]:
                value = (entity.get(key) or "").strip().rstrip("/").lower()
                if value and value == home:
                    entity.pop(key, None)

    def _consolidate(self, blueprint: DataspaceBlueprint):
        """Fold entities that ended up being the same one, in place.

        The same matching as :meth:`_merge` (name or shared URL), run again on
        the verified list: an entity whose own page died and now carries only a
        neighbour's URL, or a repaired one handed the address another entity
        already had, is a duplicate created *after* the merge — and the file
        must not carry it twice.
        """
        kept: List[dict] = []
        by_name: Dict[str, dict] = {}
        by_url: Dict[str, dict] = {}
        for entity in blueprint.entities:
            name = self._normalize_name(self._entity_name(entity))
            target = by_name.get(name) if name else None
            if target is None and name:
                target = next((candidate for known, candidate in by_name.items()
                               if self._same_entity(name, known)), None)
            if target is None:
                target = next((by_url[url] for url in self._entity_urls(entity)
                               if url in by_url), None)
            if target is None:
                kept.append(entity)
                target = entity
            else:
                for field, value in entity.items():
                    if value and not target.get(field):
                        target[field] = value
                log.info("DataspaceCollector - folded '%s' into '%s' after "
                         "verification", self._entity_name(entity),
                         self._entity_name(target))
            if name:
                by_name.setdefault(name, target)
            for url in self._entity_urls(target):
                by_url.setdefault(url, target)
        blueprint.entities = kept
        self._strip_echoed_urls(blueprint)

    @staticmethod
    def _entity_name(entity: dict) -> str:
        """The entity's name, falling back on its first URL."""
        name = str(entity.get("name") or "").strip()
        if name:
            return name
        for key in URL_KEYS:
            if entity.get(key):
                return str(entity[key]).strip()
        return ""

    @staticmethod
    def _entity_urls(entity: dict) -> List[str]:
        """The entity's URLs, normalized for comparison (case, trailing slash)."""
        return [(entity.get(key) or "").strip().rstrip("/").lower()
                for key in URL_KEYS if (entity.get(key) or "").strip()]

    @staticmethod
    def _normalize_name(name: str) -> str:
        """A name comparable across sources: no accents, case or punctuation.

        "Cinémotion Rex" and "Cinemotion  REX" are the same entity; anything
        finer (an operator prefix one source drops) is caught by the URL match
        instead, so this deliberately stays a cheap, predictable rule.
        """
        stripped = unicodedata.normalize("NFKD", name or "")
        ascii_only = "".join(c for c in stripped if not unicodedata.combining(c))
        return " ".join(re.sub(r"[^0-9a-z]+", " ", ascii_only.lower()).split())

    @staticmethod
    def _same_entity(one: str, other: str) -> bool:
        """Whether two normalized names denote the same thing.

        One source gives the official name, another the short one — "arena
        cinemas fribourg centre" and "arena cinemas fribourg" — so a name that
        is a **word-prefix** of the other, on at least two words, is treated as
        the same entity. Two words is what keeps "rex" from swallowing every
        venue whose name starts with it.

        The rule can over-merge (a chain's "cinema rex" and "cinema rex bulle"
        would join), and that direction is the safe one: merging keeps the union
        of both entities' URLs, where leaving them apart puts the same venue in
        the file twice under two of its pages.
        """
        first, second = one.split(), other.split()
        if not first or not second:
            return False
        short, long = sorted((first, second), key=len)
        return len(short) >= 2 and long[:len(short)] == short

    # ─────────────────────────────────────────────────────────────────────
    #  1. Generation
    # ─────────────────────────────────────────────────────────────────────

    def generate(self, specification: str, limit: int = 25) -> Optional[DataspaceBlueprint]:
        """Enumerate the entities and URLs ``specification`` asks for.

        ``limit`` caps how many entities the LLM returns — a handful for the
        cinemas of one city, many more for the research labs of an
        institution. Returns None when no LLM is available or the answer
        cannot be parsed: unlike a persona, there is no meaningful template
        fallback for a URL list, and inventing one would only produce a file
        full of URLs that do not exist.
        """
        llm = self._llm()
        if llm is None:
            log.warning("DataspaceCollector - no LLM available, cannot generate a URL list")
            return None
        log.info("DataspaceCollector - generating a URL list (limit=%d) for: %s",
                 limit, specification.strip()[:160])
        try:
            answer = self._respond_json(llm, self._codegen_prompt(specification, limit))
        except Exception as exc:  # noqa: BLE001 - a failed call yields no list
            log.error("DataspaceCollector - LLM generation failed: %s", exc)
            return None
        blueprint = self._parse_blueprint(answer, require_url=False)
        if blueprint is not None:
            log.info("DataspaceCollector - %d entit(ies), %d URL(s) proposed",
                     len(blueprint.entities), len(blueprint.urls))
        return blueprint

    def _codegen_prompt(self, specification: str, limit: int) -> str:
        """Assemble the URL-list request sent to the LLM."""
        return "\n".join([
            "You are the Dataspace generator of MatchMake. A Dataspace is a JSON",
            "file listing the web sources to watch: named entities, each with the",
            "URL of its site and, when they exist, the more specific pages where",
            "its content is actually published (a cinema's screening schedule, a",
            "lab's publication list).",
            "",
            "Enumerate what this request asks for:",
            "",
            specification.strip(),
            "",
            f"List up to {limit} entities. Be exhaustive within that limit and",
            "prefer real, currently reachable addresses over guesses: a URL you",
            "are unsure about is worse than one entity fewer. Give the canonical",
            "site of each entity, not a search engine, a directory listing or a",
            "social media page.",
            "",
            "Every entity must carry URLs of its OWN, and two entities may never",
            "share the same address. When several of them belong to one operator",
            "or one group, give each the page that is about IT — its own subpage",
            "on that site — and in program_url the page where ITS content is",
            "published. A list whose entities repeat one homepage is rejected:",
            "each URL becomes one stored document, so entities sharing a URL",
            "collapse into a single one and nothing tells them apart.",
            "",
            "Never assemble an address from a pattern — no invented subdomain,",
            "no guessed path. An entity you know by name but cannot address is",
            "still worth listing: give it with an empty url, and it will be",
            "searched for. A plausible address is the one thing that is worse",
            "than none.",
            "",
            "Return ONLY one JSON object with exactly these keys:",
            '  "name": short file-friendly name, lowercase with hyphens, e.g. "lausanne-cinema",',
            '  "description": one line describing what this Dataspace covers,',
            '  "collection_key": what the entity list is called in this domain,',
            '                    e.g. "cinemas", "labs", "museums", "publishers",',
            '  "context": object of scalar fields describing the whole set,',
            '             e.g. {"city": "Lausanne", "country": "Switzerland"},',
            '  "entities": list of objects, each with:',
            '        "name": the entity name,',
            '        "url": its main site,',
            '        "program_url": the page listing its current content (optional),',
            '        "listing_url": an index or archive page (optional),',
            '        "address": its postal address (optional).',
        ])

    def _parse_blueprint(self, answer: str,
                         require_url: bool = True) -> Optional[DataspaceBlueprint]:
        """Parse the LLM's JSON answer into a blueprint, None on failure.

        ``require_url=False`` keeps entities that were named without an address.
        The enumerating sources want that: an entity the model knows by name but
        cannot address is a **lead**, and :meth:`repair` searches the web for
        exactly those — far better recall than asking the model to invent an
        address or to stay silent. They never reach the saved file: whatever
        repair fails to address is dropped with the unreachable ones.
        """
        try:
            match = re.search(r"\{.*\}", answer, re.DOTALL)
            data = json.loads(match.group(0) if match else answer)
            blueprint = DataspaceBlueprint(**data)
        except Exception as exc:  # noqa: BLE001 - unparseable answer -> no list
            log.warning("DataspaceCollector - could not parse the URL list %.200r: %s",
                        answer, exc)
            return None
        blueprint.entities = [
            e for e in blueprint.entities
            if isinstance(e, dict)
            and (any(e.get(k) for k in URL_KEYS) if require_url else e.get("name"))
        ]
        return blueprint

    # ─────────────────────────────────────────────────────────────────────
    #  1b. Live web search
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def search_available() -> bool:
        """Whether the searched source can run — it needs ``GEMINI_API_KEY``.

        The key is read from the environment, loading the host's ``.env`` first
        as :class:`connectors.scraper.GeminiUrlContextScraper` does: a library
        cannot assume the host got round to loading it before the first call.
        """
        if not os.getenv("GEMINI_API_KEY"):
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except Exception:  # noqa: BLE001 - no dotenv is a valid deployment
                pass
        return bool(os.getenv("GEMINI_API_KEY"))

    def _search_model(self) -> str:
        """The Gemini model the web-grounded calls use."""
        for variable in self.SEARCH_MODEL_ENV:
            if os.getenv(variable):
                return os.getenv(variable)
        return self.DEFAULT_SEARCH_MODEL

    def research(self, specification: str, limit: int = 25) -> Optional[DataspaceBlueprint]:
        """Enumerate the entities from **live web search**.

        The precision source. Where :meth:`generate` reconstructs an address
        from what an address of that kind looks like — and produces
        ``fribourg.arena.ch``, which does not resolve — this one reads actual
        search results and copies the address out of them.

        Uses Gemini's ``google_search`` tool with ``GEMINI_API_KEY``, exactly as
        :class:`api.identity_research.IdentityResearcher` does; the shared LLM
        layer is text-only and cannot search. Returns None when the key is
        missing or the search fails: the other sources still carry the request.
        """
        if not self.search_available():
            log.info("DataspaceCollector - no GEMINI_API_KEY, skipping the "
                     "searched source")
            return None

        from google import genai
        from google.genai import types

        model = self._search_model()
        log.info("DataspaceCollector - searching the web with %s for: %s",
                 model, specification.strip()[:160])
        try:
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            response = client.models.generate_content(
                model=model,
                contents=self._research_prompt(specification, limit),
                # JSON mode cannot be combined with the search tool, so the
                # shape is asked for in the prompt and parsed leniently.
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]),
            )
        except Exception as exc:  # noqa: BLE001 - the other sources still run
            log.error("DataspaceCollector - the web search failed: %s", exc)
            return None

        blueprint = self._parse_blueprint(response.text or "", require_url=False)
        if blueprint is None:
            return None
        self._drop_unusable_urls(blueprint)
        log.info("DataspaceCollector - web search proposed %d entit(ies), "
                 "%d URL(s)", len(blueprint.entities), len(blueprint.urls))
        return blueprint if blueprint.entities else None

    def _research_prompt(self, specification: str, limit: int) -> str:
        """The searched enumeration: find them, then quote their addresses."""
        return "\n".join([
            "You are the Dataspace generator of MatchMake. Search the web and",
            "list what this request asks for. The items may be anything —",
            "venues, companies, research labs, restaurants, events, public",
            "pages — whatever the request names.",
            "",
            specification.strip(),
            "",
            "Search first, then answer from what the results actually say.",
            f"List up to {limit} items, and be exhaustive within that limit:",
            "search several times with different wordings (the local language",
            "of the place first), and consult the directories, official",
            "listings and association pages that enumerate them rather than one",
            "article about them.",
            "",
            "Every URL you write must be one you have SEEN in a result, copied",
            "exactly. Never assemble an address from a pattern — no invented",
            "subdomain, no guessed path. If you cannot see an item's own page,",
            "give the item without a url rather than a plausible one.",
            "",
            "Each item must have a page of its OWN: two items may never share",
            "the same address. When several belong to one operator or one",
            "group, give each the page that is about IT, and in program_url the",
            "page where ITS content is published — a programme, a menu, a",
            "publication list, an events calendar.",
            "",
            "Return ONLY one JSON object with exactly these keys:",
            '  "name": short file-friendly name, lowercase with hyphens,',
            '  "description": one line describing what this Dataspace covers,',
            '  "collection_key": what the item list is called in this domain,',
            '                    e.g. "cinemas", "labs", "restaurants", "concerts",',
            '  "context": object of scalar fields describing the whole set,',
            '             e.g. {"city": "Fribourg", "country": "Switzerland"},',
            '  "entities": list of objects, each with:',
            '        "name": the item name,',
            '        "url": its own page,',
            '        "program_url": the page listing its current content (optional),',
            '        "listing_url": an index or archive page (optional),',
            '        "address": its postal address (optional).',
        ])

    def repair(self, specification: str, names: List[str]) -> List[dict]:
        """Search for the real address of entities whose URLs did not answer.

        The last precision step, and the one that keeps recall: an entity the
        model got right and the URL wrong — Arena Cinemas at the invented
        ``fribourg.arena.ch`` — would otherwise be dropped by verification,
        losing a real source over a typo the web can settle in one search.

        Returns the repaired entities (name + freshly searched URLs); the
        caller re-verifies them, because a searched URL is a claim like any
        other. ``[]`` when the search is unavailable or finds nothing.
        """
        if not names or not self.search_available():
            return []

        from google import genai
        from google.genai import types

        log.info("DataspaceCollector - searching the real address of %d "
                 "entit(ies): %s", len(names), names[:8])
        try:
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            response = client.models.generate_content(
                model=self._search_model(),
                contents=self._repair_prompt(specification, names),
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]),
            )
        except Exception as exc:  # noqa: BLE001 - a failed repair drops the entities
            log.error("DataspaceCollector - the repair search failed: %s", exc)
            return []

        blueprint = self._parse_blueprint(response.text or "")
        if blueprint is None:
            return []
        self._drop_unusable_urls(blueprint)
        return blueprint.entities

    @staticmethod
    def _repair_prompt(specification: str, names: List[str]) -> str:
        """Ask for one thing only: the address these named items really have."""
        return "\n".join([
            "Search the web for the official page of each item listed below.",
            "They come from this request:",
            "",
            specification.strip(),
            "",
            "Items:",
            *(f"- {name}" for name in names),
            "",
            "The address we had for them does not exist, so give the one you",
            "SEE in the search results, copied exactly — never assembled from a",
            "pattern. Drop an item entirely rather than guessing its address,",
            "and never give two items the same address.",
            "",
            "Return ONLY one JSON object of the form",
            '{"entities": [{"name": "...", "url": "...", "program_url": "..."}]}.',
        ])

    def _drop_unusable_urls(self, blueprint: DataspaceBlueprint):
        """Strip the URLs a searched answer must never carry.

        Grounding redirects (they expire and are not the source), social
        networks and the utility pages :attr:`_SKIP_PATTERN` covers — the same
        rules the crawled source applies to its candidates, so an entity is
        judged identically wherever it came from. An entity stripped of every
        URL survives as a named lead for :meth:`repair`; one with neither name
        nor URL is nothing at all and goes.
        """
        for entity in blueprint.entities:
            for key in URL_KEYS:
                url = (entity.get(key) or "").strip()
                if url and not self._usable_url(url):
                    log.info("DataspaceCollector - dropping unusable URL %s", url)
                    entity[key] = ""
        blueprint.entities = [
            entity for entity in blueprint.entities
            if entity.get("name") or any(entity.get(key) for key in URL_KEYS)]

    def _usable_url(self, url: str) -> bool:
        """Whether ``url`` can be the page of a watched entity."""
        if not url.lower().startswith(("http://", "https://")):
            return False
        host = (urlparse(url).hostname or "").lower()
        if self._REDIRECT_HOST in host:
            return False
        if any(social in host for social in self._SOCIAL_HOSTS):
            return False
        return not self._SKIP_PATTERN.search(url)

    # ─────────────────────────────────────────────────────────────────────
    #  1c. Discovery by crawling an index page
    # ─────────────────────────────────────────────────────────────────────

    def discover(self, specification: str, index_urls: List[str] = None,
                 limit: int = 25) -> Optional[DataspaceBlueprint]:
        """Build the list from the links an index page really contains.

        The alternative to :meth:`generate` for a set the model does not know
        by heart: instead of recalling the entities, the directory page that
        lists them is scraped and the entities are chosen **among its actual
        links**, so a URL can only be one the page published.

        ``index_urls`` are the pages to crawl — typically pasted by the user
        ("https://www.epfl.ch/labs/"). When none is given the LLM proposes
        candidates, which is a far easier question than enumerating every
        entity, and each candidate is checked before it is crawled.

        Returns None when no index page can be reached or no link survives the
        selection — the caller reports that rather than saving a guess.
        """
        self.last_failure = ""
        given = [url.strip() for url in (index_urls or []) if url and url.strip()]
        proposed = [] if given else self._propose_index_urls(specification)
        if proposed:
            log.info("DataspaceCollector - proposed index page(s): %s", proposed)
        tried = given or proposed
        if not tried:
            self.last_failure = ("No index page to crawl: the model proposed none. "
                                 "Paste the URL of the page that lists the sources.")
            log.warning("DataspaceCollector - %s", self.last_failure)
            return None

        reachable = [url for url in tried if self._is_reachable(url)]
        if not reachable:
            # A proposed directory URL is still recall, and a plausible-looking
            # deep path is exactly what a model gets wrong: naming the dead
            # ones is what tells the user to paste the real one rather than
            # assume crawling is broken.
            self.last_failure = (
                ("The index page did not answer: " if given else
                 "The model proposed index pages that do not exist: ")
                + ", ".join(tried))
            log.warning("DataspaceCollector - %s", self.last_failure)
            return None

        candidates = self._crawl_candidates(reachable)
        if not candidates:
            self.last_failure = (f"{', '.join(reachable)} was crawled but carries no "
                                 f"link that could be a source.")
            log.warning("DataspaceCollector - %s", self.last_failure)
            return None

        blueprint = self._select_entities(specification, candidates, limit, reachable)
        if blueprint is None:
            self.last_failure = (f"None of the {len(candidates)} link(s) on "
                                 f"{', '.join(reachable)} matches the request.")
            log.warning("DataspaceCollector - %s", self.last_failure)
            return None
        log.info("DataspaceCollector - %d entit(ies) selected out of %d crawled link(s)",
                 len(blueprint.entities), len(candidates))
        return blueprint

    def _propose_index_urls(self, specification: str) -> List[str]:
        """Ask the LLM which page lists the entities being requested.

        Naming a directory ("the EPFL labs index") is something the model does
        reliably, unlike enumerating every entry of it — and the answer is
        checked against the live web before anything is crawled.
        """
        llm = self._llm()
        if llm is None:
            return []
        prompt = "\n".join([
            "Name the web page(s) that LIST the items in this request — a",
            "directory, an index, an 'all our X' page. Do not list the items",
            "themselves, only where they are listed.",
            "",
            specification.strip(),
            "",
            "Give at most 3 URLs, most authoritative first (the official site of",
            "the institution or city, not a search engine or a directory service).",
            'Return ONLY a JSON object of the form {"index_urls": ["https://..."]}.',
        ])
        try:
            answer = self._respond_json(llm, prompt)
            match = re.search(r"\{.*\}", answer, re.DOTALL)
            data = json.loads(match.group(0) if match else answer)
            urls = data.get("index_urls", [])
        except Exception as exc:  # noqa: BLE001 - no proposal is a valid outcome
            log.warning("DataspaceCollector - could not parse the index pages: %s", exc)
            return []
        return [str(url).strip() for url in urls
                if isinstance(url, str) and url.strip().startswith("http")][:3]

    def _crawl_candidates(self, index_urls: List[str]) -> List[dict]:
        """Scrape the index pages and return their links as ``{text, url}``.

        The link *text* matters as much as the href: it is what names the
        entity ("Computer Vision Laboratory"), and crawl4ai's normalized
        ``links`` list keeps only hrefs — so the anchors are read back out of
        the page markdown, where they survive as ``[text](url)``.
        """
        from connectors.scraper import Crawl4AIScraper

        scraper = Crawl4AIScraper()
        candidates, seen = [], set()
        for index_url in index_urls:
            try:
                page = scraper.scrape(index_url)
            except Exception as exc:  # noqa: BLE001 - one dead index must not stop the rest
                log.error("DataspaceCollector - crawl of %s failed: %s", index_url, exc)
                continue
            if not page.success:
                log.warning("DataspaceCollector - crawl of %s failed: %s",
                            index_url, page.error)
                continue
            found = 0
            for text, url in self._page_links(page, index_url):
                key = url.rstrip("/").lower()
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({"text": text, "url": url})
                found += 1
            log.info("DataspaceCollector - %d candidate link(s) from %s",
                     found, index_url)
            if len(candidates) >= self.MAX_CANDIDATES:
                break
        return candidates[:self.MAX_CANDIDATES]

    def _page_links(self, page, index_url: str):
        """The ``(text, url)`` pairs of one scraped page, filtered and absolute.

        Kept: absolute ``http(s)`` links on the same registrable domain as the
        index page. Dropped: assets, mail/phone/script links, social networks
        and the usual utility pages — none of them is ever the entity a
        Dataspace watches.
        """
        pairs, seen = [], set()
        # [text](url) out of the markdown: the anchors, with their names.
        for text, url in re.findall(r"\[([^\]\n]{1,120})\]\((https?://[^)\s]+)\)",
                                    page.markdown or ""):
            pairs.append((" ".join(text.split()), url))
        # Plus the hrefs crawl4ai normalized, for anchors the markdown lost.
        for url in (page.links or []):
            pairs.append(("", url))

        domain = self._registrable(index_url)
        for text, url in pairs:
            url = urljoin(index_url, url.strip()).split("#")[0]
            if not url.lower().startswith(("http://", "https://")):
                continue
            if self._SKIP_PATTERN.search(url):
                continue
            host = (urlparse(url).hostname or "").lower()
            if any(social in host for social in self._SOCIAL_HOSTS):
                continue
            if domain and self._registrable(url) != domain:
                continue
            key = url.rstrip("/").lower()
            if key in seen:
                continue
            seen.add(key)
            yield text, url

    @staticmethod
    def _registrable(url: str) -> str:
        """The registrable domain of a URL — ``epfl.ch`` for ``lis.epfl.ch``.

        Naive last-two-labels rule: it keeps every subdomain of one
        institution together, which is exactly what a lab index needs. It is
        wrong for multi-part suffixes (``co.uk``), where it merely narrows the
        candidates to one host instead of a whole domain — never wrong enough
        to matter here.
        """
        host = (urlparse(url).hostname or "").lower()
        parts = [p for p in host.split(".") if p]
        return ".".join(parts[-2:]) if len(parts) >= 2 else host

    def _select_entities(self, specification: str, candidates: List[dict],
                         limit: int, index_urls: List[str]) -> Optional[DataspaceBlueprint]:
        """Let the LLM pick the requested entities among the crawled links.

        The model's job here is selection and naming, not recall: every URL it
        may use is in front of it. Anything it returns that is not one of the
        crawled links is dropped — that is the whole guarantee of this mode.
        """
        llm = self._llm()
        if llm is None:
            return None
        lines = [
            "You are the Dataspace generator of MatchMake. Below are the links",
            "found on an index page. Select the ones that ARE the items the",
            "request asks for — not the navigation, not the news, not the",
            "contact or login pages.",
            "",
            "# Request",
            "",
            specification.strip(),
            "",
            "# Links found on " + ", ".join(index_urls),
            "",
        ]
        for n, candidate in enumerate(candidates, start=1):
            label = candidate["text"] or "(no link text)"
            lines.append(f"[{n}] {label} — {candidate['url']}")
        lines += [
            "",
            f"Select at most {limit} of them. Use ONLY URLs from the list above,",
            "copied exactly; never write a URL that is not listed. Give each",
            "entity a proper name (clean up the link text when needed). Never",
            "give the same URL to two entities: one URL is one stored document,",
            "so they would collapse into a single one.",
            "",
            "Return ONLY one JSON object with exactly these keys:",
            '  "name": short file-friendly name, lowercase with hyphens,',
            '  "description": one line describing what this Dataspace covers,',
            '  "collection_key": what the entity list is called in this domain,',
            '                    e.g. "labs", "cinemas", "museums",',
            '  "context": object of scalar fields describing the whole set,',
            '  "entities": list of objects, each with "name" and "url".',
        ]
        try:
            answer = self._respond_json(llm, "\n".join(lines))
        except Exception as exc:  # noqa: BLE001
            log.error("DataspaceCollector - selection failed: %s", exc)
            return None
        blueprint = self._parse_blueprint(answer)
        if blueprint is None:
            return None

        # The guarantee: keep only entities whose URL was actually crawled.
        crawled = {c["url"].rstrip("/").lower(): c["url"] for c in candidates}
        kept, invented = [], []
        for entity in blueprint.entities:
            url = (entity.get("url") or "").strip()
            real = crawled.get(url.rstrip("/").lower())
            if real is None:
                invented.append(url)
                continue
            entity["url"] = real
            kept.append(entity)
        if invented:
            log.info("DataspaceCollector - dropped %d selected URL(s) absent from "
                     "the crawled page: %s", len(invented), invented[:5])
        blueprint.entities = kept
        return blueprint if kept else None

    # ─────────────────────────────────────────────────────────────────────
    #  2. Verification
    # ─────────────────────────────────────────────────────────────────────

    def duplicate_entities(self, blueprint: DataspaceBlueprint
                           ) -> List[Tuple[str, List[str]]]:
        """The entities that carry no URL of their own, grouped by shared URL.

        Grounding stores one datapoint per URL, so an entity whose every URL is
        also some other entity's grounds nothing of its own — the three Fribourg
        cinemas that all pointed at ``cinemotion.ch`` became a single page, with
        every URL perfectly alive. Reachability cannot see this, which is why
        distinctness is its own check.

        The rule is per URL, not per entity: an entity is fine as soon as it
        carries **one** URL nobody else carries. Several venues under one
        operator's home page are therefore accepted when each keeps its own
        ``program_url`` — that page is what a scraper visits, and they still
        ground as separate documents — while a venue left with nothing but the
        shared home page is reported, including when it got there by losing its
        own page to a 404 during :meth:`verify`.

        Returns ``(shared URL, entity names)`` per group, largest first; ``[]``
        when every entity has a page of its own.
        """
        names, urls_of = [], []
        for entity in blueprint.entities:
            urls = {(entity.get(key) or "").strip().rstrip("/").lower()
                    for key in URL_KEYS if (entity.get(key) or "").strip()}
            if not urls:  # entities without a URL are dropped upstream
                continue
            names.append(str(entity.get("name") or sorted(urls)[0]))
            urls_of.append(urls)

        carriers = Counter(url for urls in urls_of for url in urls)
        groups: Dict[str, List[str]] = {}
        for name, urls in zip(names, urls_of):
            if any(carriers[url] == 1 for url in urls):
                continue  # has a page of its own
            groups.setdefault(sorted(urls)[0], []).append(name)

        collisions = sorted(groups.items(), key=lambda group: len(group[1]),
                            reverse=True)
        if collisions:
            log.info("DataspaceCollector - %d entit(ies) carry no URL of their "
                     "own: %s", sum(len(n) for _, n in collisions), collisions[:3])
        return collisions

    def verify(self, blueprint: DataspaceBlueprint
               ) -> Tuple[DataspaceBlueprint, List[str], List[Tuple[str, List[str]]]]:
        """Drop every URL that does not answer, and report the shared ones.

        Returns the pruned blueprint, the list of dropped URLs, and the groups
        of entities left sharing a URL (:meth:`duplicate_entities`) — so the UI
        can show exactly what the LLM proposed, what reality refused, and what
        would ground as one datapoint instead of several. A URL that redirects
        is kept (the redirect target is what a scraper follows too); only a
        connection failure or a 4xx/5xx answer removes it.

        The duplicate check runs **after** the pruning, deliberately: an entity
        that loses its own ``program_url`` to a 404 falls back on the URL it
        shares with its neighbours, and that collision is just as real.
        """
        urls = blueprint.urls
        if not urls:
            return blueprint, [], self.duplicate_entities(blueprint)

        with ThreadPoolExecutor(max_workers=self.VERIFY_WORKERS) as pool:
            alive = dict(zip(urls, pool.map(self._is_reachable, urls)))

        dropped = [url for url, ok in alive.items() if not ok]
        kept_entities = []
        for entity in blueprint.entities:
            pruned = {k: v for k, v in entity.items()
                      if k not in URL_KEYS or alive.get((v or "").strip(), False)}
            if any(pruned.get(k) for k in URL_KEYS):
                kept_entities.append(pruned)

        log.info("DataspaceCollector - verified %d URL(s): %d alive, %d dropped; "
                 "%d of %d entit(ies) kept",
                 len(urls), len(urls) - len(dropped), len(dropped),
                 len(kept_entities), len(blueprint.entities))
        blueprint.entities = kept_entities
        return blueprint, dropped, self.duplicate_entities(blueprint)

    def _is_reachable(self, url: str) -> bool:
        """Whether ``url`` answers.

        A rejected certificate is re-probed without TLS verification. Sites
        serving an incomplete certificate chain are common enough — the
        Cinémathèque suisse is one — and they are perfectly scrapeable: the
        headless browser the grounding uses follows them without complaint.
        Dropping them would delete real sources over a detail that never
        stops the job that consumes this file. Nothing is read from the
        unverified response but its status code.
        """
        try:
            return self._answers(url, verify_tls=True)
        except requests.exceptions.SSLError as exc:
            log.info("DataspaceCollector - %s has a rejected certificate chain "
                     "(%s); re-probing without TLS verification", url, exc)
        except Exception as exc:  # noqa: BLE001 - unreachable is a normal outcome
            log.info("DataspaceCollector - %s is not reachable: %s", url, exc)
            return False
        try:
            return self._answers(url, verify_tls=False)
        except Exception as exc:  # noqa: BLE001
            log.info("DataspaceCollector - %s is not reachable: %s", url, exc)
            return False

    def _answers(self, url: str, verify_tls: bool = True) -> bool:
        """True when ``url`` serves something. HEAD first, GET when refused.

        Plenty of sites answer 405/403 to HEAD while serving GET perfectly,
        so a negative HEAD is retried as a GET before the URL is given up on.
        """
        response = requests.head(url, timeout=self.VERIFY_TIMEOUT,
                                 allow_redirects=True, headers=_VERIFY_HEADERS,
                                 verify=verify_tls)
        if response.status_code < 400:
            return True
        response = requests.get(url, timeout=self.VERIFY_TIMEOUT,
                                allow_redirects=True, headers=_VERIFY_HEADERS,
                                verify=verify_tls, stream=True)
        response.close()
        return response.status_code < 400
