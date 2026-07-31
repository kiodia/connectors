"""The data of a Dataspace collection: the blueprint and the run report.

:class:`DataspaceBlueprint` is the file a collection produces — named entities,
each carrying the URLs to watch — and :class:`CollectionReport` is what the run
did to build it. Both are plain Pydantic models with no I/O, so a host can hold
them, render them and save them without pulling the collector in.
"""
import json
import re
from datetime import date
from typing import Dict, List, Tuple

from pydantic import BaseModel, Field

#: Keys of an entity that hold a URL, in the order the hand-written
#: dataspace files use them. ``url`` is the entity's home page — the one it
#: is identified by; the others are the more specific pages a grounding job
#: actually scrapes.
URL_KEYS = ("url", "program_url", "listing_url")

class DataspaceBlueprint(BaseModel):
    """One generated Dataspace URL list, ready to be written as JSON.

    ``entities`` are the things being watched (a cinema, a research lab, a
    museum), each a free-form dict with at least ``name`` and ``url``, plus
    whatever the domain calls for (``program_url``, ``listing_url``,
    ``address``). ``collection_key`` is what that list is called in the file —
    ``cinemas`` for the Fribourg example, ``labs`` for EPFL — and ``context``
    holds the scalar fields describing the whole set (``city``, ``country``,
    ``institution``).
    """
    name: str = Field(..., description="File-friendly name, e.g. 'lausanne-cinema'")
    description: str = Field(default="", description="One line describing what this Dataspace covers")
    collection_key: str = Field(default="entities", description="Name of the entity list in the JSON, e.g. 'cinemas'")
    context: Dict[str, str] = Field(default_factory=dict, description="Scalar fields describing the set: city, country, institution")
    entities: List[dict] = Field(default_factory=list, description="The watched entities, each with a name and its URL(s)")

    @property
    def urls(self) -> List[str]:
        """Every URL the blueprint carries, in order, deduplicated."""
        found, seen = [], set()
        for entity in self.entities:
            for key in URL_KEYS:
                url = (entity.get(key) or "").strip()
                if url and url not in seen:
                    seen.add(url)
                    found.append(url)
        return found

    def file_name(self) -> str:
        """The suggested file name, ``<name>.json``."""
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", self.name).strip("-") or "dataspace"
        return safe if safe.endswith(".json") else f"{safe}.json"

    def to_json_dict(self, query: str = "", method: str = "") -> dict:
        """The JSON written to disk, shaped like the hand-written files.

        The scalar context fields come first, then the entity list under
        ``collection_key`` — ``{"city": ..., "country": ..., "cinemas": [...]}``
        for the Fribourg example. ``query``, ``collected`` and ``generated``
        record where the file came from, which matters when it resurfaces
        months later: a crawled list is exhaustive as of that date, a recalled
        one is as good as the model's memory. Deliberately no URL goes into
        that provenance — ``load_urls`` would collect it as a source to scrape.
        """
        data = dict(self.context)
        data["description"] = self.description
        if query:
            data["query"] = query
        if method:
            data["collected"] = method
        data["generated"] = date.today().isoformat()
        data[self.collection_key or "entities"] = self.entities
        return data

    def to_json(self, query: str = "", method: str = "") -> str:
        return json.dumps(self.to_json_dict(query, method), indent=2, ensure_ascii=False)


class CollectionReport(BaseModel):
    """What :meth:`DataspaceCodeGenerator.collect` did, for the UI and the log.

    A merged collection is several things at once — searched, recalled,
    crawled, verified, repaired — and a bare entity count would hide which of
    them actually carried the list. Knowing that everything came from model
    knowledge (and nothing from search or the index) is exactly what tells the
    user how much to trust the file.
    """
    #: Entities each source proposed, keyed ``search`` / ``memory`` / ``crawl``.
    proposed: Dict[str, int] = Field(default_factory=dict)
    #: Entities kept after merging the sources.
    merged: int = 0
    #: Entities corroborated by more than one source.
    corroborated: int = 0
    #: URLs dropped because they did not answer.
    dropped: List[str] = Field(default_factory=list)
    #: Entities whose dead URL was replaced by a searched one.
    repaired: List[str] = Field(default_factory=list)
    #: Entities that lost every URL and could not be repaired.
    lost: List[str] = Field(default_factory=list)
    #: Groups of entities left without a URL of their own — see
    #: :meth:`DataspaceCodeGenerator.duplicate_entities`.
    duplicates: List[Tuple[str, List[str]]] = Field(default_factory=list)
    #: Why a source produced nothing, keyed like ``proposed``.
    failures: Dict[str, str] = Field(default_factory=dict)

    def method(self) -> str:
        """The provenance recorded in the saved file: the sources that fired."""
        names = {"search": "web search", "memory": "model knowledge",
                 "crawl": "crawled index"}
        used = [names[key] for key, count in self.proposed.items() if count]
        return " + ".join(used) if used else "no source"

    def summary(self) -> str:
        """One line for the dialogue's status text."""
        parts = [f"{self.merged} source(s) collected"]
        detail = ", ".join(f"{count} {key}" for key, count in self.proposed.items()
                           if count)
        if detail:
            parts.append(f"({detail})")
        if self.corroborated:
            parts.append(f"· {self.corroborated} confirmed by two sources")
        if self.repaired:
            parts.append(f"· {len(self.repaired)} repaired by search")
        if self.dropped:
            parts.append(f"· {len(self.dropped)} URL(s) dropped as unreachable")
        if self.lost:
            parts.append(f"· {len(self.lost)} dropped for want of a live URL")
        return " ".join(parts) + "."


