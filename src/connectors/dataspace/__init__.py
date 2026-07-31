"""Dataspaces: the URL list a grounding run scrapes, collected from the web.

A Dataspace is the set of sources being watched — the cinemas of a city, the
research labs of an institution, the restaurants of a neighbourhood, a season's
concerts — written as a JSON file of named entities, each carrying its URLs.
:class:`~connectors.dataspace.collector.DataspaceCollector` builds one from a
sentence, merging live web search, model knowledge and a crawled index, and
:class:`connectors.grounding.DataGrounding` is what then stores what those URLs
publish.

Both hosts collect the same way: MatchMake from its "New dataspace" dialogue,
keopy for the Angels it schedules.
"""

from connectors.dataspace.blueprint import (
    URL_KEYS, CollectionReport, DataspaceBlueprint,
)
from connectors.dataspace.collector import DataspaceCollector

__all__ = [
    "DataspaceCollector",
    "DataspaceBlueprint",
    "CollectionReport",
    "URL_KEYS",
]
