"""The Dataspace collector's own logic, with every source and fetch stubbed.

What is tested here is the part that makes a collection more than three lists
stapled together: merging the sources onto one entity, keeping the best URLs of
each, repairing what verification kills, and refusing a list whose entities
would ground as a single datapoint. The sources themselves (a Gemini search, a
host LLM, a crawl) are stubbed — they are network, not logic.

The cases are the real ones this was built from: the three Fribourg cinemas
that all pointed at ``cinemotion.ch``, and Arena Cinemas, recalled with the
invented ``fribourg.arena.ch`` and searchable at ``arena.ch/fr/fribourg``.
"""
import pytest

pytest.importorskip("pydantic")
pytest.importorskip("requests")

from connectors.dataspace import (  # noqa: E402
    CollectionReport, DataspaceBlueprint, DataspaceCollector,
)

ALIVE = {
    "https://www.arena.ch/fr/fribourg",
    "https://www.arena.ch/fr/fribourg/programme",
    "https://www.cineplus.ch/",
    "https://www.cinemotion.ch/rex",
    "https://www.cinemotion.ch/",
}


def blueprint(entities):
    return DataspaceBlueprint(
        name="fribourg-cinema", description="The cinemas of Fribourg",
        collection_key="cinemas", context={"city": "Fribourg"}, entities=entities)


@pytest.fixture
def collector(monkeypatch):
    """A collector that never touches the network."""
    made = DataspaceCollector()
    monkeypatch.setattr(type(made), "search_available", staticmethod(lambda: True))
    monkeypatch.setattr(made, "_is_reachable", lambda url: url in ALIVE)
    monkeypatch.setattr(made, "research", lambda spec, limit: None)
    monkeypatch.setattr(made, "generate", lambda spec, limit: None)
    monkeypatch.setattr(made, "discover", lambda spec, urls, limit: None)
    monkeypatch.setattr(made, "repair", lambda spec, names: [])
    return made


# ── Merging ───────────────────────────────────────────────────────────────

def test_sources_merge_onto_one_entity_and_keep_the_best_urls(collector, monkeypatch):
    """Search has the real home page, memory the programme page: one entity, both."""
    monkeypatch.setattr(collector, "research", lambda spec, limit: blueprint([
        {"name": "Arena Cinemas Fribourg Centre",
         "url": "https://www.arena.ch/fr/fribourg"},
    ]))
    monkeypatch.setattr(collector, "generate", lambda spec, limit: blueprint([
        {"name": "Arena Cinemas Fribourg", "url": "https://fribourg.arena.ch/",
         "program_url": "https://www.arena.ch/fr/fribourg/programme"},
    ]))

    collected, report = collector.collect("the cinemas of Fribourg")

    assert len(collected.entities) == 1
    entity = collected.entities[0]
    assert entity["url"] == "https://www.arena.ch/fr/fribourg"
    assert entity["program_url"] == "https://www.arena.ch/fr/fribourg/programme"
    # The invented address never even had to be fetched: the searched source
    # held the field before the recalled one could fill it.
    assert "https://fribourg.arena.ch/" not in report.dropped
    assert report.corroborated == 1


def test_an_entity_from_one_source_only_is_kept(collector, monkeypatch):
    monkeypatch.setattr(collector, "discover", lambda spec, urls, limit: blueprint([
        {"name": "Cineplus", "url": "https://www.cineplus.ch/"},
    ]))
    collected, report = collector.collect("the cinemas of Fribourg")
    assert [e["name"] for e in collected.entities] == ["Cineplus"]
    assert report.corroborated == 0


def test_word_prefix_names_are_the_same_entity():
    same = DataspaceCollector._same_entity
    assert same("arena cinemas fribourg", "arena cinemas fribourg centre")
    assert not same("rex", "rex bulle")            # one word is not enough
    assert not same("cinema rex", "cinema corso")


# ── Verification and repair ───────────────────────────────────────────────

def test_a_dead_url_is_dropped_and_the_entity_searched_for(collector, monkeypatch):
    monkeypatch.setattr(collector, "generate", lambda spec, limit: blueprint([
        {"name": "Cinema Rex", "url": "https://rex.invalid/"},
    ]))
    monkeypatch.setattr(collector, "repair", lambda spec, names: (
        [{"name": "Cinema Rex", "url": "https://www.cinemotion.ch/rex"}]
        if "Cinema Rex" in names else []))

    collected, report = collector.collect("the cinemas of Fribourg")

    assert report.dropped == ["https://rex.invalid/"]
    assert report.repaired == ["Cinema Rex"]
    assert collected.entities[0]["url"] == "https://www.cinemotion.ch/rex"
    assert report.lost == []


def test_an_entity_that_cannot_be_repaired_is_reported_as_lost(collector, monkeypatch):
    monkeypatch.setattr(collector, "generate", lambda spec, limit: blueprint([
        {"name": "Ghost Cinema", "url": "https://ghost.invalid/"},
        {"name": "Cineplus", "url": "https://www.cineplus.ch/"},
    ]))
    collected, report = collector.collect("the cinemas of Fribourg")
    assert report.lost == ["Ghost Cinema"]
    assert [e["name"] for e in collected.entities] == ["Cineplus"]


def test_a_named_lead_without_a_url_is_repaired_rather_than_dropped(collector, monkeypatch):
    """A source may name what it cannot address; the search then finds it."""
    monkeypatch.setattr(collector, "generate", lambda spec, limit: blueprint([
        {"name": "Cinema Rex"},
    ]))
    monkeypatch.setattr(collector, "repair", lambda spec, names: (
        [{"name": "Cinema Rex", "url": "https://www.cinemotion.ch/rex"}]
        if "Cinema Rex" in names else []))
    collected, report = collector.collect("the cinemas of Fribourg")
    assert report.repaired == ["Cinema Rex"]
    assert collected.entities[0]["url"] == "https://www.cinemotion.ch/rex"


def test_leads_never_reach_an_unverified_collection(collector, monkeypatch):
    monkeypatch.setattr(collector, "generate", lambda spec, limit: blueprint([
        {"name": "Cinema Rex"},
        {"name": "Cineplus", "url": "https://www.cineplus.ch/"},
    ]))
    collected, _ = collector.collect("the cinemas of Fribourg", verify=False)
    assert [e["name"] for e in collected.entities] == ["Cineplus"]


# ── Distinctness ──────────────────────────────────────────────────────────

def test_entities_sharing_one_home_page_are_reported():
    """The three Fribourg cinemas that ground as a single datapoint."""
    duplicates = DataspaceCollector().duplicate_entities(blueprint([
        {"name": "Cinemotion Rex", "url": "https://www.cinemotion.ch/"},
        {"name": "Cinemotion Arena", "url": "https://www.cinemotion.ch/"},
        {"name": "Cinemotion Corso", "url": "https://www.cinemotion.ch"},
    ]))
    assert len(duplicates) == 1
    url, names = duplicates[0]
    assert url == "https://www.cinemotion.ch"
    assert names == ["Cinemotion Rex", "Cinemotion Arena", "Cinemotion Corso"]


def test_a_shared_home_page_is_fine_when_each_has_its_own_content_page():
    assert DataspaceCollector().duplicate_entities(blueprint([
        {"name": "Rex", "url": "https://op.ch/", "program_url": "https://op.ch/rex"},
        {"name": "Corso", "url": "https://op.ch", "program_url": "https://op.ch/corso"},
    ])) == []


def test_an_echoed_program_url_is_stripped():
    made = blueprint([{"name": "Rex", "url": "https://op.ch/rex",
                       "program_url": "https://op.ch/rex/"}])
    DataspaceCollector._strip_echoed_urls(made)
    assert "program_url" not in made.entities[0]


# ── The report ────────────────────────────────────────────────────────────

def test_the_report_names_the_sources_that_fired():
    report = CollectionReport(proposed={"search": 3, "memory": 0, "crawl": 2}, merged=4)
    assert report.method() == "web search + crawled index"
    assert report.summary().startswith("4 source(s) collected (3 search, 2 crawl)")


def test_nothing_collected_is_reported_not_raised(collector):
    collected, report = collector.collect("something nobody publishes")
    assert collected is None
    assert report.merged == 0
    assert collector.last_failure
