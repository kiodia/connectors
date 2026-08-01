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
    monkeypatch.setattr(type(made), "cse_available", staticmethod(lambda: True))
    monkeypatch.setattr(made, "_is_reachable", lambda url: url in ALIVE)
    monkeypatch.setattr(made, "research", lambda spec, limit: None)
    monkeypatch.setattr(made, "cse", lambda spec, limit: None)
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


# ── The search engine's own results ───────────────────────────────────────

def cse_collector(monkeypatch, results, answer):
    """A collector whose CSE source returns ``results`` and whose LLM answers."""
    from connectors import googlesearch

    made = DataspaceCollector(llm=lambda prompt: answer)
    monkeypatch.setattr(googlesearch, "available", lambda: True)
    monkeypatch.setattr(googlesearch, "get_results", lambda query, pages=1: results)
    return made


def test_the_cse_source_names_entities_among_the_results(monkeypatch):
    results = [
        {"title": "Arena Cinemas Fribourg", "url": "https://www.arena.ch/fr/fribourg",
         "snippet": "10 salles au Fribourg Centre"},
        {"title": "Cineplus", "url": "https://www.cineplus.ch/", "snippet": "association"},
    ]
    answer = """{"name": "fribourg-cinema", "collection_key": "cinemas",
                 "context": {"city": "Fribourg"}, "entities": [
                   {"name": "Arena Cinemas Fribourg Centre",
                    "url": "https://www.arena.ch/fr/fribourg"},
                   {"name": "Cineplus", "url": "https://www.cineplus.ch/"}]}"""
    made = cse_collector(monkeypatch, results, answer)

    collected = made.cse("the cinemas of Fribourg")

    assert [e["url"] for e in collected.entities] == [
        "https://www.arena.ch/fr/fribourg", "https://www.cineplus.ch/"]


def test_the_cse_source_cannot_return_a_url_the_engine_did_not(monkeypatch):
    """The guarantee of a listed source: no address of the model's own making."""
    results = [{"title": "Arena Cinemas Fribourg",
                "url": "https://www.arena.ch/fr/fribourg", "snippet": ""}]
    answer = """{"name": "x", "entities": [
                   {"name": "Arena", "url": "https://www.arena.ch/fr/fribourg"},
                   {"name": "Invented", "url": "https://fribourg.arena.ch/"}]}"""
    made = cse_collector(monkeypatch, results, answer)

    collected = made.cse("the cinemas of Fribourg")

    assert [e["name"] for e in collected.entities] == ["Arena"]


def test_the_cse_source_drops_social_and_asset_results(monkeypatch):
    results = [
        {"title": "Arena on Facebook", "url": "https://www.facebook.com/arena", "snippet": ""},
        {"title": "Programme PDF", "url": "https://arena.ch/prog.pdf", "snippet": ""},
    ]
    asked = []
    made = cse_collector(monkeypatch, results, "{}")
    made.llm = lambda prompt: asked.append(prompt) or "{}"

    assert made.cse("the cinemas of Fribourg") is None
    assert asked == []          # nothing usable: the LLM is never even called


def test_the_cse_source_is_skipped_without_an_llm(monkeypatch):
    from connectors import googlesearch
    monkeypatch.setattr(googlesearch, "available", lambda: True)
    monkeypatch.setattr(googlesearch, "get_results",
                        lambda query, pages=1: [{"title": "x", "url": "https://a.ch/",
                                                 "snippet": ""}])
    assert DataspaceCollector().cse("anything") is None


def test_a_source_without_credentials_is_reported_not_run(monkeypatch):
    made = DataspaceCollector()
    monkeypatch.setattr(type(made), "search_available", staticmethod(lambda: False))
    monkeypatch.setattr(type(made), "cse_available", staticmethod(lambda: False))
    monkeypatch.setattr(made, "generate", lambda spec, limit: None)
    monkeypatch.setattr(made, "discover", lambda spec, urls, limit: None)

    _, report = made.collect("the cinemas of Fribourg")

    assert "GEMINI_API_KEY" in report.failures["search"]
    assert "GOOGLE_CSE_KEY" in report.failures["cse"]


def test_the_cse_source_merges_with_the_others(collector, monkeypatch):
    monkeypatch.setattr(collector, "cse", lambda spec, limit: blueprint([
        {"name": "Arena Cinemas Fribourg", "url": "https://www.arena.ch/fr/fribourg"},
    ]))
    monkeypatch.setattr(collector, "generate", lambda spec, limit: blueprint([
        {"name": "Arena Cinemas Fribourg Centre", "url": "https://fribourg.arena.ch/",
         "program_url": "https://www.arena.ch/fr/fribourg/programme"},
    ]))
    collected, report = collector.collect("the cinemas of Fribourg")

    assert len(collected.entities) == 1
    assert collected.entities[0]["url"] == "https://www.arena.ch/fr/fribourg"
    assert report.corroborated == 1
    assert report.proposed["cse"] == 1


# ── The report ────────────────────────────────────────────────────────────

def test_the_report_names_the_sources_that_fired():
    report = CollectionReport(
        proposed={"search": 3, "cse": 2, "memory": 0, "crawl": 2}, merged=5)
    assert report.method() == "web search + search-engine results + crawled index"
    assert report.summary().startswith("5 source(s) collected (3 search, 2 cse, 2 crawl)")


def test_nothing_collected_is_reported_not_raised(collector):
    collected, report = collector.collect("something nobody publishes")
    assert collected is None
    assert report.merged == 0
    assert collector.last_failure
