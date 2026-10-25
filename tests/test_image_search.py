"""Image search Dataspaces: the API list, the hits, the grounding of pictures.

No network: every API answer is a fake, the image download and the
multimodal embedding are stubs. What is tested is the reading of the file and
of the answers, the merging, and that the picture is what gets embedded.
"""
import hashlib
import json

import pytest

pytest.importorskip("qdrant_client")
pytest.importorskip("vectorize")

from qdrant_client import QdrantClient  # noqa: E402

from connectors import image_search as s  # noqa: E402
from vectorize.embeddings import EmbeddingSourceType, EmbeddingVector  # noqa: E402

DIM = 8


def _api(name, endpoint, results, fields, **extra):
    return {"name": name, "enabled": True, "auth": {"type": "none"}, "category": "museum",
            "endpoint": endpoint, "results": results, "fields": fields, **extra}


OPEN = _api("Open", "https://open.test/?q={query}&n={limit}", "results",
            {"image_url": "url", "page_url": "landing", "title": "title",
             "author": "creator", "license": "license", "tags": "tags.*.name"})
MUSEUM = _api("Museum", "https://museum.test/search?q={query}", "objectIDs",
              {"image_url": "primaryImage", "page_url": "objectURL", "title": "title",
               "genre": "classification", "medium": "medium", "year": "objectEndDate",
               "license": "isPublicDomain"},
              detail={"endpoint": "https://museum.test/objects/{id}"})
KEYED = _api("Keyed", "https://keyed.test/?q={query}&key=${KEYED_KEY}", "hits",
             {"image_url": "src"})


def spec(*apis):
    return {"kind": s.KIND, "apis": list(apis)}


def fake_fetch(answers):
    calls = []

    def fetch(url, headers, timeout):
        calls.append((url, dict(headers)))
        for prefix, answer in answers.items():
            if url.startswith(prefix):
                return answer
        raise OSError(f"no answer for {url}")

    fetch.calls = calls
    return fetch


ANSWERS = {
    "https://open.test/": {"results": [
        {"url": "https://img.test/a.jpg", "landing": "https://open.test/a", "title": "Sunflowers",
         "creator": "Ann", "license": "by", "tags": [{"name": "flower"}, {"name": "yellow"}]},
        {"url": "https://img.test/b.jpg", "landing": "https://open.test/b", "title": "Field"},
        {"title": "no picture"},
    ]},
    "https://museum.test/search": {"objectIDs": [1, 2]},
    "https://museum.test/objects/1": {"primaryImage": "https://img.test/m1.jpg",
                                      "objectURL": "https://museum.test/o/1", "title": "Sunflowers",
                                      "classification": "Paintings", "medium": "Oil on canvas",
                                      "objectEndDate": 1887, "isPublicDomain": True},
    "https://museum.test/objects/2": {"primaryImage": "", "title": "no image"},
}


# ── The file ─────────────────────────────────────────────────────────────

def test_load_spec_reads_only_an_image_search_file(tmp_path):
    good = tmp_path / "images.json"
    good.write_text(json.dumps(spec(OPEN)), encoding="utf-8")
    other = tmp_path / "cinemas.json"
    other.write_text(json.dumps({"urls": ["https://a.ch"]}), encoding="utf-8")
    assert s.load_spec(good)["apis"][0]["name"] == "Open"
    assert s.is_image_search(str(good))
    assert not s.is_image_search(other)
    assert not s.is_image_search(tmp_path / "missing.json")
    assert s.is_image_search(json.dumps(spec()))


def test_an_api_whose_key_is_missing_is_left_out():
    assert [a["name"] for a in s.available_apis(spec(OPEN, KEYED), env={})] == ["Open"]
    assert [a["name"] for a in s.available_apis(spec(OPEN, KEYED), env={"KEYED_KEY": "k"})] \
        == ["Open", "Keyed"]
    disabled = dict(OPEN, enabled=False)
    assert s.available_apis(spec(disabled), env={}) == []


# ── Reading an answer ────────────────────────────────────────────────────

def test_paths_index_from_the_end_and_map_over_lists():
    node = {"a": [{"b": 1}, {"b": 2}], "c": ["x", "y"]}
    assert s.get_path(node, "a.*.b") == [1, 2]
    assert s.get_path(node, "c.-1") == "y"
    assert s.get_path(node, "c.5") is None
    assert s.get_path(node, "missing.path") is None


def test_templates_read_the_hit_then_the_root_and_get_a_scheme():
    root = {"config": {"iiif_url": "https://iiif.test/2"}}
    hit = {"id": 7, "image_id": "abc"}
    assert s.field_value("template:{config.iiif_url}/{image_id}/full/843,/0/default.jpg", hit, root) \
        == "https://iiif.test/2/abc/full/843,/0/default.jpg"
    assert s.field_value("template:www.museum.test/artworks/{id}", hit, root) \
        == "https://www.museum.test/artworks/7"
    assert s.field_value("template:{nowhere}/x", hit, root) == ""
    assert s.field_value("const:CC0", hit) == "CC0"


def test_a_hit_maps_onto_the_image_entity():
    api = dict(MUSEUM, category="museum")
    hit = s.hit_from(api, ANSWERS["https://museum.test/objects/1"], query="sunflowers")
    assert hit.title == "Sunflowers" and hit.image_url == hit.thumbnail_url
    assert hit.genre == ["Paintings"] and hit.medium == "Oil on canvas" and hit.year == 1887
    assert hit.license == "public domain" and hit.provider == "Museum"
    assert hit.source_type == "museum" and hit.query == "sunflowers"
    assert s.hit_from(api, {"title": "no image"}) is None


def test_file_titles_html_and_wikimedia_categories_are_cleaned():
    api = _api("Wiki", "https://w.test/", "r", {"image_url": "u", "title": "t", "author": "a",
                                                "tags": "c", "year": "d"})
    hit = s.hit_from(api, {"u": "https://w.test/x.jpg", "t": "File:Vase_with_Sunflowers.jpg",
                           "a": '<a href="/wiki/VG">Vincent van Gogh</a>',
                           "c": "Sunflowers (Arles series, VGM)|PD-old", "d": "August 1888<div>x</div>"})
    assert hit.title == "Vase with Sunflowers"
    assert hit.author == "Vincent van Gogh"
    assert hit.tags == ["Sunflowers (Arles series, VGM)", "PD-old"]
    assert hit.year == 1888


# ── Querying ─────────────────────────────────────────────────────────────

def test_search_interleaves_the_apis_follows_details_and_dedupes():
    fetch = fake_fetch(ANSWERS)
    duplicate = _api("Mirror", "https://open.test/?mirror={query}", "results",
                     {"image_url": "url", "page_url": "landing", "title": "title"})
    hits = s.search(spec(OPEN, MUSEUM, duplicate), ["sun flowers"], limit=5, env={},
                    fetch_json=fetch)
    assert [(h.provider, h.title) for h in hits] == [
        ("Open", "Sunflowers"), ("Museum", "Sunflowers"), ("Open", "Field")]
    urls = [url for url, _ in fetch.calls]
    assert "https://open.test/?q=sun%20flowers&n=5" in urls
    assert "https://museum.test/objects/1" in urls and "https://museum.test/objects/2" in urls
    assert all(headers["User-Agent"] for _, headers in fetch.calls)


def test_one_failing_api_leaves_the_others():
    broken = _api("Broken", "https://broken.test/?q={query}", "results", {"image_url": "url"})
    hits = s.search(spec(broken, OPEN), ["x"], env={}, fetch_json=fake_fetch(ANSWERS))
    assert {h.provider for h in hits} == {"Open"}


def test_keys_and_headers_come_from_the_environment():
    api = _api("Header", "https://h.test/?q={query}&key=${H_KEY}", "results",
               {"image_url": "url"},
               auth={"type": "header", "env": "H_KEY", "header": "Authorization",
                     "value": "Client-ID ${H_KEY}"})
    fetch = fake_fetch({"https://h.test/": {"results": [{"url": "https://img.test/h.jpg"}]}})
    hits = s.search(spec(api), ["cats"], env={"H_KEY": "secret"}, fetch_json=fetch)
    assert len(hits) == 1
    url, headers = fetch.calls[0]
    assert "key=secret" in url and headers["Authorization"] == "Client-ID secret"


def test_queries_put_the_prompt_first_then_the_liked_then_the_persona():
    queries = s.queries_for({"keywords": ["impressionism", "still life"],
                             "likes": ["The paintings of Van Gogh"],
                             "intention": "I want calm art for my living room"},
                            prompt="sunflowers in a vase", liked=["Still Life", "Paintings"], n=4)
    assert queries == ["sunflowers vase", "Still Life Paintings", "impressionism still life",
                       "paintings Van Gogh"]
    assert s.queries_for({}, prompt="") == []


# ── Grounding: the picture is embedded ───────────────────────────────────

class StubEmbed:
    def encode(self, text):
        digest = hashlib.sha256(text.encode()).digest()
        vector = [digest[i] / 255.0 + 0.01 for i in range(DIM)]
        norm = sum(v * v for v in vector) ** 0.5
        return [v / norm for v in vector]


def _picture_vector(data, mime, caption):
    digest = hashlib.sha256(data + caption.encode()).digest()
    vector = [digest[i] / 255.0 + 0.01 for i in range(DIM)]
    norm = sum(v * v for v in vector) ** 0.5
    return [v / norm for v in vector]


@pytest.fixture
def grounding():
    embedding = EmbeddingVector(reference_name="stub/model",
                                source_type=EmbeddingSourceType.SENTENCE_TRANSFORMER,
                                api_key_name="NONE", dimensions=[DIM])
    pictures = []

    def embedder(data, mime, caption):
        pictures.append((data, mime, caption))
        return _picture_vector(data, mime, caption)

    def fetch(url):
        return None if "broken" in url else (url.encode(), "image/jpeg")

    instance = s.ImageGrounding(QdrantClient(":memory:"), "images", embedding,
                                vector_dimension=DIM, image_embedder=embedder, fetch=fetch)
    instance._embed = StubEmbed()
    instance.pictures = pictures
    return instance


def _hit(title, image, page):
    return s.ImageHit(title=title, image_url=image, page_url=page, provider="Open",
                      tags=["flower"])


def test_the_picture_and_its_caption_are_embedded(grounding):
    new = s.ground_images(grounding, [_hit("Sunflowers", "https://img.test/a.jpg",
                                           "https://open.test/a")])
    assert len(new) == 1
    data, mime, caption = grounding.pictures[0]
    assert data == b"https://img.test/a.jpg" and mime == "image/jpeg"
    assert caption.startswith("Sunflowers") and "flower" in caption
    point = grounding.client.scroll("images", with_vectors=True)[0][0]
    stored = point.vector if isinstance(point.vector, list) else next(iter(point.vector.values()))
    expected = _picture_vector(data, mime, caption)
    assert stored == pytest.approx(expected, abs=1e-6)
    assert "![Sunflowers](https://img.test/a.jpg)" in point.payload["markdown"]


def test_an_image_that_cannot_be_fetched_is_embedded_from_its_text(grounding):
    new = s.ground_images(grounding, [_hit("Broken", "https://broken.test/x.jpg",
                                           "https://open.test/x")])
    assert len(new) == 1 and grounding.pictures == []


def test_two_images_with_the_same_title_are_two_images(grounding):
    first = s.ground_images(grounding, [_hit("Sunflowers", "https://img.test/a.jpg",
                                             "https://open.test/a")])
    second = s.ground_images(grounding, [_hit("Sunflowers", "https://img.test/b.jpg",
                                              "https://museum.test/b"),
                                         _hit("Sunflowers", "https://img.test/a.jpg",
                                              "https://open.test/a")])
    assert len(first) == 1 and len(second) == 1
    assert grounding.count() == 2
