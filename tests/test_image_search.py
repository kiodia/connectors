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
    assert queries == ["sunflowers in a vase", "Still Life Paintings", "impressionism still life",
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


def test_the_typed_query_is_searched_as_it_was_typed():
    from connectors import image_search as s
    queries = s.queries_for({"keywords": ["impressionism"]},
                            prompt="  Mona Lisa as   modern art of the 20th century and beyond ")
    assert queries[0] == "Mona Lisa as modern art of the 20th century and beyond"


# ── Reverse image search ───────────────────────────────────────────────────

LENS = {"name": "Lens", "enabled": True, "input": "url", "auth": {"type": "query_param", "env": "LENS_KEY"},
        "endpoint": "https://lens.test/?url={image_url}&key=${LENS_KEY}",
        "result_sets": [
            {"results": "visual_matches",
             "fields": {"image_url": ["image.link", "thumbnail"], "thumbnail_url": "thumbnail",
                        "page_url": "link", "title": "title"}},
            {"results": "exact_matches",
             "fields": {"image_url": ["image.link", "thumbnail"], "page_url": "link", "title": "title"}}]}
VISION = {"name": "Vision", "enabled": True, "input": "bytes",
          "auth": {"type": "query_param", "env": "VISION_KEY"},
          "endpoint": "https://vision.test/annotate?key=${VISION_KEY}",
          "body": {"requests": [{"image": {"content": "{image_base64}"},
                                 "features": [{"type": "WEB_DETECTION", "maxResults": "{limit}"}]}]},
          "result_sets": [
              {"results": "responses.0.webDetection.pagesWithMatchingImages",
               "fields": {"image_url": ["fullMatchingImages.0.url", "partialMatchingImages.0.url"],
                          "page_url": "url",
                          "title": ["pageTitle", "root:responses.0.webDetection.bestGuessLabels.0.label"]}},
              {"results": "responses.0.webDetection.visuallySimilarImages",
               "fields": {"image_url": "url",
                          "title": "root:responses.0.webDetection.bestGuessLabels.0.label"}}]}
DESCRIBE = {"name": "Describe", "kind": "describe_then_search", "enabled": True, "input": "bytes",
            "auth": {"type": "sdk", "env": "ANTHROPIC_API_KEY"}}
LENS_ANSWER = {"visual_matches": [{"title": "Mona Lisa", "link": "https://l.test/page",
                                   "image": {"link": "https://l.test/m.jpg"}, "thumbnail": "https://t.test/m"}],
               "exact_matches": [{"title": "Louvre", "link": "https://louvre.test/",
                                  "thumbnail": "https://t.test/l"}]}
VISION_ANSWER = {"responses": [{"webDetection": {
    "bestGuessLabels": [{"label": "mona lisa"}],
    "pagesWithMatchingImages": [{"url": "https://w.test/page", "pageTitle": "<b>Mona</b> Lisa",
                                 "partialMatchingImages": [{"url": "https://w.test/m.jpg"}]}],
    "visuallySimilarImages": [{"url": "https://s.test/s.jpg"}]}}]}
EVERY_KEY = {"LENS_KEY": "k1", "VISION_KEY": "k2", "ANTHROPIC_API_KEY": "k3"}


def reverse_spec(*reverse, apis=()):
    return {"kind": "image_search", "apis": list(apis), "reverse_apis": list(reverse)}


def test_the_reverse_engines_follow_their_keys_and_what_they_take():
    both = reverse_spec(LENS, VISION, DESCRIBE)
    names = lambda apis: [a["name"] for a in apis]
    assert names(s.available_reverse_apis(both, "https://w.test/a.jpg", False, EVERY_KEY)) == ["Lens", "Vision"]
    assert names(s.available_reverse_apis(both, "", True, EVERY_KEY)) == ["Vision"]   # Lens needs a URL
    assert names(s.available_reverse_apis(both, "https://w.test/a.jpg", False, {"LENS_KEY": "k"})) == ["Lens"]
    assert s.describe_entry(both, EVERY_KEY)["name"] == "Describe"
    assert s.describe_entry(both, {}) is None


def test_a_text_search_never_asks_a_reverse_engine():
    calls = []
    s.search(reverse_spec(LENS, apis=[OPEN]), ["fox"], env=EVERY_KEY,
             fetch_json=lambda url, headers, timeout: calls.append(url) or {"results": []})
    assert calls == ["https://open.test/?q=fox&n=8"]


def test_a_url_engine_gets_the_image_url_and_reads_every_result_set():
    calls = []
    hits = s.reverse_search_api(LENS, image_url="https://w.test/a b.jpg", env=EVERY_KEY,
                                fetch_json=lambda url, headers, timeout: calls.append(url) or LENS_ANSWER)
    assert calls == ["https://lens.test/?url=https%3A%2F%2Fw.test%2Fa%20b.jpg&key=k1"]
    assert [(h.title, h.image_url, h.page_url, h.provider) for h in hits] == [
        ("Mona Lisa", "https://l.test/m.jpg", "https://l.test/page", "Lens"),
        ("Louvre", "https://t.test/l", "https://louvre.test/", "Lens")]   # no image: the thumbnail


def test_a_bytes_engine_posts_the_file_and_reads_the_root_label():
    import base64
    posted = []

    def post(url, headers, body, timeout):
        posted.append((url, body))
        return VISION_ANSWER
    hits = s.reverse_search_api(VISION, image_bytes=b"\xff\xd8pixels", limit=3, env=EVERY_KEY,
                                post_json=post)
    url, body = posted[0]
    request = body["requests"][0]
    assert url == "https://vision.test/annotate?key=k2"
    assert base64.b64decode(request["image"]["content"]) == b"\xff\xd8pixels"
    assert request["features"] == [{"type": "WEB_DETECTION", "maxResults": 3}]   # a number, not "3"
    assert [(h.title, h.image_url, h.page_url) for h in hits] == [
        ("Mona Lisa", "https://w.test/m.jpg", "https://w.test/page"),
        ("mona lisa", "https://s.test/s.jpg", "")]


LENS_UPLOAD = dict(LENS, upload={
    "url": "https://lens.test/image", "file_field": "image", "fields": {"api_key": "${LENS_KEY}"},
    "id": "image_id", "search": "https://lens.test/?image_id={image_id}&key=${LENS_KEY}"})


def test_a_url_engine_with_an_upload_takes_a_local_file():
    names = lambda apis: [a["name"] for a in apis]
    spec = reverse_spec(LENS_UPLOAD, VISION)
    assert names(s.available_reverse_apis(spec, "", True, EVERY_KEY)) == ["Lens", "Vision"]
    assert names(s.available_reverse_apis(reverse_spec(LENS), "", True, EVERY_KEY)) == []
    assert names(s.available_reverse_apis(spec, "", False, EVERY_KEY)) == []


def test_a_local_file_is_uploaded_then_searched_by_its_id():
    posted, calls = [], []

    def post_form(url, headers, fields, file_field, filename, data, mime, timeout):
        posted.append((url, fields, file_field, filename, data, mime))
        return {"message": "Image uploaded successfully.", "image_id": "abc/1"}
    hits = s.reverse_search_api(LENS_UPLOAD, image_bytes=b"\xff\xd8\xffpixels", env=EVERY_KEY,
                                fetch_json=lambda url, headers, timeout: calls.append(url) or LENS_ANSWER,
                                post_form=post_form)
    assert posted == [("https://lens.test/image", {"api_key": "k1"}, "image", "image.jpg",
                       b"\xff\xd8\xffpixels", "image/jpeg")]
    assert calls == ["https://lens.test/?image_id=abc%2F1&key=k1"]
    assert [h.title for h in hits] == ["Mona Lisa", "Louvre"]


def test_a_public_url_is_used_before_an_upload():
    calls = []
    s.reverse_search_api(LENS_UPLOAD, image_url="https://w.test/a.jpg", image_bytes=b"\xff\xd8\xffx",
                         env=EVERY_KEY, fetch_json=lambda url, h, t: calls.append(url) or LENS_ANSWER,
                         post_form=lambda *a: pytest.fail("uploaded although there is a URL"))
    assert calls[0].startswith("https://lens.test/?url=")


def test_a_refused_upload_leaves_the_engine_empty():
    hits = s.reverse_search_api(LENS_UPLOAD, image_bytes=b"\xff\xd8\xffx", env=EVERY_KEY,
                                fetch_json=lambda *a: pytest.fail("searched without an id"),
                                post_form=lambda *a: {"error": "Invalid image format."})
    assert hits == []


def _counting_form(uploads):
    def post_form(*args):
        uploads.append(args[0])
        return {"image_id": "id-1"}
    return post_form


def test_the_upload_waits_while_vision_finds_the_file():
    uploads = []
    hits = s.reverse_search(reverse_spec(LENS_UPLOAD, VISION), image_bytes=b"\xff\xd8\xffx",
                            env=EVERY_KEY, fetch_json=lambda *a: LENS_ANSWER,
                            post_json=lambda *a: VISION_ANSWER, post_form=_counting_form(uploads))
    assert uploads == [] and {h.provider for h in hits} == {"Vision"}


def test_the_file_is_uploaded_when_vision_finds_nothing():
    uploads = []
    hits = s.reverse_search(reverse_spec(LENS_UPLOAD, VISION), image_bytes=b"\xff\xd8\xffx",
                            env=EVERY_KEY, fetch_json=lambda *a: LENS_ANSWER,
                            post_json=lambda *a: {"responses": [{}]}, post_form=_counting_form(uploads))
    assert uploads == ["https://lens.test/image"] and {h.provider for h in hits} == {"Lens"}


def test_the_file_is_uploaded_when_vision_has_no_key():
    uploads = []
    hits = s.reverse_search(reverse_spec(LENS_UPLOAD, VISION), image_bytes=b"\xff\xd8\xffx",
                            env={"LENS_KEY": "k1"}, fetch_json=lambda *a: LENS_ANSWER,
                            post_json=lambda *a: pytest.fail("Vision has no key"),
                            post_form=_counting_form(uploads))
    assert uploads == ["https://lens.test/image"] and {h.provider for h in hits} == {"Lens"}


def test_a_public_url_keeps_lens_beside_vision_without_an_upload():
    uploads = []
    hits = s.reverse_search(reverse_spec(LENS_UPLOAD, VISION), image_url="https://w.test/a.jpg",
                            image_bytes=b"\xff\xd8\xffx", env=EVERY_KEY,
                            fetch_json=lambda *a: LENS_ANSWER, post_json=lambda *a: VISION_ANSWER,
                            post_form=_counting_form(uploads))
    assert uploads == [] and {h.provider for h in hits} == {"Lens", "Vision"}


def test_a_picture_too_big_for_the_upload_is_made_smaller():
    import io
    Image = pytest.importorskip("PIL.Image")
    import random
    rng = random.Random(1)
    image = Image.new("RGB", (1600, 1200))
    image.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256))
                   for _ in range(1600 * 1200)])
    out = io.BytesIO()
    image.save(out, format="PNG")
    assert len(out.getvalue()) > s.UPLOAD_MAX_BYTES
    data, mime = s.fit_upload(out.getvalue())
    assert mime == "image/jpeg" and len(data) <= s.UPLOAD_MAX_BYTES
    small = b"\x89PNG\r\n\x1a\n" + b"x" * 100
    assert s.fit_upload(small) == (small, "image/png")          # fits: sent as it is
    assert s.fit_upload(b"not a picture" * 50000) is None


def test_reverse_search_merges_the_engines():
    hits = s.reverse_search(reverse_spec(LENS, VISION), image_url="https://w.test/a.jpg",
                            image_bytes=b"\xff\xd8pixels", env=EVERY_KEY,
                            fetch_json=lambda *a: LENS_ANSWER, post_json=lambda *a: VISION_ANSWER)
    assert [h.provider for h in hits] == ["Lens", "Vision", "Lens", "Vision"]


def test_describe_then_search_is_the_fallback_for_a_local_file():
    described = []

    def describe(data, mime):
        described.append((data, mime))
        return "Mona Lisa Leonardo"
    calls = []
    hits = s.reverse_search(reverse_spec(LENS, DESCRIBE, apis=[OPEN]), image_bytes=b"\xff\xd8pixels",
                            env=EVERY_KEY, describe=describe,
                            fetch_json=lambda url, headers, timeout: calls.append(url) or
                            {"results": [{"url": "https://o.test/m.jpg", "title": "Mona"}]})
    assert described == [(b"\xff\xd8pixels", "image/jpeg")]
    assert calls == ["https://open.test/?q=Mona%20Lisa%20Leonardo&n=8"]
    assert [(h.title, h.query) for h in hits] == [("Mona", "Mona Lisa Leonardo")]


def test_no_engine_and_no_describer_finds_nothing():
    assert s.reverse_search(reverse_spec(LENS), image_bytes=b"\xff\xd8x", env=EVERY_KEY) == []
    off = dict(DESCRIBE, enabled=False)
    assert s.reverse_search(reverse_spec(off), image_bytes=b"\xff\xd8x", env=EVERY_KEY) == []
