"""HFFolder against the two kinds of content it is meant to describe.

Each case is one real file — ``tests/assets/single_image/blue_square.png`` and
``tests/assets/single_pdf/blank_note.pdf`` — because one file is enough to pin
the whole contract: the ``metadata.jsonl`` is written next to the content, it
holds one line per file, ``file_name`` comes first (HF's folder builders
consume that key to attach the content), and the record says nothing the
directory listing did not already say. A folder of a thousand scans differs
only in how many times that line is repeated.

Both cases build into ``tmp_path`` rather than in place, which is what the
"New dataspace" dialogue does with the folder its picker returned — and it is
also what keeps the assets read-only: a build in place would write a
``metadata.jsonl`` into the repository.

Nothing here needs the ``datasets`` library. Where it happens to be installed,
:func:`assert_loads_back` additionally reads the folder back the way
``HFIngest`` will.
"""
import json
from importlib.util import find_spec
from pathlib import Path

from connectors.hf_folder import HFFolder

ASSETS = Path(__file__).parent / "assets"


def read_records(metadata_path):
    """The ``metadata.jsonl`` as a list of dicts, one per line."""
    lines = metadata_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def assert_loads_back(builder, expected_rows):
    """Read the built folder back through ``datasets``, where it is installed.

    The shape of the folder is asserted on every machine by the tests
    themselves; this is the extra check that HF actually accepts what was
    written, and it is skipped silently rather than skipping its whole test
    when the optional dependency is absent.
    """
    if not find_spec("datasets"):
        return
    dataset = builder.load()
    assert dataset["train"].num_rows == expected_rows


def test_a_folder_of_one_image_becomes_a_dataset(tmp_path):
    """The image case: midjourney's folder, built from a listing instead."""
    source = ASSETS / "single_image"
    destination = tmp_path / "images"

    builder = HFFolder(source, destination)
    report = builder.build()

    assert report.records == 1
    assert report.by_media_type == {"image": 1}
    assert report.skipped == []
    assert report.duplicates == []
    assert report.in_place is False
    assert report.link_mode in (HFFolder.SYMLINK, HFFolder.HARDLINK, HFFolder.COPY)

    records = read_records(destination / HFFolder.METADATA_FILE)
    assert records == [{"file_name": "blue_square.png", "id": 0,
                        "name": "blue_square", "media_type": "image"}]
    # The HF folder builders attach the content by the first key.
    assert list(records[0])[0] == "file_name"

    # The content is next to its metadata, whether linked or copied.
    placed = destination / "blue_square.png"
    assert placed.is_file()
    assert placed.read_bytes() == (source / "blue_square.png").read_bytes()

    # The source folder is an input, not a workspace.
    assert not (source / HFFolder.METADATA_FILE).exists()

    assert_loads_back(builder, 1)


def test_a_folder_of_one_pdf_becomes_a_dataset(tmp_path):
    """The PDF case: same folder, same line, only the media type differs."""
    source = ASSETS / "single_pdf"
    destination = tmp_path / "documents"

    builder = HFFolder(source, destination)
    report = builder.build()

    assert report.records == 1
    assert report.by_media_type == {"pdf": 1}
    assert report.skipped == []
    assert report.duplicates == []
    assert report.in_place is False
    assert report.link_mode in (HFFolder.SYMLINK, HFFolder.HARDLINK, HFFolder.COPY)

    records = read_records(destination / HFFolder.METADATA_FILE)
    assert records == [{"file_name": "blank_note.pdf", "id": 0,
                        "name": "blank_note", "media_type": "pdf"}]
    assert list(records[0])[0] == "file_name"

    placed = destination / "blank_note.pdf"
    assert placed.is_file()
    assert placed.read_bytes() == (source / "blank_note.pdf").read_bytes()

    assert not (source / HFFolder.METADATA_FILE).exists()

    # A PDF folder loads through PdfFolder on a recent `datasets`, and through
    # the JSON fallback on an older one — either way, one row.
    assert_loads_back(builder, 1)
