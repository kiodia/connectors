"""connectors.pdf_text: the abstract and title of a paper."""
import pytest

from connectors.pdf_text import abstract_of, pdf_abstract

PAPER = """Attention Is All You Need
Ashish Vaswani, Noam Shazeer
Abstract
The dominant sequence transduction models are based on complex recurrent or convo-
lutional neural networks. We propose a new simple network architecture, the
Transformer, based solely on attention mechanisms.
1 Introduction
Recurrent neural networks have been firmly established.
"""


def test_the_abstract_runs_from_its_heading_to_the_introduction():
    text = abstract_of(PAPER)
    assert text.startswith("The dominant sequence")
    assert "convolutional" in text                     # the hyphenated word is joined
    assert "Recurrent neural networks have been" not in text


def test_without_a_heading_the_opening_is_kept_and_cut_at_a_sentence():
    text = abstract_of("First sentence here. " * 200, max_chars=300)
    assert len(text) <= 300 and text.endswith(".")


def test_an_unreadable_pdf_reads_as_its_file_name(tmp_path):
    pytest.importorskip("pypdf")
    broken = tmp_path / "great_paper-v2.pdf"
    broken.write_bytes(b"not a pdf")
    summary = pdf_abstract(broken)
    assert summary.title == "great paper v2" and summary.abstract == ""
    assert summary.markdown() == "### great paper v2"


def test_a_real_pdf_gives_its_text(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_metadata({"/Title": "A Study of Blank Pages"})
    path = tmp_path / "blank.pdf"
    with open(path, "wb") as f:
        writer.write(f)
    summary = pdf_abstract(path)
    assert summary.title == "A Study of Blank Pages" and summary.pages == 1
