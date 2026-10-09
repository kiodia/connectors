"""The text of a PDF, and its title and abstract.

What a host needs to show a dropped paper as an item before anything else is
known about it: :func:`pdf_abstract` reads the first pages, takes the title
from the metadata (else the first substantial line) and the abstract from the
"Abstract" section up to the introduction (else the opening paragraphs).

Requires ``pypdf`` (the ``pdf`` extra).

@author: vankomme
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import logging
log = logging.getLogger(__name__)

#: Where an abstract ends: the first section after it.
_ABSTRACT_END = re.compile(
    r"\n\s*(?:(?:1|I)\.?\s+)?(?:Introduction|Keywords|Index Terms|Background|Contents)\b",
    re.IGNORECASE)
_ABSTRACT_START = re.compile(r"\bAbstract\b[\s:.\-—–]*", re.IGNORECASE)


@dataclass
class PdfSummary:
    title: str
    abstract: str
    pages: int

    def markdown(self) -> str:
        """The paper as an item's markdown: the title as a heading, the abstract."""
        heading = f"### {self.title}\n\n" if self.title else ""
        return f"{heading}{self.abstract}".strip()


def pdf_text(path: str | Path, max_pages: int = 2) -> str:
    """The text of the first ``max_pages`` pages ("" when it cannot be read)."""
    from pypdf import PdfReader
    try:
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages[:max_pages])
    except Exception as ex:  # noqa: BLE001 - a broken PDF reads as nothing
        log.warning("pdf_text - cannot read %s: %s", path, ex)
        return ""


def pdf_abstract(path: str | Path, max_chars: int = 2000) -> PdfSummary:
    """The title and the abstract of the PDF at ``path``."""
    from pypdf import PdfReader
    title, pages, text = "", 0, ""
    try:
        reader = PdfReader(str(path))
        pages = len(reader.pages)
        meta_title = (reader.metadata.title if reader.metadata else "") or ""
        title = meta_title.strip()
        text = "\n".join((page.extract_text() or "") for page in reader.pages[:2])
    except Exception as ex:  # noqa: BLE001 - a broken PDF reads as its file name
        log.warning("pdf_abstract - cannot read %s: %s", path, ex)
    return PdfSummary(title=_title(title, text, path), abstract=abstract_of(text, max_chars),
                      pages=pages)


def abstract_of(text: str, max_chars: int = 2000) -> str:
    """The abstract in a paper's text: from "Abstract" to the next section,
    else the opening paragraphs; at most ``max_chars``, cut at a sentence."""
    text = (text or "").replace("\r", "")
    start = _ABSTRACT_START.search(text)
    if start:
        body = text[start.end():]
        end = _ABSTRACT_END.search(body)
        body = body[:end.start()] if end else body
    else:
        body = text
    body = re.sub(r"-\n(?=[a-z])", "", body)              # words hyphenated over a line
    body = re.sub(r"\s*\n\s*", " ", body).strip()
    if len(body) > max_chars:
        cut = body[:max_chars]
        stop = cut.rfind(". ")
        body = cut[:stop + 1] if stop > max_chars // 2 else cut.rstrip() + "…"
    return body


def _title(meta_title: str, text: str, path: str | Path) -> str:
    if meta_title and not re.fullmatch(r"(untitled|microsoft word.*|.*\.(docx?|tex|pdf))",
                                       meta_title, re.IGNORECASE):
        return meta_title
    for line in (text or "").splitlines():
        line = line.strip()
        if len(line) >= 12 and not _ABSTRACT_START.match(line) and not line.lower().startswith("arxiv"):
            return line
    return Path(path).stem.replace("_", " ").replace("-", " ")
