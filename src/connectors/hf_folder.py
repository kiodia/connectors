"""Build a Hugging Face dataset folder out of a plain folder of files.

The counterpart of :mod:`connectors.hf_ingest`: this module *writes* the folder
that one ingests into Qdrant. What it produces is the shape both hosts already
consume — a ``metadata.jsonl`` at the root of a folder, one JSON object per
line, each carrying the ``file_name`` of the image or PDF that lies next to it.

**Generic where imdb and midjourney were specific.** The two folders MatchMake
watches today were each written by hand:
:meth:`connectors.imdb.IMDB.hf_extract` turns movie records into lines keyed
``file_name`` beside the posters, and
:meth:`connectors.midlib.Midlib.hf_download` scrolls a Qdrant collection and
writes a line per point beside the downloaded images. Both know their domain.
This module keeps their mechanics — drop a stale ``metadata.jsonl``, append one
``json.dump`` per line, keep ``file_name`` first so the HF ``ImageFolder``
loader finds it — and drops the domain: the input is *any* folder of files, and
the only thing read is the directory listing.

**Nothing is enriched.** :meth:`HFFolder.build` never opens a content file: no
text extraction, no summary, no embedding, no thumbnail. A record is what the
listing already says — the file's relative path, its position in the sorted
listing, its stem and whether it is an image or a PDF. Enrichment is somebody
else's job and happens later, on ingestion (``HFIngest`` embeds, keopy's
``wikio_ingest`` abstracts). Keeping the record this thin also matters
downstream: MatchMake's ``WatchDataspaces._extract_hf_dataset_features`` takes
*every* key of the first line as both a payload field and an embedded field, so
a field invented here would end up inside the vectors.

**The content is linked, not copied**, so building a dataset out of 4 GB of
scans costs a directory of links rather than a second copy — which is what
MatchMake's "a metadata.jsonl alongside the symlinked image or PDF content"
describes. Symlinks need a privilege Windows only grants in developer mode, so
:meth:`HFFolder.build` falls back to a hard link and then to a real copy, and
:attr:`FolderReport.link_mode` says which one answered. Passing no destination
writes the ``metadata.jsonl`` straight into the source folder instead, the way
imdb and midjourney do.

Host: MatchMake's "New dataspace" dialogue, the **Create a HF folder** kind of
``ui.dataspace_dialogue.dataspace_prompter.DataspacePrompter`` — the prompter's
folder picker gives the destination, and the source folder is the one the user
points at::

    report = HFFolder(source_folder, destination).build(progress=progress)
    if report.records:
        ...  # register the folder as a watch

Nothing in here is MatchMake-specific, so keopy can build the same folders for
the Angels it schedules.
"""
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import logging
log = logging.getLogger(__name__)


@dataclass
class FolderReport:
    """What :meth:`HFFolder.build` wrote, for the UI and the log.

    A bare file count would hide the two things a caller actually has to know:
    which files were left out — a folder of scans usually holds a stray
    ``.docx`` or ``Thumbs.db`` — and how the content was placed, since a copy
    means the dataset now weighs as much as the source where a link does not.
    """
    #: Folder the content was described from.
    source_folder: str
    #: Folder the dataset was written to (the source folder when in place).
    dataset_folder: str
    #: The ``metadata.jsonl`` that was written.
    metadata_path: str
    #: Lines written, one per content file.
    records: int = 0
    #: Records per media type, keyed ``image`` / ``pdf``.
    by_media_type: Dict[str, int] = field(default_factory=dict)
    #: Files passed over because they are neither an image nor a PDF.
    skipped: List[str] = field(default_factory=list)
    #: ``file_name`` values that appeared more than once — they would ground as
    #: one datapoint, the way two entities sharing a URL do.
    duplicates: List[str] = field(default_factory=list)
    #: How the content was placed: ``symlink``, ``hardlink`` or ``copy``. Empty
    #: when the dataset was built in place and nothing was placed at all.
    link_mode: str = ""
    #: True when the ``metadata.jsonl`` was written into the source folder.
    in_place: bool = False

    #: How each of :class:`HFFolder`'s link modes reads in a sentence.
    PLACED = {"symlink": "symlinked", "hardlink": "hard-linked",
              "copy": "copied"}

    def summary(self) -> str:
        """One line for the dialogue's status text."""
        if not self.records:
            return (f"No image or PDF found in {self.source_folder}. "
                    f"Nothing was written.")
        detail = ", ".join(f"{count} {kind}" for kind, count
                           in sorted(self.by_media_type.items()) if count)
        parts = [f"{self.records} file(s) described"]
        if detail:
            parts.append(f"({detail})")
        parts.append("in place" if self.in_place
                     else f"· {self.PLACED.get(self.link_mode, self.link_mode)}")
        if self.skipped:
            parts.append(f"· {len(self.skipped)} file(s) skipped")
        if self.duplicates:
            parts.append(f"· {len(self.duplicates)} duplicate name(s)")
        return " ".join(parts) + "."


class HFFolder:
    """Turn a folder of images and PDFs into a Hugging Face dataset folder.

    ``source_folder`` is what the user points at and is never modified, unless
    ``dataset_folder`` is left out — then the ``metadata.jsonl`` is written
    inside it, which is what imdb and midjourney do with folders they own.

    :meth:`build` is what callers use; :meth:`load` reads the result back
    through the ``datasets`` library, the way ``HFIngest`` will, and is the
    cheapest check that the folder came out loadable.
    """

    #: The file the HF loaders look for, and the one MatchMake detects a
    #: dataset folder by (``WatchDataspaces._extract_hf_dataset_features``).
    METADATA_FILE = "metadata.jsonl"

    #: Extensions taken as image content. Exactly the ones HF's ``ImageFolder``
    #: builder accepts and Pillow opens; anything else is left to the caller to
    #: rename rather than silently mislabelled.
    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff",
                      ".webp"}

    #: Extensions taken as document content.
    PDF_SUFFIXES = {".pdf"}

    #: Values of ``media_type``, and what ``link_mode`` may ask for.
    IMAGE = "image"
    PDF = "pdf"
    SYMLINK = "symlink"
    HARDLINK = "hardlink"
    COPY = "copy"

    #: Files that are never content, whatever their extension.
    IGNORED_NAMES = {METADATA_FILE, "Thumbs.db", ".DS_Store"}

    #: Files between two ``progress`` calls. A folder of scans runs to
    #: thousands, and the callback repaints a UI.
    PROGRESS_EVERY = 25

    def __init__(self, source_folder, dataset_folder=None, recursive: bool = True,
                 link_mode: str = SYMLINK):
        """
        Args:
            source_folder: The folder of images and/or PDFs to describe.
            dataset_folder: Where the dataset is written. Left out (or equal to
                the source), the ``metadata.jsonl`` is written into the source
                folder and no content is placed.
            recursive: Walk sub-folders too. ``file_name`` then holds the path
                relative to the dataset root, which is what the HF loaders
                resolve it against.
            link_mode: How content is placed in a separate ``dataset_folder``:
                ``symlink`` (the default), ``hardlink`` or ``copy``. The first
                two fall back to the next when the filesystem refuses them.
        """
        self.source_folder = Path(source_folder)
        self.recursive = recursive
        if link_mode not in (self.SYMLINK, self.HARDLINK, self.COPY):
            raise ValueError(f"Unknown link_mode '{link_mode}': expected "
                             f"'{self.SYMLINK}', '{self.HARDLINK}' or '{self.COPY}'.")
        self.link_mode = link_mode

        if dataset_folder is None:
            self.dataset_folder = self.source_folder
        else:
            self.dataset_folder = Path(dataset_folder)
        # Resolving both is what makes "the destination is the source" true for
        # a relative path, a trailing slash or a symlinked drive as well.
        self.in_place = (self._resolved(self.dataset_folder)
                         == self._resolved(self.source_folder))
        self.metadata_path = self.dataset_folder / self.METADATA_FILE

        log.info(f"HFFolder: '{self.source_folder}' -> '{self.dataset_folder}'"
                 f"{' (in place)' if self.in_place else f' ({self.link_mode})'}")

    # ── Building ──────────────────────────────────────────────────────────

    def build(self, progress: Optional[Callable[[str], None]] = None) -> FolderReport:
        """Write the ``metadata.jsonl``, placing the content next to it.

        A stale ``metadata.jsonl`` is removed first rather than appended to, so
        rebuilding a folder whose content changed does not leave the lines of
        files that are gone — the same thing ``IMDB.hf_extract`` and
        ``Midlib.hf_download`` do before they start writing.

        Args:
            progress: Called with a one-line status every
                :attr:`PROGRESS_EVERY` files, for a UI to follow a long folder.

        Returns:
            FolderReport: what was written. ``records == 0`` means the folder
            held no image and no PDF; the caller decides what to say about it,
            nothing is raised.
        """
        if not self.source_folder.is_dir():
            raise FileNotFoundError(f"Source folder not found: {self.source_folder}")

        content, skipped = self.content_files()
        report = FolderReport(source_folder=str(self.source_folder),
                              dataset_folder=str(self.dataset_folder),
                              metadata_path=str(self.metadata_path),
                              skipped=skipped, in_place=self.in_place)
        if not content:
            log.warning(f"No image or PDF in '{self.source_folder}' — "
                        f"nothing written")
            return report

        self.dataset_folder.mkdir(parents=True, exist_ok=True)
        if self.metadata_path.exists():
            log.info(f"Removing the previous {self.metadata_path}")
            os.remove(self.metadata_path)

        seen = set()
        total = len(content)
        with open(self.metadata_path, "a", encoding="utf-8") as f:
            for index, source_file in enumerate(content):
                file_name = self.relative_name(source_file)

                # A name written twice would be one line overwriting the other's
                # content: worth reporting, not worth stopping for.
                if file_name in seen:
                    report.duplicates.append(file_name)
                    log.warning(f"Duplicated file name in the dataset: {file_name}")
                seen.add(file_name)

                if not self.in_place:
                    mode = self._place(source_file, self.dataset_folder / file_name)
                    report.link_mode = mode

                record = self.record(index, file_name, source_file)
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")

                report.records += 1
                kind = record["media_type"]
                report.by_media_type[kind] = report.by_media_type.get(kind, 0) + 1

                if progress and (index % self.PROGRESS_EVERY == 0
                                 or index == total - 1):
                    progress(f"Describing {index + 1}/{total}: {file_name}")

        log.info(f"{self.metadata_path} written — {report.summary()}")
        return report

    def content_files(self) -> Tuple[List[Path], List[str]]:
        """The files to describe, sorted, and the names of the ones left out.

        Sorted because the order is the dataset's: it fixes the ``id`` of every
        record, so rebuilding an unchanged folder rewrites the same lines.
        """
        pattern = "**/*" if self.recursive else "*"
        content, skipped = [], []
        for path in sorted(self.source_folder.glob(pattern)):
            if not path.is_file() or path.name in self.IGNORED_NAMES:
                continue
            if self.media_type(path):
                content.append(path)
            else:
                skipped.append(self.relative_name(path))
        if skipped:
            log.info(f"{len(skipped)} file(s) skipped as neither image nor PDF: "
                     f"{', '.join(skipped[:5])}"
                     f"{' ...' if len(skipped) > 5 else ''}")
        return content, skipped

    def record(self, index: int, file_name: str, path: Path) -> dict:
        """One ``metadata.jsonl`` line: what the listing says, and no more.

        ``file_name`` comes first because that is the key HF's folder builders
        consume to attach the content (``IMDB.hf_extract`` moves it to the
        front for the same reason), ``id`` is the position in the sorted
        listing, ``name`` the file's stem as it stands — untouched, so a caller
        can read a title out of it without this module having invented one —
        and ``media_type`` is what tells a viewer to render an image or a PDF.
        """
        return {
            "file_name": file_name,
            "id": index,
            "name": path.stem,
            "media_type": self.media_type(path),
        }

    def relative_name(self, path: Path) -> str:
        """A path relative to the dataset root, with forward slashes.

        The HF loaders resolve ``file_name`` against the folder holding the
        ``metadata.jsonl``, and they do it with ``/`` on every platform — a
        Windows ``sub\\scan.pdf`` would not resolve.
        """
        return path.relative_to(self.source_folder).as_posix()

    @classmethod
    def media_type(cls, path: Path) -> str:
        """``image``, ``pdf``, or empty for a file this does not describe."""
        suffix = path.suffix.lower()
        if suffix in cls.IMAGE_SUFFIXES:
            return cls.IMAGE
        if suffix in cls.PDF_SUFFIXES:
            return cls.PDF
        return ""

    # ── Placing the content ───────────────────────────────────────────────

    def _place(self, source_file: Path, target_file: Path) -> str:
        """Put one content file in the dataset folder; return the mode used.

        Tried in order of what costs least, starting at the requested mode: a
        symlink, then a hard link, then a real copy. Windows refuses symlinks
        without developer mode and hard links across volumes, and a dataset
        that cannot be built at all would be a poor answer to either.
        """
        target_file.parent.mkdir(parents=True, exist_ok=True)
        # Rebuilding must not keep a link to content that has since moved.
        if target_file.is_symlink() or target_file.exists():
            target_file.unlink()

        attempts = {self.SYMLINK: (self.SYMLINK, self.HARDLINK, self.COPY),
                    self.HARDLINK: (self.HARDLINK, self.COPY),
                    self.COPY: (self.COPY,)}[self.link_mode]

        for mode in attempts:
            try:
                if mode == self.SYMLINK:
                    os.symlink(source_file, target_file)
                elif mode == self.HARDLINK:
                    os.link(source_file, target_file)
                else:
                    shutil.copy2(source_file, target_file)
                return mode
            except (OSError, NotImplementedError) as exc:
                log.info(f"{mode} refused for {source_file.name} ({exc}); "
                         f"trying the next mode")
        raise OSError(f"Could not place {source_file} in {self.dataset_folder}")

    # ── Reading it back ───────────────────────────────────────────────────

    def load(self):
        """Load the built folder with ``datasets``, the way ``HFIngest`` does.

        The folder builders cover the media they know: ``ImageFolder`` picks a
        folder of images up, and PDFs need a ``datasets`` recent enough to
        carry ``PdfFolder`` (and its extra installed). When the folder loader
        declines, the ``metadata.jsonl`` is read as plain JSON instead — the
        records are the same, only the content column is missing, which is all
        a format check needs.

        Returns:
            The loaded ``DatasetDict``.
        """
        # Lazy on purpose: importing `datasets` before torch breaks torch's DLL
        # initialization on Windows (see connectors.arxiv).
        from datasets import load_dataset

        try:
            dataset = load_dataset(str(self.dataset_folder))
            log.info(f"Loaded as a folder dataset: {dataset}")
        except Exception as exc:  # noqa: BLE001 - any loader refusal falls back
            log.info(f"The folder loader declined ({exc}); reading "
                     f"{self.METADATA_FILE} as JSON instead")
            dataset = load_dataset("json", data_files=str(self.metadata_path))
            log.info(f"Loaded as a JSON dataset: {dataset}")
        return dataset

    @staticmethod
    def _resolved(path: Path) -> Path:
        """``Path.resolve`` that survives a folder that does not exist yet."""
        try:
            return path.resolve()
        except OSError:
            return path.absolute()
