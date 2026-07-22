import asyncio
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import List

from pydantic import BaseModel, Field

import logging
log = logging.getLogger(__name__)


class ScrapedPage(BaseModel):
    """The normalized result of scraping one URL.

    Whatever backend produced it (Crawl4AI locally or the Gemini URL-context
    tool remotely), an Angel always receives the page in this shape: markdown
    content plus the links and media found on the page.
    """
    url: str = Field(..., description="The URL that was scraped")
    title: str = Field(default="", description="Page title when the backend provides one")
    markdown: str = Field(default="", description="Page content as markdown")
    links: List[str] = Field(default_factory=list, description="Links found on the page")
    media: List[str] = Field(default_factory=list, description="Image/video references found on the page")
    fetched_at: str = Field(default="", description="ISO timestamp of the scrape")
    scraper: str = Field(default="", description="Name of the scraper backend used")
    success: bool = Field(default=True, description="False when the scrape failed")
    error: str = Field(default="", description="Error message when success is False")


class Scraper(ABC):
    """Common interface of the MatchMake scrapers.

    Concrete backends: :class:`connectors.scraper.crawl4ai_scraper.Crawl4AIScraper`
    and :class:`connectors.scraper.gemini_url_context_scraper.GeminiUrlContextScraper`.
    Angels obtain an instance via :meth:`api.angel_api.AngelAPI.get_scraper`.
    """

    #: Backend identifier, e.g. "crawl4ai" or "gemini-url-context"
    name: str = "scraper"

    @abstractmethod
    def scrape(self, url: str) -> ScrapedPage:
        """Scrape a single URL and return the normalized page."""

    def scrape_many(self, urls: List[str]) -> List[ScrapedPage]:
        """Scrape several URLs sequentially; failures never raise, they are
        returned as unsuccessful :class:`ScrapedPage` entries."""
        pages = []
        for url in urls:
            try:
                pages.append(self.scrape(url))
            except Exception as exc:  # noqa: BLE001 - one bad URL must not stop the batch
                log.error("%s - scraping %s failed: %s", self.name, url, exc)
                pages.append(self._failure(url, str(exc)))
        return pages

    def scrape_english_first(self, url: str) -> ScrapedPage:
        """Scrape ``url``, preferring the site's English rendition.

        Every scrape already asks for English through ``Accept-Language``, so
        a site that negotiates the language server-side answers in English on
        the first fetch and this costs nothing extra. This additionally covers
        the sites that put each language on its own URL: when the fetched page
        links to the English counterpart of ``url``
        (:func:`connectors.scraper.language.english_variant`), that one is
        fetched and returned instead.

        Never worse than :meth:`scrape`: the original page is returned
        whenever it is already English, offers no English link, or the second
        fetch fails.
        """
        from connectors.scraper.language import english_variant

        page = self.scrape(url)
        if not page.success:
            return page

        variant = english_variant(url, page.links)
        if not variant:
            return page

        log.info("%s - %s offers an English version, using %s", self.name, url, variant)
        try:
            english = self.scrape(variant)
        except Exception as exc:  # noqa: BLE001 - keep the page we already have
            log.warning("%s - English version %s failed (%s), keeping %s",
                        self.name, variant, exc, url)
            return page
        if english.success and english.markdown:
            return english
        log.warning("%s - English version %s returned nothing, keeping %s",
                    self.name, variant, url)
        return page

    def _failure(self, url: str, error: str) -> ScrapedPage:
        return ScrapedPage(url=url, success=False, error=error,
                           scraper=self.name, fetched_at=self._now())

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _run_async(coro):
        """Run an async coroutine from sync code.

        Works both from a plain thread (asyncio.run) and from inside a running
        event loop (a Flet handler): in the latter case the coroutine runs in
        its own thread with a fresh loop, so the caller's loop is not blocked
        by a nested run.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        holder = {}

        def _worker():
            try:
                holder["result"] = asyncio.run(coro)
            except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
                holder["error"] = exc

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        thread.join()
        if "error" in holder:
            raise holder["error"]
        return holder.get("result")
