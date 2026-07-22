from connectors.scraper.scraper_base import Scraper, ScrapedPage

import logging
log = logging.getLogger(__name__)


class Crawl4AIScraper(Scraper):
    """Scraper backed by the Crawl4AI library (https://github.com/unclecode/crawl4ai).

    Crawl4AI drives a local headless browser and returns the page as markdown,
    which is exactly the shape Angels feed into a Dataspace. The library is
    imported lazily so MatchMake starts fine when crawl4ai is not installed;
    the scrape then fails with an explanatory :class:`ScrapedPage` error.

    Install with: ``pip install crawl4ai`` (then ``crawl4ai-setup`` for the
    browser binaries).
    """

    name = "crawl4ai"

    @staticmethod
    def _english_configs():
        """Configs asking the site for English, or ``(None, None)`` when this
        crawl4ai version does not offer them.

        A multilingual site that negotiates server-side answers in English
        when the request carries ``Accept-Language`` — so preferring English
        costs no extra fetch (see :mod:`connectors.scraper.language`).
        """
        try:
            from crawl4ai import BrowserConfig, CrawlerRunConfig
            from connectors.scraper.language import ACCEPT_LANGUAGE, LOCALE
        except ImportError:
            return None, None
        try:
            return (BrowserConfig(headers={"Accept-Language": ACCEPT_LANGUAGE}),
                    CrawlerRunConfig(locale=LOCALE))
        except TypeError as exc:  # older signature without headers/locale
            log.warning("Crawl4AIScraper - cannot request English (%s); "
                        "scraping with the site's default language", exc)
            return None, None

    def scrape(self, url: str) -> ScrapedPage:
        try:
            from crawl4ai import AsyncWebCrawler
        except ImportError as exc:
            log.error("Crawl4AIScraper - crawl4ai is not installed: %s", exc)
            return self._failure(url, "crawl4ai is not installed (pip install crawl4ai)")

        browser_config, run_config = self._english_configs()

        async def _crawl():
            if browser_config is None:
                async with AsyncWebCrawler() as crawler:
                    return await crawler.arun(url=url)
            async with AsyncWebCrawler(config=browser_config) as crawler:
                return await crawler.arun(url=url, config=run_config)

        try:
            result = self._run_async(_crawl())
        except Exception as exc:  # noqa: BLE001 - the Angel decides what to do with a failure
            log.error("Crawl4AIScraper - crawl of %s failed: %s", url, exc)
            return self._failure(url, str(exc))

        return self._to_page(url, result)

    def _to_page(self, url: str, result) -> ScrapedPage:
        """Normalize a crawl4ai CrawlResult into a ScrapedPage."""
        markdown = getattr(result, "markdown", "") or ""
        # newer crawl4ai versions wrap markdown in a MarkdownGenerationResult
        raw = getattr(markdown, "raw_markdown", None)
        if raw is not None:
            markdown = raw

        metadata = getattr(result, "metadata", None) or {}
        title = str(metadata.get("title") or "")

        links = []
        for group in (getattr(result, "links", None) or {}).values():
            for link in group or []:
                href = link.get("href") if isinstance(link, dict) else str(link)
                if href:
                    links.append(href)

        media = []
        for group in (getattr(result, "media", None) or {}).values():
            for item in group or []:
                src = item.get("src") if isinstance(item, dict) else str(item)
                if src:
                    media.append(src)

        success = bool(getattr(result, "success", True))
        error = str(getattr(result, "error_message", "") or "")
        # A fetch that "succeeded" can still be an HTTP error page (e.g. a
        # Cloudflare 522 interstitial): treat status >= 400 as a failure so
        # angels fall back instead of feeding error pages to their Dataspace.
        status_code = getattr(result, "status_code", None)
        if success and status_code is not None and int(status_code) >= 400:
            success = False
            error = f"HTTP {status_code}"
        return ScrapedPage(
            url=url,
            title=title,
            markdown=str(markdown),
            links=links,
            media=media,
            fetched_at=self._now(),
            scraper=self.name,
            success=success,
            error=error,
        )
