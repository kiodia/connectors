"""Scrapers for MatchMake Angels.

Two interchangeable scraper implementations behind one interface
(:class:`connectors.scraper.scraper_base.Scraper`):

* :class:`connectors.scraper.crawl4ai_scraper.Crawl4AIScraper` — local
  crawling with the Crawl4AI library (https://github.com/unclecode/crawl4ai).
* :class:`connectors.scraper.gemini_url_context_scraper.GeminiUrlContextScraper`
  — remote extraction through the Google Gemini API "URL context" tool
  (https://ai.google.dev/gemini-api/docs/url-context).

Both return a :class:`connectors.scraper.scraper_base.ScrapedPage` and are
exposed to all Angels through :class:`api.angel_api.AngelAPI`.
"""

from connectors.scraper.scraper_base import ScrapedPage, Scraper
from connectors.scraper.crawl4ai_scraper import Crawl4AIScraper
from connectors.scraper.gemini_url_context_scraper import GeminiUrlContextScraper

__all__ = [
    "ScrapedPage",
    "Scraper",
    "Crawl4AIScraper",
    "GeminiUrlContextScraper",
]
