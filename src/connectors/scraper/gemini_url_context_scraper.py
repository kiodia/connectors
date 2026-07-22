import os

from dotenv import load_dotenv

from connectors.scraper.scraper_base import Scraper, ScrapedPage

import logging
log = logging.getLogger(__name__)


class GeminiUrlContextScraper(Scraper):
    """Scraper backed by the Google Gemini API "URL context" tool.

    Instead of crawling locally, the URL is handed to Gemini with the
    ``url_context`` tool enabled and the model returns the page content as
    markdown (https://ai.google.dev/gemini-api/docs/url-context). This gives
    Angels a scraper that needs no local browser, at the cost of a Gemini API
    key (``GEMINI_API_KEY`` or ``GOOGLE_API_KEY`` in the environment / .env).

    The ``google-genai`` SDK is imported lazily so MatchMake starts fine when
    it is not installed; the scrape then fails with an explanatory
    :class:`ScrapedPage` error. Install with: ``pip install google-genai``.
    """

    name = "gemini-url-context"

    DEFAULT_MODEL = "gemini-2.5-flash"

    DEFAULT_INSTRUCTION = (
        "Extract the full content of {url} and return it as clean markdown. "
        "Keep every title, date, time, place and link you find; do not summarize."
    )

    def __init__(self, model: str = None, api_key: str = None):
        load_dotenv()
        self.model = model or self.DEFAULT_MODEL
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    def scrape(self, url: str, instruction: str = None) -> ScrapedPage:
        """Extract one URL through the Gemini URL-context tool.

        ``instruction`` overrides the default "return the page as markdown"
        request, so an Angel can ask directly for what it needs (e.g. "list
        the movies playing this week with their showtimes").
        """
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            log.error("GeminiUrlContextScraper - google-genai is not installed: %s", exc)
            return self._failure(url, "google-genai is not installed (pip install google-genai)")

        if not self.api_key:
            return self._failure(url, "no Gemini API key (set GEMINI_API_KEY or GOOGLE_API_KEY)")

        prompt = (instruction or self.DEFAULT_INSTRUCTION).format(url=url)
        # the URL itself must appear in the prompt for the tool to fetch it
        if url not in prompt:
            prompt = f"{prompt}\n\nURL: {url}"

        try:
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(url_context=types.UrlContext())],
                ),
            )
        except Exception as exc:  # noqa: BLE001 - the Angel decides what to do with a failure
            log.error("GeminiUrlContextScraper - request for %s failed: %s", url, exc)
            return self._failure(url, str(exc))

        return self._to_page(url, response)

    def _to_page(self, url: str, response) -> ScrapedPage:
        """Normalize a Gemini generate_content response into a ScrapedPage."""
        markdown = getattr(response, "text", "") or ""

        # url_context_metadata reports the retrieval status per fetched URL
        success, error = True, ""
        try:
            candidate = (getattr(response, "candidates", None) or [None])[0]
            metadata = getattr(candidate, "url_context_metadata", None)
            for entry in getattr(metadata, "url_metadata", None) or []:
                status = str(getattr(entry, "url_retrieval_status", ""))
                if status and "SUCCESS" not in status.upper():
                    success = False
                    error = f"URL retrieval status: {status}"
        except Exception:  # noqa: BLE001 - metadata is informative only
            pass

        if not markdown and success:
            success, error = False, "Gemini returned an empty response"

        return ScrapedPage(
            url=url,
            markdown=markdown,
            fetched_at=self._now(),
            scraper=self.name,
            success=success,
            error=error,
        )
