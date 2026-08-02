from rag_assistant.scraping.fetchers.base import Fetcher
from rag_assistant.scraping.fetchers.browser_fetcher import BrowserFetcher
from rag_assistant.scraping.fetchers.factory import create_fetcher, register_fetcher
from rag_assistant.scraping.fetchers.http_fetcher import HttpFetcher

__all__ = [
    "BrowserFetcher",
    "Fetcher",
    "HttpFetcher",
    "create_fetcher",
    "register_fetcher",
]
