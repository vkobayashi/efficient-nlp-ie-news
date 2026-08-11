"""
Fetches headlines from major Dutch news outlets (public RSS feeds) and the
full article text for a selected story, for use as OneKE-style extraction
input -- mirroring OneKE's own "web news knowledge extraction" example
scenario (``examples/config`` in the OneKE repo).
"""

from __future__ import annotations

from dataclasses import dataclass

import feedparser
import requests
import trafilatura

REQUEST_TIMEOUT = 10
USER_AGENT = (
    "Mozilla/5.0 (compatible; OneKE-Dutch-News/1.0; "
    "+https://github.com/OpenSPG/OneKE)"
)

# Public RSS feeds of major Dutch news outlets. Feeds occasionally move; the
# UI lets a user paste a custom feed URL as a fallback, and a source that
# fails to parse is skipped with a warning rather than crashing the app.
NEWS_SOURCES: dict[str, str] = {
    "NOS - Algemeen nieuws": "https://feeds.nos.nl/nosnieuwsalgemeen",
    "NOS - Binnenland": "https://feeds.nos.nl/nosnieuwsbinnenland",
    "NOS - Buitenland": "https://feeds.nos.nl/nosnieuwsbuitenland",
    "NOS - Politiek": "https://feeds.nos.nl/nosnieuwspolitiek",
    "NOS - Economie": "https://feeds.nos.nl/nosnieuwseconomie",
    "NU.nl - Algemeen": "https://www.nu.nl/rss/Algemeen",
    "Telegraaf - Voorpagina": "https://www.telegraaf.nl/rss",
    "Volkskrant - Voorpagina": "https://www.volkskrant.nl/voorpagina/rss.xml",
    "NRC - Nieuws": "https://www.nrc.nl/rss/",
    "AD - Binnenland": "https://www.ad.nl/binnenland/rss.xml",
    "RTL Nieuws": "https://www.rtlnieuws.nl/rss.xml",
    "Financieele Dagblad": "https://fd.nl/?rss",
}


@dataclass
class Headline:
    title: str
    link: str
    published: str
    summary: str
    source: str


def fetch_headlines(source_name: str, feed_url: str | None = None, limit: int = 15) -> list[Headline]:
    """Parse an RSS feed into a list of Headline objects. Raises on total
    failure so the caller can decide how to surface it (we catch per-source
    in the Streamlit app so one broken feed doesn't take down the others)."""
    url = feed_url or NEWS_SOURCES[source_name]
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    parsed = feedparser.parse(resp.content)
    if parsed.bozo and not parsed.entries:
        raise ValueError(f"Could not parse feed for {source_name}: {parsed.bozo_exception}")

    headlines = []
    for entry in parsed.entries[:limit]:
        headlines.append(
            Headline(
                title=entry.get("title", "(no title)"),
                link=entry.get("link", ""),
                published=entry.get("published", entry.get("updated", "")),
                summary=_clean_summary(entry.get("summary", "")),
                source=source_name,
            )
        )
    return headlines


def _clean_summary(html_summary: str) -> str:
    if not html_summary:
        return ""
    text = trafilatura.extract(f"<html><body>{html_summary}</body></html>") or ""
    return text.strip() or html_summary


def fetch_article_text(url: str, fallback_summary: str = "") -> str:
    """Download and extract the main body text of an article using
    trafilatura (readability-style boilerplate removal). Falls back to the
    RSS summary if the full page can't be fetched or parsed -- articles
    behind a hard paywall commonly only yield the summary."""
    try:
        downloaded = trafilatura.fetch_url(url, no_ssl=True)
        if downloaded:
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
            )
            if text and len(text.strip()) > 200:
                return text.strip()
    except Exception:
        pass
    return fallback_summary
