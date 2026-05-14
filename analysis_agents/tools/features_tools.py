# -*- coding: utf-8 -*-
from __future__ import annotations

"""
features_tools.py
------------------
Tools for the features_agent (Agent 3b of the web scraping pipeline).

Replaces the HuggingFace Space dependency with direct Scrapling HTTP fetching
(+ requests fallback), making the tool robust and independent of external
HF Space availability.
"""

import re
from datetime import datetime, timezone
from typing import Dict, List
from urllib.parse import urlparse

from smolagents import tool

# ---------------------------------------------------------------------------
# Shared store
# ---------------------------------------------------------------------------
FEATURES_DATA: Dict[str, List[Dict]] = {}

# ---------------------------------------------------------------------------
# Keywords that identify pages relevant to pricing / features / services
# ---------------------------------------------------------------------------
RELEVANT_KEYWORDS = [
    "pric", "precio", "plan", "tarifa", "fee", "cost", "coste",
    "suscri", "subscri", "feature", "servicio", "service", "product",
    "producto", "oferta", "offer", "compare", "compar", "package",
    "paquete", "tier", "rates", "tipo", "account", "cuenta", "membresia",
    "membership", "premium", "pro", "business", "enterprise", "free",
    "gratis", "trial",
]


def _is_relevant_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(kw in path for kw in RELEVANT_KEYWORDS)


def _section_label(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return "homepage"
    segment = path.split("/")[-1].replace("-", " ").replace("_", " ")
    return segment or "homepage"


def _clean(text: str, max_chars: int = 3000) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def _fetch_page_content(url: str) -> str:
    """
    Fetch page and extract visible text.
    Tries Scrapling first, falls back to requests + regex tag strip.
    """
    # Attempt 1: Scrapling
    try:
        from scrapling.fetchers import Fetcher
        page = Fetcher.get(url, stealthy_headers=True, timeout=20)
        if page is not None:
            for selector in ["main", "article", ".pricing", ".plans", ".features", "body"]:
                els = page.css(selector)
                if els:
                    text = " ".join(
                        re.sub(r"\s+", " ", el.text or "").strip()
                        for el in els[:5]
                        if el.text and len(el.text.strip()) > 50
                    )
                    if text:
                        return _clean(text)
            raw = page.get_all_text(ignore_tags=("script", "style", "nav", "footer")) or ""
            return _clean(raw)
    except Exception as e:
        print(f"[features] Scrapling error for {url}: {e}")

    # Attempt 2: plain requests
    try:
        import requests
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        html = resp.text
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<[^>]+>", " ", html)
        return _clean(html)
    except Exception as e:
        print(f"[features] requests fallback error for {url}: {e}")

    return ""


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def filter_relevant_urls(company_name: str, all_links: list) -> str:
    """
    From the full list of sitemap links for a company, keep only those
    whose URL path suggests pricing, features or services content.

    Args:
        company_name: Name of the company.
        all_links:    Full list of URL strings (from WEBSITE_LINKS[company_name]).

    Returns:
        A summary of how many URLs were kept, with the filtered list.
    """
    relevant = [u for u in all_links if _is_relevant_url(u)]
    if not relevant:
        return (
            f"[features] '{company_name}': ninguna URL del sitemap coincide con "
            f"palabras clave de precios/features. URLs totales: {len(all_links)}."
        )
    lines = [f"  · {u}" for u in relevant]
    return (
        f"[features] '{company_name}': {len(relevant)}/{len(all_links)} URLs relevantes.\n"
        + "\n".join(lines)
    )


@tool
def scrape_features_page(company_name: str, page_url: str) -> str:
    """
    Scrape a single features/pricing page using Scrapling (direct HTTP, no HF Space).
    The result is appended to FEATURES_DATA.

    Args:
        company_name: Company the URL belongs to.
        page_url:     URL to scrape.

    Returns:
        Confirmation with a preview of the extracted content.
    """
    content = _fetch_page_content(page_url)

    if not content:
        return f"[features] Sin contenido en '{page_url}' para '{company_name}'."

    record = {
        "page_url": page_url,
        "section_label": _section_label(page_url),
        "raw_content": content,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }
    FEATURES_DATA.setdefault(company_name, []).append(record)

    preview = content[:300] + ("..." if len(content) > 300 else "")
    return f"[features] '{company_name}' | {page_url}\n -> {preview}"


@tool
def scrape_all_features(company_links: dict) -> str:
    """
    For every company in company_links, filter relevant URLs and scrape them.
    Populates FEATURES_DATA.

    Args:
        company_links: Dict {company_name: [url1, url2, ...]} (= WEBSITE_LINKS).

    Returns:
        Summary with pages scraped per company and total records stored.
    """
    if not company_links:
        return "[features] El diccionario de enlaces está vacío. Ejecuta el agente de sitemap primero."

    summary_lines: List[str] = []
    for company_name, links in company_links.items():
        relevant = [u for u in links if _is_relevant_url(u)]
        # Fallback: if no relevant pages found, take the first 3 links
        if not relevant:
            relevant = links[:3]

        count_ok = 0
        for url in relevant:
            msg = scrape_features_page(company_name, url)
            if "Error" not in msg and "Sin contenido" not in msg:
                count_ok += 1

        summary_lines.append(
            f"  ✔ {company_name}: {count_ok}/{len(relevant)} páginas con contenido"
        )

    total = sum(len(v) for v in FEATURES_DATA.values())
    return (
        "Scraping de features completado:\n"
        + "\n".join(summary_lines)
        + f"\n\nTotal registros en FEATURES_DATA: {total}"
    )


@tool
def get_features_summary() -> str:
    """
    Return a table showing how many feature/pricing records were collected per company.

    Returns:
        Plain-text table or a message if FEATURES_DATA is empty.
    """
    if not FEATURES_DATA:
        return "[features] FEATURES_DATA está vacío."

    header = f"{'Empresa':<30} {'Páginas':>8}"
    sep = "-" * 42
    lines = [header, sep]
    for company, records in sorted(FEATURES_DATA.items()):
        lines.append(f"{company:<30} {len(records):>8}")
    lines.append(sep)
    lines.append(f"{'TOTAL':<30} {sum(len(v) for v in FEATURES_DATA.values()):>8}")
    return "\n".join(lines)


@tool
def list_features_companies() -> str:
    """
    List the companies already present in FEATURES_DATA.

    Returns:
        Formatted list or a message if empty.
    """
    if not FEATURES_DATA:
        return "[features] FEATURES_DATA está vacío. Ejecuta scrape_all_features primero."
    lines = [f"  · {name}: {len(recs)} páginas" for name, recs in FEATURES_DATA.items()]
    return "Empresas con features scrapeadas:\n" + "\n".join(lines)
