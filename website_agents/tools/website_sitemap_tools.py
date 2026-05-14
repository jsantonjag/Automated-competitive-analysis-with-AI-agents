# -*- coding: utf-8 -*-
from __future__ import annotations
"""
website_sitemap_tools.py
-------------------------
Tools for agent 2: extract all internal links from each official website.

Strategy (in order of preference):
  1. Parse sitemap.xml (or sitemap_index.xml) directly — fast and reliable.
  2. HuggingFace Space "All links" endpoint (if available).
  3. Scrapling home-page crawl — extract <a href> links from the homepage.

Shared store
-------------
WEBSITE_LINKS: Dict[str, List[str]] = {
    "Revolut": [
        "https://www.revolut.com/pricing",
        ...
    ],
    ...
}
"""

import re
from typing import Dict, List
from urllib.parse import urljoin, urlparse
from smolagents import tool

# Shared store
WEBSITE_LINKS: Dict[str, List[str]] = {}


def _base_url(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


def _is_internal(link: str, base: str) -> bool:
    return link.startswith(base) or link.startswith("/")


def _absolute(link: str, base: str) -> str:
    if link.startswith("http"):
        return link
    return urljoin(base, link)


def _dedupe(urls: List[str]) -> List[str]:
    seen, result = set(), []
    for u in urls:
        u = u.rstrip(".,;)")
        if u not in seen and u.startswith("http"):
            seen.add(u)
            result.append(u)
    return result


def _fetch_sitemap(base_url: str) -> List[str]:
    """Try to parse sitemap.xml / sitemap_index.xml and return URLs."""
    import requests
    links: List[str] = []
    for path in ["/sitemap.xml", "/sitemap_index.xml", "/sitemap-index.xml"]:
        try:
            resp = requests.get(
                base_url.rstrip("/") + path,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
            )
            if resp.status_code == 200 and "xml" in resp.headers.get("content-type", ""):
                found = re.findall(r"<loc>(https?://[^<]+)</loc>", resp.text)
                links.extend(found)
                if links:
                    return _dedupe(links)
        except Exception:
            pass
    return []


def _fetch_homepage_links(base_url: str) -> List[str]:
    """Scrape all <a href> links from the homepage as a last resort."""
    links: List[str] = []
    base = _base_url(base_url)
    try:
        from scrapling.fetchers import Fetcher
        page = Fetcher.get(base_url, stealthy_headers=True, timeout=20)
        if page:
            for a in page.css("a"):
                href = a.attrib.get("href", "")
                if href and not href.startswith(("#", "mailto:", "tel:", "javascript:")):
                    absolute = _absolute(href, base)
                    if _is_internal(href, base):
                        links.append(absolute)
            return _dedupe(links)
    except Exception as e:
        print(f"[sitemap] Scrapling fallback error for {base_url}: {e}")

    # Plain requests fallback
    try:
        import requests
        resp = requests.get(base_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        hrefs = re.findall(r'href=["\']([^"\']+)["\']', resp.text)
        for href in hrefs:
            if href and not href.startswith(("#", "mailto:", "tel:", "javascript:")):
                absolute = _absolute(href, base)
                if _is_internal(href, base):
                    links.append(absolute)
        return _dedupe(links)
    except Exception as e:
        print(f"[sitemap] requests fallback error for {base_url}: {e}")

    return []


def _fetch_via_hf_space(url: str) -> List[str]:
    """Try HF Space endpoint — optional, may be unavailable."""
    try:
        from gradio_client import Client
        client = Client("Agents-MCP-Hackathon/web-scraper")
        result = client.predict(url=url, api_name="/all_links")
        found = re.findall(r"https?://[^\s\",<>]+", str(result))
        return _dedupe([u.rstrip(".,;)") for u in found])
    except Exception as e:
        print(f"[sitemap] HF Space unavailable: {e}")
        return []


def _get_links_for_company(company_name: str, url: str) -> List[str]:
    """
    Attempt multiple strategies to get internal links, in priority order:
      1. sitemap.xml
      2. HF Space
      3. homepage link crawl
    Falls back to [url] (just the root) if all fail.
    """
    links = _fetch_sitemap(url)
    if links:
        print(f"[sitemap] '{company_name}': sitemap.xml → {len(links)} links")
        return links

    links = _fetch_via_hf_space(url)
    if links:
        print(f"[sitemap] '{company_name}': HF Space → {len(links)} links")
        return links

    links = _fetch_homepage_links(url)
    if links:
        print(f"[sitemap] '{company_name}': homepage crawl → {len(links)} links")
        return links

    print(f"[sitemap] '{company_name}': all strategies failed, using root URL as fallback")
    return [url]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def get_sitemap_links(company_name: str, url: str) -> str:
    """
    Extract all internal links of the company's official website.
    Tries sitemap.xml, then HF Space, then homepage link crawl.
    The links are stored in WEBSITE_LINKS[company_name].

    Args:
        company_name: Company name (e.g. 'Revolut').
        url: The company's official URL (e.g. 'https://www.revolut.com/')

    Returns:
        Summary with the total number of links found and the first 10.
    """
    links = _get_links_for_company(company_name, url)
    WEBSITE_LINKS[company_name] = links

    preview = "\n".join(f"  · {link}" for link in links[:10])
    extra = f"\n ... y {len(links) - 10} más" if len(links) > 10 else ""
    return (
        f"[sitemap] '{company_name}': {len(links)} enlaces encontrados en {url}.\n"
        f"Primeros 10:\n{preview}{extra}"
    )


@tool
def get_all_sitemaps(company_urls: dict) -> str:
    """
    Retrieves internal links for all companies in the given dictionary.
    Stores the result in WEBSITE_LINKS.

    Args:
        company_urls: Dict with the format {company_name: official_url}
                      Directly passes WEBSITE_URLS from agent 1.

    Returns:
        Summary with the total number of links found per company.
    """
    if not company_urls:
        return "[sitemap] El diccionario de URLs está vacío. Ejecuta el agente 1 primero."

    lines: List[str] = []
    for company_name, url in company_urls.items():
        msg = get_sitemap_links(company_name, url)
        n = len(WEBSITE_LINKS.get(company_name, []))
        lines.append(f"  ✔ {company_name}: {n} enlaces | {url}")
        print(msg)

    return "Extracción de sitemaps completada:\n" + "\n".join(lines)


@tool
def list_sitemap_companies() -> str:
    """
    List the companies for which sitemap links have already been collected.

    Returns:
        Formatted list of companies -> number of links.
    """
    if not WEBSITE_LINKS:
        return "[sitemap] WEBSITE_LINKS está vacío. Ejecuta get_all_sitemaps primero."

    lines = [f"  · {name}: {len(links)} enlaces" for name, links in WEBSITE_LINKS.items()]
    return "Empresas con sitemap:\n" + "\n".join(lines)


@tool
def get_company_links(company_name: str) -> str:
    """
    Returns the complete list of stored links for a specific company.

    Args:
        company_name: Company name.

    Returns:
        List of URLs, one per line, or an error message if no data is found.
    """
    links = WEBSITE_LINKS.get(company_name)
    if not links:
        return f"[sitemap] No hay enlaces para '{company_name}'. Ejecuta get_sitemap_links primero."
    lines = [f"  {i+1}. {u}" for i, u in enumerate(links)]
    return f"=== {company_name} — {len(links)} enlaces ===\n" + "\n".join(lines)
