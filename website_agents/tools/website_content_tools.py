# -*- coding: utf-8 -*-
from __future__ import annotations

"""
website_content_tools.py
--------------------------
Tools for agent 3: scrape the content of each URL in the sitemap.
Uses Scrapling directly (no HF Space dependency).
"""

import csv, os, re
from datetime import datetime, timezone
from typing import Dict, List
from urllib.parse import urlparse
from smolagents import tool

# Shared store
SCRAPED_CONTENT: List[Dict] = []


def section_label(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return "homepage"
    segment = path.split("/")[-1].replace("-", " ").replace("_", " ")
    return segment or "homepage"


def clean(text: str, max_chars: int = 2000) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def _fetch_content(url: str) -> str:
    """Fetch page text via Scrapling; fallback to requests."""
    # Attempt 1: Scrapling
    try:
        from scrapling.fetchers import Fetcher
        page = Fetcher.get(url, stealthy_headers=True, timeout=20)
        if page is not None:
            for selector in ["main", "article", "body"]:
                els = page.css(selector)
                if els:
                    text = " ".join(
                        re.sub(r"\s+", " ", el.text or "").strip()
                        for el in els[:3]
                        if el.text and len(el.text.strip()) > 50
                    )
                    if text:
                        return clean(text)
            raw = page.get_all_text(ignore_tags=("script", "style", "nav", "footer")) or ""
            return clean(raw)
    except Exception as e:
        print(f"[content] Scrapling error for {url}: {e}")

    # Attempt 2: requests
    try:
        import requests
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        resp.raise_for_status()
        html = resp.text
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<[^>]+>", " ", html)
        return clean(html)
    except Exception as e:
        print(f"[content] requests error for {url}: {e}")

    return ""


# TOOLS

@tool
def scrape_page_content(company_name: str, page_url: str) -> str:
    """
    Scrape the visible text content of a page using Scrapling.
    The result is added to SCRAPED_CONTENT.

    Args:
        company_name: Name of the company to which the URL belongs.
        page_url: URL of the page to scrape.

    Returns:
        Confirmation with the first 300 characters of the extracted content.
    """
    content = _fetch_content(page_url)

    if not content:
        return f"[content] Sin contenido en '{page_url}' para '{company_name}'."

    row = {
        "company_name": company_name,
        "page_url": page_url,
        "section_label": section_label(page_url),
        "raw_content": content,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }
    SCRAPED_CONTENT.append(row)
    preview = content[:300] + ("..." if len(content) > 300 else "")
    return f"[content] '{company_name}' | {page_url}\n -> {preview}"


@tool
def scrape_all_pages(company_list: dict) -> str:
    """
    Scrape the content of all links from all companies.
    Receives the WEBSITE_LINKS dictionary from Agent 2 and populates SCRAPED_CONTENT.

    Args:
        company_list: Dict {company_name: [url1, url2, ...]} (= WEBSITE_LINKS).

    Returns:
        Summary with pages processed per company and total rows added.
    """
    if not company_list:
        return "[content] El diccionario de enlaces está vacío. Ejecuta el agente 2 primero."

    summary: List[str] = []
    for company_name, links in company_list.items():
        count_ok = 0
        for url in links:
            msg = scrape_page_content(company_name, url)
            if "Error" not in msg and "Sin contenido" not in msg:
                count_ok += 1
            print(msg)
        summary.append(f"  ✔ {company_name}: {count_ok}/{len(links)} páginas con contenido")

    total = len(SCRAPED_CONTENT)
    return (
        "Scraping de contenido completado:\n"
        + "\n".join(summary)
        + f"\n\nTotal filas en SCRAPED_CONTENT: {total}"
    )


@tool
def scrape_company_pages(company_name: str, links: list) -> str:
    """
    Scrapes the content of all links from a specific company.

    Args:
        company_name: Company name.
        links: List of URLs to scrape.

    Returns:
        Summary of processed pages.
    """
    if not links:
        return f"[content] No hay enlaces para '{company_name}'."

    count_ok = 0
    for url in links:
        msg = scrape_page_content(company_name, url)
        if "Error" not in msg and "Sin contenido" not in msg:
            count_ok += 1
        print(msg)

    return f"[content] '{company_name}': {count_ok}/{len(links)} páginas con contenido scrapeado."


@tool
def export_content_to_csv(output_path: str = "raw_website_content.csv") -> str:
    """
    Exports all scraped content (SCRAPED_CONTENT) to a CSV file.

    Args:
        output_path: Path to the output CSV file.

    Returns:
        Confirmation with the number of rows exported and the file path.
    """
    if not SCRAPED_CONTENT:
        return "[content] SCRAPED_CONTENT está vacío. Ejecuta scrape_all_pages primero."

    fieldnames = ["company_name", "page_url", "section_label", "raw_content", "scraped_at"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(SCRAPED_CONTENT)

    return (
        f"[content] {len(SCRAPED_CONTENT)} filas exportadas a '{output_path}'.\n"
        f"  Ruta absoluta: {os.path.abspath(output_path)}"
    )


@tool
def get_content_summary() -> str:
    """
    Returns a summary of the scraped content: pages and rows per company.
    """
    if not SCRAPED_CONTENT:
        return "[content] SCRAPED_CONTENT está vacío."

    from collections import defaultdict
    stats = defaultdict(lambda: {"pages": set(), "rows": 0})
    for row in SCRAPED_CONTENT:
        cn = row["company_name"]
        stats[cn]["pages"].add(row["page_url"])
        stats[cn]["rows"] += 1

    header = f"{'Empresa':<30} {'Páginas':>8} {'Filas':>6}"
    sep = "-" * 50
    lines = [header, sep]
    for cn, s in sorted(stats.items()):
        lines.append(f"{cn:<30} {len(s['pages']):>8} {s['rows']:>6}")
    lines.append(sep)
    lines.append(
        f"{'TOTAL':<30} "
        f"{sum(len(s['pages']) for s in stats.values()):>8} "
        f"{len(SCRAPED_CONTENT):>6}"
    )
    return "\n".join(lines)


@tool
def list_scraped_pages(company_name: str) -> str:
    """
    Lists the URLs already scraped for a specific company.

    Args:
        company_name: Company name.

    Returns:
        List of scraped URLs or a message if no data is found.
    """
    pages = [r["page_url"] for r in SCRAPED_CONTENT if r["company_name"] == company_name]
    if not pages:
        return f"[content] No hay páginas scrapeadas para '{company_name}'."
    lines = [f"  {i+1}. {u}" for i, u in enumerate(pages)]
    return f"=== {company_name} - {len(pages)} páginas ===\n" + "\n".join(lines)
