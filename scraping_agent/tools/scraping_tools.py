from __future__ import annotations

from typing import Dict, List
from smolagents import tool
# Fetcher for standard sites, StealthyFetcher for protected ones
from scrapling.fetchers import Fetcher, StealthyFetcher

import re

# Target news sources to scrape
NEWS_SOURCES: Dict[str, tuple] = {
    "google_news":  ("https://news.google.com/search?q={query}&hl=es&gl=ES&ceid=ES:es", True), #Protected
    "eleconomista": ("https://www.eleconomista.es/buscador/?texto={query}", False), 
    "techcrunch":   ("https://techcrunch.com/search/{query}/", False),
    "cincodias":    ("https://cincodias.elpais.com/buscador/?q={query}", False),
    "google_finance":  ("https://www.google.com/finance/search?q={query}", True), #Protected
    "yahoo_finance":  ("https://finance.yahoo.com/search?q={query}", True), #Protected
    "bbc_news":  ("https://www.bbc.co.uk/search?q={query}&filter=news", False),
    "investing":  ("https://www.investing.com/search/?q={query}", True), #Protected
    "cnbc":  ("https://www.cnbc.com/search/?q={query}", False),
    "reuters":  ("https://www.reuters.com/search/news?blob={query}", False),    
}

COMPANY_DATA: Dict[str, List[str]] = {} #Shared dict

def clean_text(text: str, max_chars: int = 600) -> str:
    """Remove excessive whitespace and truncate."""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars] if len(text) > max_chars else text

def fetch_page(url: str, stealth: bool = False, timeout: int = 30):
    """
    Fetch a page with Scrapling, returning the parsed page or None on error.
    StealthyFetcher calls are wrapped in a hard timeout (default 30 s) to
    avoid blocking the pipeline indefinitely on protected sites.
    """
    try:
        if stealth:
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    StealthyFetcher.fetch, url,
                    headless=True, network_idle=True
                )
                try:
                    return future.result(timeout=timeout)
                except FuturesTimeout:
                    print(f"[scraping_tools] Timeout ({timeout}s) fetching {url} — skipping.")
                    future.cancel()
                    return None
        return Fetcher.get(url, stealthy_headers=True, timeout=20)
    except Exception as e:
        print(f"[scraping_tools] Error fetching {url}: {e}")
        return None

# CSS selectors optimized by source to extract relevant snippets
SOURCE_SELECTORS: Dict[str, List[str]] = {
    "google_news":    ["article", "h3", "h4", "p"],
    "eleconomista":   [".noticia", "article", "h3", "p"],
    "techcrunch":     ["article", ".post-block", "h3", "p"],
    "cincodias":      [".articulo", "article", "h3", "p"],
    "google_finance": ["[data-attrid]", ".article", "a[href]", "p"],
    "yahoo_finance":  [".js-stream-content", "article", "h3", "p"],
    "bbc_news":       ["article", ".gs-c-promo-heading", "h3", "p"],
    "investing":      [".articleItem", ".js-article-item", "article", "h3", "p"],
    "cnbc":           [".SearchResult-searchResultContent", "article", "h3", "p"],
    "reuters":        [".search-result-content", "article", "h3", "p"]
}

GENERIC_SELECTORS = [
    "article", ".article", ".result", ".search-result",
    "h3", "h2", ".title", ".headline", "p"
]
            
def extract_snippets(page, source_name: str, max_items: int = 5) -> List[str]:
    """
    Extract text snippets trying source-spacific selectors first,
    the fallinf back to generic ones.
    """

    snippets: List[str] = []
    selectors = SOURCE_SELECTORS.get(source_name, GENERIC_SELECTORS)
    
    for selector in selectors:
        elements = page.css(selector)
        if elements:
            for el in elements[:max_items]:
                text = clean_text(el.text or "")
                if text and len(text) > 40: #skip noise
                    snippets.append(text)
            if snippets:
                break # stop once we have content
    
    return snippets[:max_items]

# TOOLS

@tool
def scrape_company_news(company_name: str) -> str:
    """
    Scrape news and information about a company from multiple sources:
    Google News, El Economista, TechCrunch and Cinco Días.
    Results are stord in the shared COMPANY_DATA dict and also returned as text.
    
    Args:
        company_name: Name of the company to research (e.g. 'Revolut', 'BBVA')
    
    Returns:
        A multi-line string with all collected information about the company.
    """      

    query = company_name.replace(" ", "+")
    collected: List[str] = []
    
    for source_name, (url_template, needs_stealth) in NEWS_SOURCES.items():
        url = url_template.format(query=query)
        print(f"[scraping_tools] [{source_name}] '{company_name}' -> {url}")
        
        # Google News and El Economista sometimes need stealth
        page = fetch_page(url, stealth=needs_stealth)

        if page is None:
            collected.append(f"[{source_name}] No se pudo acceder a la página.")
            continue
        
        snippets = extract_snippets(page, source_name)
        if snippets:
            block = f"[{source_name}]\n" + "\n".join(f"  · {s}" for s in snippets)
        else:
            # Last-resort: grab the full visible text (first 800 chars)
            raw = clean_text(page.get_all_text(ignore_tags=("script", "style")) or "", 800)
            block = f"[{source_name}] (texto general)\n {raw}" if raw else f"[{source_name}] Sin resultados."

        collected.append(block)
        
    result_text = f"=== {company_name} ===\n" + "\n\n".join(collected)
    
    # Persist in shared dict for the next agent
    COMPANY_DATA[company_name] = collected
    
    return result_text

@tool
def scrape_all_companies(company_list: list) -> str:
    """
    Scrape news and information for every company in the list.
    All results are accumulated in the shred COMPANY_DATA dict.
    
    Args:
        company_list: List of company name strings returned by the discovery agent.
        
    Returns: 
        A summary string confirming which companies were processed and how many 
        text blocks were collected per company.
    """

    summary_lines: List[str] = []
    
    for company in company_list:
        name = company.get("name", company) if isinstance(company, dict) else str(company)
        scrape_company_news(name) # fills COMPANY_DATA[name]
        n_blocks = len(COMPANY_DATA.get(name, []))
        summary_lines.append(f"  ✔ {name}: {n_blocks} bloques de información recogidas.")
        
    summary = "Scraping completed:\n" + "\n".join(summary_lines)
    summary += f"\n\nTotal empresas procesadas: {len(company_list)}"
    return summary

@tool
def get_company_data(company_name: str) -> str:
    """
    Retrieve previously scraped data for a company from the shared store.
    
    Args:
        company_name: Name of the company whose data you want to retrieve.
        
    Returns:
        All collected text blocks for that company joined as a single string,
        or a message indicating no data is available.
    """

    blocks = COMPANY_DATA.get(company_name)
    if not blocks:
        return f"No hay datos almacenados para '{company_name}'. Ejecuta scrape_company_news primero."
    return f"=== {company_name} ===\n" + "\n\n".join(blocks)

@tool
def list_scraped_companies() -> str:
    """
    List all companies for which scraping data has already been collected.
    
    Returns:
        A newline-separated list of company names, or a message if the store is empty. 
    """

    if not COMPANY_DATA:
        return "Almacenamiento de datos vacío.  Aún se ha escrapeado ninguna empresa."
    return "Empresas con datos:\n" + "\n".join(f"  · {k}" for k in COMPANY_DATA.keys())
