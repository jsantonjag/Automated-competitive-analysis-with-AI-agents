# -*- coding: utf-8 -*-
from __future__ import annotations

"""
website_finder_tools.py
------------------------
Tools for Agente 1: find the official URL of each company.

Shared store
-------------
WEBSITE_URLS: Dict[str, str] = {
    "Revolut": "https://www.revolut.com",
    "N25":     "https://n26.com/",
    ...
}

Agents 2 and 3 import the dict to continue the pipeline.
"""

import os, requests
from typing import Dict, List, Optional
from urllib.parse import urlparse
from dotenv import load_dotenv
from smolagents import tool

load_dotenv()

#Shared store
WEBSITE_URLS: Dict[str, str] = {} # company_name -> official base URL

EXCLUDED_DOMAINS = {
    "linkedin", "facebook", "twitter", "instagram", "wikipedia", 
    "crunchbase", "bloomberg", "reuters", "techcrunch", "forbes",
    "youtube", "glassdoor", "indeed", "trustpilot", "yelp",
    "x.com", "tiktok"
}

def find_url(company_name: str) -> Optional[str]:
    """Call Serper and return the first non-excluded organic URL (base only)"""
    
    api_key = os.getenv("SERPER_API_KEY", "")
    if not api_key:
        raise ValueError("SERPER_API_KEY is not defined in file .env")
    
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": f"{company_name} official website", "num": 6},
            timeout=15,
        )
        resp.raise_for_status()

        for o in resp.json().get("organic", []):
            url = o.get("link", "")
            parsed = urlparse(url)
            host = parsed.netloc.lower()
            if not any(ex in host for ex in EXCLUDED_DOMAINS):
                return f"{parsed.scheme}://{parsed.netloc}/"
            
    except Exception as e:
        print(f"[website_finder] Serper error para '{company_name}': {e}")
    
    return None

# TOOLS
@tool
def find_official_website(company_name: str) -> str:
    """
    Searches for and returns the official URL of a company using Serper (Google Search).
    The result is stored in WEBSITE_URLS for use by agents 2 and 3.
    
    
    Args:
        company_name: company name (ej: 'Revolut', 'BBVA').
    
    Returns:
        The official URL found or an error message if it is not found.
    """

    url = find_url(company_name)
    if url:
        WEBSITE_URLS[company_name] = url
        return url
    return f"[finder] No se encontró ninguna URL oficial para '{company_name}'."

@tool
def find_all_websites(company_list: list) -> str:
    """
    Find the official URL of all companies in the list.
    Store the results in WEBSITE_URLS.
    
    Args:
        company_list: list of company names (strings or dicts with 'name' keys)
    
    Returns:
        Summary with the URL found for each company.
    """

    lines: List[str] = []
    for item in company_list:
        name = item.get("name", item) if isinstance(item, dict) else str(item)
        url = find_url(name)
        if url:
            WEBSITE_URLS[name] = url
            lines.append(f" ✔ {name}: {url}")
        else:
            lines.append(f" ✘ {name}: no encontrada")
    
    return "Búsqueda de URLs completada:\n" + "\n".join(lines)

@tool
def list_found_website() -> str:
    """
    Lists al official URLs already found and stored in WEBSITE_URLS.
    
    Returns:
        Formatted list of company -> URL or a message if the dict is empty.
    """

    if not WEBSITE_URLS:
        return "[finder] WEBSITE_URLS está vacía. Ejecuta find_all_websites primero."
    
    lines = [f"  · {name}:{url}" for name, url in WEBSITE_URLS.items()]
    return "URLs encontradas:\n" + "\n".join(lines)

@tool
def get_website_url(company_name: str) -> str:
    """
    Retrieves the stored official URL for a specific company.
    
    Args:
        company_name: company name.
    
    Returns:
        The official URL or a message indicating that it does not exist.
    """

    url = WEBSITE_URLS.get(company_name, "")
    if url: return url
    
    return f"[finder] No hay ninguna URL almacenada para '{company_name}'. Ejecuta find_official_website primero."
