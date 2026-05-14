from __future__ import annotations

import re
from collections import Counter
from typing import List, Dict
from smolagents import tool
from dotenv import load_dotenv
import os, requests, spacy

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

try:
    NLP = spacy.load("es_core_news_md")
except OSError:
    NLP = spacy.blank("es")

@tool
def google_search(query: str) -> str:
    """
    Perform a Google search to find up-to-date information.
    
    Args:
        query: The search query for Google.
    """
    url = "https://google.serper.dev/search"
    # Usamos la variable que ya tienes en tu .env
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise ValueError("SERPER_API_KEY is not defined in file .env")
    
    payload = {"q": query, "num": 5}
    headers = {'X-API-KEY': api_key, 'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        results = response.json()
        
        # Extract snippets from organic results
        organic = results.get("organic", [])
        lines = []
        for r in organic:
            title = r.get("title", "")
            link = r.get("link", "")
            snippet = r.get("snippet" or "")[:220]
            lines.append(f"TITLE: {title}\nLINK: {link}\nSNIPPET: {snippet}\n")
        return "\n".join(lines)
    except Exception as e:
        return f"Error en la búsqueda: {str(e)}"
    
STOPWORDS = {
    # generic terms
    "Forbes", "Wikipedia", "LinkedIn", "YouTube", "Google", "Facebook", "Instagram",
    "España", "Spain", "Madrid", "Barcelona", 
    # tipical terms no-business
    "Términos", "Condiciones", "Privacy", "Cookies", "Contacto", "About",
    "Login", "Suscripción", "Aceptar", "Rechazar", "Blog", "Menú", 
    "Privacidad", "Aviso", "Legal", "Configuración", "Anuncios", 
    "Search", "Home", "Inicio", "Contacto", "Nosotros", "Iniciar sesión",
    "Company", "Companies", "Business", "Businesses", "Sector", "Sectors",
    "Industry", "Industries", "Market", "Markets", "News", "News",
    "Technology", "Technologies", "Innovation", "Innovations", "Research",
    "Research", "Development", "Developments", "Product", "Products",
    "Service", "Services", "Solution", "Solutions", "Platform", "Platforms",
    "System", "Systems", "Network", "Networks", "Database", "Databases",
    "Software", "Softwares", "Hardware", "Hardwares", "Cloud", "Clouds",
    "AI", "Machine Learning", "Deep Learning", "LLM", "LLMs", "GPT", "ChatGPT",
    "Berlin", "Berlín", "Múnich", "Frankfurt", "Hamburg", 
    "Cologne", "Colonia", "Düsseldorf", "Milán", "Roma", "París", "Paris", 
    "Washington", "Miami", "Ciudad de México", "México City", "Turín", "Varsovia", "Holanda",
    "Bolonia", "Pisa", "Plataforma", "redes", "red", "Servicios", "Investgación", 
    "Invest", "Inversión"
}

LEGAL_SUFFIXES = (
    "S.A.", "SA", "S.L.", "SL", "Inc", "LLC", "Ltd", "GmbH", "PLC", "BV", "SAS"
)

def _clean_name(name: str) -> str:
    name = name.strip()
    # removes final legal suffixes
    for suf in LEGAL_SUFFIXES: 
        name = re.sub(rf"\b{re.escape(suf)}\b\.?$", "", name).strip()
    # normalize spaces
    name = re.sub(r"\s+", " ", name).strip() 
    return name


@tool
def extract_company_candidates(texts: List[str] | str) -> List[str]:
    """
    Extract likely company names from a list of short texts (titles/snippets).
    
    Args:
        texts: list of strings (titles/snippets) to extract company candidates from.
    
    Returns:
        A list of candidate company names (may include duplicates).
    """
    
    candidates: List[str] = []

    if isinstance(texts, str):
        texts = [texts]

    # pattern: 1-4 words such as "Banco Santander", "OpenAI", "BBVA"
    pattern = re.compile(r"\b([A-ZÁÉÍÓÚÑ][a-zâêîôûáéíóúñ]{2,15}(?:\s+[A-ZÁÉÍÓÚÑ][a-zâêîôûáéíóúñ]{1,15}){0,2})\b")

    for t in texts:
        t = str(t)
        
        # 1) spaCy: only ORG (it ignores GPE/LOC automatically)
        doc = NLP(t)
        for ent in doc.ents:
            if ent.label_ == "ORG":
                name = _clean_name(ent.text)
                if len(name) < 2:
                    continue
                if name.title() in STOPWORDS or name.upper() in STOPWORDS:
                    continue
                candidates.append(name.title())
        
        # 2) Fallback regex: if spaCy doesn't detect nothing, it uses the pattern
        if not doc.ents:
            for m in pattern.findall(t):
                name = _clean_name(m)
                if len(name) < 2:
                    continue
                if name.title() in STOPWORDS or name.upper() in STOPWORDS:
                    continue
                candidates.append(name.title())

    return candidates

@tool
def rank_candidates(candidates: List[str], top_k: int = 5) -> List[Dict[str, int]]:
    """
    Count and rank candidates by frequency. Returns list of {"name": ..., "count": ...}
    
    Args:
        candidates: candidates names (may include duplicates)
        top_k: maximum number of companies to return
    
    Returns:
        A list of dicts with keys 'name' and 'count'.
    """
    c = Counter(
        _clean_name(x).title()
        for x in candidates
        if x and _clean_name(x).title() not in STOPWORDS and _clean_name(x).upper() not in STOPWORDS
    )
    ranked = c.most_common(top_k)
    return [{"name": n, "count": int(k)} for n, k in ranked]


@tool
def format_top(top: List[Dict[str, int]]) -> str:
    """
    Format the top list nicely as a text block.
    
    Args:
        top: List of items with keys 'name' and 'count'.
    
    Returns:
        A formatted string with the final ranking.
    """
    if not top:
        return ""
    lines = []
    for i, item in enumerate(top, start=1):
        lines.append(f"{i}. {item['name']}")
    return "\n".join(lines)