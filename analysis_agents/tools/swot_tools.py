# -*- coding: utf-8 -*-
from __future__ import annotations

"""
swot_tools.py
--------------
Tools for the swot_agent (Agent 6).

Responsibility: read the news snippets stored in company_intelligence.duckdb
(produced by scraping_storage_agent), ask an LLM to classify each company's
information into Strengths / Weaknesses / Opportunities / Threats, and build
a SWOT matrix stored in SWOT_DATA.

Shared store
-------------
SWOT_DATA: Dict[str, Dict[str, List[str]]] = {
    "Revolut": {
        "strengths":     ["..."],
        "weaknesses":    ["..."],
        "opportunities": ["..."],
        "threats":       ["..."],
    },
    ...
}
"""

import json
import re
from typing import Dict, List

import duckdb
from smolagents import tool

# ---------------------------------------------------------------------------
# Lazy import de COMPANY_DATA para evitar importaciones circulares
# ---------------------------------------------------------------------------
def _get_company_data_store() -> Dict[str, List[str]]:
    """Importa COMPANY_DATA desde scraping_tools (store en memoria)."""
    try:
        from scraping_agent.tools.scraping_tools import COMPANY_DATA
        return COMPANY_DATA
    except Exception:
        return {}

# ---------------------------------------------------------------------------
# Shared store
# ---------------------------------------------------------------------------
SWOT_DATA: Dict[str, Dict[str, List[str]]] = {}

# ---------------------------------------------------------------------------
# DuckDB path (must match scraping_storage_tools.py)
# ---------------------------------------------------------------------------
DB_PATH = "company_intelligence.duckdb"

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
_SWOT_PROMPT = """
Eres un consultor de estrategia empresarial experto en análisis DAFO.

A continuación tienes fragmentos de noticias y artículos recientes sobre la empresa "{company}".
Analiza el texto y clasifica los hallazgos en las cuatro categorías DAFO.

Devuelve ÚNICAMENTE un JSON válido con esta estructura:
{{
  "strengths":     ["punto 1", "punto 2", ...],
  "weaknesses":    ["punto 1", "punto 2", ...],
  "opportunities": ["punto 1", "punto 2", ...],
  "threats":       ["punto 1", "punto 2", ...]
}}

Cada lista debe contener entre 2 y 5 puntos concisos (máx. 120 caracteres cada uno).
Si no hay información suficiente para una categoría, incluye al menos un punto
indicando "Información insuficiente".

Fragmentos de noticias:
{snippets}
"""


def _call_llm_swot(company: str, snippets_text: str) -> Dict[str, List[str]]:
    """Call the Anthropic API to generate a SWOT for one company."""
    import os
    import urllib.request
    import urllib.error
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("[swot] ANTHROPIC_API_KEY no está definida en .env")

    prompt = _SWOT_PROMPT.format(company=company, snippets=snippets_text[:4000])
    payload = json.dumps(
        {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    empty = {
        "strengths": ["Información insuficiente"],
        "weaknesses": ["Información insuficiente"],
        "opportunities": ["Información insuficiente"],
        "threats": ["Información insuficiente"],
    }

    import time
    max_retries = 4
    wait = 30
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            raw_text = "".join(
                block.get("text", "") for block in data.get("content", [])
            )
            raw_text = re.sub(r"```json|```", "", raw_text).strip()
            result = json.loads(raw_text)
            for key in ("strengths", "weaknesses", "opportunities", "threats"):
                if key not in result:
                    result[key] = ["Información insuficiente"]
            return result
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < max_retries - 1:
                print(f"[swot] Rate limit (429) — esperando {wait}s (intento {attempt+1}/{max_retries})...")
                time.sleep(wait)
                wait *= 2
                continue
            print(f"[swot] LLM error for '{company}': HTTP {exc.code}")
            return empty
        except Exception as exc:
            print(f"[swot] LLM error for '{company}': {exc}")
            return empty
    return empty


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def build_swot_for_company(company_name: str) -> str:
    """
    Read all news snippets for a company from company_intelligence.duckdb,
    send them to the LLM, and store the resulting SWOT in SWOT_DATA.

    Args:
        company_name: Name of the company to analyse.

    Returns:
        The SWOT matrix as a formatted text block.
    """
    snippets_text = ""

    # ── Fuente 1: DuckDB ──────────────────────────────────────────────────
    try:
        conn = duckdb.connect(DB_PATH, read_only=False)
        rows = conn.execute(
            """
            SELECT clean_text FROM companies
            WHERE lower(company_name) = lower(?)
            ORDER BY scraped_at
            """,
            [company_name],
        ).fetchall()
        conn.close()
        if rows:
            snippets_text = "\n".join(f"• {r[0]}" for r in rows if r[0])
            print(f"[swot] '{company_name}': {len(rows)} snippets desde DuckDB.")
    except Exception as exc:
        print(f"[swot] DuckDB no disponible para '{company_name}': {exc}")

    # ── Fuente 2: COMPANY_DATA en memoria (fallback) ──────────────────────
    if not snippets_text:
        company_data_store = _get_company_data_store()
        # Búsqueda insensible a mayúsculas
        matched_key = next(
            (k for k in company_data_store if k.lower() == company_name.lower()),
            None,
        )
        if matched_key:
            blocks = company_data_store[matched_key]
            snippets_text = "\n".join(f"• {b}" for b in blocks if b)
            print(f"[swot] '{company_name}': {len(blocks)} bloques desde COMPANY_DATA (memoria).")

    if not snippets_text:
        return (
            f"[swot] No hay datos disponibles para '{company_name}' "
            "(ni en DuckDB ni en COMPANY_DATA). "
            "Asegúrate de que el scraping se ha ejecutado correctamente."
        )
    swot = _call_llm_swot(company_name, snippets_text)
    SWOT_DATA[company_name] = swot

    lines = [f"=== DAFO: {company_name} ==="]
    labels = {
        "strengths": "✅ FORTALEZAS",
        "weaknesses": "⚠️  DEBILIDADES",
        "opportunities": "🚀 OPORTUNIDADES",
        "threats": "🔴 AMENAZAS",
    }
    for key, label in labels.items():
        lines.append(f"\n{label}")
        for point in swot.get(key, []):
            lines.append(f"  · {point}")
    return "\n".join(lines)


@tool
def build_all_swots(company_names: list) -> str:
    """
    Build the SWOT analysis for every company in the list.
    Populates SWOT_DATA for all companies.
    Las llamadas al LLM se hacen en paralelo (una por empresa).

    Args:
        company_names: List of company name strings.

    Returns:
        Full SWOT matrix for all companies.
    """
    if not company_names:
        return "[swot] La lista de empresas está vacía."

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results_map: dict = {}
    with ThreadPoolExecutor(max_workers=min(5, len(company_names))) as executor:
        futures = {executor.submit(build_swot_for_company, name): name for name in company_names}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                results_map[name] = fut.result()
            except Exception as exc:
                results_map[name] = f"[swot] Error para '{name}': {exc}"

    # Preserve original order
    results = [results_map[name] for name in company_names if name in results_map]
    return "\n\n".join(results)


@tool
def get_swot_matrix_text() -> str:
    """
    Return all SWOT analyses stored in SWOT_DATA as a formatted report block.

    Returns:
        Multi-company SWOT text or a message if empty.
    """
    if not SWOT_DATA:
        return "[swot] SWOT_DATA está vacío. Ejecuta build_all_swots primero."

    lines = ["=" * 70, "ANÁLISIS DAFO — TODAS LAS EMPRESAS", "=" * 70]
    labels = {
        "strengths": "✅ FORTALEZAS",
        "weaknesses": "⚠️  DEBILIDADES",
        "opportunities": "🚀 OPORTUNIDADES",
        "threats": "🔴 AMENAZAS",
    }
    for company, swot in sorted(SWOT_DATA.items()):
        lines.append(f"\n{'─'*60}")
        lines.append(f"🏢 {company.upper()}")
        lines.append(f"{'─'*60}")
        for key, label in labels.items():
            lines.append(f"\n{label}")
            for point in swot.get(key, []):
                lines.append(f"  · {point}")
    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


@tool
def list_swot_companies() -> str:
    """
    List which companies already have a SWOT in SWOT_DATA.

    Returns:
        Formatted list or a message if empty.
    """
    if not SWOT_DATA:
        return "[swot] SWOT_DATA está vacío."
    lines = [f"  · {c}" for c in sorted(SWOT_DATA)]
    return "Empresas con DAFO generado:\n" + "\n".join(lines)
