# -*- coding: utf-8 -*-
from __future__ import annotations

"""
structuring_tools.py  (versión RAG)
-------------------------------------
Tools para el structuring_agent (Agente 5).

CAMBIO RAG
-----------
Antes: contenido de FEATURES_DATA truncado a 2500 chars → LLM.
Ahora: búsqueda semántica en ChromaDB → top-k fragmentos → LLM.
       Fallback automático a truncación si ChromaDB no está disponible.
"""

import csv, json, os, re
from typing import Dict, List
from smolagents import tool

STRUCTURED_DATA: List[Dict] = []

_RAG_QUERY = (
    "planes precios suscripción tarifas servicios funcionalidades "
    "pricing subscription plans fees tiers"
)

_EXTRACT_PROMPT = """
Eres un analista de mercado experto. Tienes fragmentos de contenido web
de la empresa "{company}" sobre precios, planes y servicios.

Extrae TODA la información sobre:
- Nombres de planes o niveles de suscripción
- Precios (mensuales, anuales, por uso, etc.)
- Servicios o funcionalidades incluidas en cada plan
- Aspectos destacados o diferenciadores

Devuelve ÚNICAMENTE un JSON válido (array de planes):
[
  {{
    "plan_name": "Nombre del plan o 'General' si no hay planes definidos",
    "price": "Precio exacto o 'No especificado'",
    "services": "Lista de servicios separados por coma",
    "highlights": "Puntos clave diferenciadores"
  }}
]

Si no hay información relevante devuelve:
[{{"plan_name": "Sin datos", "price": "N/A", "services": "N/A", "highlights": "N/A"}}]

Contenido:
{content}
"""


def _retrieve_rag(company: str, top_k: int = 6) -> str:
    """Recupera fragmentos de ChromaDB. Devuelve '' si no está disponible."""
    try:
        from analysis_agents.tools.rag_tools import _get_embedder, _get_collection
        embedder   = _get_embedder()
        collection = _get_collection()
        if collection.count() == 0:
            return ""
        qemb    = embedder.encode(_RAG_QUERY).tolist()
        results = collection.query(
            query_embeddings=[qemb],
            n_results=min(top_k, collection.count()),
            where={"company": company},
        )
        docs = results.get("documents", [[]])[0]
        if not docs:
            return ""
        return "\n\n---\n\n".join(f"[fragmento {i+1}]\n{d}" for i, d in enumerate(docs))
    except Exception as exc:
        print(f"[structuring] RAG no disponible para '{company}': {exc}")
        return ""


def _fallback_content(features_data: dict, company: str, max_chars: int = 2500) -> str:
    """Fallback: concatena raw_content con truncación (comportamiento pre-RAG)."""
    records  = features_data.get(company, [])
    combined = "\n\n".join(r.get("raw_content", "") for r in records if r.get("raw_content"))
    return combined[:max_chars]


def _call_llm_extract(company: str, content: str) -> List[Dict]:
    """Llama a la API de Anthropic para extraer JSON de planes."""
    import os, urllib.request, time
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[structuring] ANTHROPIC_API_KEY no definida")
        return []

    prompt  = _EXTRACT_PROMPT.format(company=company, content=content)
    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    max_retries, wait = 4, 30
    for attempt in range(max_retries):
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
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            raw  = "".join(b.get("text", "") for b in data.get("content", []))
            raw  = re.sub(r"```json|```", "", raw).strip()
            plans = json.loads(raw)
            return plans if isinstance(plans, list) else []
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < max_retries - 1:
                print(f"[structuring] Rate limit (429) — esperando {wait}s (intento {attempt+1}/{max_retries})...")
                time.sleep(wait)
                wait *= 2
                continue
            print(f"[structuring] LLM HTTP error para '{company}': {exc.code}")
            return []
        except Exception as exc:
            print(f"[structuring] LLM error para '{company}': {exc}")
            return []
    return []


def _process_one_company(company: str, records: list, features_data: dict) -> tuple:
    """Procesa una empresa: recupera RAG/fallback y llama al LLM. Thread-safe."""
    content = _retrieve_rag(company)
    method  = "RAG"
    if not content:
        content = _fallback_content(features_data, company)
        method  = "fallback-truncación"
    if not content:
        return company, [], method, records
    plans = _call_llm_extract(company, content)
    return company, plans, method, records


@tool
def extract_structure_from_features(features_data: dict) -> str:
    """
    Para cada empresa en features_data, recupera fragmentos de ChromaDB (RAG)
    o usa el contenido directo como fallback, luego llama al LLM para extraer
    planes, precios y servicios en JSON estructurado.
    Acumula los resultados en STRUCTURED_DATA.
    Las llamadas al LLM se hacen en paralelo (una por empresa) para reducir latencia.

    Args:
        features_data: Dict {company_name: [{"page_url", "section_label",
                        "raw_content", "scraped_at"}, ...]} — pasar FEATURES_DATA.

    Returns:
        Resumen con planes extraídos por empresa y método usado (RAG o fallback).
    """
    if not features_data:
        return "[structuring] features_data vacío."

    from concurrent.futures import ThreadPoolExecutor, as_completed

    summary: List[str] = []
    futures_map = {}

    with ThreadPoolExecutor(max_workers=min(5, len(features_data))) as executor:
        for company, records in features_data.items():
            fut = executor.submit(_process_one_company, company, records, features_data)
            futures_map[fut] = company

        for fut in as_completed(futures_map):
            company, plans, method, records = fut.result()
            if not plans:
                summary.append(f"  ✘ {company}: sin contenido")
                continue
            count = 0
            for plan in plans:
                if plan.get("plan_name") == "Sin datos":
                    continue
                STRUCTURED_DATA.append({
                    "company":    company,
                    "plan_name":  plan.get("plan_name", ""),
                    "price":      plan.get("price", "No especificado"),
                    "services":   plan.get("services", ""),
                    "highlights": plan.get("highlights", ""),
                    "source_url": records[0].get("page_url", "") if records else "",
                    "rag_source": method,
                })
                count += 1
            summary.append(f"  ✔ {company}: {count} planes [{method}]")

    return (
        "Estructuración completada:\n"
        + "\n".join(summary)
        + f"\n\nTotal registros en STRUCTURED_DATA: {len(STRUCTURED_DATA)}"
    )


@tool
def get_comparative_matrix() -> str:
    """
    Devuelve la matriz comparativa completa como texto plano,
    indicando el método de recuperación (RAG o fallback) por empresa.

    Returns:
        Tabla multi-empresa o mensaje si STRUCTURED_DATA está vacío.
    """
    if not STRUCTURED_DATA:
        return "[structuring] STRUCTURED_DATA vacío. Ejecuta extract_structure_from_features primero."

    by_company: Dict[str, List[Dict]] = {}
    for row in STRUCTURED_DATA:
        by_company.setdefault(row["company"], []).append(row)

    lines = ["=" * 80, "MATRIZ COMPARATIVA DE EMPRESAS", "=" * 80]
    for company, plans in sorted(by_company.items()):
        lines.append(f"\n{company.upper()}")
        lines.append("-" * 60)
        for p in plans:
            lines.append(f"  Plan:        {p['plan_name']}")
            lines.append(f"  Precio:      {p['price']}")
            lines.append(f"  Servicios:   {p['services']}")
            lines.append(f"  Destacados:  {p['highlights']}")
            lines.append(f"  Fuente:      {p['source_url']}")
            lines.append(f"  Metodo RAG:  {p.get('rag_source', 'N/A')}")
            lines.append("")
    return "\n".join(lines)


@tool
def export_matrix_to_csv(output_path: str = "comparative_matrix.csv") -> str:
    """
    Exporta STRUCTURED_DATA a un CSV.
    Columnas: company, plan_name, price, services, highlights, source_url, rag_source.

    Args:
        output_path: Ruta del CSV (default: 'comparative_matrix.csv').

    Returns:
        Confirmación con filas escritas y ruta absoluta.
    """
    if not STRUCTURED_DATA:
        return "[structuring] STRUCTURED_DATA vacío."
    fieldnames = ["company", "plan_name", "price", "services", "highlights", "source_url", "rag_source"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(STRUCTURED_DATA)
    return (
        f"[structuring] {len(STRUCTURED_DATA)} filas → '{output_path}'\n"
        f"  Ruta absoluta: {os.path.abspath(output_path)}"
    )


@tool
def get_structuring_summary() -> str:
    """Resumen compacto de STRUCTURED_DATA: empresas y planes."""
    if not STRUCTURED_DATA:
        return "[structuring] STRUCTURED_DATA vacío."
    by_company: Dict[str, int] = {}
    for row in STRUCTURED_DATA:
        by_company[row["company"]] = by_company.get(row["company"], 0) + 1
    lines = [f"  · {c}: {n} plan(es)" for c, n in sorted(by_company.items())]
    return f"STRUCTURED_DATA — {len(STRUCTURED_DATA)} registros:\n" + "\n".join(lines)
