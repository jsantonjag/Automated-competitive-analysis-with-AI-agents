# -*- coding: utf-8 -*-
from __future__ import annotations

"""
graph.py  —  Orquestador LangGraph con sistema de evaluación integrado.
"""

import sys, os
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from pipeline_state import PipelineState

# ── Evaluación ────────────────────────────────────────────────────────────────
from evaluation.tracker import PipelineTracker
_TRACKER: PipelineTracker | None = None

# ── Agentes ───────────────────────────────────────────────────────────────────
from company_agent.company_agent import build_agent
from scraping_agent.scraping_agent import build_scraping_agent
from website_agents.website_finder_agent import build_website_finder_agent
from website_agents.website_sitemap_agent import build_website_sitemap_agent
from website_agents.website_storage_agent import build_website_storage_agent
from analysis_agents.features_agent import build_features_agent
from analysis_agents.structuring_agent import build_structuring_agent
from analysis_agents.swot_agent import build_swot_agent

# ── Stores compartidos ────────────────────────────────────────────────────────
from scraping_agent.tools.scraping_tools import COMPANY_DATA
from website_agents.tools.website_finder_tools import WEBSITE_URLS
from website_agents.tools.website_sitemap_tools import WEBSITE_LINKS
from website_agents.tools.website_content_tools import SCRAPED_CONTENT
from analysis_agents.tools.features_tools import FEATURES_DATA
from analysis_agents.tools.rag_tools import CHROMA_DIR, _get_collection
from analysis_agents.tools.structuring_tools import STRUCTURED_DATA, get_comparative_matrix
from analysis_agents.tools.swot_tools import SWOT_DATA, get_swot_matrix_text

# ── Tools directas ───────────────────────────────────────────────────────────
from scraping_agent.tools.scraping_storage_tools import store_all_companies
from analysis_agents.tools.rag_tools import index_features_data, get_rag_summary
from analysis_agents.tools.report_tools import generate_full_output


# ══════════════════════════════════════════════════════════════════════════════
# RESET
# ══════════════════════════════════════════════════════════════════════════════

def reset_all_stores() -> None:
    import shutil
    import analysis_agents.tools.rag_tools as rag_mod
    import scraping_agent.tools.scraping_storage_tools as scraping_db_mod

    if scraping_db_mod._conn is not None:
        try: scraping_db_mod._conn.close()
        except Exception: pass
        scraping_db_mod._conn = None

    try:
        import website_agents.tools.website_storage_tools as web_db_mod
        if web_db_mod._conn is not None:
            try: web_db_mod._conn.close()
            except Exception: pass
            web_db_mod._conn = None
    except Exception:
        pass

    COMPANY_DATA.clear(); WEBSITE_URLS.clear(); WEBSITE_LINKS.clear()
    SCRAPED_CONTENT.clear(); FEATURES_DATA.clear()
    STRUCTURED_DATA.clear(); SWOT_DATA.clear()

    rag_mod._collection = None
    rag_mod._embedder   = None
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print(f"[reset] Índice RAG '{CHROMA_DIR}/' eliminado.")

    for db_file in ("company_intelligence.duckdb", "website_intel.duckdb"):
        if os.path.exists(db_file):
            os.remove(db_file)
            print(f"[reset] Base de datos '{db_file}' eliminada.")

    print("[reset] Todos los stores han sido reiniciados. ✓\n")


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

PROMPT_DISCOVERY = """
Eres un experto en inteligencia de mercado. Tu objetivo es encontrar las 5
empresas más relevantes para: "{category}".

Instrucciones:
1. Realiza una búsqueda amplia con google_search.
2. Si los resultados contienen listas ("Las 10 mejores..."), usa visit_webpage
   en esos links para extraer los nombres reales.
3. Filtra nombres que parezcan secciones de web (ej: "Contacto", "Login").
4. Usa extract_company_candidates con el texto completo obtenido.
5. Usa validate_and_fix_companies(company_names=<lista>, category="{category}")
   para limpiar y validar los nombres antes de devolver el resultado final.

Devuelve la lista numerada con los 5 nombres finales validados:
1. Empresa A
2. Empresa B
...
"""

PROMPT_SCRAPING = """
Tienes la siguiente lista de empresas: {company_names}
Tarea: recopilar información actualizada desde fuentes de noticias.
1. Usa scrape_all_companies pasándole la lista.
2. Usa list_scraped_companies para confirmar qué empresas se procesaron.
3. Si alguna no tiene datos, usa scrape_company_news individualmente.
"""

PROMPT_FINDER = """
Lista de empresas: {company_names}
Tarea: encontrar la URL oficial de cada empresa.
1. Usa find_all_websites.
2. Usa list_found_website para confirmar.
3. Si alguna falta, reintenta con find_official_website.
"""

PROMPT_SITEMAP = """
URLs oficiales: {website_urls}
Tarea: extraer enlaces internos de cada web.
1. Usa get_all_sitemaps.
2. Usa list_sitemap_companies para confirmar.
3. Si alguna tiene menos de 5 enlaces, reintenta con get_sitemap_links.
"""

PROMPT_FEATURES = """
Enlaces internos por empresa: {website_links}
Tarea: scrapear páginas de precios, planes y servicios.
1. Usa scrape_all_features.
2. Usa get_features_summary.
3. Si alguna tiene 0 registros, usa scrape_features_page manualmente.
"""

PROMPT_WEB_STORAGE = """
Contenido scrapeado de: {company_names}
1. Llama a validate_scraped_content.
2. Llama a store_and_export.
3. Llama a get_website_storage_summary.
"""

PROMPT_STRUCTURING = """
FEATURES_DATA e índice RAG disponibles para: {company_names}

Tarea: construir la matriz comparativa normalizada.
1. Usa extract_structure_from_features pasándole el diccionario FEATURES_DATA
   (importado del módulo analysis_agents.tools.features_tools).
   Ejemplo de código:
     from analysis_agents.tools.features_tools import FEATURES_DATA
     result = extract_structure_from_features(FEATURES_DATA)
2. Usa get_comparative_matrix para visualizar la matriz.
3. Usa export_matrix_to_csv para exportar a 'comparative_matrix.csv'.
"""

PROMPT_SWOT = """
Datos de noticias en company_intelligence.duckdb para: {company_names}

Tarea: generar análisis DAFO por empresa.
1. Usa build_all_swots.
2. Usa list_swot_companies para confirmar.
3. Si alguna falta, reintenta con build_swot_for_company.
4. Devuelve get_swot_matrix_text.
"""


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: tracker global
# ══════════════════════════════════════════════════════════════════════════════

def _tracker() -> PipelineTracker:
    global _TRACKER
    if _TRACKER is None:
        _TRACKER = PipelineTracker(category="<unknown>", run_id="resume")
    return _TRACKER


# ══════════════════════════════════════════════════════════════════════════════
# NODOS DEL GRAFO
# ══════════════════════════════════════════════════════════════════════════════

def company_node(state: PipelineState) -> dict:
    print("\n[1/6] Buscando empresas relevantes...\n")
    import re
    from company_agent.tools.company_guardrails import ensure_five_companies

    category = state["category"]
    guardrail_activated = False

    with _tracker().measure("company_node"):
        raw_names: list = []
        try:
            agent  = build_agent()
            result = agent.run(PROMPT_DISCOVERY.format(category=category))
            if isinstance(result, list):
                result = "\n".join(str(r) for r in result)
            elif not isinstance(result, str):
                result = str(result)
            print(result)

            names = re.findall(r"\d+[\.\)]\s*(.+)", result)
            cleaned = []
            for n in names:
                n = n.strip()
                n = re.sub(r"\*+", "", n).strip()
                n = re.split(r"\s[—\-–]\s|\s\(|\s{2,}|:\s", n)[0].strip()
                n = n.rstrip(".,;:")
                if n and len(n) > 1:
                    cleaned.append(n)
            raw_names = cleaned

            if not raw_names:
                import json
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, list):
                        raw_names = [
                            item.get("name", item) if isinstance(item, dict) else str(item)
                            for item in parsed
                        ]
                except Exception:
                    pass

            if not raw_names:
                raw_names = [
                    l.strip() for l in result.splitlines()
                    if l.strip() and len(l.strip()) > 2
                    and not l.strip().startswith(("http", "#", "-", "*"))
                    and not l.strip().lower().startswith(("las ", "los ", "top ", "here", "these", "the "))
                ][:5]

        except Exception as exc:
            print(f"[1/6] Agente falló: {exc}. Activando guardrails directamente.")
            guardrail_activated = True

        names = ensure_five_companies(raw_names=raw_names, category=category, target=5)
        if len(raw_names) < 5:
            guardrail_activated = True

        if not names:
            raise RuntimeError(
                f"[company_node] No se pudieron obtener empresas para '{category}' "
                "después de todos los fallbacks."
            )
        print(f"[1/6] Empresas validadas: {names}")

    _tracker().record_quality(
        "company_node",
        n_companies_found=len(names),
        guardrail_activated=int(guardrail_activated),
        companies=", ".join(names),
    )
    return {"company_names": names}


def parallel_node(state: PipelineState) -> dict:
    company_names = state.get("company_names", [])
    errors        = list(state.get("errors", []))
    result_news, result_web = {}, {}

    with _tracker().measure("parallel_node"):
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_news = executor.submit(_run_news_branch, company_names)
            fut_web  = executor.submit(_run_web_branch,  company_names)
            for fut in as_completed([fut_news, fut_web]):
                try:
                    partial = fut.result()
                    if fut is fut_news:
                        result_news = partial
                        print("\n[Rama A] Noticias completadas.")
                    else:
                        result_web = partial
                        print("\n[Rama B] Web scraping completado.")
                except Exception as exc:
                    errors.append(f"Error en rama paralela: {exc}")
                    print(f"\n[ERROR] {exc}")

    return {
        "company_data":    result_news.get("company_data", {}),
        "website_urls":    result_web.get("website_urls", {}),
        "website_links":   result_web.get("website_links", {}),
        "features_data":   result_web.get("features_data", {}),
        "scraped_content": result_web.get("scraped_content", []),
        "errors":          errors,
    }


def embedding_node(state: PipelineState) -> dict:
    print("\n[3/6] Construyendo índice RAG (ChromaDB)...\n")
    features_data = state.get("features_data", {})
    errors        = list(state.get("errors", []))

    with _tracker().measure("embedding_node"):
        if not features_data:
            errors.append("embedding_node: features_data vacío, saltando RAG.")
            _tracker().record_quality("embedding_node", chunks_indexed=0, skipped=1)
            return {"rag_index_path": "", "rag_chunk_count": 0, "errors": errors}

        result = index_features_data(features_data)
        print(result)
        print(get_rag_summary())

        try:
            chunk_count = _get_collection().count()
        except Exception:
            chunk_count = 0

    _tracker().record_quality("embedding_node", chunks_indexed=chunk_count)
    return {
        "rag_index_path":  CHROMA_DIR,
        "rag_chunk_count": chunk_count,
        "errors":          errors,
    }


def structuring_node(state: PipelineState) -> dict:
    print("\n[4/6] Construyendo matriz comparativa (con RAG)...\n")
    features_data = state.get("features_data", {})
    company_names = state.get("company_names", [])
    errors        = list(state.get("errors", []))

    with _tracker().measure("structuring_node"):
        if not features_data:
            errors.append("structuring_node: features_data vacío.")
            return {"structured_data": [], "matrix_text": "", "errors": errors}

        agent = build_structuring_agent()
        agent.run(PROMPT_STRUCTURING.format(company_names=company_names))
        matrix_text = get_comparative_matrix()
        print(matrix_text)

    return {"structured_data": list(STRUCTURED_DATA), "matrix_text": matrix_text, "errors": errors}


def swot_node(state: PipelineState) -> dict:
    print("\n[5/6] Generando análisis DAFO...\n")
    company_names = state.get("company_names", [])
    errors        = list(state.get("errors", []))

    with _tracker().measure("swot_node"):
        if not company_names:
            errors.append("swot_node: company_names vacío.")
            return {"swot_data": {}, "swot_text": "", "errors": errors}

        agent = build_swot_agent()
        agent.run(PROMPT_SWOT.format(company_names=company_names))
        swot_text = get_swot_matrix_text()
        print(swot_text)

    return {"swot_data": dict(SWOT_DATA), "swot_text": swot_text, "errors": errors}


def structuring_swot_node(state: PipelineState) -> dict:
    """Ejecuta structuring y swot en paralelo (ambos son independientes tras embedding)."""
    print("\n[4-5/6] Construyendo matriz comparativa y análisis DAFO en paralelo...\n")
    errors = list(state.get("errors", []))
    result_struct, result_swot = {}, {}

    with _tracker().measure("structuring_swot_node"):
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_struct = executor.submit(structuring_node, state)
            fut_swot   = executor.submit(swot_node,        state)
            for fut in as_completed([fut_struct, fut_swot]):
                try:
                    partial = fut.result()
                    if fut is fut_struct:
                        result_struct = partial
                        print("\n[Rama Structuring] Matriz completada.")
                    else:
                        result_swot = partial
                        print("\n[Rama SWOT] DAFO completado.")
                except Exception as exc:
                    errors.append(f"Error en nodo paralelo struct/swot: {exc}")
                    print(f"\n[ERROR struct/swot] {exc}")

    return {
        "structured_data": result_struct.get("structured_data", []),
        "matrix_text":     result_struct.get("matrix_text", ""),
        "swot_data":       result_swot.get("swot_data", {}),
        "swot_text":       result_swot.get("swot_text", ""),
        "errors":          errors + result_struct.get("errors", []) + result_swot.get("errors", []),
    }


def report_node(state: PipelineState) -> dict:
    print("\n[6/6] Generando informe de mercado (con RAG enrichment)...\n")
    errors        = list(state.get("errors", []))
    matrix_text   = state.get("matrix_text", "")
    swot_text     = state.get("swot_text", "")
    company_names = state.get("company_names", [])

    with _tracker().measure("report_node"):
        if not matrix_text and not swot_text:
            errors.append("report_node: sin datos suficientes para el informe.")
            return {"errors": errors}

        result = generate_full_output(
            category=state["category"],
            matrix_text=matrix_text,
            swot_text=swot_text,
            company_names=company_names,
        )
        print(result)
        if "Error" in result:
            errors.append(f"report_node: {result}")

    return {"errors": errors}


# ══════════════════════════════════════════════════════════════════════════════
# RAMAS PARALELAS
# ══════════════════════════════════════════════════════════════════════════════

def _run_news_branch(company_names: list) -> dict:
    print("\n[Rama A · a] Scrapeando noticias...\n")
    build_scraping_agent().run(PROMPT_SCRAPING.format(company_names=company_names))
    if not COMPANY_DATA:
        return {"company_data": {}}
    print("\n[Rama A · b] Almacenando noticias en DuckDB...\n")
    try:
        result = store_all_companies(dict(COMPANY_DATA))
        print(result)
    except Exception as exc:
        print(f"[Rama A] Error almacenando en DuckDB: {exc}")
    return {"company_data": dict(COMPANY_DATA)}


def _run_web_branch(company_names: list) -> dict:
    print("\n[Rama B · a] Buscando URLs oficiales...\n")
    build_website_finder_agent().run(PROMPT_FINDER.format(company_names=company_names))
    if not WEBSITE_URLS:
        print("[Rama B] ADVERTENCIA: no se encontraron URLs oficiales.")
        return {"website_urls": {}, "website_links": {}, "features_data": {}, "scraped_content": []}

    print("\n[Rama B · b] Extrayendo sitemaps...\n")
    build_website_sitemap_agent().run(PROMPT_SITEMAP.format(website_urls=dict(WEBSITE_URLS)))

    if not WEBSITE_LINKS:
        print("[Rama B] ADVERTENCIA: sitemap vacío — usando URLs raíz como fallback.")
        for company, url in WEBSITE_URLS.items():
            WEBSITE_LINKS[company] = [url]

    print(f"[Rama B] WEBSITE_LINKS: { {k: len(v) for k, v in WEBSITE_LINKS.items()} }")

    print("\n[Rama B · c] Scrapeando páginas de precios y features...\n")
    build_features_agent().run(PROMPT_FEATURES.format(website_links=dict(WEBSITE_LINKS)))
    print(f"[Rama B] FEATURES_DATA: { {k: len(v) for k, v in FEATURES_DATA.items()} }")

    if SCRAPED_CONTENT:
        print("\n[Rama B · d] Almacenando contenido web en DuckDB...\n")
        build_website_storage_agent().run(PROMPT_WEB_STORAGE.format(
            company_names=list({r["company_name"] for r in SCRAPED_CONTENT}),
        ))

    return {
        "website_urls":    dict(WEBSITE_URLS),
        "website_links":   dict(WEBSITE_LINKS),
        "features_data":   dict(FEATURES_DATA),
        "scraped_content": list(SCRAPED_CONTENT),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DEL GRAFO
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline():
    graph = StateGraph(PipelineState)
    graph.add_node("company",           company_node)
    graph.add_node("parallel",          parallel_node)
    graph.add_node("embedding",         embedding_node)
    graph.add_node("structuring_swot",  structuring_swot_node)
    graph.add_node("report",            report_node)
    graph.set_entry_point("company")
    graph.add_edge("company",          "parallel")
    graph.add_edge("parallel",         "embedding")
    graph.add_edge("embedding",        "structuring_swot")
    graph.add_edge("structuring_swot", "report")
    graph.add_edge("report",           END)
    return graph.compile(checkpointer=MemorySaver())


# ══════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global _TRACKER

    thread_id, resume = "run-001", False
    if "--resume" in sys.argv:
        idx = sys.argv.index("--resume")
        if idx + 1 < len(sys.argv):
            thread_id, resume = sys.argv[idx + 1], True

    pipeline = build_pipeline()
    config   = {"configurable": {"thread_id": thread_id}}

    if resume:
        print(f"\nReanudando pipeline (thread_id={thread_id})...\n")
        initial_state = {}
        _TRACKER = PipelineTracker(category="<resume>", run_id=thread_id)
    else:
        category = input("Dime una categoría (ej: 'Neobancos en España'): ").strip()
        if not category:
            category = "Neobancos en España"
        _TRACKER = PipelineTracker(category=category, run_id=thread_id)
        _TRACKER.meta["thread_id"] = thread_id
        print("\n[reset] Reiniciando stores y BD para nueva búsqueda...")
        reset_all_stores()
        initial_state = {"category": category, "errors": []}

    # ── Ejecutar pipeline ─────────────────────────────────────────────────────
    final_state = pipeline.invoke(initial_state, config=config)

    # ── Cerrar tracker + persistir + mostrar ──────────────────────────────────
    _TRACKER.finish(final_state)
    _TRACKER.save()
    _TRACKER.print_summary()
    _TRACKER.compare_with_history()

    # ── Resumen original ──────────────────────────────────────────────────────
    errors = final_state.get("errors", [])
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETADO")
    print("=" * 60)
    print(f"  Categoría            : {final_state.get('category', '')}")
    print(f"  Empresas analizadas  : {final_state.get('company_names', [])}")
    print(f"  Chunks RAG indexados : {final_state.get('rag_chunk_count', 0)}")
    print(f"  Noticias DuckDB      : company_intelligence.duckdb")
    print(f"  Contenido web DuckDB : website_intel.duckdb")
    print(f"  Índice RAG           : {final_state.get('rag_index_path', 'N/A')}/")
    print(f"  Matriz CSV           : comparative_matrix.csv")
    print(f"  Informe Markdown     : market_report.md")
    print(f"  Informe PDF          : market_report.pdf")
    if errors:
        print(f"\n  Errores no fatales ({len(errors)}):")
        for e in errors:
            print(f"    · {e}")
    print("=" * 60)


if __name__ == "__main__":
    main()
