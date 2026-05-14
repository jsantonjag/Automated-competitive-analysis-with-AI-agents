# -*- coding: utf-8 -*-
from __future__ import annotations

"""
rag_tools.py
-------------
Herramientas RAG para el embedding_node del pipeline LangGraph.

Responsabilidad: vectorizar el contenido de FEATURES_DATA con
sentence-transformers y almacenarlo en ChromaDB (vector store embebido,
sin servidor). Los agentes de análisis posteriores recuperan fragmentos
semánticamente relevantes en lugar de truncar por longitud.

Flujo RAG
----------
    FEATURES_DATA (páginas de precios/servicios)
          │
    [embedding_node]  → ChromaDB (features_index/)
          │
    ┌─────┴─────┐
    │           │
structuring   report
   agent       agent
    │           │
  query       query
 semántica   semántica
    │           │
 top-k frags  top-k frags
    │           │
   LLM         LLM

Modelo de embeddings
---------------------
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  · Multilingüe (ES + EN, ambos presentes en el contenido)
  · Ligero: 118 MB, corre en CPU sin GPU
  · Dimensión de embedding: 384
  · Gratuito: descarga automática desde HuggingFace Hub la primera vez

Vector store
-------------
ChromaDB (embebido, sin servidor)
  · Persiste en disco en features_index/
  · Similitud coseno
  · Sin infraestructura adicional
"""

import os
import re
import unicodedata
from typing import Dict, List

from smolagents import tool

# ── Constantes ────────────────────────────────────────────────────────────────
CHROMA_DIR        = "features_index"
COLLECTION_NAME   = "features"
EMBEDDING_MODEL   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K_DEFAULT     = 5
MIN_CONTENT_CHARS = 80   # mínimo para indexar un fragmento
CHUNK_SIZE        = 500  # caracteres por chunk
CHUNK_OVERLAP     = 100  # solapamiento entre chunks

# ── Estado interno del módulo ─────────────────────────────────────────────────
_collection = None
_embedder   = None


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS INTERNOS (no son @tool — uso interno del módulo)
# ══════════════════════════════════════════════════════════════════════════════

def _get_embedder():
    """Carga el modelo de embeddings una sola vez (lazy init)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print(f"[rag] Cargando modelo: {EMBEDDING_MODEL} ...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        print("[rag] Modelo listo.")
    return _embedder


def _get_collection():
    """Abre o crea la colección ChromaDB (lazy init)."""
    global _collection
    if _collection is None:
        import chromadb
        client      = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[rag] ChromaDB: '{CHROMA_DIR}/' "
              f"({_collection.count()} documentos existentes).")
    return _collection


def _normalize(text: str) -> str:
    """NFC + colapsar espacios en blanco."""
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def _chunk_text(text: str) -> List[str]:
    """
    Divide texto en chunks solapados.
    chunk_size=500 chars, overlap=100 chars.
    El solapamiento evita perder contexto en los cortes.
    """
    text = _normalize(text)
    if len(text) <= CHUNK_SIZE:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@tool
def index_features_data(features_data: dict) -> str:
    """
    Vectoriza FEATURES_DATA y lo indexa en ChromaDB.

    Para cada página scrapeada de cada empresa:
      1. Divide el contenido en chunks solapados (500 chars, overlap 100).
      2. Genera embeddings con sentence-transformers (modelo multilingüe).
      3. Almacena en ChromaDB con metadatos: company, page_url, section_label.

    Este proceso reemplaza la truncación por longitud que hacía
    extract_structure_from_features en la versión sin RAG.

    Args:
        features_data: Dict {company_name: [{"page_url", "section_label",
                        "raw_content", "scraped_at"}, ...]} — pasar FEATURES_DATA.

    Returns:
        Resumen con chunks indexados por empresa y total.
    """
    if not features_data:
        return "[rag] features_data vacío. Ejecuta features_agent primero."

    embedder   = _get_embedder()
    collection = _get_collection()
    summary: List[str] = []
    total_chunks = 0

    # Collect all chunks first, then encode in one batch (much faster than one-by-one)
    all_ids:       List[str] = []
    all_chunks:    List[str] = []
    all_metadatas: List[dict] = []
    per_company_counts: dict = {}

    for company, records in features_data.items():
        per_company_counts[company] = 0
        for rec in records:
            raw = rec.get("raw_content", "")
            if len(_normalize(raw)) < MIN_CONTENT_CHARS:
                continue
            for i, chunk in enumerate(_chunk_text(raw)):
                doc_id = (
                    f"{company}::{rec.get('page_url', '')}::chunk{i}"
                    .replace(" ", "_")[:180]
                )
                # Evitar duplicados en re-indexaciones
                try:
                    if collection.get(ids=[doc_id])["ids"]:
                        continue
                except Exception:
                    pass
                all_ids.append(doc_id)
                all_chunks.append(chunk)
                all_metadatas.append({
                    "company":       company,
                    "page_url":      rec.get("page_url", ""),
                    "section_label": rec.get("section_label", ""),
                    "chunk_index":   i,
                })
                per_company_counts[company] += 1

    # Batch encode all chunks at once — sentence-transformers is significantly
    # faster encoding a list than calling .encode() in a loop.
    BATCH_SIZE = 128
    if all_chunks:
        print(f"[rag] Codificando {len(all_chunks)} chunks en batch...")
        all_embeddings: List[list] = []
        for start in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[start : start + BATCH_SIZE]
            vecs  = embedder.encode(batch, show_progress_bar=False)
            all_embeddings.extend(v.tolist() for v in vecs)

        # Insert in batches to avoid ChromaDB memory spikes
        INSERT_BATCH = 500
        for start in range(0, len(all_ids), INSERT_BATCH):
            collection.add(
                ids        = all_ids[start : start + INSERT_BATCH],
                embeddings = all_embeddings[start : start + INSERT_BATCH],
                documents  = all_chunks[start : start + INSERT_BATCH],
                metadatas  = all_metadatas[start : start + INSERT_BATCH],
            )

    for company, count in per_company_counts.items():
        total_chunks += count
        summary.append(f"  ✔ {company}: {count} chunks indexados")

    return (
        f"Indexación RAG completada en '{CHROMA_DIR}/':\n"
        + "\n".join(summary)
        + f"\n\nTotal chunks en ChromaDB: {total_chunks}"
    )


@tool
def retrieve_for_company(company: str, query: str, top_k: int = TOP_K_DEFAULT) -> str:
    """
    Recupera los fragmentos más relevantes para una empresa y una consulta.

    Genera el embedding de la query y busca los top_k chunks más similares
    en ChromaDB filtrados por empresa.

    Args:
        company: Nombre de la empresa (debe coincidir con el indexado).
        query:   Consulta en lenguaje natural.
                 Ejemplo: "planes precios suscripción servicios".
        top_k:   Número de fragmentos a recuperar (default: 5).

    Returns:
        Los top_k fragmentos más relevantes listos para incluir en el prompt.
    """
    embedder   = _get_embedder()
    collection = _get_collection()

    if collection.count() == 0:
        return (
            f"[rag] ChromaDB vacío para '{company}'. "
            "Ejecuta index_features_data primero."
        )

    query_emb = embedder.encode(query).tolist()
    results   = collection.query(
        query_embeddings=[query_emb],
        n_results=min(top_k, collection.count()),
        where={"company": company},
    )
    docs = results.get("documents", [[]])[0]
    if not docs:
        return f"[rag] Sin resultados para '{company}' con query: '{query}'"

    fragments = "\n\n---\n\n".join(
        f"[fragmento {i+1}]\n{doc}" for i, doc in enumerate(docs)
    )
    return (
        f"=== RAG: {company} | query: '{query}' | {len(docs)} fragmentos ===\n\n"
        + fragments
    )


@tool
def retrieve_for_all_companies(company_names: list, query: str, top_k: int = TOP_K_DEFAULT) -> str:
    """
    Recupera fragmentos relevantes para todas las empresas con la misma query.

    Útil para el report_agent, que necesita contexto de todas las empresas
    a la vez para redactar el informe comparativo.

    Args:
        company_names: Lista de nombres de empresa.
        query:         Consulta en lenguaje natural.
        top_k:         Fragmentos por empresa (default: 5).

    Returns:
        Fragmentos de todas las empresas organizados por empresa.
    """
    if not company_names:
        return "[rag] Lista de empresas vacía."

    results: List[str] = []
    for company in company_names:
        results.append(retrieve_for_company(company, query, top_k))
    return ("\n\n" + "=" * 60 + "\n\n").join(results)


@tool
def get_rag_summary() -> str:
    """
    Muestra el estado del índice RAG: empresas indexadas y total de chunks.

    Returns:
        Tabla empresa → chunks, o mensaje si el índice está vacío.
    """
    collection = _get_collection()
    total = collection.count()
    if total == 0:
        return "[rag] ChromaDB vacío. Ejecuta index_features_data primero."

    all_meta = collection.get(include=["metadatas"])["metadatas"]
    counts: Dict[str, int] = {}
    for meta in all_meta:
        c = meta.get("company", "desconocido")
        counts[c] = counts.get(c, 0) + 1

    header = f"{'Empresa':<30} {'Chunks':>8}"
    sep    = "-" * 42
    lines  = [f"Índice RAG — {CHROMA_DIR}/", header, sep]
    for company, n in sorted(counts.items()):
        lines.append(f"{company:<30} {n:>8}")
    lines += [sep, f"{'TOTAL':<30} {total:>8}"]
    return "\n".join(lines)


@tool
def clear_rag_index() -> str:
    """
    Elimina el índice RAG (útil para re-indexar desde cero).

    Returns:
        Confirmación con el número de documentos eliminados.
    """
    import shutil
    global _collection
    n = _get_collection().count()
    _collection = None
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    return f"[rag] Índice eliminado ({n} docs). Directorio '{CHROMA_DIR}/' borrado."
