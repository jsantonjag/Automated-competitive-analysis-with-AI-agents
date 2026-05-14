# -*- coding: utf-8 -*-
from __future__ import annotations

"""
pipeline_state.py
------------------
Estado compartido del pipeline LangGraph.

Viaja entre todos los nodos del grafo. Cada nodo recibe el estado completo,
modifica solo sus campos, y devuelve un dict con esos cambios.
LangGraph hace el merge automáticamente y persiste tras cada nodo.

Campos añadidos en la versión RAG
-----------------------------------
  rag_index_path : str  → ruta al directorio ChromaDB (features_index/)
  rag_chunk_count: int  → total de chunks indexados
"""

from typing import Any, Dict, List, TypedDict


class PipelineState(TypedDict, total=False):

    # ── Input ─────────────────────────────────────────────────────────────────
    category: str

    # ── Stage 1: Descubrimiento ───────────────────────────────────────────────
    company_names: List[str]

    # ── Stage 2-3: Noticias (Rama A) ─────────────────────────────────────────
    company_data: Dict[str, List[str]]

    # ── Stage 4-7: Web scraping (Rama B) ─────────────────────────────────────
    website_urls:    Dict[str, str]
    website_links:   Dict[str, List[str]]
    scraped_content: List[Dict[str, Any]]
    features_data:   Dict[str, List[Dict[str, Any]]]

    # ── Stage RAG ← NUEVO ────────────────────────────────────────────────────
    rag_index_path:  str
    rag_chunk_count: int

    # ── Stage 8: Estructuración ───────────────────────────────────────────────
    structured_data: List[Dict[str, Any]]
    matrix_text:     str

    # ── Stage 9: DAFO ─────────────────────────────────────────────────────────
    swot_data: Dict[str, Dict[str, List[str]]]
    swot_text: str

    # ── Meta ──────────────────────────────────────────────────────────────────
    errors: List[str]
