# -*- coding: utf-8 -*-
from __future__ import annotations

"""
rag_agent.py
-------------
Agente de indexación RAG (embedding_node del grafo LangGraph).

Responsabilidad: recibir FEATURES_DATA del parallel_node,
vectorizar todo el contenido con sentence-transformers y
almacenarlo en ChromaDB para que structuring_agent y
report_agent puedan hacer búsqueda semántica.
"""

import os
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel

from analysis_agents.tools.rag_tools import (
    index_features_data,
    get_rag_summary,
    clear_rag_index,
)

load_dotenv()

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"


def build_rag_agent() -> CodeAgent:
    """
    Construye y devuelve el CodeAgent de indexación RAG.

    El agente:
    1. Recibe FEATURES_DATA con el contenido scrapeado de precios/servicios.
    2. Llama a index_features_data para vectorizar y almacenar en ChromaDB.
    3. Llama a get_rag_summary para confirmar el índice construido.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY no está definida en .env")

    model = OpenAIServerModel(
        model_id="claude-sonnet-4-20250514",
        api_base="https://api.anthropic.com/v1",
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        extra_headers={"anthropic-version": "2023-06-01"}
    )

    agent = CodeAgent(
        model=model,
        tools=[
            index_features_data,
            get_rag_summary,
            clear_rag_index,
        ],
        max_steps=5,
        additional_authorized_imports=[
            "re", "os", "unicodedata", "chromadb", "sentence_transformers",
        ],
        verbosity_level=2,
    )

    return agent
