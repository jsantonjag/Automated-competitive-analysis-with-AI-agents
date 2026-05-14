# -*- coding: utf-8 -*-
from __future__ import annotations

"""
structuring_agent.py
---------------------
Agent 5 of the pipeline.

Responsibility: take FEATURES_DATA from the features agent, use an LLM
to extract structured fields (plan names, prices, services, highlights)
from each scraped page, and build a normalised comparative matrix stored
in STRUCTURED_DATA and exported to comparative_matrix.csv.
"""

import os
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel

from analysis_agents.tools.structuring_tools import (
    extract_structure_from_features,
    get_comparative_matrix,
    export_matrix_to_csv,
    get_structuring_summary,
)

load_dotenv()

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"


def build_structuring_agent() -> CodeAgent:
    """
    Build and return a CodeAgent that produces the comparative matrix.

    The agent:
    1. Receives FEATURES_DATA from the features agent.
    2. Calls extract_structure_from_features to parse plans/prices/services.
    3. Calls get_comparative_matrix to display the full matrix.
    4. Calls export_matrix_to_csv to persist results.
    5. Returns get_structuring_summary as confirmation.
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
            extract_structure_from_features,
            get_comparative_matrix,
            export_matrix_to_csv,
            get_structuring_summary,
        ],
        max_steps=8,
        additional_authorized_imports=["re", "json", "csv", "os"],
        verbosity_level=2,
    )

    return agent
