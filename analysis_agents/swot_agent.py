# -*- coding: utf-8 -*-
from __future__ import annotations

"""
swot_agent.py
--------------
Agent 6 of the pipeline.

Responsibility: read the news snippets from company_intelligence.duckdb
and generate a SWOT (Strengths / Weaknesses / Opportunities / Threats)
analysis for each of the 5 companies, storing results in SWOT_DATA.
"""

import os
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel

from analysis_agents.tools.swot_tools import (
    build_swot_for_company,
    build_all_swots,
    get_swot_matrix_text,
    list_swot_companies,
)

load_dotenv()

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"


def build_swot_agent() -> CodeAgent:
    """
    Build and return a CodeAgent that generates SWOT analyses.

    The agent:
    1. Receives the list of company names.
    2. Calls build_all_swots to process all companies at once.
    3. Calls list_swot_companies to confirm coverage.
    4. If any company is missing, retries with build_swot_for_company individually.
    5. Returns get_swot_matrix_text as final output for the report agent.
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
            build_swot_for_company,
            build_all_swots,
            get_swot_matrix_text,
            list_swot_companies,
        ],
        max_steps=10,
        additional_authorized_imports=["re", "json"],
        verbosity_level=2,
    )

    return agent
