from __future__ import annotations

"""
storage_agent.py
────────────────
Agent that takes the raw COMPANY_DATA produced by scraping_agent,
cleans every text block and persists them into a local DuckDB database.

Usage (from main.py):
    from agents.storage_agent import build_storage_agent
    from tools.scraping_tools import COMPANY_DATA

    storage_agent = build_storage_agent()
    result = storage_agent.run(
        PROMPT_STORAGE_TEMPLATE.format(company_data=COMPANY_DATA)
    )
"""

import os
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel

from scraping_agent.tools.scraping_storage_tools import (
    clean_and_store_company,
    store_all_companies,
    query_company_data,
    get_storage_summary,
    export_to_csv,
)

load_dotenv()

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"


def build_storage_agent() -> CodeAgent:
    """
    Build and return a CodeAgent equipped with DuckDB storage tools.

    The agent:
    1. Receives COMPANY_DATA (dict) from the scraping agent.
    2. Cleans each raw text block (normalises unicode, removes boilerplate, truncates).
    3. Inserts clean snippets into DuckDB (file: company_intelligence.duckdb).
    4. Updates a summary table per company.
    5. Optionally exports everything to CSV.

    Downstream agents can query the DB directly via query_company_data()
    or by connecting to the .duckdb file.
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
            clean_and_store_company,
            store_all_companies,
            query_company_data,
            get_storage_summary,
            export_to_csv,
        ],
        max_steps=10,
        additional_authorized_imports=["re", "json", "datetime", "unicodedata"],
        verbosity_level=2,
    )

    return agent
