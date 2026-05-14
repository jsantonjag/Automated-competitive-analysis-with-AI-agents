# -*- coding: utf-8 -*-
from __future__ import annotations
 
"""
website_storage_agent.py
────────────────────────
Agent 4 of the webscraping pipeline.

Responsability: take SCRAPED_CONTENT produced by website_content_agent (Agent 3),
validate every row, discard those that contain error messages or empty/boilerplate text
and persist the clean data to:
    - website_intel.duckdb (tables: website_intel, website_summary)
    - website_content.csv
"""

import os

from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel
from website_agents.tools.website_storage_tools import (
    validate_scraped_content,
    store_valid_content,
    export_valid_content_to_csv,
    store_and_export,
    get_website_storage_summary,
    query_website_intel
)

load_dotenv()

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"

def build_website_storage_agent() -> CodeAgent:
    """
    Build and return a CodeAgent that validates and stores website content.
    
    The agent: 
     1. Calls validate_scraped_content() for a dry-run report on data quality.
    2. Calls store_and_export() to persist valid rows into DuckDB and CSV in
       one shot (or uses store_valid_content + export_valid_content_to_csv
       individually if it needs finer control).
    3. Calls get_website_storage_summary() to confirm what was stored.
    4. Optionally calls query_website_intel() to spot-check a company.
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
        model = model,
        tools = [
            validate_scraped_content,
            store_valid_content,
            export_valid_content_to_csv,
            store_and_export,
            get_website_storage_summary,
            query_website_intel
        ],
        max_steps = 8,
        additional_authorized_imports = ["re", "csv", "os", "datetime", "unicodedata"],
        verbosity_level = 2
    )

    return agent

