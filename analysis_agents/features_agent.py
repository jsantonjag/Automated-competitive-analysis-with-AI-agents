# -*- coding: utf-8 -*-
from __future__ import annotations

"""
features_agent.py
------------------
Agent 3b of the web scraping pipeline (replaces the generic content agent
for the features/pricing use-case required by the spec).

Responsibility: given WEBSITE_LINKS from the sitemap agent, identify and
scrape only the pages related to pricing, features and services for each
company, storing the results in FEATURES_DATA.
"""

import os
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel

from analysis_agents.tools.features_tools import (
    filter_relevant_urls,
    scrape_features_page,
    scrape_all_features,
    get_features_summary,
    list_features_companies,
)

load_dotenv()

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"


def build_features_agent() -> CodeAgent:
    """
    Build and return a CodeAgent that scrapes features/pricing pages.

    The agent:
    1. Receives WEBSITE_LINKS {company: [urls]} from the sitemap agent.
    2. Calls scrape_all_features to filter relevant URLs and scrape them.
    3. If a company has 0 records, uses filter_relevant_urls + scrape_features_page
       individually to retry.
    4. Returns FEATURES_DATA populated for the structuring agent.
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
            filter_relevant_urls,
            scrape_features_page,
            scrape_all_features,
            get_features_summary,
            list_features_companies,
        ],
        max_steps=15,
        additional_authorized_imports=["re", "urllib.parse", "datetime"],
        verbosity_level=2,
    )

    return agent
