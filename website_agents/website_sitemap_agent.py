# -*- coding: utf-8 -*-
from __future__ import annotations

"""
website_sitemap_agent.py
-------------------------
Agent 2 of the web scraping pipeline.

Responsability: given the WEBSITE_URLS dictionario produced by agent 1,
use "All links (sitemap)" endpoint in the HuggingFace Space
https://huggingface.co/spaces/Agents-MCP-Hackathon/web-scraper
to extract all internal links from each official website and store them
in the shared WEBSITE_LINKS dictionary.
"""

import os
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel

from website_agents.tools.website_sitemap_tools import (
    get_sitemap_links,
    get_all_sitemaps,
    list_sitemap_companies,
    get_company_links
)

load_dotenv()

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"

def build_website_sitemap_agent() -> CodeAgent:
    """
    Build and return a CodeAgent that extracts the sitemap from each official website.
    
    The agent:
    1. Receives the dict {company_name: official_url} from agent 1.
    2. Calls get_all_sitemaps to process all companies in bulk.
    3. If any company has few links, it retries with get_sitemap_links individually.
    4. Uses list_sitemap_companies to confirm which companies were processed.
    5. Leaves WEBSITE_LINKS populated for agent 3 to consume:
        from tools.website_sitemap_tools import WEBSITE_LINKS
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
            get_sitemap_links,
            get_all_sitemaps,
            list_sitemap_companies,
            get_company_links
        ],
        max_steps=12, # 1 call get_all + to 1 try per company x 5
        additional_authorized_imports=["re", "urllib.parse"],
        verbosity_level=2
    )

    return agent