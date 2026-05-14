# -*- coding: utf-8 -*-
from __future__ import annotations

"""
website_content_agent.py
-------------------------
Agent 3 of the web scraping pipeline.

Responsability: given the WEBSITE_LINKS dictionary produced by Agent 2,
use the "Content scraper" endpoint from the HuggingFace Space
https://huggingface.co/spaces/Agents-MCP-Hackathon/web-scraper to extract
to content from each URL, accumulate the results in SCRAPED_CONTENT and
export them to a CSV file.

CVS columns:
company_name | page_url | section_label | raw_content | scraped_at
"""

import os
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel

from website_agents.tools.website_content_tools import (
    scrape_page_content,
    scrape_all_pages,
    scrape_company_pages,
    export_content_to_csv,
    get_content_summary,
    list_scraped_pages
)

load_dotenv()

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"

def build_website_content_agent() -> CodeAgent:
    """
    
    
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
            scrape_page_content,
            scrape_all_pages,
            scrape_company_pages,
            export_content_to_csv,
            get_content_summary,
            list_scraped_pages
        ],
        max_steps=20, #Many possible URLs, wide margin
        additional_authorized_imports=["re", "csv", "os", "datetime", "urllib.parse", "collections"],
        verbosity_level=2
    )

    return agent