# -*- coding: utf-8 -*-
from __future__ import annotations

"""
website_finder_agent.py
------------------------
Agent 1 of the scraping web pipeline.

Responsability: given the list of companies discovered by company_agent,
it locates the official URL of each one using Serper (Google Search) and
stores it in the shared dictionary WEBSITE_URLs.
"""

import os
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel

from website_agents.tools.website_finder_tools import(
    find_official_website,
    find_all_websites,
    list_found_website,
    get_website_url
)

load_dotenv()

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"

def build_website_finder_agent():
    """
    Build and return a CodeAgent that searchs official companies URLs.
    
    The agent:
    1. Receives a list of company names from the discovery agent.
    2. Calls 'find_all_website' to retrieve all URLs in a single pass.
    3. If any company is missing a URL, it retries with 'find_official_website' individually. 
    4. Returns the populated WEBSITE_URLS dictionary for Agent 2 to use:
        'from tools.website_finder_tools import WEBSITE_URLS'
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
            find_official_website,
            find_all_websites,
            list_found_website,
            get_website_url
        ],
        max_steps = 8,
        additional_authorized_imports=["re", "urllib.parse"],
        verbosity_level = 2
    )
    
    return agent

