from __future__ import annotations

import os
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel

from scraping_agent.tools.scraping_tools import (
    scrape_company_news,
    scrape_all_companies,
    get_company_data,
    list_scraped_companies,
    COMPANY_DATA # shared dict - other agents can import this directly
)

load_dotenv()

def build_scraping_agent() -> CodeAgent:
    """
    Build and return a CodeAgent equipped with Scrapling-based tools.
    
    The agent scrapes Google News, El Economista, TechCrunch and Cinco Días for each
    company discovered by the company_agent. All collected text is stored in the 
    module-level COMPANY_DATA dict, which downstream agents can import and use directly:
        
        from tools.scraping_tools import COMPANY_DATA
    
    """
    
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    
    model = OpenAIServerModel(
        model_id="claude-sonnet-4-20250514",
        api_base="https://api.anthropic.com/v1",
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        extra_headers={"anthropic-version": "2023-06-01"}
    )

    agent = CodeAgent(
        model=model,
        tools=[
            scrape_company_news,
            scrape_all_companies,
            get_company_data,
            list_scraped_companies
        ],
        max_steps=12,
        # Important: we allow pandas so the agent can structure data
        additional_authorized_imports=["re", "json"],
        verbosity_level=2 
    )
    
    return agent