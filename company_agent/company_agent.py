from __future__ import annotations

import os
from dotenv import load_dotenv

from smolagents import CodeAgent, OpenAIServerModel, VisitWebpageTool
from company_agent.tools.company_tools import (
    extract_company_candidates,
    rank_candidates,
    format_top,
    google_search,
    validate_and_fix_companies,
)

load_dotenv()

def build_agent() -> CodeAgent:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    

    model = OpenAIServerModel(
        model_id="claude-sonnet-4-20250514",
        api_base="https://api.anthropic.com/v1",
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        extra_headers={"anthropic-version": "2023-06-01"}
    )

    
    visit_webpage = VisitWebpageTool()

    agent = CodeAgent(
        model=model,
        tools=[
            google_search,
            visit_webpage,
            extract_company_candidates,
            rank_candidates,
            format_top,
            validate_and_fix_companies,
        ],
        max_steps=8, # avoiding loopings
        # if the agent needs basic modules in the code
        additional_authorized_imports=["re", "collections"],
        verbosity_level=2 
    )
    return agent