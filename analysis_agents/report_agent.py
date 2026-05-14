# -*- coding: utf-8 -*-
from __future__ import annotations

"""
report_agent.py
----------------
Agent 7 of the pipeline (final output agent).

Responsibility: combine the comparative matrix (STRUCTURED_DATA) and SWOT
analysis (SWOT_DATA) to generate a consulting-style Markdown report and a
downloadable PDF file.
"""

import os
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel

from analysis_agents.tools.report_tools import (
    generate_report,
    export_report_to_pdf,
    generate_full_output,
)

load_dotenv()

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"


def build_report_agent() -> CodeAgent:
    """
    Build and return a CodeAgent that generates the final market report.

    The agent:
    1. Receives category, matrix_text (from structuring_agent) and
       swot_text (from swot_agent).
    2. Calls generate_full_output to produce market_report.md and
       market_report.pdf in a single step.
    3. Returns confirmation with file paths.
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
            generate_report,
            export_report_to_pdf,
            generate_full_output,
        ],
        max_steps=6,
        additional_authorized_imports=["re", "json", "os", "datetime"],
        verbosity_level=2,
    )

    return agent
