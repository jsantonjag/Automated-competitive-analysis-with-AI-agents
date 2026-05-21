# Automated competitive analysis with AI agents
A multi-agent market intelligence system that, given a business category, automatically discovers the most relevant companies, collects data from their websites and news sources, and generates a full competitive report in Markdown and PDF.

---

## Overview
The pipeline orchestrates a chain of specialized agents using **LangGraph** each with its own set of tools. From a single input ("Neobanks in Spain", "UI design tools", etc.) the system:

1. Discovers the 5 most relevant companies in the sector.
2. Collects recent news and website content for each company **in parallel**.
3. Indexes all collected data into a vector store (ChromaDB) for semantic search (RAG).
4. Generates a comparative feature matrix and a SWOT analysis.
5. Produces a RAG-enriched market report in both `.md` and `.pdf`.

The entire process is accessible through a real-time web interface built with FastAPI and Server-Sent Events.

---

## Architecture

Input (category)
        │
        ▼
        
┌─────────────────┐

│  company_node   │  → Discovers the top 5 companies

└────────┬────────┘
         │
         ▼
         
┌─────────────────────────────────────────────┐

│              parallel_node                  │

│  ┌─────────────────┐  ┌──────────────────┐  │

│  │   Branch A      │  │    Branch B      │  │

│  │   (News)        │  │  (Web Scraping)  │  │

│  │ scraping_agent  │  │ website_finder   │  │

│  │ → DuckDB        │  │ website_sitemap  │  │

│  └─────────────────┘  │ features_agent   │  │

│                       │ → DuckDB         │  │

│                       └──────────────────┘  │

└───────────────┬─────────────────────────────┘
                │                
                ▼
                
┌───────────────────────┐

│    embedding_node     │  → Indexes data into ChromaDB (RAG)

└──────────┬────────────┘
           │           
           ▼
           
┌─────────────────────────────┐

│    structuring_swot_node    │  → Feature matrix + SWOT (parallel)

└──────────┬──────────────────┘
           │           
           ▼
           
┌─────────────────────┐

│     report_node     │  → Generates .md and .pdf report (RAG-enriched)

└─────────────────────┘

### Key modules

| Directory | Responsibility |
|---|---|
| `company_agent/` | Company discovery with validation guardrails |
| `scraping_agent/` | News scraping and storage in DuckDB |
| `website_agents/` | Official URL lookup, sitemap extraction, and web content scraping |
| `analysis_agents/` | Features extraction, RAG (ChromaDB), structuring, SWOT, and report generation |
| `evaluation/` | Per-node metrics tracking (time, CPU, RAM, quality) |
| `frontend/` | Single-page web interface (HTML/JS) |
| `app.py` | FastAPI server with SSE for real-time status updates |
| `graph.py` | LangGraph graph definition and all node functions |
| `pipeline_state.py` | Shared state (`TypedDict`) passed between nodes |

---

## ⚙️ Requirements

- Python 3.12+
- Anthropic API key

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd <repo-name>

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the Spanish spaCy model
python -m spacy download es_core_news_md
```

### Environment variables

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=sk-ant-...
```

---

## 🚀 Usage

### Web interface (recommended)

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open [http://localhost:8000](http://localhost:8000), type a category, and watch the pipeline progress in real time.

### Command line

```bash
python graph.py
# Enter the category when prompted, e.g.: "Neobanks in Spain"
```

To resume an interrupted run:

```bash
python graph.py --resume <thread_id>
```

---

## 📤 Generated outputs

After each run the following files are written to the project root:

| File | Contents |
|---|---|
| `market_report.md` | Full market report in Markdown |
| `market_report.pdf` | Same report in PDF format |
| `comparative_matrix.csv` | Per-company feature comparison matrix |
| `company_intelligence.duckdb` | Scraped news database |
| `website_intel.duckdb` | Extracted web content database |
| `features_index/` | ChromaDB vector index |
| `eval_runs.jsonl` | Per-run evaluation metrics history |

---

## 📊 Evaluation system

The `evaluation/` module automatically records performance and quality metrics for every node:

- **Performance:** wall-clock execution time, CPU time consumed, RAM delta.
- **Quality:** companies found, pages scraped, chunks indexed, matrix completeness, SWOT coverage, report word count, PDF size.

A summary is printed to the console at the end of each run. When two or more runs exist for the same sector, an automatic comparison between them is shown.

---

## REST API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/run` | Start the pipeline with `{"category": "..."}` |
| `GET` | `/api/status/{run_id}` | SSE stream with real-time logs |
| `GET` | `/api/result/{run_id}` | Final result as JSON |
| `GET` | `/api/history` | List of past runs |
| `GET` | `/api/download/{filename}` | Download a generated file |
| `GET` | `/` | Web interface |

---

## 🛠️ Tech stack

| Category | Technology |
|---|---|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM | [Anthropic Claude](https://www.anthropic.com) via [smolagents](https://github.com/huggingface/smolagents) |
| Vector store (RAG) | [ChromaDB](https://www.trychroma.com) |
| Embeddings | [sentence-transformers](https://www.sbert.net) (multilingual, CPU) |
| NLP | [spaCy](https://spacy.io) (`es_core_news_md`) |
| Database | [DuckDB](https://duckdb.org) |
| Web scraping | [Scrapling](https://github.com/D4Vinci/Scrapling) |
| Web API | [FastAPI](https://fastapi.tiangolo.com) + [Uvicorn](https://www.uvicorn.org) |
| PDF generation | ReportLab / fpdf2 |

---

## 📁 Project structure

├── app.py                        # FastAPI server
├── graph.py                      # LangGraph orchestrator
├── pipeline_state.py             # Shared pipeline state
├── eval_report.py                # Evaluation report
├── requirements.txt
├── .gitignore
│
├── company_agent/
│   ├── company_agent.py
│   └── tools/
│       ├── company_tools.py
│       └── company_guardrails.py
│
├── scraping_agent/
│   ├── scraping_agent.py
│   ├── scraping_storage_agent.py
│   └── tools/
│
├── website_agents/
│   ├── website_finder_agent.py
│   ├── website_sitemap_agent.py
│   ├── website_content_agent.py
│   ├── website_storage_agent.py
│   └── tools/
│
├── analysis_agents/
│   ├── features_agent.py
│   ├── rag_agent.py
│   ├── structuring_agent.py
│   ├── swot_agent.py
│   ├── report_agent.py
│   └── tools/
│       ├── features_tools.py
│       ├── rag_tools.py
│       ├── structuring_tools.py
│       ├── swot_tools.py
│       └── report_tools.py
│
├── evaluation/
│   ├── tracker.py
│   ├── quality.py
│   ├── llm_judge.py
│   └── __init__.py
│
└── frontend/
    └── index.html

---

## License

This project is distributed under a License. See the `LICENSE` file for details.









