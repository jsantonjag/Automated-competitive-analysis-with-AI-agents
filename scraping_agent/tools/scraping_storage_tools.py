from __future__ import annotations

"""
storage_tools.py
────────────────
Tools to clean raw scraped text blocks and persist them into a DuckDB database.

Schema
──────
Table: companies
  - id             INTEGER PRIMARY KEY AUTOINCREMENT
  - company_name   VARCHAR
  - source         VARCHAR   (e.g. 'google_news', 'techcrunch', …)
  - raw_text       VARCHAR   (original snippet)
  - clean_text     VARCHAR   (normalised snippet)
  - scraped_at     TIMESTAMP (UTC, set at insert time)

Table: company_summary
  - company_name   VARCHAR PRIMARY KEY
  - n_sources      INTEGER   (how many sources had data)
  - n_snippets     INTEGER   (total snippets stored)
  - last_updated   TIMESTAMP
"""

import re
import unicodedata
from datetime import datetime, timezone
from typing import Dict, List

import duckdb
from smolagents import tool

# database connection (in-process, file-backed)
DB_PATH = "company_intelligence.duckdb"
_conn: duckdb.DuckDBPyConnection | None = None

def get_conn() -> duckdb.DuckDBPyConnection:
    """Return (and lazily create) the shared DuckDB connection."""
    global _conn
    if _conn is None:
        _conn = duckdb.connect(DB_PATH)
        _init_schema(_conn)
    return _conn


def _init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create tables if they do not exist yet."""
    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS companies_id_seq START 1;
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            id           INTEGER DEFAULT nextval('companies_id_seq') PRIMARY KEY,
            company_name VARCHAR NOT NULL,
            source       VARCHAR NOT NULL,
            raw_text     VARCHAR,
            clean_text   VARCHAR,
            scraped_at   TIMESTAMP DEFAULT now()
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS company_summary (
            company_name VARCHAR PRIMARY KEY,
            n_sources    INTEGER,
            n_snippets   INTEGER,
            last_updated TIMESTAMP
        )
    """)

# Text cleaning 

_BOILERPLATE = re.compile(
    r"(cookies?|aviso legal|privacidad|términos|suscri[bp]|iniciar sesión"
    r"|newsletter|publicidad|anunci|menú|navegación|skip to)",
    re.IGNORECASE
)

def clean_text(text: str) -> str:
    """
    Normalise a raw scraped text block:
    1. Unicode NFC normalisation
    2. Collapse whitespace
    3. Strip leading/trailing spaces
    4. Remove lines that are pure boilerplate
    5. Truncate to 800 chars
    """
    # NFC unicode normalisation
    text = unicodedata.normalize("NFC", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Drop boilerplate sentences
    sentences = [s.strip() for s in re.split(r"[.·•\n]", text) if s.strip()]
    sentences = [s for s in sentences if not _BOILERPLATE.search(s)]
    text = ". ".join(sentences)
    # Final truncation
    return text[:800]

def parse_source_blocks(raw_blocks: List[str]) -> List[Dict[str, str]]:
    """
    Convert the list of raw text blocks stored by scraping_tools
    (format: "[source_name]\n  · snippet\n  · snippet …")
    into a list of {"source": ..., "text": ...} dicts.
    """
    parsed: List[Dict[str, str]] = []
    source_re = re.compile(r"^\[([^\]]+)\]")

    for block in raw_blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue

        # First line contains the source tag, e.g. "[techcrunch]"
        m = source_re.match(lines[0])
        source = m.group(1) if m else "unknown"

        # Remaining lines are the actual snippets
        snippets = [
            re.sub(r"^\s*[·•\-]\s*", "", ln).strip()
            for ln in lines[1:]
            if ln.strip() and not ln.strip().startswith("[")
        ]

        for snippet in snippets:
            if snippet:
                parsed.append({"source": source, "text": snippet})

    return parsed

# smolagents tools

@tool
def clean_and_store_company(company_name: str, raw_blocks: List[str]) -> str:
    """
    Clean the raw scraped text blocks for ONE company and persist them in DuckDB.

    Args:
        company_name: Name of the company (e.g. 'Revolut').
        raw_blocks:   List of raw text blocks as produced by scraping_tools.COMPANY_DATA[company_name].

    Returns:
        A confirmation string with the number of rows inserted.
    """
    
    conn = get_conn()
    parsed = parse_source_blocks(raw_blocks)
    if not parsed:
        return f"[storage] '{company_name}': no se encontraron snippets válidos tras la limpieza."
    
    now = datetime.now(timezone.utc)
    rows_inserted = 0
    sources_seen: set[str] = set()
    
    for item in parsed:
        raw_text = item["text"]
        clean = clean_text(raw_text)
        if not clean: 
            continue
        
        conn.execute(
            """
            INSERT INTO companies (company_name, source, raw_text, clean_text, scraped_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [company_name, item["source"], raw_text, clean, now]
        )
        
        rows_inserted +=1
        sources_seen.add(item["source"])
        
        
    #insert summarry row
    conn.execute(
        """
        INSERT INTO company_summary (company_name, n_sources, n_snippets, last_updated)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (company_name) DO UPDATE SET
            n_sources    = excluded.n_sources,
            n_snippets   = excluded.n_snippets,
            last_updated = excluded.last_updated
        """,
        [company_name, len(sources_seen), rows_inserted, now],
    )
        
    return(
        f"[storage] '{company_name}': {rows_inserted} snippets almacenados."
        f" de {len(sources_seen)} enlaces -> {DB_PATH}"
    )
    
@tool
def store_all_companies(company_data: dict) -> str:
    """
    Clean and store scraped data for ALL companies at once.
    Receives the full COMPANY_DATA dict from scraping_tools.

    Args:
        company_data: Dict mapping company_name -> list of raw text blocks.
                      Pass it as: store_all_companies(COMPANY_DATA)

    Returns:
        A summary string with total rows inserted per company.
    """
    
    if not company_data:
        return "[storage] company_data está vacío. Ejecuta scraping_agent primero."
    
    lines: List[str] = []
    for company_name, raw_blocks in company_data.items():
        result = clean_and_store_company(company_name, raw_blocks)
        lines.append(result)
        
    return "Almacenamiento completado:\n" + "\n".join(lines)

@tool
def query_company_data(company_name: str) -> str:
    """
    Query DuckDB and return all clean snippets stored for a given company.

    Args:
        company_name: Name of the company to query.

    Returns:
        Formatted text with all stored snippets grouped by source.
    """
    
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT source, clean_text, scraped_at
        FROM companies
        WHERE lower(company_name) = lower(?)
        ORDER BY source, scraped_at
        """,
        [company_name],
    ).fetchall()
    
    if not rows:
        return f"[storage] There is no data in DuckDB for '{company_name}'."

    lines = [f"=== {company_name} ({len(rows)} snippets) ==="]
    current_source = None
    for source, text, ts in rows:
        if source != current_source:
            lines.append(f"\n[{source}]")
            current_source = source
        lines.append(f"  • {text}")

    return "\n".join(lines)
    
@tool
def get_storage_summary() -> str:
    """
    Return a summary table of all companies stored in DuckDB
    (company name, number of sources, total snippets, last update).

    Returns:
        A formatted text table.
    """
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT company_name, n_sources, n_snippets, last_updated
        FROM company_summary
        ORDER BY n_snippets DESC
        """
    ).fetchall()

    if not rows:
        return "[storage] Database is empty."

    header = f"{'Company':<30} {'Source':>7} {'Snippets':>9} {'Last update'}"
    sep = "─" * 70
    lines = [header, sep]
    for name, n_src, n_snip, ts in rows:
        ts_str = ts.strftime("%Y-%m-%d %H:%M") if ts else "—"
        lines.append(f"{name:<30} {n_src:>7} {n_snip:>9} {ts_str}")

    return "\n".join(lines)


@tool
def export_to_csv(output_path: str = "company_intelligence.csv") -> str:
    """
    Export the full companies table from DuckDB to a CSV file.

    Args:
        output_path: Path for the output CSV file (default: company_intelligence.csv).

    Returns:
        Confirmation message with the number of rows exported.
    """
    conn = get_conn()
    conn.execute(f"COPY companies TO '{output_path}' (HEADER, DELIMITER ',')")
    n = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
    return f"[storage] {n} filas exportadas a '{output_path}'"
