# -*- coding: utf-8 -*-
from __future__ import annotations
import time
 
"""
website_storage_tools.py
────────────────────────
Tools for the website_storage_agent (Agent 4 of the web pipeline).

Responsabilities:
1. Validate each row in SCRAPED_CONTENT, discarding rows whose content is empty, too short, or matches known "no-data" patterns
(e.g. error message, boilerplate, placeholder text).
2. Persist validated rows into DuckDB (file: website_intel.duckdb)
3. Export and clean data to website_content.csv.

DuckDB Schema
──────
Table: website_intel
  - id             INTEGER PRIMARY KEY (autoincrement)
  - company_name   VARCHAR
  - page_url       VARCHAR 
  - section_label  VARCHAR
  - clean_content  VARCHAR
  - scraped_at     TIMESTAMP (UTC)
 
Table: website_summary
  - company_name   VARCHAR PRIMARY KEY
  - n_pages        INTEGER   
  - n_rows         INTEGER   
  - last_updated   TIMESTAMP
"""

import csv, os, re, unicodedata, duckdb
from datetime import datetime, timezone
from typing import Dict, List
from smolagents import tool
from website_agents.tools.website_content_tools import SCRAPED_CONTENT

DB_PATH="website_intel.duckdb"
_conn: duckdb.DuckDBPyConnection | None = None

def _init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create tables if they do not exist yet"""
    conn.execute("CREATE SEQUENCE IF NOT EXISTS website_intel_id_seq START 1;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS website_intel (
            id            INTEGER  DEFAULT nextval('website_intel_id_seq') PRIMARY KEY,
            company_name  VARCHAR  NOT NULL,
            page_url      VARCHAR  NOT NULL,
            section_label VARCHAR,
            clean_content VARCHAR,
            scraped_at    TIMESTAMP DEFAULT now()
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS website_summary (
            company_name  VARCHAR  PRIMARY KEY,
            n_pages       INTEGER,
            n_rows        INTEGER,
            last_updated  TIMESTAMP
        )
    """)

def get_conn() -> duckdb.DuckDBPyConnection:
    """Return (and lazily create) the shared DuckDB connection"""
    global _conn
    if _conn is None: 
        _conn = duckdb.connect(DB_PATH)
        _init_schema(_conn)
    return _conn

INVALID_PATTERNS = re.compile(
    r"(no\s+se\s+pudo\s+(encontrar|acceder|obtener|cargar)"
    r"|no\s+(se\s+encontr[oó]|hay|tiene|hubo)\s+(información|contenido|datos|resultados?|texto)"
    r"|la\s+p[aá]gina\s+(no\s+ten[ií]a|no\s+conten[ií]a|est[aá]\s+vac[ií]a)"
    r"|p[aá]gina\s+no\s+encontrada"
    r"|404\s*(not\s*found|error|p[aá]gina)?"
    r"|access\s+denied|forbidden|error\s+\d{3}"
    r"|could\s+not\s+(fetch|retrieve|load|find)"
    r"|no\s+content\s+(found|available)"
    r"|sin\s+(contenido|información|resultados?)"
    r"|informaci[oó]n\s+no\s+disponible"
    r"|contenido\s+no\s+disponible"
    r"|this\s+page\s+(does\s+not\s+exist|is\s+empty|could\s+not)"
    r"|cloudflare|captcha|just\s+a\s+moment"
    r"|\[content\]\s+(error|sin\s+contenido))"
    ,
    re.IGNORECASE,
)

#Minimum meaningful content length (characters)
MIN_CONTENT_LENGTH = 80 

def normalize(text: str) -> str:
    """NFC normalise + collapse whitespace"""
    text = unicodedata.normalize("NFC", text)
    return re.sub(R"\s+", " ", text).strip()

def is_valid_content(text: str) -> bool:
    """
    Return True if the text looks like real page content:
        - Long enogh to be meaningful.
        - Does not match any known invalid/error pattern.
    """

    text = normalize(text)
    if len(text) < MIN_CONTENT_LENGTH: return False
    if INVALID_PATTERNS.search(text): return False
    return True

def validate_rows(rows: List[Dict]) -> tuple[List[Dict], List[Dict]]:
    """
    Split a list of SCRAPED_CONTENT rows into (valid, invalid).
    Each valid row gets a new 'clean_content' key with the normalised text.
    """
    
    valid, invalid = [], []
    for row in rows:
        content = normalize(row.get("raw_content", ""))
        if is_valid_content(content):
            clean_row = dict(row)
            clean_row["clean_content"] = content
            valid.append(clean_row)
        else:
            invalid.append(row)
    
    return valid, invalid

#TOOLS

@tool
def validate_scraped_content() -> str:
    """
    Inspect SCRAPED_CONTENT (produced by website_content_agent) and report how many rows are
    valid vs. invalid, with examples of rejected rows.
    
    No data is wirtten anywhere; this is a dry-run check.
    
    Returns:
        A text report with counts and up to 5 examples of invalid rows.
    """
    
    if not SCRAPED_CONTENT:
        return (
            "[ws_storage] SCRAPED_CONTENT está vacío."
            "Ejecuta website_content_Agent primero."
        )
        
    valid, invalid = validate_rows(SCRAPED_CONTENT)
    lines = [
        "Validación completada:",
        f"  ✔ Filas válidas: {len(valid)}",
        f"  ✘ Filas inválidas {len(invalid)}"
    ]
    
    if invalid:
        lines.append("\nEjemplos de filas rechazadas (máx. 5):")
        for row in invalid[:5]:
            preview = (row.get("raw_content") or "")[:120].replace("\n", " ")
            lines.append(f"  · [{row.get('company_name')}] {row.get('gae_url')}")
            lines.append(f"    Contenido: \"{preview}\"")
    
    return "\n".join(lines)

@tool
def store_valid_content() -> str:
    """
    Validate all rows in SCRAPED_CONTENT and persist the valid ones into DuckDB (website_intel.duckdb),
    updating the website_summary table.
    
    Returns:
        Confirmation with the number of rows inserted per company.
    """
    
    if not SCRAPED_CONTENT:
        return (
            "[ws_storage] SCRAPED_CONTENT está vacío. "
            "Ejecuta website_content_agent primero."
        )
    
    valid, invalid = validate_rows(SCRAPED_CONTENT)
    if not valid:
        return (
            f"[ws_storage] ninguna fila pasó la validación "
            f"({len(invalid)} rechazadas). Revisa el scraping."
        )
    
    conn = get_conn()
    now = datetime.now(timezone.utc)
    company_stats: Dict[str, Dict] = {}
    
    for row in valid:
        company = row["company_name"]
        conn.execute(
             """
            INSERT INTO website_intel
                (company_name, page_url, section_label, clean_content, scraped_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                company,
                row["page_url"],
                row.get("section_label", ""),
                row["clean_content"],
                now
            ],
        )
        stats = company_stats.setdefault(company, {"pages": set(), "rows": 0})
        stats["pages"].add(row["page_url"])
        stats["rows"] += 1
    
    #Upsert summary row per company
    for company, stats in company_stats.items():
        conn.execute(
            """
            INSERT INTO website_summary (company_name, n_pages, n_rows, last_updated)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (company_name) DO UPDATE SET
                n_pages      = excluded.n_pages,
                n_rows       = excluded.n_rows,
                last_updated = excluded.last_updated
            """,
            [company, len(stats["pages"]), stats["rows"], now]
        )
    
    lines = [
        f"Almacenamiento en DuckDB completado ({DB_PATH}):",
        f"  ✔ {len(valid)} filas insertadas",
        f"  ✘ {len(invalid)} filas descartadas por validación",
        "",
        "Detalle por empresa:"
    ]
    
    for company, stats in sorted(company_stats.items()):
        lines.append(
            f"  · {company}: {stats['rows']} filas "
            f"| {len(stats['pages'])} páginas únicas"
        )

    return "\n".join(lines)

@tool
def export_valid_content_to_csv(output_path: str = "website_content.csv") -> str:
    """
    Validate all rows in SCRAPED_CONTENT and export only the valid ones to a CSV file
    (default name: website_content.csv).
    
    Columns: company_name, page_url, section_label, clean_content, scraped_at.
    
    Args:
        output_path: Destination CSV path (default: 'website_content.csv').
        
    Returns:
        Confirmation with the number of rows written and the absolute path.
    """

    if not SCRAPED_CONTENT:
        return (
            "[ws_storage] SCRAPED_CONTENT está vacío."
            "Ejecuta website_content_agent primero."
        )
    
    valid, invalid = validate_rows(SCRAPED_CONTENT)
    if not valid:
        return (
            f"[ws_storage] Ninguna fila válida para exportar."
            f"({len(invalid)} rechazar)."
        )
    
    fieldnames = ["company_name", "page_url", "section_label", "clean_content", "scraped_at"]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(valid)
    
    return(
        f"[ws_storage] {len(valid)} filas exportadas a '{output_path}' "
        f"({len(invalid)} descartadas).\n"
        f"  Ruta absoluta: {os.path.abspath(output_path)}"
    )
    
@tool
def store_and_export(output_path: str = "website_content.csv") -> str:
    """
    Convenience tool: validates SCRAPED_CONTENT, persist valid rows in DuckDB and exports
    them to website_content.csv in a single call.
    
    This is the recommend entry point for the agent to call.
    
    Args:
        output_path: Destination CSV path (default: 'website_content.csv')
    
    Returns:
        Combined confirmation from both the DuckDB and CSV operations.
    """

    conn = get_conn()
    rows = conn.execute(
        """
        SELECT company_name, n_pages, n_rows, last_updated
        FROM website_summary
        ORDER BY n_rows DESC
        """
    ).fetchall()
    
    if not rows:
        return "[ws_storage] La base de datos website_intel.duckdb está vacía."

    header = f"{'Empresa':<30} {'Páginas':>8} {'Filas':>6} {'Última actualización'}"
    sep = "-" * 72
    lines = [header, sep]
    for name, n_pages, n_rows, ts in rows:
        ts_str = ts.strftime("%Y-%m-%d %H:%M") if ts else "-"
        lines.append(f"{name:<30} {n_pages:>8} {n_rows:>6}, {ts_str}")
    
    total_rows = conn.execute(
        "SELECT COUNT(*) FROM website_intel"
    ).fetchone()[0]
    lines += [sep, f"{'TOTAL': <30} {'': >8} {total_rows:>6}"]
    
    return "\n".join(lines)

@tool
def get_website_storage_summary() -> str:
    """
    Return a formatted summary table of all companies stored in
    website_intel.duckdb (company, pages, rows, last update).
 
    Returns:
        A plain-text table or a message if the database is empty.
    """
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT company_name, n_pages, n_rows, last_updated
        FROM website_summary
        ORDER BY n_rows DESC
        """
    ).fetchall()
 
    if not rows:
        return "[ws_storage] La base de datos website_intel.duckdb está vacía."
 
    header = f"{'Empresa':<30} {'Páginas':>8} {'Filas':>6}  {'Última actualización'}"
    sep = "─" * 72
    lines = [header, sep]
    for name, n_pages, n_rows, ts in rows:
        ts_str = ts.strftime("%Y-%m-%d %H:%M") if ts else "—"
        lines.append(f"{name:<30} {n_pages:>8} {n_rows:>6}  {ts_str}")
 
    total_rows = conn.execute(
        "SELECT COUNT(*) FROM website_intel"
    ).fetchone()[0]
    lines += [sep, f"{'TOTAL':<30} {'':>8} {total_rows:>6}"]
 
    return "\n".join(lines)
    
@tool
def query_website_intel(company_name: str, limit: int = 10) -> str:
    """
    Query DuckDB and return stored clean content for a given company.
    
    Args:
        company_name: Company name to look up.
        limit: Maximum number of rows to return (default: 10).
    
    Returns:
        Formatted text with URL, section and a content preview per row.
    """
    
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT page_url, section_label, clean_content, scraped_at
        FROM website_intel
        WHERE lower(company_name) = lower(?)
        ORDER BY scraped_at
        LIMIT ?
        """,
        [company_name, limit]
    ).fetchall()

    if not rows:
        return (
            f"[ws_storage] No hay datos en DuckDB para '{company_name}'. "
            "Ejecuta store_valid_content primero."
        )
    
    lines = [f"=== {company_name} ({len(rows)} filas) ==="]
    for url, section, content, ts in rows:
        preview = (content or "")[:200] + ("..." if len(content or "") > 200 else "")
        lines.append(f"\n[{section}] {url}")
        lines.append(f"  {preview}")
        
    return "\n".join(lines)    
        