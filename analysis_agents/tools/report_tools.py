# -*- coding: utf-8 -*-
from __future__ import annotations

"""
report_tools.py  (versión RAG)
--------------------------------
Tools para el report_agent (Agente 7).

CAMBIO RAG
-----------
Antes: matrix_text + swot_text truncados a 3000 chars → LLM.
Ahora: matrix_text + swot_text + fragmentos RAG adicionales → LLM.
       Se hacen 3 búsquedas semánticas orientadas a estrategia para
       enriquecer el contexto antes de generar el informe.
       Fallback graceful si ChromaDB no está disponible.

Outputs: market_report.md + market_report.pdf
"""

import json, os, re
from datetime import datetime, timezone
from typing import List
from smolagents import tool


_RAG_QUERIES_REPORT = [
    "propuesta de valor diferenciadores clave ventaja competitiva",
    "estrategia expansión crecimiento mercado",
    "tecnología innovación producto características únicas",
]


def _retrieve_rag_enrichment(company_names: List[str], top_k: int = 3) -> str:
    """Recupera fragmentos estratégicos de ChromaDB para todas las empresas en paralelo."""
    try:
        from analysis_agents.tools.rag_tools import _get_embedder, _get_collection
        from concurrent.futures import ThreadPoolExecutor, as_completed

        embedder   = _get_embedder()
        collection = _get_collection()
        if collection.count() == 0:
            return ""

        # Pre-compute all query embeddings at once (batch)
        query_embeddings = [embedder.encode(q).tolist() for q in _RAG_QUERIES_REPORT]

        def _fetch_company(company: str) -> tuple:
            seen, frags = set(), []
            for qemb in query_embeddings:
                results = collection.query(
                    query_embeddings=[qemb],
                    n_results=min(top_k, collection.count()),
                    where={"company": company},
                )
                for doc in results.get("documents", [[]])[0]:
                    if doc not in seen:
                        seen.add(doc)
                        frags.append(doc)
            return company, frags

        sections: List[str] = []
        with ThreadPoolExecutor(max_workers=min(5, len(company_names))) as executor:
            futures = {executor.submit(_fetch_company, c): c for c in company_names}
            # Preserve order
            results_map = {}
            for fut in as_completed(futures):
                company, frags = fut.result()
                results_map[company] = frags
        for company in company_names:
            frags = results_map.get(company, [])
            if frags:
                sections.append(f"### {company}\n" + "\n---\n".join(frags[:6]))
        return "\n\n".join(sections)
    except Exception as exc:
        print(f"[report] RAG enrichment no disponible: {exc}")
        return ""


def _call_llm_report(category: str, matrix_text: str, swot_text: str, rag_enrichment: str, company_names: list) -> str:
    """Llama a la API de Anthropic para generar el informe completo."""
    import os, urllib.request, urllib.error
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "[report] Error: ANTHROPIC_API_KEY no está definida en .env"

    # ── Truncar cada sección para no superar ~12 000 tokens de entrada ────────
    # Regla práctica: 1 token ≈ 4 chars en español.
    # Reservamos ~8 000 tokens para matrix+swot (32 000 chars), 2 000 para RAG.
    matrix_text   = matrix_text[:16_000]
    swot_text     = swot_text[:16_000]
    rag_enrichment = rag_enrichment[:8_000] if rag_enrichment else ""

    rag_section = (
        f"\n\n## Contexto adicional (RAG — fragmentos estratégicos)\n{rag_enrichment}"
        if rag_enrichment else ""
    )
    # Lista explícita de empresas para que el LLM no omita ninguna
    companies_str = ", ".join(company_names) if company_names else "las empresas del sector"

    prompt = f"""
Eres un consultor estratégico senior. Redacta un informe de mercado profesional
sobre el sector: "{category}".

IMPORTANTE: El análisis debe cubrir OBLIGATORIAMENTE estas {len(company_names)} empresas: {companies_str}.
No omitas ninguna de ellas en ninguna sección del informe.

## Matriz comparativa de productos y precios
{matrix_text}

## Análisis DAFO de cada empresa
{swot_text}
{rag_section}

El informe debe incluir:
1. **Resumen ejecutivo** (3-4 párrafos): estado del mercado y principales hallazgos.
2. **Análisis competitivo**: comparativa de propuestas de valor, precios y servicios.
   Incluye una subsección por empresa: {companies_str}.
3. **Fortalezas y debilidades del sector**: patrones comunes entre las empresas.
4. **Oportunidades y amenazas del mercado**: tendencias externas relevantes.
5. **Conclusiones e insights estratégicos**: 5 recomendaciones accionables.

Tono: consultoría de negocio. Sé concreto y usa datos cuando estén disponibles.
Formato Markdown con cabeceras ##, negrita **texto** y listas con guion -.
No uses cabeceras ###, usa solo ## para secciones principales y **negrita** para subsecciones.
"""
    print(f"[report] Tamaño del prompt: {len(prompt):,} chars "
          f"(≈{len(prompt)//4:,} tokens estimados)")

    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4000,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    # Retry con backoff exponencial para errores 429 (rate limit)
    import time
    max_retries = 4
    wait = 30  # segundos iniciales de espera
    for attempt in range(max_retries):
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            return "".join(b.get("text", "") for b in data.get("content", []))
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            if exc.code == 429 and attempt < max_retries - 1:
                print(f"[report] Rate limit (429) — esperando {wait}s antes de reintentar "
                      f"(intento {attempt+1}/{max_retries})...")
                time.sleep(wait)
                wait *= 2  # backoff exponencial: 30s → 60s → 120s → 240s
                continue
            return f"[report] Error generando informe: HTTP {exc.code} — {body[:400]}"
        except Exception as exc:
            return f"[report] Error generando informe: {exc}"
    return "[report] Error: se agotaron los reintentos por rate limit."


def _markdown_to_pdf(markdown_text: str, output_path: str, category: str) -> None:
    """
    Convierte Markdown a PDF probando tres motores en orden:
      1. ReportLab  (pip install reportlab)
      2. fpdf2      (pip install fpdf2)
      3. weasyprint (pip install weasyprint)

    Lanza RuntimeError si ninguno está disponible.
    """
    ts = datetime.now(timezone.utc).strftime("%d/%m/%Y")

    # ── Motor 1: ReportLab ────────────────────────────────────────────────────
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable

        doc = SimpleDocTemplate(
            output_path, pagesize=A4,
            leftMargin=2.5*cm, rightMargin=2.5*cm,
            topMargin=2.5*cm, bottomMargin=2.5*cm,
            title=f"Informe de mercado: {category}",
            author="TFG de Ingeniería Informática (CUNEF Universidad)",
        )
        styles = getSampleStyleSheet()
        title_style  = ParagraphStyle("RT", parent=styles["Title"],
            fontSize=22, textColor=colors.HexColor("#1a1a2e"), spaceAfter=6)
        sub_style    = ParagraphStyle("RS", parent=styles["Normal"],
            fontSize=11, textColor=colors.HexColor("#4a4a8a"), spaceAfter=20)
        h2_style     = ParagraphStyle("H2", parent=styles["Heading2"],
            fontSize=14, textColor=colors.HexColor("#1a1a2e"), spaceBefore=14, spaceAfter=6)
        h3_style     = ParagraphStyle("H3", parent=styles["Heading3"],
            fontSize=12, textColor=colors.HexColor("#2a2a5a"), spaceBefore=8, spaceAfter=4)
        body_style   = ParagraphStyle("B",  parent=styles["Normal"],
            fontName="Times-Roman", fontSize=11, leading=16, spaceAfter=6, alignment=4)
        bullet_style = ParagraphStyle("BL", parent=styles["Normal"],
            fontName="Times-Roman", fontSize=11, leading=15, leftIndent=16, spaceAfter=3, bulletIndent=6, alignment=4)

        story = []
        story.append(Paragraph("Informe de Mercado", title_style))
        story.append(Paragraph(f"Sector: {category}", sub_style))
        story.append(Paragraph(f"Generado: {ts} · TFG de Ingeniería Informática (CUNEF Universidad)", sub_style))
        story.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor("#4a4a8a")))
        story.append(Spacer(1, 0.4*cm))

        def _safe(text: str) -> str:
            """Escapa caracteres especiales XML para ReportLab."""
            return (text
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;"))

        for line in markdown_text.split("\n"):
            line = line.rstrip()
            if line.startswith("## ") or line.startswith("# "):
                story.append(Spacer(1, 0.3*cm))
                heading = _safe(re.sub(r"^#+\s*", "", line))
                story.append(Paragraph(heading, h2_style))
            elif line.startswith("### "):
                heading = _safe(re.sub(r"^#+\s*", "", line))
                story.append(Paragraph(heading, h3_style))
            elif line.startswith("- ") or line.startswith("* "):
                text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", _safe(line[2:]))
                story.append(Paragraph(f"• {text}", bullet_style))
            elif line.strip() in ("", "---"):
                story.append(Spacer(1, 0.2*cm))
            else:
                text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", _safe(line))
                if text.strip():
                    story.append(Paragraph(text, body_style))

        doc.build(story)
        print("[report] PDF generado con ReportLab.")
        return

    except ImportError:
        print("[report] ReportLab no instalado, probando fpdf2...")
    except Exception as exc:
        print(f"[report] ReportLab falló ({exc}), probando fpdf2...")

    # ── Motor 2: fpdf2 ────────────────────────────────────────────────────────
    try:
        from fpdf import FPDF

        class _PDF(FPDF):
            def header(self):
                self.set_font("Helvetica", "B", 10)
                self.set_text_color(74, 74, 138)
                self.cell(0, 8, f"Informe de Mercado — {category}", ln=True, align="C")
                self.ln(2)

            def footer(self):
                self.set_y(-12)
                self.set_font("Helvetica", "", 8)
                self.set_text_color(150, 150, 150)
                self.cell(0, 8, f"Página {self.page_no()} | {ts}", align="C")

        pdf = _PDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        for line in markdown_text.split("\n"):
            line = line.rstrip()
            # Quitar markdown bold para fpdf2 (no soporta XML inline aquí)
            plain = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
            plain = re.sub(r"\*(.+?)\*",   r"\1", plain)

            if line.startswith("# ") and not line.startswith("## "):
                pdf.set_font("Georgia", "B", 16)
                pdf.set_text_color(26, 26, 46)
                pdf.multi_cell(0, 9, re.sub(r"^#+\s*", "", plain))
                pdf.ln(2)
            elif line.startswith("## "):
                pdf.set_font("Georgia", "B", 13)
                pdf.set_text_color(26, 26, 46)
                pdf.multi_cell(0, 8, re.sub(r"^#+\s*", "", plain))
                pdf.ln(1)
            elif line.startswith("### "):
                pdf.set_font("Georgia", "B", 11)
                pdf.set_text_color(42, 42, 90)
                pdf.multi_cell(0, 7, re.sub(r"^#+\s*", "", plain))
                pdf.ln(1)
            elif line.startswith("- ") or line.startswith("* "):
                pdf.set_font("Georgia", "", 10)
                pdf.set_text_color(40, 40, 40)
                pdf.multi_cell(0, 6, "  •  " + plain[2:])
            elif line.strip() in ("", "---"):
                pdf.ln(3)
            else:
                pdf.set_font("Georgia", "", 10)
                pdf.set_text_color(40, 40, 40)
                if plain.strip():
                    pdf.multi_cell(0, 6, plain)

        pdf.output(output_path)
        print("[report] PDF generado con fpdf2.")
        return

    except ImportError:
        print("[report] fpdf2 no instalado, probando weasyprint...")
    except Exception as exc:
        print(f"[report] fpdf2 falló ({exc}), probando weasyprint...")

    # ── Motor 3: weasyprint ───────────────────────────────────────────────────
    try:
        from weasyprint import HTML

        # Convertir Markdown a HTML básico
        html_lines = [
            "<html><head><meta charset='utf-8'>",
            f"<title>Informe de mercado: {category}</title>",
            "<style>",
            "  body { font-family: Arial, sans-serif; font-size: 11pt; margin: 2cm; color: #222; }",
            "  h1 { font-size: 20pt; color: #1a1a2e; }",
            "  h2 { font-size: 14pt; color: #1a1a2e; border-bottom: 1px solid #ccc; padding-bottom: 4px; }",
            "  h3 { font-size: 12pt; color: #333; }",
            "  li { margin-bottom: 4px; }",
            "  .cover { text-align: center; padding: 2cm 0; color: #4a4a8a; }",
            "</style></head><body>",
            f"<div class='cover'><h1>Informe de Mercado</h1>",
            f"<p>Sector: <b>{category}</b><br>Generado: {ts}</p></div><hr>",
        ]
        for line in markdown_text.split("\n"):
            line = line.rstrip()
            line_html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
            line_html = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", line_html)
            if line.startswith("# ") and not line.startswith("## "):
                html_lines.append(f"<h1>{line_html.lstrip('# ')}</h1>")
            elif line.startswith("## "):
                html_lines.append(f"<h2>{line_html.lstrip('# ')}</h2>")
            elif line.startswith("### "):
                html_lines.append(f"<h3>{line_html.lstrip('# ')}</h3>")
            elif line.startswith("- ") or line.startswith("* "):
                html_lines.append(f"<li>{line_html[2:]}</li>")
            elif line.strip() == "---":
                html_lines.append("<hr>")
            elif line.strip() == "":
                html_lines.append("<br>")
            else:
                html_lines.append(f"<p>{line_html}</p>")
        html_lines.append("</body></html>")

        HTML(string="\n".join(html_lines)).write_pdf(output_path)
        print("[report] PDF generado con weasyprint.")
        return

    except ImportError:
        pass
    except Exception as exc:
        print(f"[report] weasyprint falló: {exc}")

    # ── Ningún motor disponible ───────────────────────────────────────────────
    raise RuntimeError(
        "No se encontró ningún motor PDF. Instala alguno de estos:\n"
        "  pip install reportlab        # recomendado\n"
        "  pip install fpdf2            # alternativa ligera\n"
        "  pip install weasyprint       # alternativa HTML→PDF\n"
        "O añádelos a requirements.txt y ejecuta: pip install -r requirements.txt"
    )


@tool
def generate_report(
    category: str,
    matrix_text: str,
    swot_text: str,
    company_names: list,
    md_output: str = "market_report.md",
) -> str:
    """
    Genera el informe de mercado en Markdown enriquecido con RAG.

    Proceso:
    1. Recupera fragmentos estratégicos adicionales de ChromaDB (RAG enrichment).
    2. Combina matrix_text + swot_text + fragmentos RAG en el prompt.
    3. Llama al LLM para redactar el informe de consultoría.
    4. Guarda el resultado en market_report.md.

    Args:
        category:      Categoría de mercado del usuario.
        matrix_text:   Output de get_comparative_matrix().
        swot_text:     Output de get_swot_matrix_text().
        company_names: Lista de nombres de empresa para la búsqueda RAG.
        md_output:     Ruta del archivo Markdown (default: 'market_report.md').

    Returns:
        Confirmación con ruta y vista previa del informe.
    """
    rag_enrichment = _retrieve_rag_enrichment(company_names)
    rag_label = f"{len(rag_enrichment.splitlines())} líneas" if rag_enrichment else "no disponible"
    print(f"[report] RAG enrichment: {rag_label}")

    report_md = _call_llm_report(category, matrix_text, swot_text, rag_enrichment, company_names)
    if report_md.startswith("[report] Error"):
        return report_md

    with open(md_output, "w", encoding="utf-8") as f:
        f.write(report_md)

    excerpt = report_md[:400].replace("\n", " ") + "..."
    return (
        f"[report] Informe Markdown: '{md_output}'\n"
        f"  RAG enrichment: {rag_label}\n"
        f"  Ruta absoluta: {os.path.abspath(md_output)}\n\n"
        f"Vista previa:\n{excerpt}"
    )


@tool
def export_report_to_pdf(
    category: str,
    md_path: str = "market_report.md",
    pdf_output: str = "market_report.pdf",
) -> str:
    """
    Convierte el informe Markdown a PDF.
    Prueba tres motores en orden: ReportLab → fpdf2 → weasyprint.
    Instala el que prefieras: pip install reportlab  /  pip install fpdf2

    Args:
        category:   Categoría de mercado (usada en la portada).
        md_path:    Ruta del Markdown (default: 'market_report.md').
        pdf_output: Ruta del PDF (default: 'market_report.pdf').

    Returns:
        Confirmación con ruta absoluta y tamaño del PDF.
    """
    if not os.path.exists(md_path):
        return f"[report] No se encontró '{md_path}'. Ejecuta generate_report primero."
    with open(md_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()
    try:
        _markdown_to_pdf(markdown_text, pdf_output, category)
    except RuntimeError as exc:
        return f"[report] Error generando PDF: {exc}"
    except Exception as exc:
        return f"[report] Error generando PDF: {exc}"
    size_kb = os.path.getsize(pdf_output) // 1024
    return (
        f"[report] PDF: '{pdf_output}' ({size_kb} KB)\n"
        f"  Ruta absoluta: {os.path.abspath(pdf_output)}"
    )


@tool
def generate_full_output(
    category: str,
    matrix_text: str,
    swot_text: str,
    company_names: list,
) -> str:
    """
    Genera el Markdown y el PDF en una sola llamada.

    Args:
        category:      Categoría de mercado.
        matrix_text:   Output de get_comparative_matrix().
        swot_text:     Output de get_swot_matrix_text().
        company_names: Lista de nombres de empresa para RAG enrichment.

    Returns:
        Confirmación combinada de ambos pasos.
    """
    md_result = generate_report(category, matrix_text, swot_text, company_names)
    if "Error" in md_result:
        return md_result
    return md_result + "\n\n" + export_report_to_pdf(category)
