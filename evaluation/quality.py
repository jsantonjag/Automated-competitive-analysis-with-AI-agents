# -*- coding: utf-8 -*-
from __future__ import annotations

"""
evaluation/quality.py
----------------------
Evaluación de calidad de los outputs del pipeline.

Qué evalúa
-----------
Cada función recibe el output de un nodo y devuelve un QualityResult con:
  · score        — float 0-100 (promedio ponderado de las dimensiones)
  · dimensiones  — dict con scores individuales y sus justificaciones
  · warnings     — lista de problemas detectados
  · passed       — True si score >= threshold (configurable, default 60)

Dimensiones evaluadas por nodo
-------------------------------
  company_node
    · relevance      — ¿los nombres son empresas reales del sector?  (regex + heurísticas)
    · diversity      — ¿hay variedad o son todas iguales?            (ratio único/total)
    · cleanliness    — ¿no contienen stopwords o ruido?              (blacklist)

  parallel_node (noticias)
    · coverage       — % empresas con ≥1 snippet
    · snippet_depth  — media de snippets por empresa
    · recency        — heurística: presencia de años recientes en texto

  parallel_node (web / features)
    · url_coverage   — % empresas con URL oficial encontrada
    · page_depth     — media de páginas scrapeadas por empresa
    · price_presence — % empresas con algún dato de precio detectado

  embedding_node
    · density        — chunks por empresa (objetivo ≥ 5)
    · balance        — std normalizada entre empresas (baja = mejor)

  structuring_node (matriz)
    · completeness   — % celdas no vacías / N/A
    · price_fill     — % empresas con precio no "No especificado"
    · plan_diversity — media de planes por empresa (más = mejor hasta 5)
    · coherence      — ¿plan_name + price + services son coherentes entre sí?

  swot_node
    · coverage       — % empresas con las 4 dimensiones DAFO
    · depth          — media de ítems por dimensión
    · specificity    — ratio de ítems que no son "Información insuficiente"
    · balance        — distribución equilibrada F/D/O/A por empresa

  report_node (informe Markdown)
    · structure      — presencia de las 5 secciones obligatorias del prompt
    · company_coverage — % empresas mencionadas explícitamente
    · length         — nº palabras (objetivo 800-3000)
    · data_density   — presencia de cifras/porcentajes
    · recommendations — presencia de ≥3 recomendaciones accionables

Uso
----
  from evaluation.quality import evaluate_all_outputs, QualityResult
  results = evaluate_all_outputs(final_state, company_names)
  for node, r in results.items():
      print(r.summary())
"""

import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════════════
# Umbral de calidad mínima por nodo
# ══════════════════════════════════════════════════════════════════════════════

THRESHOLDS: Dict[str, float] = {
    "company_node":     70.0,
    "news_branch":      55.0,
    "web_branch":       55.0,
    "embedding_node":   60.0,
    "structuring_node": 60.0,
    "swot_node":        65.0,
    "report_node":      70.0,
}

# Palabras que NUNCA deben aparecer como nombre de empresa
_COMPANY_BLACKLIST = {
    "top", "best", "mejores", "principales", "lista", "ranking",
    "empresas", "compañías", "companies", "startups", "players",
    "forbes", "wikipedia", "linkedin", "youtube", "google", "facebook",
    "contacto", "login", "inicio", "home", "blog", "noticias", "news",
    "sobre", "about", "services", "servicios", "productos", "products",
}

# Secciones que el prompt del informe exige explícitamente
_REQUIRED_REPORT_SECTIONS = [
    "resumen ejecutivo",
    "análisis competitivo",
    "fortalezas",       # "Fortalezas y debilidades"
    "oportunidades",    # "Oportunidades y amenazas"
    "conclusiones",     # "Conclusiones e insights"
]

# ══════════════════════════════════════════════════════════════════════════════
# Estructura de resultado
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QualityResult:
    node: str
    score: float                                    # 0–100
    dimensions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    warnings: List[str]                   = field(default_factory=list)
    threshold: float                      = 60.0
    passed: bool                          = False

    def __post_init__(self):
        self.passed = self.score >= self.threshold

    def summary(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        lines  = [
            f"\n  [{status}] {self.node}  score={self.score:.1f}/100"
            f"  (umbral={self.threshold})"
        ]
        for dim, info in self.dimensions.items():
            s   = info.get("score", 0)
            why = info.get("reason", "")
            bar = _mini_bar(s)
            lines.append(f"    {dim:<28} {s:>5.1f}  {bar}  {why}")
        for w in self.warnings:
            lines.append(f"    ⚠  {w}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "node":       self.node,
            "score":      round(self.score, 2),
            "passed":     self.passed,
            "threshold":  self.threshold,
            "dimensions": {
                k: {kk: (round(vv, 2) if isinstance(vv, float) else vv)
                    for kk, vv in v.items()}
                for k, v in self.dimensions.items()
            },
            "warnings": self.warnings,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Punto de entrada principal
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_all_outputs(final_state: dict, company_names: List[str]) -> Dict[str, QualityResult]:
    """
    Evalúa todos los outputs del pipeline a partir del estado final.

    Devuelve un dict {node_name: QualityResult}.
    """
    results: Dict[str, QualityResult] = {}

    results["company_node"] = evaluate_companies(
        company_names=company_names,
        category=final_state.get("category", ""),
    )
    results["news_branch"] = evaluate_news(
        company_data=final_state.get("company_data", {}),
        company_names=company_names,
    )
    results["web_branch"] = evaluate_web(
        website_urls=final_state.get("website_urls", {}),
        features_data=final_state.get("features_data", {}),
        company_names=company_names,
    )
    results["embedding_node"] = evaluate_embeddings(
        rag_chunk_count=final_state.get("rag_chunk_count", 0),
        features_data=final_state.get("features_data", {}),
        company_names=company_names,
    )
    results["structuring_node"] = evaluate_matrix(
        structured_data=final_state.get("structured_data", []),
        company_names=company_names,
    )
    results["swot_node"] = evaluate_swot(
        swot_data=final_state.get("swot_data", {}),
        company_names=company_names,
    )
    results["report_node"] = evaluate_report(
        company_names=company_names,
        category=final_state.get("category", ""),
    )

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Evaluadores por nodo
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_companies(company_names: List[str], category: str) -> QualityResult:
    """Evalúa la calidad de los nombres de empresa descubiertos."""
    dims: Dict[str, Dict] = {}
    warnings: List[str]   = []
    n = len(company_names)

    # ── Cantidad ──────────────────────────────────────────────────────────────
    count_score = min(n / 5 * 100, 100)
    dims["quantity"] = {
        "score":  count_score,
        "reason": f"{n}/5 empresas encontradas",
        "weight": 0.25,
    }
    if n < 5:
        warnings.append(f"Solo {n} empresas encontradas (se esperan 5)")

    # ── Limpieza (sin stopwords ni ruido) ─────────────────────────────────────
    dirty = [
        c for c in company_names
        if any(sw in c.lower().split() for sw in _COMPANY_BLACKLIST)
        or len(c) < 3
        or c.startswith(("http", "www"))
    ]
    clean_score = (1 - len(dirty) / max(n, 1)) * 100
    dims["cleanliness"] = {
        "score":  clean_score,
        "reason": f"{len(dirty)} nombres con ruido detectado",
        "weight": 0.30,
    }
    for d in dirty:
        warnings.append(f"Nombre sospechoso: '{d}'")

    # ── Diversidad (no duplicados ni muy parecidos) ────────────────────────────
    unique_prefixes = len({c[:6].lower() for c in company_names})
    diversity_score = (unique_prefixes / max(n, 1)) * 100
    dims["diversity"] = {
        "score":  diversity_score,
        "reason": f"{unique_prefixes} prefijos únicos de {n}",
        "weight": 0.20,
    }

    # ── Relevancia al sector (palabras del category en ningún nombre = posible mismatch) ──
    # Heurística: si la categoría menciona un término geográfico o de sector,
    # al menos 1 empresa debería no ser un gigante genérico global
    sector_words = {w.lower() for w in category.split() if len(w) > 3}
    # No podemos verificar sector real sin internet, así que chequeamos formato
    # como proxy de calidad: ¿los nombres parecen entidades comerciales?
    well_formed = [
        c for c in company_names
        if re.match(r"^[A-ZÁÉÍÓÚÑ][a-záéíóúñA-Z0-9& .\-]{1,50}$", c)
    ]
    relevance_score = (len(well_formed) / max(n, 1)) * 100
    dims["format_quality"] = {
        "score":  relevance_score,
        "reason": f"{len(well_formed)}/{n} nombres bien formados",
        "weight": 0.25,
    }

    score = _weighted_score(dims)
    return QualityResult(
        node="company_node", score=score, dimensions=dims,
        warnings=warnings, threshold=THRESHOLDS["company_node"],
    )


def evaluate_news(company_data: Dict[str, List], company_names: List[str]) -> QualityResult:
    """Evalúa la calidad del scraping de noticias (Rama A)."""
    dims:     Dict[str, Dict] = {}
    warnings: List[str]       = []
    n = max(len(company_names), 1)

    # ── Cobertura ─────────────────────────────────────────────────────────────
    covered = [c for c in company_names if c in company_data and company_data[c]]
    coverage_score = len(covered) / n * 100
    dims["coverage"] = {
        "score":  coverage_score,
        "reason": f"{len(covered)}/{n} empresas con noticias",
        "weight": 0.40,
    }
    for c in company_names:
        if c not in company_data or not company_data[c]:
            warnings.append(f"Sin noticias: {c}")

    # ── Profundidad (media de snippets por empresa cubierta) ──────────────────
    counts  = [len(v) for v in company_data.values() if v]
    avg_snip = statistics.mean(counts) if counts else 0
    depth_score = min(avg_snip / 5 * 100, 100)   # 5 snippets = 100 pts
    dims["snippet_depth"] = {
        "score":  depth_score,
        "reason": f"media {avg_snip:.1f} snippets/empresa (objetivo ≥5)",
        "weight": 0.35,
    }
    if avg_snip < 2:
        warnings.append("Muy pocos snippets por empresa — calidad de noticias baja")

    # ── Longitud media de snippets (proxy de contenido real vs. títulos vacíos) ──
    all_snippets = [s for lst in company_data.values() for s in lst if isinstance(s, str)]
    avg_len      = statistics.mean([len(s) for s in all_snippets]) if all_snippets else 0
    length_score = min(avg_len / 200 * 100, 100)  # 200 chars = 100 pts
    dims["snippet_length"] = {
        "score":  length_score,
        "reason": f"longitud media {avg_len:.0f} chars (objetivo ≥200)",
        "weight": 0.25,
    }
    if avg_len < 80:
        warnings.append("Snippets muy cortos — posible scraping incompleto")

    score = _weighted_score(dims)
    return QualityResult(
        node="news_branch", score=score, dimensions=dims,
        warnings=warnings, threshold=THRESHOLDS["news_branch"],
    )


def evaluate_web(
    website_urls: Dict[str, str],
    features_data: Dict[str, List],
    company_names: List[str],
) -> QualityResult:
    """Evalúa la calidad del scraping web y extracción de features (Rama B)."""
    dims:     Dict[str, Dict] = {}
    warnings: List[str]       = []
    n = max(len(company_names), 1)

    # ── Cobertura de URLs ──────────────────────────────────────────────────────
    urls_found  = [c for c in company_names if c in website_urls and website_urls[c]]
    url_score   = len(urls_found) / n * 100
    dims["url_coverage"] = {
        "score":  url_score,
        "reason": f"{len(urls_found)}/{n} URLs oficiales encontradas",
        "weight": 0.25,
    }
    for c in company_names:
        if c not in website_urls or not website_urls.get(c):
            warnings.append(f"Sin URL oficial: {c}")

    # ── Cobertura de features ──────────────────────────────────────────────────
    feat_ok   = [c for c in company_names if c in features_data and features_data[c]]
    feat_score = len(feat_ok) / n * 100
    dims["feature_coverage"] = {
        "score":  feat_score,
        "reason": f"{len(feat_ok)}/{n} empresas con features scrapeadas",
        "weight": 0.30,
    }

    # ── Profundidad de páginas ─────────────────────────────────────────────────
    page_counts = [len(v) for v in features_data.values() if v]
    avg_pages   = statistics.mean(page_counts) if page_counts else 0
    depth_score = min(avg_pages / 3 * 100, 100)   # 3 páginas = 100 pts
    dims["page_depth"] = {
        "score":  depth_score,
        "reason": f"media {avg_pages:.1f} páginas/empresa (objetivo ≥3)",
        "weight": 0.20,
    }

    # ── Detección de precios en el contenido raw ───────────────────────────────
    price_pattern = re.compile(
        r"(\d[\d.,]*\s*€|\$\s*\d[\d.,]*|€\s*\d[\d.,]*|USD|EUR|gratis|free|precio|price|plan)",
        re.I,
    )
    with_price = 0
    for company, pages in features_data.items():
        for page in pages:
            raw = page.get("raw_content", "")
            if price_pattern.search(raw):
                with_price += 1
                break
    price_score = with_price / n * 100
    dims["price_presence"] = {
        "score":  price_score,
        "reason": f"{with_price}/{n} empresas con datos de precio detectados",
        "weight": 0.25,
    }
    if with_price < n // 2:
        warnings.append("Menos de la mitad de las empresas tienen información de precios")

    score = _weighted_score(dims)
    return QualityResult(
        node="web_branch", score=score, dimensions=dims,
        warnings=warnings, threshold=THRESHOLDS["web_branch"],
    )


def evaluate_embeddings(
    rag_chunk_count: int,
    features_data: Dict[str, List],
    company_names: List[str],
) -> QualityResult:
    """Evalúa la calidad del índice RAG en ChromaDB."""
    dims:     Dict[str, Dict] = {}
    warnings: List[str]       = []
    n = max(len(company_names), 1)

    # ── Densidad global (chunks totales / empresas) ────────────────────────────
    avg_chunks  = rag_chunk_count / n
    density_score = min(avg_chunks / 5 * 100, 100)   # 5 chunks/empresa = 100 pts
    dims["chunk_density"] = {
        "score":  density_score,
        "reason": f"{rag_chunk_count} chunks totales, media {avg_chunks:.1f}/empresa",
        "weight": 0.50,
    }
    if rag_chunk_count == 0:
        warnings.append("ChromaDB vacío — RAG desactivado, calidad de structuring reducida")
    elif avg_chunks < 3:
        warnings.append(f"Pocos chunks por empresa ({avg_chunks:.1f}) — considera aumentar páginas de scraping")

    # ── Balance entre empresas (proxy: features_data tiene datos para todas) ───
    covered = sum(1 for c in company_names if c in features_data and features_data[c])
    balance_score = covered / n * 100
    dims["company_balance"] = {
        "score":  balance_score,
        "reason": f"{covered}/{n} empresas con datos en el índice",
        "weight": 0.50,
    }

    score = _weighted_score(dims)
    return QualityResult(
        node="embedding_node", score=score, dimensions=dims,
        warnings=warnings, threshold=THRESHOLDS["embedding_node"],
    )


def evaluate_matrix(
    structured_data: List[Dict],
    company_names: List[str],
) -> QualityResult:
    """Evalúa la calidad de la matriz comparativa (STRUCTURED_DATA)."""
    dims:     Dict[str, Dict] = {}
    warnings: List[str]       = []
    n = max(len(company_names), 1)

    if not structured_data:
        warnings.append("STRUCTURED_DATA vacío — la matriz no se generó")
        empty = QualityResult(
            node="structuring_node", score=0.0,
            warnings=warnings, threshold=THRESHOLDS["structuring_node"],
        )
        return empty

    # Agrupar por empresa
    by_company: Dict[str, List[Dict]] = {}
    for row in structured_data:
        c = row.get("company", "")
        by_company.setdefault(c, []).append(row)

    # ── Cobertura de empresas ──────────────────────────────────────────────────
    covered    = sum(1 for c in company_names if c in by_company)
    cov_score  = covered / n * 100
    dims["company_coverage"] = {
        "score":  cov_score,
        "reason": f"{covered}/{n} empresas en la matriz",
        "weight": 0.25,
    }
    for c in company_names:
        if c not in by_company:
            warnings.append(f"Empresa ausente en la matriz: {c}")

    # ── Completitud de celdas (no vacías, no N/A, no "No especificado") ────────
    null_vals   = {"", "n/a", "na", "no especificado", "sin datos", "none"}
    total_cells = len(structured_data) * 4   # plan_name, price, services, highlights
    empty_cells = sum(
        1 for row in structured_data
        for key in ("plan_name", "price", "services", "highlights")
        if str(row.get(key, "")).strip().lower() in null_vals
    )
    fill_pct   = (1 - empty_cells / max(total_cells, 1)) * 100
    dims["cell_completeness"] = {
        "score":  fill_pct,
        "reason": f"{fill_pct:.1f}% celdas con datos reales (total {total_cells})",
        "weight": 0.25,
    }
    if fill_pct < 50:
        warnings.append(f"Más de la mitad de las celdas están vacías o con N/A ({fill_pct:.0f}%)")

    # ── Cobertura de precios ───────────────────────────────────────────────────
    with_price = sum(
        1 for c, rows in by_company.items()
        if any(
            str(r.get("price", "")).strip().lower() not in null_vals
            for r in rows
        )
    )
    price_score = with_price / n * 100
    dims["price_coverage"] = {
        "score":  price_score,
        "reason": f"{with_price}/{n} empresas con precio definido",
        "weight": 0.25,
    }
    if with_price < n // 2:
        warnings.append("Menos de la mitad de las empresas tienen precio definido")

    # ── Diversidad de planes (media de planes por empresa) ─────────────────────
    plan_counts = [len(rows) for rows in by_company.values()]
    avg_plans   = statistics.mean(plan_counts) if plan_counts else 0
    plan_score  = min(avg_plans / 3 * 100, 100)   # 3 planes/empresa = 100 pts
    dims["plan_diversity"] = {
        "score":  plan_score,
        "reason": f"media {avg_plans:.1f} planes/empresa (objetivo ≥3)",
        "weight": 0.25,
    }

    score = _weighted_score(dims)
    return QualityResult(
        node="structuring_node", score=score, dimensions=dims,
        warnings=warnings, threshold=THRESHOLDS["structuring_node"],
    )


def evaluate_swot(
    swot_data: Dict[str, Dict[str, List[str]]],
    company_names: List[str],
) -> QualityResult:
    """Evalúa la calidad del análisis DAFO."""
    dims:     Dict[str, Dict] = {}
    warnings: List[str]       = []
    n = max(len(company_names), 1)
    DIMS = {"strengths", "weaknesses", "opportunities", "threats"}

    if not swot_data:
        warnings.append("SWOT_DATA vacío — el análisis DAFO no se generó")
        return QualityResult(
            node="swot_node", score=0.0,
            warnings=warnings, threshold=THRESHOLDS["swot_node"],
        )

    # ── Cobertura de empresas ──────────────────────────────────────────────────
    covered = sum(1 for c in company_names if c in swot_data)
    dims["company_coverage"] = {
        "score":  covered / n * 100,
        "reason": f"{covered}/{n} empresas con DAFO",
        "weight": 0.25,
    }
    for c in company_names:
        if c not in swot_data:
            warnings.append(f"Sin DAFO: {c}")

    # ── Completitud de las 4 dimensiones por empresa ───────────────────────────
    complete = sum(
        1 for c_data in swot_data.values()
        if set(c_data.keys()) >= DIMS
    )
    dims["dimension_completeness"] = {
        "score":  complete / max(len(swot_data), 1) * 100,
        "reason": f"{complete}/{len(swot_data)} empresas con las 4 dimensiones DAFO",
        "weight": 0.25,
    }

    # ── Profundidad (media de ítems por dimensión) ─────────────────────────────
    all_counts = []
    for c_data in swot_data.values():
        for dim in DIMS:
            items = c_data.get(dim, [])
            all_counts.append(len(items))
    avg_items   = statistics.mean(all_counts) if all_counts else 0
    depth_score = min(avg_items / 3 * 100, 100)   # 3 ítems/dim = 100 pts
    dims["depth"] = {
        "score":  depth_score,
        "reason": f"media {avg_items:.1f} ítems/dimensión (objetivo ≥3)",
        "weight": 0.25,
    }
    if avg_items < 2:
        warnings.append("Muy pocos ítems por dimensión DAFO — análisis superficial")

    # ── Especificidad (no "Información insuficiente") ─────────────────────────
    all_items   = [
        item
        for c_data in swot_data.values()
        for dim in DIMS
        for item in c_data.get(dim, [])
    ]
    insuf       = sum(1 for i in all_items if "insuficiente" in i.lower() or "insufficient" in i.lower())
    spec_score  = (1 - insuf / max(len(all_items), 1)) * 100
    dims["specificity"] = {
        "score":  spec_score,
        "reason": f"{insuf}/{len(all_items)} ítems genéricos 'Información insuficiente'",
        "weight": 0.25,
    }
    if insuf > len(all_items) * 0.3:
        warnings.append(f"Más del 30% de los ítems DAFO son 'Información insuficiente' ({insuf})")

    score = _weighted_score(dims)
    return QualityResult(
        node="swot_node", score=score, dimensions=dims,
        warnings=warnings, threshold=THRESHOLDS["swot_node"],
    )


def evaluate_report(
    company_names: List[str],
    category: str,
    md_path: str = "market_report.md",
) -> QualityResult:
    """Evalúa la calidad del informe Markdown final."""
    from pathlib import Path
    dims:     Dict[str, Dict] = {}
    warnings: List[str]       = []
    n = max(len(company_names), 1)

    path = Path(md_path)
    if not path.exists():
        warnings.append(f"'{md_path}' no encontrado — informe no generado")
        return QualityResult(
            node="report_node", score=0.0,
            warnings=warnings, threshold=THRESHOLDS["report_node"],
        )

    text = path.read_text(encoding="utf-8", errors="ignore")
    text_lower = text.lower()

    # ── Estructura: secciones obligatorias ───────────────────────────────────
    found_sections = [s for s in _REQUIRED_REPORT_SECTIONS if s in text_lower]
    struct_score   = len(found_sections) / len(_REQUIRED_REPORT_SECTIONS) * 100
    dims["structure"] = {
        "score":  struct_score,
        "reason": f"{len(found_sections)}/{len(_REQUIRED_REPORT_SECTIONS)} secciones presentes"
                  f": {found_sections}",
        "weight": 0.25,
    }
    missing_sections = [s for s in _REQUIRED_REPORT_SECTIONS if s not in text_lower]
    for s in missing_sections:
        warnings.append(f"Sección ausente en el informe: '{s}'")

    # ── Cobertura de empresas (cada empresa debe aparecer al menos 2 veces) ───
    mention_counts = {
        c: len(re.findall(re.escape(c), text, re.I))
        for c in company_names
    }
    well_covered = sum(1 for cnt in mention_counts.values() if cnt >= 2)
    company_score = well_covered / n * 100
    dims["company_coverage"] = {
        "score":  company_score,
        "reason": f"{well_covered}/{n} empresas mencionadas ≥2 veces",
        "weight": 0.25,
    }
    for c, cnt in mention_counts.items():
        if cnt < 2:
            warnings.append(f"Empresa poco presente en el informe: '{c}' ({cnt} menciones)")

    # ── Longitud (objetivo 800-3000 palabras) ──────────────────────────────────
    words = len(text.split())
    if words < 400:
        length_score = words / 400 * 50          # muy corto
        warnings.append(f"Informe muy corto ({words} palabras) — objetivo 800-3000")
    elif words > 5000:
        length_score = max(0, 100 - (words - 5000) / 100)   # penaliza exceso
        warnings.append(f"Informe muy largo ({words} palabras) — puede perder foco")
    else:
        # Pico en 1500 palabras
        if words <= 1500:
            length_score = 50 + (words - 400) / 1100 * 50
        else:
            length_score = 100 - (words - 1500) / 3500 * 20
    dims["length"] = {
        "score":  max(0, min(length_score, 100)),
        "reason": f"{words} palabras",
        "weight": 0.20,
    }

    # ── Densidad de datos (cifras y porcentajes) ──────────────────────────────
    numbers = re.findall(r"\d[\d.,]*\s*(?:%|€|\$|USD|EUR|millones?|mil)", text, re.I)
    # Escala: ≥10 cifras = 100 pts
    data_score = min(len(numbers) / 10 * 100, 100)
    dims["data_density"] = {
        "score":  data_score,
        "reason": f"{len(numbers)} cifras/porcentajes detectados (objetivo ≥10)",
        "weight": 0.15,
    }
    if len(numbers) < 3:
        warnings.append("Muy pocas cifras en el informe — falta soporte cuantitativo")

    # ── Recomendaciones accionables ───────────────────────────────────────────
    rec_patterns = [
        r"recomend[a-z]*",
        r"(se\s+)?sugier[a-z]*",
        r"(se\s+)?propon[a-z]*",
        r"estrategia",
        r"acción|accion",
        r"oportunidad[a-z]*\s+de",
    ]
    rec_count = sum(
        len(re.findall(p, text_lower))
        for p in rec_patterns
    )
    # ≥5 hits = 100 pts
    rec_score = min(rec_count / 5 * 100, 100)
    dims["recommendations"] = {
        "score":  rec_score,
        "reason": f"{rec_count} referencias a recomendaciones/estrategia",
        "weight": 0.15,
    }
    if rec_count < 3:
        warnings.append("Pocas recomendaciones estratégicas detectadas en el informe")

    score = _weighted_score(dims)
    return QualityResult(
        node="report_node", score=score, dimensions=dims,
        warnings=warnings, threshold=THRESHOLDS["report_node"],
    )


# ══════════════════════════════════════════════════════════════════════════════
# Resumen global de calidad
# ══════════════════════════════════════════════════════════════════════════════

def print_quality_report(results: Dict[str, QualityResult]):
    """Imprime el informe de calidad completo en consola."""
    _sep("EVALUACIÓN DE CALIDAD DE OUTPUTS")
    passed = sum(1 for r in results.values() if r.passed)
    total  = len(results)
    global_score = statistics.mean([r.score for r in results.values()]) if results else 0
    print(f"  Score global: {global_score:.1f}/100   Nodos aprobados: {passed}/{total}\n")

    for result in results.values():
        print(result.summary())

    # Warnings consolidados
    all_warnings = [
        (name, w)
        for name, result in results.items()
        for w in result.warnings
    ]
    if all_warnings:
        _sep("ADVERTENCIAS CONSOLIDADAS")
        for name, w in all_warnings:
            print(f"  [{name}] ⚠  {w}")
    print()


def quality_results_to_dict(results: Dict[str, QualityResult]) -> dict:
    """Serializa todos los resultados a dict (para guardar en eval_runs.jsonl)."""
    return {name: r.to_dict() for name, r in results.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers internos
# ══════════════════════════════════════════════════════════════════════════════

def _weighted_score(dims: Dict[str, Dict]) -> float:
    """Calcula score ponderado. Si no hay pesos, usa media simple."""
    total_w, total_s = 0.0, 0.0
    for info in dims.values():
        w = info.get("weight", 1.0)
        s = max(0.0, min(100.0, info.get("score", 0.0)))
        total_s += s * w
        total_w += w
    return round(total_s / total_w, 2) if total_w else 0.0


def _mini_bar(score: float, width: int = 10) -> str:
    filled = round(score / 100 * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _sep(title: str = ""):
    w = 68
    print("\n" + "═" * w)
    if title:
        print(f"  {title}")
        print("═" * w)
