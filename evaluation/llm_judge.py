# -*- coding: utf-8 -*-
from __future__ import annotations

"""
evaluation/llm_judge.py
------------------------
Evaluación semántica de outputs mediante LLM-as-judge.

Complementa quality.py (heurísticas) con juicios cualitativos que
ninguna heurística puede capturar: coherencia argumental, utilidad
real de las recomendaciones, consistencia interna del DAFO, etc.

Arquitectura
------------
  Cada evaluador construye un prompt de juicio, llama a Claude con
  temperatura 0 (determinismo máximo) y parsea la respuesta JSON:

  {
    "scores": {
      "dimension_1": <int 0-10>,
      "dimension_2": <int 0-10>,
      ...
    },
    "overall": <int 0-10>,
    "strengths": ["..."],
    "weaknesses": ["..."],
    "verdict": "PASS | FAIL"
  }

  El overall se normaliza a 0-100 para alinearse con quality.py.

Evaluadores disponibles
-----------------------
  judge_swot(swot_data, company_names)
    · consistency   — ¿F/D/O/A son distintas y no se solapan?
    · evidence      — ¿los ítems citan hechos concretos, no generalidades?
    · actionability — ¿las amenazas y oportunidades son accionables?
    · balance       — ¿ninguna dimensión está sobredimensionada?

  judge_matrix(structured_data, company_names)
    · accuracy      — ¿precios y planes parecen reales y coherentes?
    · differentiation — ¿las empresas aparecen realmente diferenciadas?
    · completeness  — ¿hay datos suficientes para comparar?

  judge_report(category, company_names, md_path)
    · executive_value   — ¿el resumen ejecutivo aporta insight real?
    · analytical_depth  — ¿el análisis va más allá de describir datos?
    · consistency       — ¿el informe es coherente con DAFO y matriz?
    · recommendation_quality — ¿las recomendaciones son concretas y priorizadas?
    · writing_quality   — claridad, ausencia de relleno, tono consultivo

  run_all_judges(final_state, company_names) → Dict[str, JudgeResult]
    Ejecuta los tres jueces y devuelve resultados combinados.

Uso
----
  from evaluation.llm_judge import run_all_judges
  judge_results = run_all_judges(final_state, company_names)
  print_judge_report(judge_results)

Coste
-----
  Cada llamada consume ~800-1500 tokens de entrada + ~400 de salida.
  El judge llama a la API 3 veces por run (SWOT + matriz + informe).
  Puede desactivarse con EVAL_SKIP_LLM_JUDGE=1 en el entorno.
"""

import json
import os
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# Configuración
# ══════════════════════════════════════════════════════════════════════════════

_JUDGE_MODEL   = "claude-sonnet-4-20250514"
_JUDGE_TOKENS  = 700
_JUDGE_TEMP    = 0          # temperatura 0 = máximo determinismo
_API_URL       = "https://api.anthropic.com/v1/messages"
_THRESHOLD     = 6.0        # sobre 10; PASS si overall >= 6

# ══════════════════════════════════════════════════════════════════════════════
# Estructura de resultado
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class JudgeResult:
    evaluator: str                              # "swot" | "matrix" | "report"
    overall: float                              # 0-10
    score_100: float                            # overall normalizado a 0-100
    passed: bool
    scores: Dict[str, float]   = field(default_factory=dict)   # dim → 0-10
    strengths: List[str]       = field(default_factory=list)
    weaknesses: List[str]      = field(default_factory=list)
    error: Optional[str]       = None
    skipped: bool              = False

    def summary(self) -> str:
        if self.skipped:
            return f"\n  [SKIP] judge_{self.evaluator}  ({self.error})"
        status = "✓ PASS" if self.passed else "✗ FAIL"
        lines  = [
            f"\n  [{status}] judge_{self.evaluator}"
            f"  overall={self.overall:.1f}/10  ({self.score_100:.0f}/100)"
        ]
        for dim, s in self.scores.items():
            bar = _mini_bar(s * 10)
            lines.append(f"    {dim:<32} {s:.1f}/10  {bar}")
        if self.strengths:
            lines.append(f"    + {self.strengths[0]}")
        if self.weaknesses:
            lines.append(f"    - {self.weaknesses[0]}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "evaluator": self.evaluator,
            "overall":   round(self.overall, 2),
            "score_100": round(self.score_100, 2),
            "passed":    self.passed,
            "scores":    {k: round(v, 2) for k, v in self.scores.items()},
            "strengths":  self.strengths,
            "weaknesses": self.weaknesses,
            "error":      self.error,
            "skipped":    self.skipped,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Punto de entrada
# ══════════════════════════════════════════════════════════════════════════════

def run_all_judges(
    final_state: dict,
    company_names: List[str],
) -> Dict[str, JudgeResult]:
    """
    Ejecuta los tres jueces LLM y devuelve resultados.
    Si EVAL_SKIP_LLM_JUDGE=1 en el entorno, devuelve resultados marcados
    como skipped sin consumir tokens.
    """
    if os.getenv("EVAL_SKIP_LLM_JUDGE", "0") == "1":
        skip = lambda name: JudgeResult(
            evaluator=name, overall=0, score_100=0, passed=False,
            skipped=True, error="EVAL_SKIP_LLM_JUDGE=1",
        )
        return {
            "swot":   skip("swot"),
            "matrix": skip("matrix"),
            "report": skip("report"),
        }

    results: Dict[str, JudgeResult] = {}

    print("\n[judge] Evaluando calidad semántica con LLM-as-judge...")

    results["swot"] = judge_swot(
        swot_data=final_state.get("swot_data", {}),
        company_names=company_names,
    )
    results["matrix"] = judge_matrix(
        structured_data=final_state.get("structured_data", []),
        company_names=company_names,
    )
    results["report"] = judge_report(
        category=final_state.get("category", ""),
        company_names=company_names,
    )

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Juez 1 — DAFO
# ══════════════════════════════════════════════════════════════════════════════

_SWOT_JUDGE_PROMPT = """\
Eres un evaluador experto en análisis estratégico empresarial.
Tu tarea es evaluar la calidad de un análisis DAFO generado automáticamente.

## DAFO a evaluar
{swot_text}

## Instrucciones de evaluación
Puntúa del 0 al 10 cada dimensión de calidad:

- **consistency** (0-10): ¿Las Fortalezas, Debilidades, Oportunidades y Amenazas
  son conceptualmente distintas? ¿No se repiten ítems entre cuadrantes?
  0 = solapamientos graves, 10 = perfectamente diferenciados.

- **evidence** (0-10): ¿Los ítems citan hechos, datos o eventos concretos
  (productos, cifras, eventos recientes), o son generalidades vacías como
  "empresa innovadora" / "mercado competitivo"?
  0 = todo genérico, 10 = todos los ítems tienen evidencia concreta.

- **actionability** (0-10): ¿Las Oportunidades y Amenazas son accionables?
  ¿Una empresa podría tomar decisiones reales basándose en ellas?
  0 = demasiado vago, 10 = perfectamente accionable.

- **balance** (0-10): ¿El análisis dedica atención similar a las 4 dimensiones
  y a las distintas empresas? ¿O hay dimensiones/empresas dominantes?
  0 = muy desequilibrado, 10 = perfectamente equilibrado.

## Formato de respuesta
Responde ÚNICAMENTE con JSON válido, sin texto adicional:
{{
  "scores": {{
    "consistency": <0-10>,
    "evidence": <0-10>,
    "actionability": <0-10>,
    "balance": <0-10>
  }},
  "overall": <0-10>,
  "strengths": ["<punto fuerte del DAFO en 1 frase>"],
  "weaknesses": ["<punto débil del DAFO en 1 frase>"],
  "verdict": "<PASS|FAIL>"
}}
"""

def judge_swot(
    swot_data: Dict[str, Dict[str, List[str]]],
    company_names: List[str],
) -> JudgeResult:
    """Evalúa semánticamente el análisis DAFO."""
    if not swot_data:
        return JudgeResult(
            evaluator="swot", overall=0, score_100=0, passed=False,
            skipped=True, error="SWOT_DATA vacío",
        )

    # Serializar DAFO como texto legible para el juez
    swot_text = _format_swot_for_judge(swot_data)
    # Limitar para no exceder contexto
    swot_text = swot_text[:6000]

    prompt = _SWOT_JUDGE_PROMPT.format(swot_text=swot_text)
    return _call_judge("swot", prompt)


def _format_swot_for_judge(swot_data: dict) -> str:
    lines = []
    dim_labels = {
        "strengths": "Fortalezas",
        "weaknesses": "Debilidades",
        "opportunities": "Oportunidades",
        "threats": "Amenazas",
    }
    for company, dims in swot_data.items():
        lines.append(f"\n### {company}")
        for key, label in dim_labels.items():
            items = dims.get(key, [])
            lines.append(f"**{label}:**")
            for item in items:
                lines.append(f"  - {item}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Juez 2 — Matriz comparativa
# ══════════════════════════════════════════════════════════════════════════════

_MATRIX_JUDGE_PROMPT = """\
Eres un analista de mercado experto. Evalúa la calidad de una matriz
comparativa de empresas generada automáticamente.

## Sector analizado
{category}

## Empresas esperadas
{companies}

## Matriz comparativa
{matrix_text}

## Instrucciones de evaluación
Puntúa del 0 al 10:

- **accuracy** (0-10): ¿Los precios y nombres de plan parecen reales y
  coherentes con el sector? ¿O hay valores inventados, contradictorios
  o claramente incorrectos?
  0 = datos falsos o incoherentes, 10 = todo parece preciso y real.

- **differentiation** (0-10): ¿La matriz muestra diferencias reales entre
  empresas? ¿O todas tienen el mismo tipo de datos y parece que el LLM
  copió el mismo patrón para todas?
  0 = todas idénticas, 10 = cada empresa claramente diferenciada.

- **usefulness** (0-10): ¿Un analista de mercado podría usar esta matriz
  para tomar decisiones reales de negocio?
  0 = inútil, 10 = directamente accionable para decisiones estratégicas.

## Formato de respuesta
Responde ÚNICAMENTE con JSON válido, sin texto adicional:
{{
  "scores": {{
    "accuracy": <0-10>,
    "differentiation": <0-10>,
    "usefulness": <0-10>
  }},
  "overall": <0-10>,
  "strengths": ["<punto fuerte de la matriz en 1 frase>"],
  "weaknesses": ["<punto débil de la matriz en 1 frase>"],
  "verdict": "<PASS|FAIL>"
}}
"""

def judge_matrix(
    structured_data: List[Dict],
    company_names: List[str],
    category: str = "",
) -> JudgeResult:
    """Evalúa semánticamente la matriz comparativa."""
    if not structured_data:
        return JudgeResult(
            evaluator="matrix", overall=0, score_100=0, passed=False,
            skipped=True, error="STRUCTURED_DATA vacío",
        )

    matrix_text = _format_matrix_for_judge(structured_data)
    matrix_text = matrix_text[:5000]

    prompt = _MATRIX_JUDGE_PROMPT.format(
        category=category or "No especificado",
        companies=", ".join(company_names),
        matrix_text=matrix_text,
    )
    return _call_judge("matrix", prompt)


def _format_matrix_for_judge(structured_data: List[Dict]) -> str:
    by_company: Dict[str, List] = {}
    for row in structured_data:
        c = row.get("company", "Desconocida")
        by_company.setdefault(c, []).append(row)

    lines = []
    for company, plans in by_company.items():
        lines.append(f"\n### {company}")
        for p in plans:
            lines.append(
                f"  Plan: {p.get('plan_name', 'N/A')} | "
                f"Precio: {p.get('price', 'N/A')} | "
                f"Servicios: {p.get('services', 'N/A')[:120]}"
            )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Juez 3 — Informe final
# ══════════════════════════════════════════════════════════════════════════════

_REPORT_JUDGE_PROMPT = """\
Eres un consultor de estrategia senior evaluando un informe de mercado
generado por un sistema automático de inteligencia competitiva.

## Sector analizado
{category}

## Empresas cubiertas
{companies}

## Informe a evaluar
{report_excerpt}

## Instrucciones de evaluación
Puntúa del 0 al 10:

- **executive_value** (0-10): ¿El resumen ejecutivo aporta insight real
  y sintetiza los hallazgos clave? ¿O es solo un listado descriptivo?
  0 = no aporta valor, 10 = insight denso y accionable.

- **analytical_depth** (0-10): ¿El análisis competitivo va más allá
  de describir los datos? ¿Hay interpretación, comparaciones directas
  y conclusiones propias del analista?
  0 = puramente descriptivo, 10 = análisis profundo con juicio propio.

- **consistency** (0-10): ¿El informe es internamente coherente?
  ¿Las conclusiones están respaldadas por el análisis previo?
  ¿No hay contradicciones entre secciones?
  0 = inconsistente, 10 = perfectamente coherente.

- **recommendation_quality** (0-10): ¿Las recomendaciones son concretas,
  priorizadas y aplicables? ¿O son genéricas ("invertir en tecnología")?
  0 = recomendaciones vacías, 10 = recomendaciones específicas y priorizadas.

- **writing_quality** (0-10): ¿El texto es claro, sin relleno innecesario
  y con tono apropiado para consultoría de negocio?
  0 = confuso o con relleno, 10 = redacción profesional y precisa.

## Formato de respuesta
Responde ÚNICAMENTE con JSON válido, sin texto adicional:
{{
  "scores": {{
    "executive_value": <0-10>,
    "analytical_depth": <0-10>,
    "consistency": <0-10>,
    "recommendation_quality": <0-10>,
    "writing_quality": <0-10>
  }},
  "overall": <0-10>,
  "strengths": ["<punto fuerte del informe en 1 frase>"],
  "weaknesses": ["<punto débil del informe en 1 frase>"],
  "verdict": "<PASS|FAIL>"
}}
"""

def judge_report(
    category: str,
    company_names: List[str],
    md_path: str = "market_report.md",
) -> JudgeResult:
    """Evalúa semánticamente el informe Markdown final."""
    path = Path(md_path)
    if not path.exists():
        return JudgeResult(
            evaluator="report", overall=0, score_100=0, passed=False,
            skipped=True, error=f"'{md_path}' no encontrado",
        )

    text = path.read_text(encoding="utf-8", errors="ignore")
    # Enviamos los primeros ~4500 chars: suficiente para capturar
    # resumen ejecutivo + inicio del análisis competitivo
    excerpt = text[:4500]

    prompt = _REPORT_JUDGE_PROMPT.format(
        category=category or "No especificado",
        companies=", ".join(company_names),
        report_excerpt=excerpt,
    )
    return _call_judge("report", prompt)


# ══════════════════════════════════════════════════════════════════════════════
# Llamada a la API (compartida por los tres jueces)
# ══════════════════════════════════════════════════════════════════════════════

def _call_judge(evaluator: str, prompt: str) -> JudgeResult:
    """Llama a la API de Anthropic con el prompt del juez y parsea la respuesta."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return JudgeResult(
            evaluator=evaluator, overall=0, score_100=0, passed=False,
            skipped=True, error="ANTHROPIC_API_KEY no definida",
        )

    payload = json.dumps({
        "model":       _JUDGE_MODEL,
        "max_tokens":  _JUDGE_TOKENS,
        "temperature": _JUDGE_TEMP,
        "messages":    [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    max_retries, wait = 3, 20
    last_error = ""

    for attempt in range(max_retries):
        req = urllib.request.Request(
            _API_URL, data=payload,
            headers={
                "Content-Type":      "application/json; charset=utf-8",
                "x-api-key":         api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())

            raw = "".join(b.get("text", "") for b in data.get("content", []))
            return _parse_judge_response(evaluator, raw)

        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            if exc.code == 429 and attempt < max_retries - 1:
                print(f"[judge/{evaluator}] Rate limit — esperando {wait}s...")
                time.sleep(wait)
                wait *= 2
                continue
            break
        except Exception as exc:
            last_error = str(exc)
            break

    return JudgeResult(
        evaluator=evaluator, overall=0, score_100=0, passed=False,
        skipped=True, error=f"API error: {last_error}",
    )


def _parse_judge_response(evaluator: str, raw: str) -> JudgeResult:
    """Parsea la respuesta JSON del juez y construye un JudgeResult."""
    # Limpiar posibles bloques ```json
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Intentar extraer JSON con regex como fallback
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
            except Exception:
                return JudgeResult(
                    evaluator=evaluator, overall=0, score_100=0, passed=False,
                    skipped=True, error=f"JSON parse error: {raw[:200]}",
                )
        else:
            return JudgeResult(
                evaluator=evaluator, overall=0, score_100=0, passed=False,
                skipped=True, error=f"No JSON en respuesta: {raw[:200]}",
            )

    scores   = {k: float(v) for k, v in data.get("scores", {}).items()}
    overall  = float(data.get("overall", 0))
    # Calcular overall como media de dimensiones si el LLM no lo calculó bien
    if scores and (overall == 0 or overall > 10):
        overall = sum(scores.values()) / len(scores)
    overall   = max(0.0, min(10.0, overall))
    score_100 = overall * 10

    return JudgeResult(
        evaluator  = evaluator,
        overall    = round(overall, 2),
        score_100  = round(score_100, 2),
        passed     = overall >= _THRESHOLD,
        scores     = scores,
        strengths  = data.get("strengths", []),
        weaknesses = data.get("weaknesses", []),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Presentación
# ══════════════════════════════════════════════════════════════════════════════

def print_judge_report(results: Dict[str, JudgeResult]):
    """Imprime el resumen del LLM-as-judge en consola."""
    _sep("EVALUACIÓN SEMÁNTICA (LLM-as-judge)")

    active   = [r for r in results.values() if not r.skipped]
    skipped  = [r for r in results.values() if r.skipped]
    passed   = sum(1 for r in active if r.passed)

    if active:
        avg = sum(r.overall for r in active) / len(active)
        print(f"  Overall promedio: {avg:.1f}/10   Jueces aprobados: {passed}/{len(active)}\n")
    else:
        print("  Todos los jueces omitidos.\n")

    for r in results.values():
        print(r.summary())

    if skipped:
        print(f"\n  Jueces omitidos ({len(skipped)}):")
        for r in skipped:
            print(f"    · judge_{r.evaluator}: {r.error}")
    print()


def judge_results_to_dict(results: Dict[str, JudgeResult]) -> dict:
    """Serializa para persistencia en eval_runs.jsonl."""
    return {name: r.to_dict() for name, r in results.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _mini_bar(score_0_100: float, width: int = 10) -> str:
    filled = round(score_0_100 / 100 * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _sep(title: str = ""):
    w = 68
    print("\n" + "═" * w)
    if title:
        print(f"  {title}")
        print("═" * w)
