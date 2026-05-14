# -*- coding: utf-8 -*-
from __future__ import annotations

"""
evaluation/tracker.py
----------------------
Sistema de evaluación y tracking de rendimiento del pipeline.

Qué mide
---------
  Por nodo
    · elapsed_s          — tiempo real de ejecución (wall clock)
    · cpu_delta_s        — tiempo de CPU consumido
    · mem_delta_mb       — incremento de RAM durante el nodo
    · success            — True / False

  Por nodo (métricas de calidad específicas)
    · company_node       — n_companies_found, guardrail_activated
    · parallel_node      — news_companies_ok, web_companies_ok,
                           news_snippets_total, features_pages_total
    · embedding_node     — chunks_indexed
    · structuring_node   — matrix_rows, matrix_cols, matrix_fill_pct
    · swot_node          — swot_companies_ok, swot_dims_complete_pct,
                           avg_items_per_dim
    · report_node        — report_words, report_sections, pdf_kb

  Global
    · total_elapsed_s    — duración total del pipeline
    · run_id             — UUID reproducible a partir de la categoría + timestamp
    · model_swot         — modelo usado en swot_agent
    · model_report       — modelo usado en report_node

Persistencia
------------
  Cada run se añade a `eval_runs.jsonl` (una línea JSON por ejecución).
  Al terminar se imprime un resumen en consola y, si existen ≥2 runs del
  mismo sector, se muestra una comparación automática.

Uso
----
  from evaluation.tracker import PipelineTracker
  tracker = PipelineTracker(category="Neobancos en España")

  with tracker.measure("company_node"):
      result = company_node(state)

  tracker.record_quality("company_node", n_companies_found=5, guardrail_activated=False)
  tracker.save()
  tracker.print_summary()
  tracker.compare_with_history()
"""

import json
import os
import time
import uuid
# resource es solo Unix — usamos psutil si está disponible, si no time.process_time
try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _psutil = None
    _HAS_PSUTIL = False
import statistics
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

EVAL_FILE = Path("eval_runs.jsonl")

# ══════════════════════════════════════════════════════════════════════════════
# Estructura de datos de un nodo
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class NodeMetrics:
    node: str
    elapsed_s: float = 0.0
    cpu_delta_s: float = 0.0
    mem_delta_mb: float = 0.0
    success: bool = True
    error: Optional[str] = None
    quality: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ══════════════════════════════════════════════════════════════════════════════
# Tracker principal
# ══════════════════════════════════════════════════════════════════════════════

class PipelineTracker:
    """
    Instrumenta cada nodo del pipeline y persiste las métricas.

    Ejemplo mínimo
    --------------
    tracker = PipelineTracker(category="Neobancos en España")

    with tracker.measure("company_node"):
        result = company_node(state)
    tracker.record_quality("company_node", n_companies_found=5)

    tracker.finish(final_state)
    tracker.save()
    tracker.print_summary()
    tracker.compare_with_history()
    """

    NODES_ORDER = [
        "company_node",
        "parallel_node",
        "embedding_node",
        "structuring_node",
        "swot_node",
        "report_node",
    ]

    def __init__(self, category: str, run_id: Optional[str] = None):
        self.category  = category
        self.run_id    = run_id or str(uuid.uuid4())[:8]
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._wall_start = time.perf_counter()
        self._nodes: Dict[str, NodeMetrics] = {}
        self.meta: Dict[str, Any] = {}   # modelos, thread_id, etc.

    # ── Medición de tiempo y recursos ──────────────────────────────────────

    @contextmanager
    def measure(self, node_name: str):
        """
        Context manager que mide elapsed, CPU y RAM de un nodo.

        with tracker.measure("swot_node"):
            result = swot_node(state)
        """
        metrics = NodeMetrics(node=node_name)
        self._nodes[node_name] = metrics

        cpu_before  = _get_cpu_time()
        mem_before  = _get_rss_mb()
        wall_before = time.perf_counter()

        try:
            yield metrics
        except Exception as exc:
            metrics.success = False
            metrics.error   = str(exc)
            raise
        finally:
            metrics.elapsed_s   = round(time.perf_counter() - wall_before, 3)
            metrics.cpu_delta_s = round(_get_cpu_time() - cpu_before, 3)
            metrics.mem_delta_mb = round(_get_rss_mb() - mem_before, 2)

    # ── Métricas de calidad ────────────────────────────────────────────────

    def record_quality(self, node_name: str, **kwargs):
        """
        Añade métricas de calidad al nodo indicado.

        tracker.record_quality("company_node", n_companies_found=5, guardrail_activated=False)
        """
        if node_name not in self._nodes:
            self._nodes[node_name] = NodeMetrics(node=node_name)
        self._nodes[node_name].quality.update(kwargs)

    # ── Cierre del pipeline ────────────────────────────────────────────────

    def finish(self, final_state: Optional[dict] = None):
        """
        Calcula métricas globales, extrae calidad operacional y lanza
        evaluación de calidad de outputs. Llamar al terminar el pipeline.
        """
        self.total_elapsed_s = round(time.perf_counter() - self._wall_start, 3)
        self._quality_results: dict = {}

        if final_state:
            _extract_quality_from_state(self, final_state)

            company_names = final_state.get("company_names", [])

            # ── Evaluación heurística ──────────────────────────────────────
            try:
                from evaluation.quality import (
                    evaluate_all_outputs,
                    print_quality_report,
                    quality_results_to_dict,
                )
                q_results = evaluate_all_outputs(final_state, company_names)
                print_quality_report(q_results)
                self._quality_results = quality_results_to_dict(q_results)
            except Exception as exc:
                print(f"[eval] Evaluación heurística falló: {exc}")

            # ── Evaluación semántica (LLM-as-judge) ───────────────────────
            try:
                from evaluation.llm_judge import (
                    run_all_judges,
                    print_judge_report,
                    judge_results_to_dict,
                )
                j_results = run_all_judges(final_state, company_names)
                print_judge_report(j_results)
                self._judge_results = judge_results_to_dict(j_results)
            except Exception as exc:
                print(f"[eval] LLM-as-judge falló: {exc}")
                self._judge_results = {}

    # ── Persistencia ───────────────────────────────────────────────────────

    def save(self):
        """Añade este run a eval_runs.jsonl (incluye scores de calidad)."""
        record = {
            "run_id":          self.run_id,
            "category":        self.category,
            "started_at":      self.started_at,
            "total_elapsed_s": getattr(self, "total_elapsed_s", 0.0),
            "meta":            self.meta,
            "nodes":           {n: m.to_dict() for n, m in self._nodes.items()},
            "output_quality":  getattr(self, "_quality_results", {}),
            "llm_judge":       getattr(self, "_judge_results", {}),
        }
        with EVAL_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\n[eval] Run guardado → {EVAL_FILE}  (run_id={self.run_id})")

    # ── Resumen en consola ─────────────────────────────────────────────────

    def print_summary(self):
        """Imprime tabla de tiempos y calidad en consola."""
        _print_separator("EVALUACIÓN DEL PIPELINE")
        print(f"  Run ID   : {self.run_id}")
        print(f"  Categoría: {self.category}")
        print(f"  Inicio   : {self.started_at}")
        total = getattr(self, "total_elapsed_s", 0.0)
        print(f"  Total    : {_fmt_s(total)}\n")

        # ── Tabla de tiempos por nodo
        header = f"{'Nodo':<22} {'Elapsed':>9} {'CPU':>9} {'RAM Δ':>9} {'%Total':>8}  Estado"
        print(header)
        print("─" * len(header))
        for name in self.NODES_ORDER:
            m = self._nodes.get(name)
            if not m:
                continue
            pct = (m.elapsed_s / total * 100) if total else 0
            bar = _mini_bar(pct)
            status = "✓" if m.success else f"✗ {m.error or ''}"[:30]
            print(
                f"  {name:<20} {_fmt_s(m.elapsed_s):>9} {_fmt_s(m.cpu_delta_s):>9}"
                f" {m.mem_delta_mb:>7.1f}MB {pct:>7.1f}% {bar} {status}"
            )

        # ── Métricas de calidad
        _print_separator("MÉTRICAS DE CALIDAD")
        for name in self.NODES_ORDER:
            m = self._nodes.get(name)
            if not m or not m.quality:
                continue
            print(f"  {name}:")
            for k, v in m.quality.items():
                if isinstance(v, float):
                    print(f"    · {k:<35} {v:.2f}")
                else:
                    print(f"    · {k:<35} {v}")
        print()

    # ── Comparación histórica ──────────────────────────────────────────────

    def compare_with_history(self, n_last: int = 5):
        """
        Lee eval_runs.jsonl y muestra comparación con runs anteriores
        del mismo sector. Solo si hay ≥2 runs.
        """
        if not EVAL_FILE.exists():
            return

        history = _load_runs_for_category(self.category, exclude_run=self.run_id)
        if not history:
            print("[eval] Sin historial previo para esta categoría — sin comparación.\n")
            return

        recent = history[-n_last:]
        _print_separator(f"COMPARACIÓN HISTÓRICA — {self.category} (últimos {len(recent)} runs)")

        # Tiempos globales
        prev_totals  = [r["total_elapsed_s"] for r in recent]
        curr_total   = getattr(self, "total_elapsed_s", 0.0)
        avg_prev     = statistics.mean(prev_totals) if prev_totals else 0
        delta_global = curr_total - avg_prev

        print(f"  Total actual      : {_fmt_s(curr_total)}")
        print(f"  Media histórica   : {_fmt_s(avg_prev)}  ({len(recent)} runs)")
        print(f"  Diferencia global : {_delta_str(delta_global)}\n")

        # Tiempos por nodo
        print(f"  {'Nodo':<22} {'Actual':>9} {'Media prev.':>12} {'Δ':>10}")
        print("  " + "─" * 58)
        for name in self.NODES_ORDER:
            curr_m = self._nodes.get(name)
            if not curr_m:
                continue
            prev_times = [
                r["nodes"].get(name, {}).get("elapsed_s", 0.0)
                for r in recent
                if r["nodes"].get(name)
            ]
            if not prev_times:
                continue
            avg_p = statistics.mean(prev_times)
            delta = curr_m.elapsed_s - avg_p
            print(
                f"  {name:<22} {_fmt_s(curr_m.elapsed_s):>9}"
                f" {_fmt_s(avg_p):>12} {_delta_str(delta):>10}"
            )

        # Calidad: comparar métricas numéricas
        _print_separator("TENDENCIA DE CALIDAD")
        for name in self.NODES_ORDER:
            curr_m = self._nodes.get(name)
            if not curr_m or not curr_m.quality:
                continue
            for key, val in curr_m.quality.items():
                if not isinstance(val, (int, float)):
                    continue
                prev_vals = [
                    r["nodes"].get(name, {}).get("quality", {}).get(key)
                    for r in recent
                ]
                prev_vals = [v for v in prev_vals if v is not None]
                if not prev_vals:
                    continue
                avg_q = statistics.mean(prev_vals)
                delta_q = val - avg_q
                print(
                    f"  {name}.{key:<40} {val:>8}  (prev avg {avg_q:.2f}"
                    f"  {_delta_str(delta_q, invert=False)})"
                )
        print()


# ══════════════════════════════════════════════════════════════════════════════
# Helpers internos
# ══════════════════════════════════════════════════════════════════════════════

def _get_cpu_time() -> float:
    """Tiempo de CPU (user + sys). Multiplataforma: psutil > time.process_time."""
    try:
        if _HAS_PSUTIL:
            p = _psutil.Process()
            ct = p.cpu_times()
            return ct.user + ct.system
        return time.process_time()
    except Exception:
        return 0.0


def _get_rss_mb() -> float:
    """RAM residente en MB. Multiplataforma: psutil > lectura de /proc."""
    try:
        if _HAS_PSUTIL:
            return _psutil.Process().memory_info().rss / 1024 / 1024
        # Fallback: /proc/self/status (Linux sin psutil)
        proc = '/proc/self/status'
        if os.path.exists(proc):
            for line in open(proc):
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
        return 0.0
    except Exception:
        return 0.0


def _extract_quality_from_state(tracker: PipelineTracker, state: dict):
    """Lee el estado final del pipeline y rellena métricas de calidad."""

    # company_node
    company_names = state.get("company_names", [])
    tracker.record_quality(
        "company_node",
        n_companies_found=len(company_names),
        companies=", ".join(company_names),
    )

    # parallel_node
    company_data    = state.get("company_data", {})
    features_data   = state.get("features_data", {})
    scraped_content = state.get("scraped_content", [])
    n_companies     = len(company_names) or 1
    tracker.record_quality(
        "parallel_node",
        news_companies_ok=len(company_data),
        news_companies_pct=round(len(company_data) / n_companies * 100, 1),
        news_snippets_total=sum(len(v) for v in company_data.values()),
        web_companies_ok=len(features_data),
        web_companies_pct=round(len(features_data) / n_companies * 100, 1),
        features_pages_total=sum(len(v) for v in features_data.values()),
        scraped_content_records=len(scraped_content),
    )

    # embedding_node
    tracker.record_quality(
        "embedding_node",
        chunks_indexed=state.get("rag_chunk_count", 0),
        rag_index_path=state.get("rag_index_path", ""),
    )

    # structuring_node
    matrix_text = state.get("matrix_text", "")
    _analyse_matrix(tracker, matrix_text)

    # swot_node
    swot_data = state.get("swot_data", {})
    _analyse_swot(tracker, swot_data, n_companies)

    # report_node  (métricas del fichero generado, si existe)
    _analyse_report(tracker)


def _analyse_matrix(tracker: PipelineTracker, matrix_text: str):
    """Extrae métricas de la matriz comparativa."""
    if not matrix_text:
        tracker.record_quality("structuring_node", matrix_rows=0, matrix_cols=0,
                               matrix_fill_pct=0.0)
        return
    lines = [l for l in matrix_text.splitlines() if "|" in l and "---" not in l]
    rows  = len(lines) - 1  # descuenta encabezado
    cols  = max((l.count("|") - 1 for l in lines), default=0)
    total_cells = rows * cols if rows > 0 and cols > 0 else 1
    empty = sum(
        1 for l in lines[1:]
        for cell in l.split("|")[1:-1]
        if not cell.strip() or cell.strip() in ("-", "N/A", "n/a", "")
    )
    fill_pct = round((1 - empty / total_cells) * 100, 1)
    tracker.record_quality(
        "structuring_node",
        matrix_rows=rows,
        matrix_cols=cols,
        matrix_fill_pct=fill_pct,
    )


def _analyse_swot(tracker: PipelineTracker, swot_data: dict, n_companies: int):
    """Extrae métricas del análisis DAFO."""
    if not swot_data:
        tracker.record_quality("swot_node", swot_companies_ok=0,
                               swot_dims_complete_pct=0.0, avg_items_per_dim=0.0)
        return
    dims = {"strengths", "weaknesses", "opportunities", "threats"}
    complete, total_items, n_dims = 0, 0, 0
    for company_data in swot_data.values():
        present = set(company_data.keys()) & dims
        if present == dims:
            complete += 1
        for dim in present:
            items = company_data[dim]
            total_items += len(items)
            n_dims += 1
    tracker.record_quality(
        "swot_node",
        swot_companies_ok=len(swot_data),
        swot_companies_pct=round(len(swot_data) / max(n_companies, 1) * 100, 1),
        swot_dims_complete_pct=round(complete / max(len(swot_data), 1) * 100, 1),
        avg_items_per_dim=round(total_items / max(n_dims, 1), 2),
    )


def _analyse_report(tracker: PipelineTracker):
    """Lee market_report.md y market_report.pdf para obtener métricas."""
    md_path  = Path("market_report.md")
    pdf_path = Path("market_report.pdf")

    words, sections, pdf_kb = 0, 0, 0.0
    if md_path.exists():
        text     = md_path.read_text(encoding="utf-8", errors="ignore")
        words    = len(text.split())
        sections = text.count("\n## ")
    if pdf_path.exists():
        pdf_kb = round(pdf_path.stat().st_size / 1024, 1)

    tracker.record_quality(
        "report_node",
        report_words=words,
        report_sections=sections,
        pdf_kb=pdf_kb,
    )


def _load_runs_for_category(category: str, exclude_run: str) -> List[dict]:
    runs = []
    with EVAL_FILE.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("category") == category and r.get("run_id") != exclude_run:
                    runs.append(r)
            except Exception:
                pass
    return runs


def _fmt_s(seconds: float) -> str:
    if seconds >= 60:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"
    return f"{seconds:.1f}s"


def _mini_bar(pct: float, width: int = 8) -> str:
    filled = round(pct / 100 * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _delta_str(delta: float, invert: bool = False) -> str:
    """Verde si mejora (delta < 0 para tiempo, delta > 0 para calidad)."""
    if abs(delta) < 0.5:
        return "  ≈ igual"
    sign = "+" if delta > 0 else "-"
    val  = f"{sign}{abs(delta):.1f}s"
    # Para tiempo: delta < 0 es bueno; para calidad: delta > 0 es bueno
    good = (delta < 0) if not invert else (delta > 0)
    return f"▲ {val}" if not good else f"▼ {val}"


def _print_separator(title: str = ""):
    w = 64
    print("\n" + "═" * w)
    if title:
        print(f"  {title}")
        print("═" * w)
