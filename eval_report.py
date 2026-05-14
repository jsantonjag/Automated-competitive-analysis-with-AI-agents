#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_report.py
--------------
Herramienta de análisis del historial de evaluaciones.

Uso
----
    python eval_report.py                        # resumen de todos los runs
    python eval_report.py --category "Neobancos" # filtra por categoría
    python eval_report.py --last 3               # solo los N últimos runs
    python eval_report.py --node parallel_node   # detalle de un nodo
    python eval_report.py --csv                  # exporta a eval_summary.csv
"""

import argparse
import json
import csv
import statistics
from pathlib import Path

EVAL_FILE = Path("eval_runs.jsonl")

NODES = [
    "company_node",
    "parallel_node",
    "embedding_node",
    "structuring_node",
    "swot_node",
    "report_node",
]


def load_runs(category: str = None, last: int = None) -> list[dict]:
    if not EVAL_FILE.exists():
        print(f"[eval_report] No se encontró {EVAL_FILE}. Ejecuta el pipeline al menos una vez.")
        return []
    runs = []
    with EVAL_FILE.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if category and category.lower() not in r.get("category", "").lower():
                    continue
                runs.append(r)
            except Exception:
                pass
    if last:
        runs = runs[-last:]
    return runs


def fmt_s(s: float) -> str:
    if s >= 60:
        m, sec = divmod(int(s), 60)
        return f"{m}m{sec:02d}s"
    return f"{s:.1f}s"


def delta_indicator(val: float, ref: float, lower_is_better: bool = True) -> str:
    diff = val - ref
    if abs(diff) < 0.5:
        return "≈"
    if lower_is_better:
        return "▼" if diff < 0 else "▲"
    else:
        return "▲" if diff > 0 else "▼"


def print_runs_table(runs: list[dict]):
    print("\n" + "═" * 80)
    print(f"  HISTORIAL DE RUNS  ({len(runs)} ejecuciones)")
    print("═" * 80)
    print(f"  {'run_id':<10} {'Categoría':<30} {'Fecha':<22} {'Total':>8}  {'Nodos OK':>8}")
    print("  " + "─" * 76)
    for r in runs:
        total   = r.get("total_elapsed_s", 0)
        nodes   = r.get("nodes", {})
        ok      = sum(1 for n in nodes.values() if n.get("success", True))
        total_n = len(nodes)
        fecha   = r.get("started_at", "")[:19].replace("T", " ")
        cat     = r.get("category", "")[:29]
        print(f"  {r['run_id']:<10} {cat:<30} {fecha:<22} {fmt_s(total):>8}  {ok}/{total_n}")
    print()


def print_node_trend(runs: list[dict], node_name: str):
    print("\n" + "═" * 70)
    print(f"  TENDENCIA: {node_name}  ({len(runs)} runs)")
    print("═" * 70)

    times = []
    for r in runs:
        n = r.get("nodes", {}).get(node_name)
        if not n:
            continue
        t     = n.get("elapsed_s", 0)
        fecha = r.get("started_at", "")[:19].replace("T", " ")
        times.append(t)
        qual  = n.get("quality", {})
        qual_str = "  ".join(f"{k}={v}" for k, v in list(qual.items())[:4])
        print(f"  {fecha}  {fmt_s(t):>8}   {qual_str}")

    if len(times) >= 2:
        avg = statistics.mean(times)
        mn  = min(times)
        mx  = max(times)
        std = statistics.stdev(times) if len(times) > 1 else 0
        print(f"\n  Media: {fmt_s(avg)}  |  Mín: {fmt_s(mn)}  |  Máx: {fmt_s(mx)}  |  σ: {fmt_s(std)}")
    print()


def print_quality_trends(runs: list[dict]):
    print("\n" + "═" * 70)
    print(f"  TENDENCIA DE CALIDAD  ({len(runs)} runs)")
    print("═" * 70)

    # Recoge todas las métricas de calidad numéricas disponibles
    all_keys: dict[str, list[float]] = {}
    for r in runs:
        for node_name in NODES:
            n = r.get("nodes", {}).get(node_name)
            if not n:
                continue
            for k, v in n.get("quality", {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    label = f"{node_name}.{k}"
                    all_keys.setdefault(label, []).append(float(v))

    for label, vals in all_keys.items():
        if len(vals) < 2:
            continue
        avg = statistics.mean(vals)
        trend = vals[-1] - vals[0]
        arrow = "↗" if trend > 0.5 else ("↘" if trend < -0.5 else "→")
        print(f"  {label:<55}  últimos: {vals[-3:]!s:<30}  {arrow}")
    print()


def export_csv(runs: list[dict], out: str = "eval_summary.csv"):
    if not runs:
        return
    rows = []
    for r in runs:
        base = {
            "run_id":    r.get("run_id"),
            "category":  r.get("category"),
            "started_at": r.get("started_at", "")[:19],
            "total_s":   r.get("total_elapsed_s", 0),
        }
        for node_name in NODES:
            n = r.get("nodes", {}).get(node_name, {})
            base[f"{node_name}_elapsed_s"] = n.get("elapsed_s", "")
            base[f"{node_name}_ok"]        = n.get("success", "")
            for k, v in n.get("quality", {}).items():
                base[f"{node_name}__{k}"] = v
        rows.append(base)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval_report] CSV exportado → {out}  ({len(rows)} filas)")




def print_output_quality_history(runs: list[dict]):
    """Muestra la evolución de los scores de calidad de output por nodo."""
    print("\n" + "═" * 70)
    print("  CALIDAD DE OUTPUTS (scores 0-100 por nodo)")
    print("═" * 70)

    QUALITY_NODES = [
        "company_node", "news_branch", "web_branch",
        "embedding_node", "structuring_node", "swot_node", "report_node",
    ]

    for node in QUALITY_NODES:
        scores = []
        for r in runs:
            q = r.get("output_quality", {}).get(node, {})
            s = q.get("score")
            if s is not None:
                scores.append(float(s))
        if not scores:
            continue

        last    = scores[-1]
        avg     = sum(scores) / len(scores)
        trend   = scores[-1] - scores[0] if len(scores) > 1 else 0
        arrow   = "↗" if trend > 2 else ("↘" if trend < -2 else "→")
        bar     = "[" + "█" * round(last/10) + "░" * (10 - round(last/10)) + "]"
        passed  = last >= 60
        status  = "✓" if passed else "✗"

        print(
            f"  {status} {node:<22}  último={last:>5.1f}  avg={avg:>5.1f}"
            f"  {bar}  {arrow}"
        )

    # Detalle de warnings del último run
    last_run = runs[-1] if runs else {}
    warnings_found = False
    for node in QUALITY_NODES:
        q = last_run.get("output_quality", {}).get(node, {})
        for w in q.get("warnings", []):
            if not warnings_found:
                print("\n  Advertencias del último run:")
                warnings_found = True
            print(f"    [{node}] ⚠  {w}")
    print()

def print_judge_history(runs: list[dict]):
    """Muestra la evolución de los scores LLM-as-judge por evaluador."""
    print("\n" + "═" * 70)
    print("  LLM-AS-JUDGE — CALIDAD SEMÁNTICA (scores 0-10)")
    print("═" * 70)

    EVALUATORS = ["swot", "matrix", "report"]
    DIM_LABELS = {
        "swot":   ["consistency", "evidence", "actionability", "balance"],
        "matrix": ["accuracy", "differentiation", "usefulness"],
        "report": ["executive_value", "analytical_depth", "consistency",
                   "recommendation_quality", "writing_quality"],
    }

    for ev in EVALUATORS:
        overalls = []
        for r in runs:
            j = r.get("llm_judge", {}).get(ev, {})
            if j.get("skipped"):
                continue
            o = j.get("overall")
            if o is not None:
                overalls.append(float(o))

        if not overalls:
            print(f"  judge_{ev:<10}  (sin datos)")
            continue

        last  = overalls[-1]
        avg   = sum(overalls) / len(overalls)
        trend = overalls[-1] - overalls[0] if len(overalls) > 1 else 0
        arrow = "↗" if trend > 0.3 else ("↘" if trend < -0.3 else "→")
        bar   = "[" + "█" * round(last) + "░" * (10 - round(last)) + "]"
        status = "✓" if last >= 6 else "✗"
        print(
            f"  {status} judge_{ev:<10}  último={last:.1f}/10  "
            f"avg={avg:.1f}/10  {bar}  {arrow}"
        )

        # Dimensiones del último run con datos
        last_run_with_judge = next(
            (r for r in reversed(runs)
             if not r.get("llm_judge", {}).get(ev, {}).get("skipped")),
            None
        )
        if last_run_with_judge:
            dims = last_run_with_judge.get("llm_judge", {}).get(ev, {}).get("scores", {})
            for dim in DIM_LABELS.get(ev, []):
                val = dims.get(dim)
                if val is not None:
                    bar_d = "[" + "█" * round(float(val)) + "░" * (10 - round(float(val))) + "]"
                    print(f"      {dim:<30}  {float(val):.1f}/10  {bar_d}")

    # Weaknesses del último run
    last_run = runs[-1] if runs else {}
    printed_header = False
    for ev in EVALUATORS:
        j = last_run.get("llm_judge", {}).get(ev, {})
        for w in j.get("weaknesses", []):
            if not printed_header:
                print("\n  Puntos débiles detectados (último run):")
                printed_header = True
            print(f"    [judge_{ev}] - {w}")
    print()

def main():
    parser = argparse.ArgumentParser(description="Análisis del historial de evaluaciones del pipeline.")
    parser.add_argument("--category", "-c", default=None, help="Filtrar por categoría (parcial)")
    parser.add_argument("--last",     "-n", type=int, default=None, help="Mostrar solo los N últimos runs")
    parser.add_argument("--node",     "-d", default=None, help="Detalle de tendencia de un nodo concreto")
    parser.add_argument("--csv",            action="store_true",  help="Exportar resumen a eval_summary.csv")
    args = parser.parse_args()

    runs = load_runs(category=args.category, last=args.last)
    if not runs:
        return

    print_runs_table(runs)

    if args.node:
        print_node_trend(runs, args.node)
    else:
        # Tendencia de tiempos por nodo
        print("═" * 70)
        print("  TIEMPOS POR NODO (promedio de todos los runs mostrados)")
        print("═" * 70)
        print(f"  {'Nodo':<22} {'Runs':>5} {'Media':>9} {'Mín':>8} {'Máx':>8} {'σ':>8}")
        print("  " + "─" * 64)
        for node_name in NODES:
            times = [
                r["nodes"][node_name]["elapsed_s"]
                for r in runs
                if node_name in r.get("nodes", {}) and r["nodes"][node_name].get("elapsed_s")
            ]
            if not times:
                continue
            avg = statistics.mean(times)
            mn  = min(times)
            mx  = max(times)
            std = statistics.stdev(times) if len(times) > 1 else 0.0
            print(
                f"  {node_name:<22} {len(times):>5} {fmt_s(avg):>9}"
                f" {fmt_s(mn):>8} {fmt_s(mx):>8} {fmt_s(std):>8}"
            )
        print()
        print_quality_trends(runs)

    print_output_quality_history(runs)
    print_judge_history(runs)

    if args.csv:
        export_csv(runs)


if __name__ == "__main__":
    main()
