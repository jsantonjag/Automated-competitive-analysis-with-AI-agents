# -*- coding: utf-8 -*-
"""
app.py — Servidor FastAPI para la interfaz web del pipeline de inteligencia de mercado.

Endpoints:
  POST /api/run          → Lanza el pipeline con una categoría
  GET  /api/status/{id}  → Estado en tiempo real (SSE)
  GET  /api/result/{id}  → Resultado final (JSON)
  GET  /api/history      → Historial de ejecuciones
  GET  /                 → Interfaz web
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Market Intelligence Pipeline", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Almacén en memoria ────────────────────────────────────────────────────────
runs: Dict[str, Dict[str, Any]] = {}


# ── Modelos ───────────────────────────────────────────────────────────────────
class RunRequest(BaseModel):
    category: str


class RunResponse(BaseModel):
    run_id: str
    status: str


# ── Ejecutar pipeline en hilo separado ───────────────────────────────────────
def _run_pipeline(run_id: str, category: str) -> None:
    """Ejecuta el pipeline completo y actualiza el estado en `runs`."""
    run = runs[run_id]
    run["status"] = "running"
    run["logs"] = []
    run["started_at"] = datetime.utcnow().isoformat()

    # Captura stdout para emitir como logs en tiempo real
    import io

    class LogCapture(io.TextIOBase):
        def write(self, s):
            if s.strip():
                run["logs"].append({"ts": datetime.utcnow().isoformat(), "msg": s.rstrip()})
            return len(s)

        def flush(self):
            pass

    old_stdout = sys.stdout
    sys.stdout = LogCapture()

    try:
        # Importar pipeline aquí para evitar importaciones circulares a nivel módulo
        from graph import build_pipeline, reset_all_stores

        run["logs"].append({"ts": datetime.utcnow().isoformat(), "msg": f"[inicio] Categoría: {category}"})

        reset_all_stores()

        pipeline = build_pipeline()
        config = {"configurable": {"thread_id": run_id}}

        from evaluation.tracker import PipelineTracker
        import graph as graph_mod
        graph_mod._TRACKER = PipelineTracker(category=category, run_id=run_id)
        graph_mod._TRACKER.meta["thread_id"] = run_id

        final_state = pipeline.invoke({"category": category, "errors": []}, config=config)

        graph_mod._TRACKER.finish(final_state)
        graph_mod._TRACKER.save()

        # Leer outputs generados
        result: Dict[str, Any] = {
            "category": final_state.get("category", category),
            "companies": final_state.get("company_names", []),
            "rag_chunks": final_state.get("rag_chunk_count", 0),
            "errors": final_state.get("errors", []),
            "matrix_text": final_state.get("matrix_text", ""),
            "swot_text": final_state.get("swot_text", ""),
        }

        # Leer informe si existe
        for fname in ("market_report.md", "market_report.pdf", "comparative_matrix.csv"):
            if Path(fname).exists():
                result[fname] = True

        if Path("market_report.md").exists():
            result["report_md"] = Path("market_report.md").read_text(encoding="utf-8")

        run["result"] = result
        run["status"] = "completed"

    except Exception as exc:
        run["status"] = "error"
        run["error"] = str(exc)
        run["logs"].append({"ts": datetime.utcnow().isoformat(), "msg": f"[ERROR] {exc}"})
    finally:
        sys.stdout = old_stdout
        run["finished_at"] = datetime.utcnow().isoformat()


# ── Endpoints API ─────────────────────────────────────────────────────────────
@app.post("/api/run", response_model=RunResponse)
async def start_run(req: RunRequest):
    if not req.category.strip():
        raise HTTPException(status_code=400, detail="La categoría no puede estar vacía")

    run_id = str(uuid.uuid4())[:8]
    runs[run_id] = {
        "run_id": run_id,
        "category": req.category,
        "status": "queued",
        "logs": [],
        "result": None,
        "created_at": datetime.utcnow().isoformat(),
    }

    thread = threading.Thread(target=_run_pipeline, args=(run_id, req.category), daemon=True)
    thread.start()

    return RunResponse(run_id=run_id, status="queued")


@app.get("/api/status/{run_id}")
async def stream_status(run_id: str):
    """Server-Sent Events: emite logs en tiempo real."""
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run no encontrado")

    async def event_generator():
        sent = 0
        while True:
            run = runs[run_id]
            logs = run.get("logs", [])

            while sent < len(logs):
                entry = logs[sent]
                data = json.dumps({"type": "log", "ts": entry["ts"], "msg": entry["msg"]})
                yield f"data: {data}\n\n"
                sent += 1

            status = run["status"]
            if status in ("completed", "error"):
                payload = {"type": "done", "status": status}
                if status == "error":
                    payload["error"] = run.get("error", "Error desconocido")
                yield f"data: {json.dumps(payload)}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/result/{run_id}")
async def get_result(run_id: str):
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run no encontrado")
    run = runs[run_id]
    if run["status"] != "completed":
        raise HTTPException(status_code=202, detail=f"Estado actual: {run['status']}")
    return run["result"]


@app.get("/api/history")
async def get_history():
    return [
        {
            "run_id": r["run_id"],
            "category": r["category"],
            "status": r["status"],
            "created_at": r.get("created_at"),
            "finished_at": r.get("finished_at"),
        }
        for r in reversed(list(runs.values()))
    ]


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    safe = Path(filename).name
    path = Path(safe)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    return FileResponse(path, filename=safe)


# ── Frontend ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path("frontend/index.html")
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Frontend no encontrado. Coloca index.html en /frontend/</h1>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
