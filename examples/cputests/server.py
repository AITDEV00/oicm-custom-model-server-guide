"""
Diagnostic web server.

Endpoints
---------
  GET  /                     -> endpoint index
  GET  /health-check         -> liveness + capability probe
  POST /run                  -> run benchmark, STREAMING NDJSON progress
                                (also written to a per-run .log file)
  GET  /runs                 -> all stored run outputs (history)
  GET  /runs/{run_id}        -> a single run's full output
  GET  /runs/{run_id}/log    -> a single run's raw progress log

History is persisted as JSON files under $DATA_DIR (default /data), so mount a
volume there to keep results across container restarts.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

import bench

DATA_DIR = os.environ.get("DATA_DIR", "/data")
RUNS_DIR = os.path.join(DATA_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

app = FastAPI(title="CPU + Silero-VAD Diagnostic", version="1.0.0")
_write_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# request model
# --------------------------------------------------------------------------- #

class RunRequest(BaseModel):
    repetitions: int = Field(1, ge=1, le=50, description="how many times to repeat")
    cpu_threads: int | None = Field(None, ge=1, le=512)
    cpu_max_prime: int = Field(20000, ge=1000, le=1_000_000)
    vad_chunks: int = Field(2000, ge=10, le=200_000)
    vad_threads: int = Field(1, ge=1, le=128)
    skip_cpu: bool = False
    skip_vad: bool = False


# --------------------------------------------------------------------------- #
# persistence helpers
# --------------------------------------------------------------------------- #

def _run_paths(run_id: str) -> tuple[str, str]:
    return (
        os.path.join(RUNS_DIR, f"{run_id}.json"),
        os.path.join(RUNS_DIR, f"{run_id}.log"),
    )


def _save_record(record: Dict[str, Any]) -> None:
    json_path, _ = _run_paths(record["run_id"])
    tmp = json_path + ".tmp"
    with _write_lock:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, json_path)


def _load_record(run_id: str) -> Dict[str, Any] | None:
    json_path, _ = _run_paths(run_id)
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _list_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for name in os.listdir(RUNS_DIR):
        if name.endswith(".json"):
            try:
                with open(os.path.join(RUNS_DIR, name), encoding="utf-8") as fh:
                    records.append(json.load(fh))
            except (OSError, json.JSONDecodeError):
                continue
    records.sort(key=lambda r: r.get("started_at", ""), reverse=True)
    return records


# --------------------------------------------------------------------------- #
# endpoints
# --------------------------------------------------------------------------- #

@app.get("/")
def index() -> Dict[str, Any]:
    return {
        "service": "cpu-vad-diagnostic",
        "endpoints": {
            "GET /health-check": "liveness + capability probe",
            "POST /run": "run benchmark; streams NDJSON progress, persists result",
            "GET /runs": "list all stored run outputs (history)",
            "GET /runs/{run_id}": "single run full output",
            "GET /runs/{run_id}/log": "single run raw progress log",
        },
        "data_dir": DATA_DIR,
    }


@app.get("/health-check")
def health_check() -> Dict[str, Any]:
    caps = {
        "sysbench": shutil.which("sysbench") is not None,
        "lscpu": shutil.which("lscpu") is not None,
        "lstopo": shutil.which("lstopo-no-graphics") is not None,
    }
    vad_ok, vad_detail = True, None
    try:
        import onnxruntime  # noqa: F401
        bench._find_silero_onnx()
    except Exception as exc:  # noqa: BLE001
        vad_ok, vad_detail = False, str(exc)

    return {
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat(),
        "logical_cpus": os.cpu_count(),
        "capabilities": caps,
        "vad_available": vad_ok,
        "vad_detail": vad_detail,
        "stored_runs": len([n for n in os.listdir(RUNS_DIR) if n.endswith(".json")]),
    }


@app.post("/run")
def run(req: RunRequest) -> StreamingResponse:
    """
    Stream NDJSON progress events while the benchmark runs.

    Each line is one JSON event. Progress is *also* appended to
    {run_id}.log, and the final aggregated record is saved to {run_id}.json.
    """
    run_id = uuid.uuid4().hex[:12]
    started_at = datetime.now(timezone.utc).isoformat()
    _, log_path = _run_paths(run_id)

    record: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at,
        "status": "running",
        "params": req.model_dump(),
        "events": [],
        "result": None,
    }
    _save_record(record)

    def stream() -> Iterator[bytes]:
        log_fh = open(log_path, "w", encoding="utf-8")
        try:
            # emit run_id first so the client can correlate immediately
            head = {"event": "accepted", "run_id": run_id, "started_at": started_at}
            log_fh.write(json.dumps(head) + "\n")
            log_fh.flush()
            yield (json.dumps(head) + "\n").encode()

            for event in bench.execute_run(req.model_dump()):
                line = json.dumps(event, ensure_ascii=False)
                log_fh.write(line + "\n")
                log_fh.flush()
                record["events"].append(
                    {k: v for k, v in event.items() if k != "result"}
                )
                if event.get("event") == "result":
                    record["result"] = event["result"]
                yield (line + "\n").encode()

            record["status"] = "completed"
        except Exception as exc:  # noqa: BLE001
            record["status"] = "error"
            record["error"] = str(exc)
            err = {"event": "error", "message": str(exc)}
            log_fh.write(json.dumps(err) + "\n")
            yield (json.dumps(err) + "\n").encode()
        finally:
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            _save_record(record)
            log_fh.close()

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={"X-Run-Id": run_id, "Cache-Control": "no-cache"},
    )


@app.get("/runs")
def list_runs(full: bool = False) -> JSONResponse:
    """All stored run outputs. `?full=true` includes every per-rep detail."""
    records = _list_records()
    if not full:
        slim = []
        for r in records:
            slim.append({
                "run_id": r.get("run_id"),
                "started_at": r.get("started_at"),
                "finished_at": r.get("finished_at"),
                "status": r.get("status"),
                "params": r.get("params"),
                "summary": (r.get("result") or {}).get("summary"),
            })
        return JSONResponse({"count": len(slim), "runs": slim})
    return JSONResponse({"count": len(records), "runs": records})


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> JSONResponse:
    rec = _load_record(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return JSONResponse(rec)


@app.get("/runs/{run_id}/log", response_class=PlainTextResponse)
def get_run_log(run_id: str) -> str:
    _, log_path = _run_paths(run_id)
    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail=f"log for {run_id} not found")
    with open(log_path, "r", encoding="utf-8") as fh:
        return fh.read()
