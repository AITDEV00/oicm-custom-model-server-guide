#!/usr/bin/env python3
"""Tiny keep-alive server for the OICM diagnostic.

Endpoints:
  GET  /health (and aliases)  -> 200 'ok' (for k8s probes)
  GET  /                      -> full diagnostic report + observed-paths summary
  POST /test-args             -> normalize a raw EXTRA_ARGS string and return the
                                 same verdict section 16 would produce. Body is
                                 either raw text (Content-Type: text/plain) or
                                 JSON {"args": "..."} (Content-Type: application/json).
                                 The run is stored in a bounded ring buffer.
  GET  /test-runs             -> list of stored runs (most recent first).
                                 Query params:
                                   limit=N      (default 50, max 500)
                                   offset=N     (default 0)
                                   clear=1      delete all stored runs first
                                   verdict=...  filter by verdict substring

Why these two endpoints exist: section 16 only sees the EXTRA_ARGS the pod was
deployed with. /test-args lets you fire any number of variant inputs at the
SAME normalizer code without redeploying, and /test-runs gives you a single
place to read back what you tested and what each input produced.
"""
import json
import os
import threading
import time
import urllib.parse
from collections import deque

import http.server
import socketserver

# Local module: shared with startup.sh section 16 so the two paths can't drift.
import arg_normalizer


PORT = int(os.environ.get("PORT", "8080"))
REPORT_FILE = os.environ.get("REPORT_FILE", "")

# Bounded ring buffer for /test-args runs. Sized so a sustained 1 req/s flood
# couldn't OOM the pod -- each entry is well under a kilobyte.
MAX_RUNS = int(os.environ.get("DIAG_MAX_RUNS", "500"))

HEALTH_PATHS = {"/health", "/health-check", "/healthz", "/livez", "/readyz", "/ping"}

# Observed-path tracking (unchanged from before; documents the probe path).
OBSERVED = {}

# Test-runs storage. deque enforces the bound; lock keeps multi-client safety.
RUNS = deque(maxlen=MAX_RUNS)
RUNS_LOCK = threading.Lock()
_RUN_SEQ = 0  # monotonically-increasing id, never reused even after eviction


def record(method, raw_path, client, agent):
    path = raw_path.split("?", 1)[0].rstrip("/") or "/"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    entry = OBSERVED.get(path)
    if entry is None:
        entry = {"count": 0, "methods": set(), "first": now, "last": now, "agents": set()}
        OBSERVED[path] = entry
        print("[probe] FIRST request: method=%s path=%s client=%s ua=%r time=%s"
              % (method, path, client, agent, now), flush=True)
    entry["count"] += 1
    entry["methods"].add(method)
    entry["last"] = now
    if agent:
        entry["agents"].add(agent)
    return path


def observed_summary():
    bar = "=" * 63
    lines = ["", bar,
             "  OBSERVED PROBE / REQUEST PATHS (live, since server start)",
             "  -> the path your k8s/OICM health check hits will appear here",
             bar]
    if not OBSERVED:
        lines.append("  (no requests received yet)")
    else:
        for p, e in sorted(OBSERVED.items(), key=lambda kv: -kv[1]["count"]):
            tag = "   <== handled as HEALTH (200 ok)" if p in HEALTH_PATHS else ""
            lines.append("  %-22s count=%-6d methods=%s  first=%s  last=%s%s"
                         % (p, e["count"], ",".join(sorted(e["methods"])),
                            e["first"], e["last"], tag))
            if e["agents"]:
                lines.append("      user-agents: %s" % " | ".join(sorted(e["agents"])))
    lines.append(bar)
    lines.append("  Try the normalizer interactively:")
    lines.append("    POST /test-args   body: raw arg string (text or {\"args\":\"...\"})")
    lines.append("    GET  /test-runs   list previous /test-args invocations")
    lines.append(bar)
    return ("\n".join(lines) + "\n").encode()


# ---------------------------------------------------------------------------
# /test-args + /test-runs implementation
# ---------------------------------------------------------------------------

# Body cap: refuse anything pathological. 64 KB is far more than any sane args
# string and protects the pod from accidental file-uploads to this endpoint.
MAX_BODY_BYTES = 64 * 1024


def _parse_test_args_body(content_type, body_bytes):
    """Accept either JSON {"args": "..."} or raw text/plain. Returns the
    string to normalize, or raises ValueError with a helpful message."""
    ctype = (content_type or "").split(";", 1)[0].strip().lower()
    text = body_bytes.decode("utf-8", errors="replace")
    if ctype == "application/json":
        try:
            obj = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError as e:
            raise ValueError("invalid JSON body: %s" % e)
        if not isinstance(obj, dict) or "args" not in obj:
            raise ValueError("JSON body must be an object with an 'args' field")
        if not isinstance(obj["args"], str):
            raise ValueError("'args' must be a string")
        return obj["args"]
    # text/plain (or unknown): treat the whole body as the arg string.
    return text


def _store_run(raw, result, client, label):
    """Persist a run record. Returns the stored entry."""
    global _RUN_SEQ
    with RUNS_LOCK:
        _RUN_SEQ += 1
        entry = {
            "id": _RUN_SEQ,
            "received_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "client": client,
            "label": label,                          # optional caller-supplied tag
            "input": raw,
            "result": result,                        # the full normalize() output
        }
        RUNS.append(entry)
        return entry


def _list_runs(limit, offset, verdict_filter):
    """Most-recent first slice. Filter happens before pagination."""
    with RUNS_LOCK:
        items = list(RUNS)
    items.reverse()  # newest first
    if verdict_filter:
        items = [r for r in items if verdict_filter in r["result"]["verdict"]]
    total = len(items)
    sliced = items[offset:offset + limit]
    return sliced, total


def _clear_runs():
    with RUNS_LOCK:
        n = len(RUNS)
        RUNS.clear()
    return n


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(http.server.BaseHTTPRequestHandler):
    def _client(self):
        try:
            return self.client_address[0]
        except Exception:
            return "?"

    def _send_json(self, status, payload):
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _send_text(self, status, text, content_type="text/plain; charset=utf-8"):
        body = text if isinstance(text, (bytes, bytearray)) else text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    # --- health & report ---------------------------------------------------

    def _send_health(self):
        self._send_text(200, b"ok\n")

    def _send_report(self):
        body = b"OICM diagnostic running. Report file not available; see pod logs.\n"
        try:
            if REPORT_FILE and os.path.exists(REPORT_FILE):
                with open(REPORT_FILE, "rb") as fh:
                    body = fh.read()
        except Exception as exc:  # noqa: BLE001
            body = ("error reading report: %s\n" % exc).encode()
        body += observed_summary()
        self._send_text(200, body)

    # --- test-args ---------------------------------------------------------

    def _handle_test_args_post(self):
        # Read at most MAX_BODY_BYTES.
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._send_json(400, {"error": "invalid Content-Length"})
        if length > MAX_BODY_BYTES:
            return self._send_json(413, {"error": "body too large",
                                         "limit_bytes": MAX_BODY_BYTES})
        body = self.rfile.read(length) if length > 0 else b""

        # Optional ?label=... so a caller can tag runs (e.g. case names).
        qs = urllib.parse.urlparse(self.path).query
        label = urllib.parse.parse_qs(qs).get("label", [""])[0]

        try:
            raw = _parse_test_args_body(self.headers.get("Content-Type"), body)
        except ValueError as e:
            return self._send_json(400, {"error": str(e)})

        result = arg_normalizer.normalize(raw)
        entry = _store_run(raw=raw, result=result, client=self._client(), label=label)
        return self._send_json(200, entry)

    # --- test-runs ---------------------------------------------------------

    def _handle_test_runs_get(self):
        qs = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(qs)

        if params.get("clear", ["0"])[0] in ("1", "true", "yes"):
            n = _clear_runs()
            return self._send_json(200, {"cleared": n, "runs": [], "total": 0})

        try:
            limit = max(1, min(500, int(params.get("limit", ["50"])[0])))
            offset = max(0, int(params.get("offset", ["0"])[0]))
        except ValueError:
            return self._send_json(400, {"error": "limit/offset must be integers"})
        verdict_filter = params.get("verdict", [""])[0]

        runs, total = _list_runs(limit, offset, verdict_filter)
        return self._send_json(200, {
            "runs": runs,
            "returned": len(runs),
            "total": total,
            "limit": limit,
            "offset": offset,
            "max_capacity": MAX_RUNS,
        })

    # --- dispatch ----------------------------------------------------------

    def do_GET(self):
        agent = self.headers.get("User-Agent", "")
        path = record("GET", self.path, self._client(), agent)
        if path in HEALTH_PATHS:
            return self._send_health()
        if path == "/test-runs":
            return self._handle_test_runs_get()
        if path == "/test-args":
            # Help text on accidental GET -- the endpoint is POST-only.
            return self._send_json(405, {
                "error": "method not allowed; use POST",
                "usage": {
                    "POST /test-args": "body is the raw EXTRA_ARGS string "
                                       "(Content-Type: text/plain) "
                                       "or JSON {\"args\": \"...\"} "
                                       "(Content-Type: application/json). "
                                       "Optional ?label= to tag the run.",
                    "GET /test-runs": "list stored runs; params: limit, offset, "
                                      "verdict, clear=1",
                },
            })
        return self._send_report()

    def do_POST(self):
        agent = self.headers.get("User-Agent", "")
        path = record("POST", self.path, self._client(), agent)
        if path == "/test-args":
            return self._handle_test_args_post()
        return self._send_json(404, {"error": "no such endpoint", "path": path})

    def do_HEAD(self):
        # k8s httpGet probes use GET, but record + 200 HEAD too just in case.
        agent = self.headers.get("User-Agent", "")
        record("HEAD", self.path, self._client(), agent)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        return  # suppress default per-request noise; record() does concise logging


class Server(socketserver.ThreadingTCPServer):
    """Threaded so /test-args requests don't block the k8s health probe."""
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    with Server(("0.0.0.0", PORT), Handler) as httpd:
        print("[diag] serving on :%d (health paths -> 200 'ok'; / -> report)" % PORT,
              flush=True)
        print("[diag] new: POST /test-args  GET /test-runs  (run live normalizer "
              "tests; max %d stored runs)" % MAX_RUNS, flush=True)
        print("[diag] logging the FIRST request to each distinct path so you can see "
              "which path OICM's health probe uses.", flush=True)
        httpd.serve_forever()
