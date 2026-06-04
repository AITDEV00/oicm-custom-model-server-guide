#!/usr/bin/env python3
"""Tiny keep-alive server for the OICM diagnostic.

- Answers health paths (/health, /health-check, ...) with a fast 200 'ok'.
- Serves the full diagnostic report on any other path.
- RECORDS which paths are requested -- so you can see exactly which path
  OICM's health probe uses. The FIRST request to each distinct path is
  printed to stdout (visible in `kubectl logs`), and a live summary of all
  observed paths is appended to the report (visible when you curl '/').
"""
import os
import time
import http.server
import socketserver

PORT = int(os.environ.get("PORT", "8080"))
REPORT_FILE = os.environ.get("REPORT_FILE", "")

HEALTH_PATHS = {"/health", "/health-check", "/healthz", "/livez", "/readyz", "/ping"}

# path -> {count, methods:set, first, last, agents:set}
OBSERVED = {}


def record(method, raw_path, client, agent):
    path = raw_path.split("?", 1)[0].rstrip("/") or "/"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    entry = OBSERVED.get(path)
    if entry is None:
        entry = {"count": 0, "methods": set(), "first": now, "last": now, "agents": set()}
        OBSERVED[path] = entry
        # Log only the FIRST hit per path -> reveals the probe path, no log spam.
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
    return ("\n".join(lines) + "\n").encode()


class Handler(http.server.BaseHTTPRequestHandler):
    def _client(self):
        try:
            return self.client_address[0]
        except Exception:
            return "?"

    def _send_health(self):
        body = b"ok\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _send_report(self):
        body = b"OICM diagnostic running. Report file not available; see pod logs.\n"
        try:
            if REPORT_FILE and os.path.exists(REPORT_FILE):
                with open(REPORT_FILE, "rb") as fh:
                    body = fh.read()
        except Exception as exc:  # noqa: BLE001
            body = ("error reading report: %s\n" % exc).encode()
        body += observed_summary()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def do_GET(self):
        agent = self.headers.get("User-Agent", "")
        path = record("GET", self.path, self._client(), agent)
        if path in HEALTH_PATHS:
            self._send_health()
        else:
            self._send_report()

    def do_HEAD(self):
        # k8s httpGet probes use GET, but record + 200 HEAD too, just in case.
        agent = self.headers.get("User-Agent", "")
        record("HEAD", self.path, self._client(), agent)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        return  # suppress default per-request noise; record() does concise logging


class Server(socketserver.TCPServer):
    allow_reuse_address = True


if __name__ == "__main__":
    with Server(("0.0.0.0", PORT), Handler) as httpd:
        print("[diag] serving on :%d (health paths -> 200 'ok'; other paths -> report)" % PORT,
              flush=True)
        print("[diag] logging the FIRST request to each distinct path so you can see "
              "which path OICM's health probe uses.", flush=True)
        httpd.serve_forever()