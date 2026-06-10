#!/usr/bin/env python3
"""Driver for the diagnostic image's /test-args + /test-runs endpoints.

Reads a file of one-arg-string-per-line, POSTs each to /test-args with a
label, and prints a one-line summary per case (verdict, protected, dropped,
json-invalid, final-token count). Comments (# ...) and blank lines are
skipped. At the end, fetches /test-runs and prints a final tally.

Stdlib only -- no `requests` install needed.

USAGE
-----
    # 1. Rotate the OICM bearer token first if it has been pasted anywhere.
    # 2. Put it in the environment (never on the command line):
    export OICM_TOKEN='sk-...'
    export OICM_BASE='https://inference.adeoaiengine.ecouncil.ae/models/<deployment-id>/proxy'

    python3 run_arg_tests.py cases.txt
    python3 run_arg_tests.py cases.txt --probe-only       # check connectivity
    python3 run_arg_tests.py cases.txt --clear-first      # wipe runs before testing
    python3 run_arg_tests.py cases.txt --dump-failures    # show full results for FAILs
    python3 run_arg_tests.py cases.txt --insecure         # skip TLS verify
"""
import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


# ---------- HTTP plumbing (stdlib only) ------------------------------------

def _build_opener(insecure):
    """Build a urllib opener. If insecure=True, skip TLS verification --
    useful if the OICM proxy uses a corporate CA the host doesn't trust."""
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        handler = urllib.request.HTTPSHandler(context=ctx)
        return urllib.request.build_opener(handler)
    return urllib.request.build_opener()


def _request(opener, method, url, token, headers=None, body=None, timeout=30):
    """One HTTP call. Returns (status, body_bytes). On HTTPError we still
    return the status + body so the caller can show the server's message."""
    hdrs = {"Authorization": "Bearer " + token, "Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=body, method=method, headers=hdrs)
    try:
        with opener.open(req, timeout=timeout) as resp:
            return resp.getcode(), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read() or b""
    except urllib.error.URLError as e:
        # Network-level failure (DNS, refused, etc). Re-raise with context.
        raise SystemExit(f"network error calling {url}: {e.reason}")


# ---------- Case file parsing ----------------------------------------------

def load_cases(path):
    """Return [(label, arg_string), ...]. Lines starting with # are
    comments. A line of the form `# CATEGORY: name` updates the auto-label
    prefix for the cases that follow."""
    cases = []
    category = "case"
    seq_in_cat = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                # Recognize `# CATEGORY: <name>` to auto-label following cases.
                header = stripped.lstrip("#").strip()
                if header.lower().startswith("category:"):
                    category = header.split(":", 1)[1].strip().replace(" ", "-").lower() or "case"
                    seq_in_cat = 0
                continue
            seq_in_cat += 1
            label = f"{category}-{seq_in_cat:02d}"
            cases.append((label, line))    # use raw line; preserves any leading whitespace meaning
    return cases


# ---------- Endpoint operations --------------------------------------------

def probe(opener, base, token):
    """Confirm the proxy lets us reach BOTH endpoints. Many proxies allow
    only /health -- in that case POST /test-args returns 404 or 502, which
    is the signal to use kubectl port-forward instead."""
    print("=== reachability probe ===")
    # GET /test-runs is harmless (read-only) and confirms POST routes work too,
    # since the proxy almost always treats GET and POST the same per-path.
    status, body = _request(opener, "GET", base + "/test-runs?limit=1", token)
    if 200 <= status < 300:
        try:
            data = json.loads(body)
            print(f"  GET /test-runs -> {status} OK (total stored: {data.get('total')}, cap: {data.get('max_capacity')})")
            return True
        except Exception:
            print(f"  GET /test-runs -> {status} but body wasn't JSON: {body[:200]!r}")
            return False
    print(f"  GET /test-runs -> HTTP {status}")
    print(f"  body: {body[:400]!r}")
    print("  hint: if you get 404/502 here, the OICM proxy isn't forwarding non-health paths.")
    print("        Use `kubectl port-forward pod/<name> 8080:8080` and set OICM_BASE=http://127.0.0.1:8080")
    return False


def clear_runs(opener, base, token):
    status, body = _request(opener, "GET", base + "/test-runs?clear=1", token)
    if 200 <= status < 300:
        data = json.loads(body)
        print(f"=== cleared {data.get('cleared', 0)} previous run(s) ===")
    else:
        print(f"WARN: clear returned HTTP {status}: {body[:200]!r}")


def post_case(opener, base, token, label, arg_string):
    """POST one case. Returns the parsed result dict (or None on hard failure)."""
    qs = urllib.parse.urlencode({"label": label})
    url = f"{base}/test-args?{qs}"
    body = arg_string.encode("utf-8")
    status, resp = _request(opener, "POST", url, token,
                            headers={"Content-Type": "text/plain; charset=utf-8"},
                            body=body)
    if not (200 <= status < 300):
        return {"_http_error": status, "_body": resp.decode("utf-8", "replace")[:400]}
    try:
        return json.loads(resp)
    except json.JSONDecodeError:
        return {"_http_error": status, "_body": resp.decode("utf-8", "replace")[:400]}


def summarize(entry):
    """Compact one-line summary for the per-case print."""
    if "_http_error" in entry:
        return f"HTTP {entry['_http_error']}  {entry['_body'][:140]!r}"
    r = entry.get("result", {})
    return (f"{r.get('verdict','?'):<18} "
            f"tokens={len(r.get('tokens',[])):>3}  "
            f"protected={r.get('protected',0):>2}  "
            f"dropped={len(r.get('dropped',[])):>2}  "
            f"json_invalid={len(r.get('json_invalid',[])):>2}")


# ---------- Main -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cases_file", nargs="?",
                    help="Path to a file with one arg-string per line; # for comments")
    ap.add_argument("--base", default=os.environ.get("OICM_BASE"),
                    help="Base URL of the diag pod (default: $OICM_BASE)")
    ap.add_argument("--token", default=os.environ.get("OICM_TOKEN"),
                    help="Bearer token (default: $OICM_TOKEN). PREFER the env var.")
    ap.add_argument("--insecure", action="store_true",
                    help="Skip TLS verification (use for corporate CAs)")
    ap.add_argument("--probe-only", action="store_true",
                    help="Test connectivity to the endpoints and exit")
    ap.add_argument("--clear-first", action="store_true",
                    help="Wipe previous runs before posting cases")
    ap.add_argument("--dump-failures", action="store_true",
                    help="After the run, print full JSON for every FAIL")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="Seconds to sleep between requests (default 0)")
    args = ap.parse_args()

    if not args.base:
        sys.exit("ERROR: set OICM_BASE (or pass --base). E.g. "
                 "https://inference.adeoaiengine.ecouncil.ae/models/<id>/proxy")
    if not args.token:
        sys.exit("ERROR: set OICM_TOKEN in the environment "
                 "(don't put the bearer on the command line).")

    base = args.base.rstrip("/")
    opener = _build_opener(args.insecure)

    if not probe(opener, base, args.token):
        sys.exit(2)
    if args.probe_only:
        return

    if not args.cases_file:
        sys.exit("ERROR: need a cases file (or use --probe-only). "
                 "Try: python3 run_arg_tests.py cases.txt")
    if not os.path.exists(args.cases_file):
        sys.exit(f"ERROR: file not found: {args.cases_file}")
    cases = load_cases(args.cases_file)
    if not cases:
        sys.exit("ERROR: cases file has no non-comment lines")

    if args.clear_first:
        clear_runs(opener, base, args.token)

    print(f"\n=== posting {len(cases)} case(s) ===")
    counts = {"PASS": 0, "PASS (auto-fixed)": 0, "FAIL": 0, "OTHER": 0, "HTTP_ERR": 0}
    failures = []
    for label, arg_string in cases:
        entry = post_case(opener, base, args.token, label, arg_string)
        if "_http_error" in entry:
            counts["HTTP_ERR"] += 1
            print(f"  {label:<30} -> {summarize(entry)}")
            failures.append((label, arg_string, entry))
            if args.sleep:
                time.sleep(args.sleep)
            continue
        verdict = entry.get("result", {}).get("verdict", "OTHER")
        counts[verdict] = counts.get(verdict, 0) + 1
        print(f"  {label:<30} -> {summarize(entry)}")
        if verdict == "FAIL":
            failures.append((label, arg_string, entry))
        if args.sleep:
            time.sleep(args.sleep)

    print("\n=== tally ===")
    for k in ("PASS", "PASS (auto-fixed)", "FAIL", "OTHER", "HTTP_ERR"):
        print(f"  {k:<18} {counts.get(k, 0)}")

    if args.dump_failures and failures:
        print("\n=== failure details ===")
        for label, arg_string, entry in failures:
            print(f"\n--- {label} ---")
            print(f"input:  {arg_string!r}")
            print(json.dumps(entry, indent=2))

    # Final cross-check: how many runs the server thinks it stored.
    status, body = _request(opener, "GET", base + f"/test-runs?limit={len(cases)+10}",
                            args.token)
    if 200 <= status < 300:
        data = json.loads(body)
        print(f"\n=== server side: {data.get('total')} run(s) stored "
              f"(cap {data.get('max_capacity')}) ===")
        print(f"  fetch all: curl -H 'Authorization: Bearer $OICM_TOKEN' '{base}/test-runs?limit=500'")
        print(f"  failures : curl -H 'Authorization: Bearer $OICM_TOKEN' '{base}/test-runs?verdict=FAIL'")


if __name__ == "__main__":
    main()
