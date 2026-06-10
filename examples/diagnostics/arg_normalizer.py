#!/usr/bin/env python3
"""Production-mirroring argument normalizer.

Single source of truth for how the OICM 'Model Server Arguments' field
(EXTRA_ARGS) is turned into the argv vLLM actually receives.

Used by:
  - startup.sh section 16 (subprocess: `python3 -m arg_normalizer --report`)
  - serve_report.py POST /test-args (in-process import: `normalize(raw)`)

Goals:
  * Accept every form an operator might type -- =/space/quoted/unquoted JSON,
    scalar flags, dot-notation -- so they don't have to remember a rule.
  * Auto-fix what's recoverable: protect unquoted JSON before shlex strips its
    inner ", dedup repeated flags last-wins (vLLM's argparse-style).
  * Diagnose what's NOT recoverable: malformed JSON, unbalanced quotes -- raise
    nothing; report it. The caller decides whether to crash or continue.
  * Pure stdlib. No `eval`. No third-party deps. No Dockerfile changes.
"""
import json
import shlex
import sys


# We deliberately do NOT keep a hardcoded list of "JSON-valued flags".
# vLLM grows new ones every release (--speculative-config, --compilation-config,
# --reasoning-config, --attention-config, --kv-transfer-config,
# --structured-outputs-config, --additional-config, --hf-overrides,
# --override-generation-config, --default-chat-template-kwargs,
# --mm-processor-kwargs, --limit-mm-per-prompt, --kernel-config, ...) and any
# such list will drift behind the version actually running in the image.
#
# Instead, we detect JSON values STRUCTURALLY: a token whose value starts with
# '{' or '['. That's the only thing that matters for the two failure modes this
# normalizer fixes (shlex stripping the inner " from an unquoted JSON, and an
# operator hand-writing malformed JSON). It works for every current vLLM flag
# AND every future one, with no list to maintain.
#
# All vLLM JSON flags also accept the equivalent dot-notation form (e.g.
# --speculative-config.method=mtp), which is a regular scalar string -- the
# normalizer routes those through the normal scalar path automatically; no
# special-case needed.


# ---------------------------------------------------------------------------
# Stage 1: protect unquoted JSON values
# ---------------------------------------------------------------------------

def _consume_balanced(s, i):
    """Walk s starting at s[i] (which is { or [), return the balanced span and
    the index just past it. Honours quotes inside so a brace inside a string
    doesn't close the structure early. If unbalanced, returns the rest of the
    string -- json.loads will then report the precise error to the caller."""
    depth = 0
    j = i
    in_single = in_double = False
    while j < len(s):
        c = s[j]
        if in_single:
            if c == "'":
                in_single = False
        elif in_double:
            if c == '"':
                in_double = False
        else:
            if c == "'":
                in_single = True
            elif c == '"':
                in_double = True
            elif c in "{[":
                depth += 1
            elif c in "}]":
                depth -= 1
                if depth == 0:
                    return s[i:j + 1], j + 1
        j += 1
    return s[i:], len(s)


def _protect_unquoted_json(s):
    """Wrap UNQUOTED JSON values with shlex.quote so the inner " survives
    shlex.split. Already-quoted values are left untouched; shlex.quote
    guarantees shlex.split round-trips its output verbatim.

    Returns (new_string, count_of_values_protected).
    """
    out = []
    i = 0
    n = len(s)
    in_single = in_double = False
    protected = 0
    while i < n:
        c = s[i]
        if in_single:
            out.append(c)
            i += 1
            if c == "'":
                in_single = False
            continue
        if in_double:
            out.append(c)
            i += 1
            if c == '"':
                in_double = False
            continue
        if c == "'":
            in_single = True
            out.append(c)
            i += 1
            continue
        if c == '"':
            in_double = True
            out.append(c)
            i += 1
            continue
        if c in "{[":
            # "Value position" = preceded by '=' or whitespace (or start of string).
            # Avoids wrapping braces that show up inside a value somehow (rare).
            prev = out[-1] if out else None
            if prev is None or prev == "=" or prev.isspace():
                span, j = _consume_balanced(s, i)
                out.append(shlex.quote(span))
                protected += 1
                i = j
                continue
        out.append(c)
        i += 1
    return "".join(out), protected


# ---------------------------------------------------------------------------
# Stage 2: fold + dedup tokens into final argv
# ---------------------------------------------------------------------------

def _is_flag(t):
    """A '--flag' or '-f', but never a negative number ('-0.92' is a value)."""
    return t.startswith("--") or (len(t) > 1 and t[0] == "-" and not t[1].isdigit())


def _fold_and_dedup(toks):
    """Fold space-form `--flag value` into `--flag=value`; keep booleans bare;
    dedup repeated flags keeping the LAST occurrence (argparse-style); preserve
    positionals in order. Returns (final_tokens, dropped_tokens)."""
    items = []   # list of (key_or_None, token_string)
    i = 0
    while i < len(toks):
        t = toks[i]
        if _is_flag(t):
            if "=" in t:
                items.append((t.split("=", 1)[0], t))
                i += 1
            else:
                nxt = toks[i + 1] if i + 1 < len(toks) else None
                if nxt is not None and not _is_flag(nxt):
                    items.append((t, t + "=" + nxt))
                    i += 2
                else:
                    items.append((t, t))  # bare boolean
                    i += 1
        else:
            items.append((None, t))   # positional
            i += 1

    # Build last-occurrence index per flag key.
    last = {}
    for idx, (k, _t) in enumerate(items):
        if k is not None:
            last[k] = idx

    final, dropped = [], []
    for idx, (k, tok) in enumerate(items):
        if k is None or last[k] == idx:
            final.append(tok)
        else:
            dropped.append(tok)
    return final, dropped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize(raw):
    """Normalize a raw EXTRA_ARGS string into a structured report + final argv.

    Returns a dict with keys:
      raw            : the input string (unchanged)
      protected      : number of unquoted JSON values that were auto-protected
      tokens         : list of normalized argv tokens (what vLLM would receive)
      dropped        : list of duplicate-flag tokens that were removed
      json_invalid   : list of {flag, value, error} for values still not valid JSON
      tokenize_error : string error from shlex, or None
      verdict        : "PASS" | "PASS (auto-fixed)" | "FAIL"
      notes          : list of human-readable notes about what happened

    The function NEVER raises on operator input; everything bad shows up as
    a verdict + notes so callers can decide whether to use the result.
    """
    raw = raw or ""
    out = {
        "raw": raw,
        "protected": 0,
        "tokens": [],
        "dropped": [],
        "json_invalid": [],
        "tokenize_error": None,
        "verdict": "PASS",
        "notes": [],
    }

    stripped = raw.strip()
    if not stripped:
        out["notes"].append("input is empty; nothing to normalize")
        return out

    protected_src, n_protected = _protect_unquoted_json(stripped)
    out["protected"] = n_protected

    try:
        toks = shlex.split(protected_src)
    except ValueError as e:
        # Fall back to the raw input -- the protection pass may itself be the
        # cause if quotes were unbalanced. Either way, report it as the failure.
        out["tokenize_error"] = str(e)
        try:
            toks = shlex.split(stripped)
            out["notes"].append("protected tokenize failed (%s); fell back to raw" % e)
        except ValueError as e2:
            out["tokenize_error"] = str(e2)
            out["verdict"] = "FAIL"
            out["notes"].append("shlex cannot tokenize the input -- check for unbalanced quotes")
            return out

    final, dropped = _fold_and_dedup(toks)
    out["tokens"] = final
    out["dropped"] = dropped

    # Validate any remaining JSON-looking values.
    for tok in final:
        if tok.startswith("-") and "=" in tok:
            flag, val = tok.split("=", 1)
            if val[:1] in "{[":
                try:
                    json.loads(val)
                except Exception as e:
                    out["json_invalid"].append({"flag": flag, "value": val, "error": str(e)})

    if out["json_invalid"]:
        out["verdict"] = "FAIL"
        out["notes"].append(
            "%d JSON value(s) still invalid after auto-fix -- the JSON itself is malformed"
            % len(out["json_invalid"])
        )
    elif n_protected or dropped:
        out["verdict"] = "PASS (auto-fixed)"
        if n_protected:
            out["notes"].append("auto-protected %d unquoted JSON value(s)" % n_protected)
        if dropped:
            out["notes"].append(
                "dropped %d duplicate flag(s) last-wins -- confirm the kept values are intended"
                % len(dropped)
            )
    else:
        out["verdict"] = "PASS"
        out["notes"].append("clean input, no normalization required")
    return out


# ---------------------------------------------------------------------------
# CLI: called by startup.sh section 16 for the in-report verdict
# ---------------------------------------------------------------------------

def _render_text_report(result):
    lines = []
    line = lines.append
    line("  raw EXTRA_ARGS+VLLM_EXTRA_ARGS : %r" % result["raw"])
    if not result["raw"].strip():
        line("  => empty; production passes no extra args. VERDICT: PASS")
        return "\n".join(lines)

    line("")
    line("  AUTO-FIXES the normalizer applied:")
    line("    unquoted JSON values protected      : %d" % result["protected"])
    line("    duplicate flags dropped (last-wins) : %d" % len(result["dropped"]))
    for d in result["dropped"]:
        line("      - %s" % d)

    line("")
    line("  FINAL argv -> what production hands to vLLM (%d tokens):" % len(result["tokens"]))
    bad_set = {(b["flag"], b["value"]) for b in result["json_invalid"]}
    for t in result["tokens"]:
        tag = ""
        if t.startswith("-") and "=" in t:
            flag, val = t.split("=", 1)
            if val[:1] in "{[":
                if (flag, val) in bad_set:
                    tag = "   [JSON INVALID]"
                else:
                    tag = "   [JSON OK]"
        line("    %s%s" % (t, tag))

    if result["tokenize_error"]:
        line("")
        line("  TOKENIZE ERROR: %s" % result["tokenize_error"])

    if result["json_invalid"]:
        line("")
        line("  STILL-INVALID JSON values (normalizer can't invent valid JSON):")
        for b in result["json_invalid"]:
            line("    - %s : %s" % (b["flag"], b["error"]))

    line("")
    line("  VERDICT: %s" % result["verdict"])
    for n in result["notes"]:
        line("    %s" % n)
    return "\n".join(lines)


def main(argv):
    import os
    raw = (os.environ.get("EXTRA_ARGS", "") + " "
           + os.environ.get("VLLM_EXTRA_ARGS", "")).strip()
    result = normalize(raw)
    if "--json" in argv:
        sys.stdout.write(json.dumps(result, indent=2))
    elif "--argv-nul" in argv:
        # Production startup.sh consumes this: NUL-separated normalized tokens.
        sys.stdout.write("\0".join(result["tokens"]))
    else:
        sys.stdout.write(_render_text_report(result) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
