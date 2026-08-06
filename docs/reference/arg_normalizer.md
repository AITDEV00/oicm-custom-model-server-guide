# arg_normalizer.py

A **pure-stdlib** normalizer (no dependencies, no `eval`) that turns the OICM
"Model Server Arguments" field (`EXTRA_ARGS`) into the final argv passed to the
server.

## What it does

- Protects unquoted JSON values
- Folds `--flag value` → `--flag=value`
- Dedups repeated flags (last-wins)
- Validates JSON
- Returns a PASS/FAIL verdict

The SGLang variants enumerate SGLang's JSON flags (e.g.
`--json-model-override-args`) and note that SGLang speculative decoding uses
scalar flags, not JSON.

## Two input formats

1. **Dot-notation** (recommended — avoids quote escaping):
   ```
   --speculative-config.method mtp --speculative-config.num_speculative_tokens 5
   ```
2. **JSON with escaped quotes** (legacy):
   ```
   --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":5}'
   ```