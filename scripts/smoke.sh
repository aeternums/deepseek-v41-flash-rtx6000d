#!/usr/bin/env bash
# Minimal end-to-end check against a running server.
set -euo pipefail
BASE=${BASE:-http://127.0.0.1:8000}
MODEL=${MODEL:-DeepSeek-V4.1-Flash}

echo 'models:'
curl -fsS "$BASE/v1/models" | jq -r '.data[].id'

echo 'chat:'
curl -fsS -w '\nttft=%{time_starttransfer}s total=%{time_total}s\n' \
  "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":32}" \
  | tail -c 400
