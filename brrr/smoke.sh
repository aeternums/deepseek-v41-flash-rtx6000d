#!/usr/bin/env bash
set -euo pipefail
base=${1:-http://127.0.0.1:18136}
curl -fsS --max-time 5 "$base/health"
curl -fsS --max-time 5 "$base/v1/models" |
  jq -e '.data[] | select(.id=="DeepSeek-V4.1-Flash-1017" and .max_model_len==1048576)'
curl -fsS --max-time 30 "$base/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"What is 17 times 19? Answer with only the number."}],"temperature":0,"max_tokens":16,"chat_template_kwargs":{"thinking":false}}' |
  jq -er '.choices[0].message.content | select(test("^\\s*323\\s*$"))'
