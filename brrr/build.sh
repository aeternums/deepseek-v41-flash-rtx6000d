#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
src=$(mktemp -d)
trap 'rm -rf -- "$src"' EXIT
git -C "$src" init -q
git -C "$src" fetch -q --depth=1 https://github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR.git 2aff97254f4b9005e181b91386c120505e091c18
git -C "$src" checkout -q --detach FETCH_HEAD
git -C "$src" apply "$PWD/engram-buffered.patch"
DOCKER_BUILDKIT=1 docker build --platform linux/amd64 --network=none \
  --build-arg BASE_IMAGE=ghcr.io/local-inference-lab/vllm@sha256:376745af9ebffe0ecb3a9b7ee3d2040027834e58e3248898e52487255f33125a \
  --build-arg UV_IMAGE=ghcr.io/astral-sh/uv@sha256:cbe0a44ba994e327b8fe7ed72beef1aaa7d2c4c795fd406d1dbf328bacb2f1c5 \
  -t deepseek-v41-brrr:2aff972-buffered "$src"
