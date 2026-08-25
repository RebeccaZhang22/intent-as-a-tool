#!/usr/bin/env bash
set -euo pipefail

exec python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-27B \
    --served-model-name qwen3.5-27b \
    --tensor-parallel-size 2 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --max-logprobs 50 \
    "$@"
