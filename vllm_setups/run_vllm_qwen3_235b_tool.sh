#!/usr/bin/env bash
set -euo pipefail

exec python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-235B-A22B-Thinking-2507 \
    --served-model-name Qwen3-235B-A22B-Thinking-2507 \
    --tensor-parallel-size 8 \
    --reasoning-parser deepseek_r1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-logprobs 50 \
    "$@"
