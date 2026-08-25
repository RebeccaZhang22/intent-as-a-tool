#!/usr/bin/env bash
set -euo pipefail

exec python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B \
    --served-model-name qwen3-32b \
    --tensor-parallel-size 4 \
    --reasoning-parser deepseek_r1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-logprobs 50 \
    "$@"
