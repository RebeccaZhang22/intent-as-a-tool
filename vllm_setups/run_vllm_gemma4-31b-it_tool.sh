#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-4-31B-it \
    --served-model-name gemma4-31b-it \
    --tensor-parallel-size 2 \
    --reasoning-parser gemma4 \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --chat-template "$SCRIPT_DIR/gemma4_chat_template.jinja" \
    --max-logprobs 50 \
    "$@"
