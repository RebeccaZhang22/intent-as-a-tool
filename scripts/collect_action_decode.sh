#!/bin/bash
# Collect full traces and truncated tool-call probabilities.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

INFERENCE_TYPE="${INFERENCE_TYPE:-vllm}"
MODEL="${MODEL:-qwen3-8b}"  # qwen3-8b, qwen3.5-27b, qwen3-32b, gemma4-31b-it, Qwen3-235B-A22B-Thinking-2507
NUM_SAMPLES="${NUM_SAMPLES:-1}"

SCENARIOS="${SCENARIOS:-${1:-blackmail leaking murder}}" # blackmail leaking murder all clean
STAGE="${STAGE:-1}"
TOOL_MODE="${TOOL_MODE:-signal}" # neutral, signal, or all

if [ "$INFERENCE_TYPE" = "api" ]; then
    LOAD_TOKENIZER="${LOAD_TOKENIZER:-false}"
else
    LOAD_TOKENIZER="${LOAD_TOKENIZER:-true}"
fi

TEMPERATURE="${TEMPERATURE:-1}"
MAX_WORKERS="${MAX_WORKERS:-32}"
TRUNCATION_GRANULARITY="${TRUNCATION_GRANULARITY:-token}"
BATCH_SIZE="${BATCH_SIZE:-32}"

VERBOSE=true

for SCENARIO in $SCENARIOS; do

    case "$SCENARIO" in
        blackmail|leaking|murder|all|clean) ;;
        *) echo "ERROR: Unknown scenario: $SCENARIO"; exit 1 ;;
    esac

    SCENE_NAME="agentic_misalignment"
    prompt_suffix=""
    PROMPTS_DIR="${PROMPTS_DIR:-$PROJECT_ROOT/data/agentic_misalignment}"
    [ ! -d "$PROMPTS_DIR" ] && echo "ERROR: $PROMPTS_DIR not found" && exit 1
    DEFAULT_STAGE1_DIR="$PROJECT_ROOT/results/${SCENE_NAME}_decode${prompt_suffix}_${TOOL_MODE}/${MODEL}/stage1"
    DEFAULT_STAGE2_DIR="$PROJECT_ROOT/results/${SCENE_NAME}_decode${prompt_suffix}_${TOOL_MODE}/${MODEL}/stage2_${TRUNCATION_GRANULARITY}"

    if [ "$STAGE" = "1" ]; then
        OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_STAGE1_DIR}"
        mkdir -p "$OUTPUT_DIR"
    elif [ "$STAGE" = "2" ]; then
        STAGE1_INPUT="${STAGE1_INPUT:-$DEFAULT_STAGE1_DIR}"
        OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_STAGE2_DIR}"
        [ ! -e "$STAGE1_INPUT" ] && echo "ERROR: Stage-1 input not found: $STAGE1_INPUT" && exit 1
        mkdir -p "$OUTPUT_DIR"
    else
        echo "ERROR: STAGE must be 1 or 2 (got: $STAGE)"
        exit 1
    fi

    echo "=================================================================================================="
    echo " Scenario: $SCENARIO  |  Stage: $STAGE  |  Model: $MODEL  |  Tool Mode: $TOOL_MODE"
    echo "=================================================================================================="
    echo "Output Dir:     $OUTPUT_DIR"
    [ "$STAGE" = "2" ] && echo "Stage1 Input:   $STAGE1_INPUT"
    echo "=================================================================================================="

    CMD=(
        python pipeline/collect_action_decode.py
        --stage "$STAGE"
        --scenario "$SCENARIO"
        --tool-mode "$TOOL_MODE"
        --model "$MODEL"
        --inference-type "$INFERENCE_TYPE"
        --temperature "$TEMPERATURE"
        --num-samples "$NUM_SAMPLES"
        --max-workers "$MAX_WORKERS"
        --truncation-granularity "$TRUNCATION_GRANULARITY"
        --resume
    )
    if [ "$STAGE" = "1" ]; then
        CMD+=(--prompts-dir "$PROMPTS_DIR" --output-dir "$OUTPUT_DIR")
    else
        CMD+=(--stage1-input "$STAGE1_INPUT" --output-dir "$OUTPUT_DIR")
    fi
    [ "$VERBOSE" = true ]   && CMD+=(--verbose)
    [ "$TRUNCATION_GRANULARITY" = "token" ] && CMD+=(--batch-size "$BATCH_SIZE")
    if [ "$LOAD_TOKENIZER" = "true" ]; then
        CMD+=(--load-tokenizer)
    fi

    set +e
    "${CMD[@]}"
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Scenario '$SCENARIO' stage $STAGE failed (exit code: $EXIT_CODE)"
        exit $EXIT_CODE
    fi

    echo ""
    echo "Done: $SCENARIO -> $OUTPUT_DIR"
    echo ""
done
