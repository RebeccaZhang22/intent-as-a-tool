#!/bin/bash
# Run online-intervention or system-prompt defense experiments.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL="${MODEL:-qwen3-8b}"  # qwen3-8b, qwen3.5-27b, qwen3-32b
                            # gemma4-31b-it, Qwen3-235B-A22B-Thinking-2507
NUM_SAMPLES="${NUM_SAMPLES:-3}"

SCENARIOS="${SCENARIOS:-${1:-blackmail leaking murder}}" # blackmail leaking murder all
TOOL_MODE="${TOOL_MODE:-signal}" # neutral or signal
DEFENSE_METHOD="${DEFENSE_METHOD:-prompting}" # intervention or prompting

TEMPERATURE="${TEMPERATURE:-1}"
MAX_WORKERS="${MAX_WORKERS:-32}"

VERBOSE="${VERBOSE:-true}"

for SCENARIO in $SCENARIOS; do

    case "$SCENARIO" in
        blackmail|leaking|murder|all) ;;
        *) echo "ERROR: Unknown scenario: $SCENARIO"; exit 1 ;;
    esac
    case "$DEFENSE_METHOD" in
        intervention|prompting) ;;
        *) echo "ERROR: Unknown defense method: $DEFENSE_METHOD"; exit 1 ;;
    esac

    SCENE_NAME="agentic_misalignment"
    PROMPTS_DIR="${PROMPTS_DIR:-$PROJECT_ROOT/data/agentic_misalignment}"
    [ ! -d "$PROMPTS_DIR" ] && echo "ERROR: $PROMPTS_DIR not found" && exit 1
    RUN_TAG="${TOOL_MODE}"

    DEFAULT_OUTPUT_DIR="$PROJECT_ROOT/results/${SCENE_NAME}_decode_${RUN_TAG}/${MODEL}/defense/${DEFENSE_METHOD}"
    OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
    mkdir -p "$OUTPUT_DIR"

    echo "=================================================================================================="
    echo " Scenario: $SCENARIO  |  Model: $MODEL  |  Tool Mode: $TOOL_MODE  |  Defense: $DEFENSE_METHOD"
    echo " Output Dir:     $OUTPUT_DIR"
    echo "=================================================================================================="

    CMD=(
        python defense/collect_action_decode_defense.py
        --prompts-dir "$PROMPTS_DIR"
        --output-dir "$OUTPUT_DIR"
        --scenario "$SCENARIO"
        --tool-mode "$TOOL_MODE"
        --model "$MODEL"
        --temperature "$TEMPERATURE"
        --num-samples "$NUM_SAMPLES"
        --max-workers "$MAX_WORKERS"
        --defense-method "$DEFENSE_METHOD"
        --resume
    )
    [ "$VERBOSE" = true ]   && CMD+=(--verbose)

    set +e
    "${CMD[@]}"
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Scenario '$SCENARIO' defense stage failed (exit code: $EXIT_CODE)"
        exit $EXIT_CODE
    fi

    echo ""
    echo "Done: $SCENARIO  ->  $OUTPUT_DIR"
    echo ""
done
