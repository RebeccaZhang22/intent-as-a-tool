#!/bin/bash
# Run LLM-as-judge classification.
set -euo pipefail

INFERENCE_MODE="${INFERENCE_MODE:-api}"
CLASSIFIER_MODEL="${CLASSIFIER_MODEL:-gpt-4.1}"
CONCURRENCY="${CONCURRENCY:-50}"

BEHAVIOR_MODEL="${BEHAVIOR_MODEL:-qwen3-8b}" # qwen3-8b, qwen3.5-27b, qwen3-32b, gemma4-31b-it, Qwen3-235B-A22B-Thinking-2507
TOOL_MODE="${TOOL_MODE:-signal}" # neutral, signal, or all
BASE_DATA_DIR="${BASE_DATA_DIR:-results/agentic_misalignment_decode_${TOOL_MODE}/${BEHAVIOR_MODEL}}"

SCENARIOS="${SCENARIOS:-blackmail leaking murder}" # blackmail leaking murder
MAX_SAMPLES_PER_CONDITION="${MAX_SAMPLES_PER_CONDITION:-3}"

MODE="${MODE:-stage1}" # stage1, stage2_token, stage2_sentence, intervention, prompting

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

if [ "$MODE" = "intervention" ] || [ "$MODE" = "prompting" ]; then
    INPUT_DIR="${INPUT_DIR:-${BASE_DATA_DIR}/defense/${MODE}}"
    OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DATA_DIR}/defense/${MODE}_final_judge}"
else
    INPUT_DIR="${INPUT_DIR:-${BASE_DATA_DIR}/${MODE}}"
    OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DATA_DIR}/${MODE}_final_judge}"
fi


if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory not found: $INPUT_DIR"
    exit 1
fi

echo "============================================================"
echo "LLM-as-Judge Classification ($TOOL_MODE)"
echo "============================================================"
echo "Input:          $INPUT_DIR"
echo "Output:         $OUTPUT_DIR"
echo "Mode:           $MODE"
echo "Model:          $CLASSIFIER_MODEL"
echo "Concurrency:    $CONCURRENCY"
echo "Scenarios:      ${SCENARIOS}"
echo "Samples per condition: ${MAX_SAMPLES_PER_CONDITION:-all}"
echo "============================================================"
echo ""

mkdir -p "$OUTPUT_DIR"

EXTRA_ARGS=()
[ -n "$SCENARIOS" ]      && EXTRA_ARGS+=(--scenarios "$SCENARIOS")
[ -n "$MAX_SAMPLES_PER_CONDITION" ] && EXTRA_ARGS+=(--max-samples-per-condition "$MAX_SAMPLES_PER_CONDITION")
EXTRA_ARGS+=(--tool-mode "$TOOL_MODE")

python "$PROJECT_DIR/classifiers/behavioral_judge.py" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --mode "$MODE" \
    --inference-mode "$INFERENCE_MODE" \
    --classifier-model "$CLASSIFIER_MODEL" \
    --concurrency "$CONCURRENCY" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "============================================================"
echo "Done! Results: $OUTPUT_DIR"
echo "============================================================"
