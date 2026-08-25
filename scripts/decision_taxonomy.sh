#!/bin/bash
# Run hierarchical decision-process classification.
#
# Level 3 follows TOOL_MODE: intent declaration for signal/all, and behavior
# execution for neutral.
set -euo pipefail

BEHAVIOR_MODEL="${BEHAVIOR_MODEL:-qwen3-8b}" # qwen3-8b, gemma4-31b-it, qwen3-32b, Qwen3-235B-A22B-Thinking-2507, qwen3.5-27b
TOOL_MODE="${TOOL_MODE:-signal}" # neutral, signal, or all
SCENARIOS="${SCENARIOS:-blackmail leaking murder}" # Scenarios to process

case "$TOOL_MODE" in
    signal|all) VARIANT="w_button" ;;
    neutral)    VARIANT="wo_button" ;;
    *) echo "ERROR: Unknown TOOL_MODE: $TOOL_MODE (expected neutral, signal, or all)"; exit 1 ;;
esac

RAW_DATA_ROOT="${RAW_DATA_ROOT:-results}"
BASE_DATA_DIR="${BASE_DATA_DIR:-${RAW_DATA_ROOT}/agentic_misalignment_decode_${TOOL_MODE}}"
INPUT_DIR="${INPUT_DIR:-${BASE_DATA_DIR}/${BEHAVIOR_MODEL}/stage1}"

CLASSIFIER_MODEL="${CLASSIFIER_MODEL:-gpt-4.1}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DATA_DIR}/${BEHAVIOR_MODEL}/decision_process_taxonomy_final_${CLASSIFIER_MODEL}}"

MAX_SAMPLES="${MAX_SAMPLES:-3}"
MAX_WORKERS="${MAX_WORKERS:-32}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

echo "============================================================"
echo "Hierarchical Decision Process Taxonomy Classification"
echo "============================================================"
echo "Model:            $BEHAVIOR_MODEL"
echo "Tool Mode:        $TOOL_MODE"
echo "Taxonomy Variant: $VARIANT"
echo "Scenarios:        $SCENARIOS"
echo "Raw Data Root:    $RAW_DATA_ROOT"
echo "Input Dir:        $INPUT_DIR"
echo "Output Dir:       $OUTPUT_DIR"
echo "Classifier Model: $CLASSIFIER_MODEL"
echo "Max Workers:      $MAX_WORKERS"
if [ -z "${MAX_SAMPLES}" ] || [ "${MAX_SAMPLES}" = "0" ]; then
    echo "Max Samples:      all (per item_id)"
else
    echo "Max Samples:      ${MAX_SAMPLES} (per item_id)"
fi
echo "============================================================"
echo ""

cmd_args=(
    "--input-dir" "$INPUT_DIR"
    "--output-dir" "$OUTPUT_DIR"
    "--model" "$CLASSIFIER_MODEL"
    "--scenarios" "$SCENARIOS"
    "--max-workers" "$MAX_WORKERS"
    "--variant" "$VARIANT"
    "--resume"
)

if [ -n "${MAX_SAMPLES}" ] && [ "${MAX_SAMPLES}" != "0" ]; then
    cmd_args+=("--max-samples" "$MAX_SAMPLES")
fi

python "$PROJECT_DIR/classifiers/decision_taxonomy.py" "${cmd_args[@]}"

echo ""
echo "============================================================"
echo "Done! Results saved to: $OUTPUT_DIR"
echo "============================================================"
