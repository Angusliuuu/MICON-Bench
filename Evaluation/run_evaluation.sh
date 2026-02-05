#!/bin/bash
# End-to-end evaluation script
# Supports evaluating a single model or all models in batch

set -e

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
GLOBAL_CONFIG="${GLOBAL_CONFIG:-${PROJECT_ROOT}/configs/global.yaml}"
JUDGE_CONFIG="${JUDGE_CONFIG:-${PROJECT_ROOT}/configs/judge_models.yaml}"

# Usage
show_usage() {
    echo "Usage: $0 [MODEL_NAME|--all] [TASK_TYPE] [JUDGE_NAME]"
    echo ""
    echo "Arguments:"
    echo "  MODEL_NAME    Name of the model to evaluate (e.g., UNO, GPT-5-Image)"
    echo "  --all         Evaluate all models in generations_root (default: outputs/generations)"
    echo "  TASK_TYPE     Task type to evaluate (default: all)"
    echo "  JUDGE_NAME    Judge model to use (default: local-qwen-judge)"
    echo "                Options: local-qwen-judge, local-internvl-judge"
    echo ""
    echo "Environment overrides:"
    echo "  PROJECT_ROOT   Project root (default: script directory)"
    echo "  GLOBAL_CONFIG  Path to configs/global.yaml"
    echo "  JUDGE_CONFIG   Path to configs/judge_models.yaml"
    echo "  GENERATIONS_DIR Override generations root"
    echo "  EVAL_BASE_DIR  Override eval results root"
    echo ""
    echo "Examples:"
    echo "  $0 UNO all                              # Evaluate single model with qwen"
    echo "  $0 UNO story_infer local-internvl-judge # Evaluate single model with internvl"
    echo "  $0 --all story_infer                    # Evaluate all models on story_infer"
    echo "  $0 --all all local-internvl-judge       # Evaluate all models with internvl"
    echo ""
    exit 1
}

get_yaml_value() {
    local key="$1"
    local default="$2"
    python - "$GLOBAL_CONFIG" "$key" "$default" <<'PY'
import sys
try:
    import yaml
except Exception:
    print(sys.argv[3])
    sys.exit(0)

config_path, key, default = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
except Exception:
    print(default)
    sys.exit(0)

cur = cfg
for part in key.split("."):
    if isinstance(cur, dict) and part in cur:
        cur = cur[part]
    else:
        cur = None
        break

print(cur if cur is not None else default)
PY
}

resolve_path() {
    local path_value="$1"
    if [[ -z "$path_value" ]]; then
        echo ""
    elif [[ "$path_value" = /* ]]; then
        echo "$path_value"
    else
        echo "${PROJECT_ROOT}/${path_value}"
    fi
}

GLOBAL_CONFIG="$(resolve_path "$GLOBAL_CONFIG")"
JUDGE_CONFIG="$(resolve_path "$JUDGE_CONFIG")"
EVAL_BASE_DIR="${EVAL_BASE_DIR:-$(get_yaml_value "project.paths.eval_results_root" "outputs/eval_res")}"
GENERATIONS_DIR="${GENERATIONS_DIR:-$(get_yaml_value "project.paths.generations_root" "outputs/generations")}"
EVAL_BASE_DIR="$(resolve_path "$EVAL_BASE_DIR")"
GENERATIONS_DIR="$(resolve_path "$GENERATIONS_DIR")"

# Evaluate a single model
evaluate_single_model() {
    local MODEL_NAME="$1"
    local TASK_TYPE="$2"
    local JUDGE_NAME="$3"
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    TIMESTAMP_DIR="${MODEL_NAME}_${TIMESTAMP}"
    
    MODEL_BASE_DIR="${EVAL_BASE_DIR}/${MODEL_NAME}"
    FULL_OUTPUT_DIR="${MODEL_BASE_DIR}/${TIMESTAMP_DIR}"
    
    mkdir -p "$FULL_OUTPUT_DIR"
    
    echo ""
    echo "============================================"
    echo "Multi-Image Benchmark Evaluation Pipeline"
    echo "============================================"
    echo "Model: $MODEL_NAME"
    echo "Task Type: $TASK_TYPE"
    echo "Judge Model: $JUDGE_NAME"
    echo "Project Root: $PROJECT_ROOT"
    echo "Output Directory: $FULL_OUTPUT_DIR"
    echo "============================================"
    echo ""
    
    echo "[Step 1/3] Building judge prompts..."
    cd "$PROJECT_ROOT"
    python scripts/build_prompts.py \
        --project-root "$PROJECT_ROOT" \
        --global-config "$GLOBAL_CONFIG" \
        --task-type "$TASK_TYPE"
    
    echo ""
    echo "✓ Prompts built successfully"
    echo ""
    
    echo "[Step 2/3] Running judge evaluation..."
    python scripts/run_judge.py \
        --project-root "$PROJECT_ROOT" \
        --model-name "$MODEL_NAME" \
        --task-type "$TASK_TYPE" \
        --timestamp-dir "$TIMESTAMP_DIR" \
        --output-base-dir "$EVAL_BASE_DIR" \
        --global-config "$GLOBAL_CONFIG" \
        --config "$JUDGE_CONFIG" \
        --judge-name "$JUDGE_NAME"
    
    echo ""
    echo "✓ Evaluation completed"
    echo ""
    
    echo "[Step 3/3] Aggregating scores..."
    python scripts/aggregate_scores.py \
        --project-root "$PROJECT_ROOT" \
        --model-name "$MODEL_NAME" \
        --timestamp-dir "$TIMESTAMP_DIR" \
        --output-base-dir "$EVAL_BASE_DIR" \
        --global-config "$GLOBAL_CONFIG"
    
    echo ""
    echo "✓ Score aggregation completed"
    echo ""
    
    echo "============================================"
    echo "Evaluation Complete for $MODEL_NAME!"
    echo "============================================"
    echo "Reports saved to:"
    echo "  - ${FULL_OUTPUT_DIR}/reports/${MODEL_NAME}_summary.csv"
    echo "  - ${FULL_OUTPUT_DIR}/reports/${MODEL_NAME}_report.txt"
    echo "============================================"
    echo ""
    
    if [ -f "${FULL_OUTPUT_DIR}/reports/${MODEL_NAME}_report.txt" ]; then
        echo "Brief Summary:"
        head -20 "${FULL_OUTPUT_DIR}/reports/${MODEL_NAME}_report.txt"
    fi
}

# Main
# Parse arguments
if [ $# -eq 0 ]; then
    show_usage
fi

MODE="$1"
TASK_TYPE="${2:-all}"
JUDGE_NAME="${3:-local-qwen-judge}"

# Batch evaluate all models
if [ "$MODE" == "--all" ]; then
    echo "=========================================="
    echo "Batch Evaluation Mode: Evaluating ALL models"
    echo "Task Type: $TASK_TYPE"
    echo "Judge Model: $JUDGE_NAME"
    echo "=========================================="
    echo ""
    
    # Get all model directories
    if [ ! -d "$GENERATIONS_DIR" ]; then
        echo "Error: Generations directory not found: $GENERATIONS_DIR"
        exit 1
    fi
    
    # Get model names (exclude hidden)
    MODEL_LIST=()
    for model_dir in "$GENERATIONS_DIR"/*; do
        if [ -d "$model_dir" ]; then
            model_name=$(basename "$model_dir")
            # Skip hidden directories
            if [[ ! "$model_name" =~ ^\. ]]; then
                MODEL_LIST+=("$model_name")
            fi
        fi
    done
    
    if [ ${#MODEL_LIST[@]} -eq 0 ]; then
        echo "Error: No models found in $GENERATIONS_DIR"
        exit 1
    fi
    
    echo "Found ${#MODEL_LIST[@]} models to evaluate:"
    for model in "${MODEL_LIST[@]}"; do
        echo "  - $model"
    done
    echo ""
    
    read -p "Continue with batch evaluation? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Batch evaluation cancelled."
        exit 0
    fi
    
    # Record start time
    BATCH_START_TIME=$(date +%s)
    TOTAL_MODELS=${#MODEL_LIST[@]}
    CURRENT_MODEL=0
    SUCCESSFUL_MODELS=0
    FAILED_MODELS=()
    
    # Evaluate models one by one
    for model_name in "${MODEL_LIST[@]}"; do
        CURRENT_MODEL=$((CURRENT_MODEL + 1))
        
        echo ""
        echo "######################################################"
        echo "# Processing Model [$CURRENT_MODEL/$TOTAL_MODELS]: $model_name"
        echo "######################################################"
        echo ""
        
        # Evaluate a single model and capture errors
        if evaluate_single_model "$model_name" "$TASK_TYPE" "$JUDGE_NAME"; then
            SUCCESSFUL_MODELS=$((SUCCESSFUL_MODELS + 1))
            echo "✓ Model $model_name completed successfully"
        else
            FAILED_MODELS+=("$model_name")
            echo "✗ Model $model_name failed"
        fi
        
        # Show progress
        echo ""
        echo "Progress: $CURRENT_MODEL/$TOTAL_MODELS models processed"
        echo "Successful: $SUCCESSFUL_MODELS, Failed: ${#FAILED_MODELS[@]}"
        echo ""
        
        # Sleep if not the last model
        if [ $CURRENT_MODEL -lt $TOTAL_MODELS ]; then
            echo "Waiting 3 seconds before next model..."
            sleep 3
        fi
    done
    
    # Compute total duration
    BATCH_END_TIME=$(date +%s)
    BATCH_DURATION=$((BATCH_END_TIME - BATCH_START_TIME))
    HOURS=$((BATCH_DURATION / 3600))
    MINUTES=$(((BATCH_DURATION % 3600) / 60))
    SECONDS=$((BATCH_DURATION % 60))
    
    # Print final summary
    echo ""
    echo "=========================================="
    echo "Batch Evaluation Complete!"
    echo "=========================================="
    echo "Total Models: $TOTAL_MODELS"
    echo "Successful: $SUCCESSFUL_MODELS"
    echo "Failed: ${#FAILED_MODELS[@]}"
    echo "Total Time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo ""
    
    if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
        echo "Failed Models:"
        for failed_model in "${FAILED_MODELS[@]}"; do
            echo "  - $failed_model"
        done
        echo ""
    fi
    
    echo "All results saved in: ${EVAL_BASE_DIR}/"
    echo "=========================================="
    
elif [ "$MODE" == "-h" ] || [ "$MODE" == "--help" ]; then
    show_usage
else
    # Single model evaluation
    MODEL_NAME="$MODE"
    
    # Check if model directory exists
    if [ ! -d "${GENERATIONS_DIR}/${MODEL_NAME}" ]; then
        echo "Warning: Model directory not found: ${GENERATIONS_DIR}/${MODEL_NAME}"
        echo "Available models:"
        ls -1 "$GENERATIONS_DIR" 2>/dev/null | grep -v "^\." || echo "  (none found)"
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    evaluate_single_model "$MODEL_NAME" "$TASK_TYPE" "$JUDGE_NAME"
fi
