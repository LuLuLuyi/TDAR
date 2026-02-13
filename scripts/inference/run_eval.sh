#!/bin/bash
# ============================================
# Distributed Inference Evaluation Script
# Most parameters managed through YAML config
# ============================================

set -e  # Exit on error

# ============================================
# 🎨 Color Output Definitions
# ============================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() { echo -e "${CYAN}===================================${NC}\n${CYAN}$1${NC}\n${CYAN}===================================${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

# ============================================
# 📝 Core Configuration - Only modify these
# ============================================

# --- Environment Config ---
CONDA_ENV_PATH="/path/to/your/conda/env"
PROJECT_ROOT="/path/to/project/root"

# --- Path Config ---
MODEL_PATH="${PROJECT_ROOT}/models/your-model-name"
CONFIG_FILE="${PROJECT_ROOT}/configs/eval_config.yaml"  # All inference params here

# --- Output Path Config ---
LOG_BASE_DIR="${PROJECT_ROOT}/logs"
RESULT_BASE_DIR="${PROJECT_ROOT}/results"

# --- GPU Config ---
NUM_GPUS_TO_USE=8  # Number of GPUs to use

# --- Distributed Config (for multi-node) ---
NUM_NODE=1
NODE_INDEX=0

# ============================================
# 📋 Dataset Configuration
# Format: "dataset_name|data_type|samples_per_question|run_id"
# ============================================
declare -a DATASET_CONFIGS=(
    "AIME2024|math|8|pass8"
    "AIME2025|math|8|pass8"
    "MATH500|math|1|single"
    "GPQA|option|1|single"
    "LiveCodeBench|code|1|single"
    # Add more datasets...
)

# ============================================
# 🚀 Script Execution Start
# ============================================

print_header "Distributed Inference Evaluation Script Started"

# --- Activate Conda Environment ---
if [ -d "$CONDA_ENV_PATH" ]; then
    source activate "$CONDA_ENV_PATH"
    export PATH="${CONDA_ENV_PATH}/bin:$PATH"
    print_success "Conda environment activated"
else
    print_error "Conda environment path not found: $CONDA_ENV_PATH"
    exit 1
fi

# --- Generate Timestamp ---
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
print_info "Run timestamp: ${TIMESTAMP}"

# --- Verify Paths ---
if [ ! -d "$MODEL_PATH" ]; then
    print_error "Model path not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

# Remove existing chat_template.jinja if present
TEMPLATE_FILE="${MODEL_PATH}/chat_template.jinja"
[ -f "$TEMPLATE_FILE" ] && rm "$TEMPLATE_FILE" && print_warning "Removed old template file"

# --- Configure GPUs ---
AVAILABLE_GPUS=($(seq 0 $((NUM_GPUS_TO_USE - 1))))
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${AVAILABLE_GPUS[*]}")
print_info "GPU config: Using ${NUM_GPUS_TO_USE} GPUs [${CUDA_VISIBLE_DEVICES}]"

# --- Extract Model Name ---
MODEL_NAME="$(basename "$(dirname "$MODEL_PATH")")-$(basename "$MODEL_PATH")"

# --- Create Output Directories (using config name as identifier) ---
CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
LOG_DIR="${LOG_BASE_DIR}/${MODEL_NAME}/${CONFIG_NAME}/${TIMESTAMP}"
RESULT_DIR="${RESULT_BASE_DIR}/${MODEL_NAME}/${CONFIG_NAME}/${TIMESTAMP}"

mkdir -p "$LOG_DIR" "$RESULT_DIR"
print_success "Output directories created"
print_info "Log directory: $LOG_DIR"
print_info "Result directory: $RESULT_DIR"

# --- Print Configuration Summary ---
print_header "Configuration Summary"
echo "📂 Model: $MODEL_NAME"
echo "⚙️  Config File: $(basename $CONFIG_FILE)"
echo "🎮 GPU Count: ${NUM_GPUS_TO_USE}"
echo "📋 Test Tasks: ${#DATASET_CONFIGS[@]}"
echo ""

# ============================================
# 🛠️ Single Dataset Evaluation Function
# ============================================
run_single_dataset() {
    local GPU_ID=$1
    local EVAL_DATASET=$2
    local DATA_TYPE=$3
    local NUM_RESPONSE=$4
    local RUN_ID=$5
    local INDEX=$6
    
    local EXP_NAME="${EVAL_DATASET}_${RUN_ID}_${TIMESTAMP}"
    local LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"
    
    print_info "[$INDEX] GPU-${GPU_ID} Starting: $EVAL_DATASET [$RUN_ID]"
    
    (
        export CUDA_VISIBLE_DEVICES=$GPU_ID
        
        echo "==================================="
        echo "📝 Experiment: $EVAL_DATASET [$RUN_ID]"
        echo "==================================="
        echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "GPU: $GPU_ID"
        echo ""
        
        # 🎯 Core Command - Most params from config
        python "${PROJECT_ROOT}/src/eval.py" \
            config="${CONFIG_FILE}" \
            model="${MODEL_PATH}" \
            experiment.name="${EXP_NAME}" \
            experiment.num_node="${NUM_NODE}" \
            experiment.node_index="${NODE_INDEX}" \
            paths.result_base="${RESULT_DIR}" \
            dataset.eval_dataset="${EVAL_DATASET}" \
            dataset.data_type="${DATA_TYPE}" \
            rollout.num_response_per_task="${NUM_RESPONSE}"
        
        local EXIT_CODE=$?
        
        echo ""
        echo "==================================="
        [ $EXIT_CODE -eq 0 ] && echo "✅ Success" || echo "❌ Failed (code: $EXIT_CODE)"
        echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "==================================="
        
        exit $EXIT_CODE
        
    ) > "$LOG_FILE" 2>&1
    
    local EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        print_success "[$INDEX] GPU-${GPU_ID} $EVAL_DATASET [$RUN_ID] Completed"
    else
        print_error "[$INDEX] GPU-${GPU_ID} $EVAL_DATASET [$RUN_ID] Failed"
    fi
    
    return $EXIT_CODE
}

# ============================================
# 🚀 Parallel Evaluation Execution
# ============================================
print_header "Starting Parallel Evaluation"

TOTAL=${#DATASET_CONFIGS[@]}
NUM_GPUS=${#AVAILABLE_GPUS[@]}
BATCH_START=0
BATCH_NUM=1

while [ $BATCH_START -lt $TOTAL ]; do
    BATCH_END=$((BATCH_START + NUM_GPUS))
    [ $BATCH_END -gt $TOTAL ] && BATCH_END=$TOTAL
    BATCH_SIZE=$((BATCH_END - BATCH_START))
    
    echo ""
    print_info "📦 Batch ${BATCH_NUM}: Launching ${BATCH_SIZE} tasks"
    
    # Launch Tasks
    PIDS=()
    for i in $(seq 0 $((BATCH_SIZE - 1))); do
        IDX=$((BATCH_START + i))
        GPU_ID=${AVAILABLE_GPUS[$i]}
        
        IFS='|' read -r dataset datatype num_resp run_id <<< "${DATASET_CONFIGS[$IDX]}"
        
        run_single_dataset "$GPU_ID" "$dataset" "$datatype" "$num_resp" "$run_id" "$((IDX + 1))" &
        PIDS+=($!)
    done
    
    # Wait for Completion
    print_warning "⏳ Waiting for batch completion..."
    FAILED=0
    for pid in "${PIDS[@]}"; do
        wait $pid || ((FAILED++))
    done
    
    print_success "Batch ${BATCH_NUM} completed: Success $((BATCH_SIZE - FAILED))/${BATCH_SIZE}"
    
    BATCH_START=$BATCH_END
    ((BATCH_NUM++))
    
    # Inter-batch Interval
    [ $BATCH_START -lt $TOTAL ] && sleep 10
done

# ============================================
# 📊 Completion Report
# ============================================
print_header "Evaluation Complete"
echo "🕐 Timestamp: ${TIMESTAMP}"
echo "📁 Logs: $LOG_DIR"
echo "📁 Results: $RESULT_DIR"
print_success "All tasks completed!"

exit 0