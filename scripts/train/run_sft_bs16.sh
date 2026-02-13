#!/bin/bash

# ========== Activate Environment ==========
source activate /path/to/your/conda/env
export PATH=/path/to/your/conda/env/bin:$PATH

export TOKENIZERS_PARALLELISM=false
export HADOOP_HOME=/opt/meituan/hadoop
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_IF_BASE_PORT=65100

main_output=$(python3 /path/to/your/project/third_party/llama_factory_sdar/print_dist.py)
echo $main_output
IFS=' ' read -r rank main port num_machines num_processes <<< "$main_output"
echo "MASTER_ADDR: $main"
echo "MASTER_PORT: $port"
echo "NNODES: $num_machines"
echo "NODE_RANK: $rank"
echo "NUM_PROCESSES: $num_processes"

ln -s /path/to/your/datasets/tmp 
export TMPDIR="/tmp/processing"
export HF_DATASETS_CACHE="/tmp/hf_cache"
cache_dir=/path/to/your/datasets/tmp/hf_cache
# Create directories
mkdir -p $HF_DATASETS_CACHE
mkdir -p $TMPDIR

# ========== Training Parameters ==========
# Custom hyperparameters
cutoff_len=20480
block_length=16
per_device_train_batch_size=1
gradient_accumulation_steps=1
learning_rate=3.0e-5
num_train_epochs=6.0
save_steps=1024
batch_size=$(( num_processes * per_device_train_batch_size * gradient_accumulation_steps ))
PARAM=seqlen${cutoff_len}-bs${block_length}-z3-${batch_size}bsz
MODE=longcat_sft_final # debug or samplexx or full

# DeepSpeed config
# deepspeed=/path/to/configs/train_config/deepspeed/ds_z2_config.json
deepspeed=/path/to/configs/train_config/deepspeed/ds_z3_config.json

# WandB parameters
export WANDB_PROJECT="dllm-sft-tdar_8b-longcat_cot"
export WANDB_ENTITY="your_wandb_username"
export WANDB_NAME="${WANDB_PROJECT}-${PARAM}-${MODE}"
export WANDB_MODE="offline"
export WANDB_API_KEY="your_wandb_api_key"

MODEL_DIR=/path/to/your/checkpoints
dataset=longcat_sft
tokenized_path=/path/to/your/datasets/tokenized_datasets/${dataset}_${cutoff_len}_${block_length}
export PYTHONPATH="${MODEL_DIR}:$PYTHONPATH"

OUTPUT_DIR=/path/to/your/output/${WANDB_PROJECT}/${WANDB_NAME}
LOG_FILE="/path/to/your/logs/${WANDB_NAME}-node${NODE_RANK}.log"
export WANDB_DIR=${OUTPUT_DIR}/wandb

mkdir -p $WANDB_DIR $(dirname $LOG_FILE) $OUTPUT_DIR

echo "📁 Output: $OUTPUT_DIR"
echo "📄 Log: $LOG_FILE"
echo ""

# Redirect to log file
exec >> "$LOG_FILE" 2>&1

cd /path/to/llama_factory

FORCE_TORCHRUN=1 NNODES=$num_machines NODE_RANK=$rank MASTER_ADDR=$main MASTER_PORT=$port llamafactory-cli train \
    /path/to/configs/train_config/tdar_8b_sft.yaml \
    model_name_or_path=${MODEL_DIR} \
    cutoff_len=${cutoff_len} \
    block_length=${block_length} \
    deepspeed=${deepspeed} \
    output_dir=${OUTPUT_DIR} \
    run_name=${WANDB_NAME} \
    per_device_train_batch_size=$per_device_train_batch_size \
    gradient_accumulation_steps=$gradient_accumulation_steps \
    learning_rate=$learning_rate \
    dataset=$dataset \
    tokenized_path=$tokenized_path \
    cache_dir=$cache_dir \
    num_train_epochs=${num_train_epochs} \
    save_steps=${save_steps}

EXIT_CODE=$?

echo ""
echo "=========================================="
[ $EXIT_CODE -eq 0 ] && echo "✅ Training completed" || echo "❌ Training failed (exit code: $EXIT_CODE)"
echo "  Time: $(date)"
echo "  Node: $HOSTNAME (rank $NODE_RANK)"
echo "=========================================="

exit $EXIT_CODE