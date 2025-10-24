#!/bin/bash

# This script runs the SimCSE stage of the LLM2Rec model.

# --- Configuration ---
# Get the directory of this script
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Set the project root directory
PROJECT_ROOT="$SCRIPT_DIR"

# Path to the SimCSE configuration file
CONFIG_FILE="$PROJECT_ROOT/llm2rec/train_simcse_config.json"

# --- Functions ---
# Function to extract a value from the JSON config file
get_config_value() {
    python3 -c "import json; print(json.load(open('$1'))['$2'])"
}

# --- Main Execution ---
echo "Starting SimCSE stage..."

# Extract parameters from the config file
# MODEL_NAME_OR_PATH=$(get_config_value "$CONFIG_FILE" "model_name_or_path")
BASE_MODEL_PATH="Qwen/Qwen2-0.5B"  # Set the base model path
PEFT_MODEL_PATH=$(get_config_value "$CONFIG_FILE" "model_name_or_path")
OUTPUT_DIR=$(get_config_value "$CONFIG_FILE" "output_dir")
DATASET_FILE_PATH=$(get_config_value "$CONFIG_FILE" "dataset_file_path")
LEARNING_RATE=$(get_config_value "$CONFIG_FILE" "learning_rate")
NUM_TRAIN_EPOCHS=$(get_config_value "$CONFIG_FILE" "num_train_epochs")
PER_DEVICE_TRAIN_BATCH_SIZE=$(get_config_value "$CONFIG_FILE" "per_device_train_batch_size")
MAX_SEQ_LENGTH=$(get_config_value "$CONFIG_FILE" "max_seq_length")
SEED=$(get_config_value "$CONFIG_FILE" "seed")
SIMCSE_DROPOUT=$(get_config_value "$CONFIG_FILE" "simcse_dropout")
STOP_AFTER_N_STEPS=$(get_config_value "$CONFIG_FILE" "stop_after_n_steps")
LOSS_SCALE=$(get_config_value "$CONFIG_FILE" "loss_scale")

# Check if the input model checkpoint exists
# if [ ! -d "$MODEL_NAME_OR_PATH" ]; then
#     echo "Error: Input model checkpoint not found at '$MODEL_NAME_OR_PATH'"
#     exit 1
# fi
if [ ! -d "$PEFT_MODEL_PATH" ]; then
    echo "Error: Input PEFT model checkpoint not found at '$PEFT_MODEL_PATH'"
    exit 1
fi

# echo "Input model: $MODEL_NAME_OR_PATH"
echo "Base model: $BASE_MODEL_PATH"
echo "PEFT model: $PEFT_MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"

# Set a different port for this stage to avoid conflicts
export MASTER_PORT=29503

# Run the SimCSE training script
torchrun --nproc_per_node=1 --master_port=$MASTER_PORT "$PROJECT_ROOT/llm2rec/run_unsupervised_SimCSE.py" \
    --model_name_or_path "$BASE_MODEL_PATH" \
    --peft_model_name_or_path "$PEFT_MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_name "ItemTitles" \
    --dataset_file_path "$DATASET_FILE_PATH" \
    --remove_unused_columns false \
    --learning_rate "$LEARNING_RATE" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --warmup_steps 300 \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps 1 \
    --do_train \
    --disable_tqdm false \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --overwrite_output_dir \
    --logging_steps 1 \
    --max_grad_norm 1.0 \
    --simcse_dropout "$SIMCSE_DROPOUT" \
    --save_only_model false \
    --save_steps 100 \
    --stop_after_n_steps "$STOP_AFTER_N_STEPS" \
    --loss_scale "$LOSS_SCALE" \
    --gradient_checkpointing \
    --torch_dtype "bfloat16" \
    --attn_implementation "flash_attention_2" \
    --run_name "simcse_modified" \
    --seed "$SEED"

# Check the exit code of the training script
if [ $? -eq 0 ]; then
    echo "SimCSE stage finished successfully."
    echo "Model saved in $OUTPUT_DIR"
else
    echo "Error: SimCSE stage failed."
    exit 1
fi