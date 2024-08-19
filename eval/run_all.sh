#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to handle errors
handle_error() {
    echo "Error occurred in command: $BASH_COMMAND"
    echo "Error occurred on line $1"
    exit 1
}

# Set up error handling
trap 'handle_error $LINENO' ERR

# Check if all required arguments are provided
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <model_name> <subdir_name> <CODE_GEN> <gpu_id> <adapter_dir>"
    exit 1
fi

# Assign input arguments to variables
MODEL_NAME=$1
SUBDIR_NAME=$2
CODE_GEN=$3
GPU_ID=$4
ADAPTER_DIR=$5

# Extract the parent directory and checkpoint name from MODEL_NAME
PARENT_DIR=$(dirname "$MODEL_NAME")
CHECKPOINT_NAME=$(basename "$MODEL_NAME")

# Set the model path
MODEL_PATH="models/${MODEL_NAME}"

# Create directory for YAML file
YAML_DIR="merge_lora/${PARENT_DIR}"
mkdir -p "$YAML_DIR"
JSON_DIR="jsons/${PARENT_DIR}"
mkdir -p "$JSON_DIR"

# Generate the YAML file
YAML_FILE="${YAML_DIR}/${CHECKPOINT_NAME}.yaml"
cat << EOF > "$YAML_FILE"
### Note: DO NOT use quantized model or quantization_bit when merging lora adapters
### model
model_name_or_path: microsoft/Phi-3-mini-4k-instruct
adapter_name_or_path: /shared/model_outputs/${ADAPTER_DIR}/${MODEL_NAME}
template: phi
finetuning_type: lora
### export
export_dir: ${MODEL_PATH}
export_size: 2
export_device: cpu
export_legacy_format: false
EOF

echo "Generated YAML file: $YAML_FILE"

# Run the merge command
CUDA_VISIBLE_DEVICES=${GPU_ID} llamafactory-cli export "$YAML_FILE"

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate vllm_env

# Run the existing commands
CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 rule_inference_json.py --name "${MODEL_NAME}" --model_path "${MODEL_PATH}" --temp 0 --tensor_parallel 1
CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 clean_rules_json.py --input_file "${JSON_DIR}/${CHECKPOINT_NAME}_rules.json"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 gen_eval_json.py --json_file "${JSON_DIR}/cleaned-${CHECKPOINT_NAME}_rules.json" --model "${CODE_GEN}" --gpu 2 --subdir "${SUBDIR_NAME}" --key cleaned-output

# Deactivate the conda environment
conda activate llamafactory

rm -rf "${MODEL_PATH}"
echo "Merged Model Removed From: ${MODEL_PATH}"

echo "All commands executed successfully."