#!/bin/bash
# Test Docker container locally before deploying to SageMaker
# This script simulates the SageMaker environment on your local machine

set -e

# Configuration
IMAGE_NAME="${1:-rfdetr-sagemaker-training}"
IMAGE_TAG="${2:-latest}"
LOCAL_DATA_PATH="${3:-/tmp/test_dataset}"
LOCAL_OUTPUT_PATH="${4:-/tmp/sagemaker-local/output}"
LOCAL_MODEL_PATH="${5:-/tmp/sagemaker-local/model}"

echo "========================================="
echo "Testing SageMaker Docker Container Locally"
echo "========================================="
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Data Path: ${LOCAL_DATA_PATH}"
echo "Output Path: ${LOCAL_OUTPUT_PATH}"
echo "Model Path: ${LOCAL_MODEL_PATH}"
echo "========================================="

# Validate data directory exists
if [ ! -d "${LOCAL_DATA_PATH}" ]; then
    echo "Error: Data directory does not exist: ${LOCAL_DATA_PATH}"
    echo "Please provide a valid data directory with train/ and valid/ subdirectories"
    exit 1
fi

# Check for required subdirectories
if [ ! -d "${LOCAL_DATA_PATH}/train" ]; then
    echo "Error: Missing train/ subdirectory in ${LOCAL_DATA_PATH}"
    exit 1
fi

if [ ! -d "${LOCAL_DATA_PATH}/valid" ]; then
    echo "Error: Missing valid/ subdirectory in ${LOCAL_DATA_PATH}"
    exit 1
fi

# Create local output directories to simulate SageMaker structure
echo "Creating local SageMaker directory structure..."
mkdir -p ${LOCAL_OUTPUT_PATH}
mkdir -p ${LOCAL_MODEL_PATH}
mkdir -p /tmp/sagemaker-local/input/config

# Create a mock hyperparameters.json file
cat > /tmp/sagemaker-local/input/config/hyperparameters.json << EOF
{
    "epochs": "2",
    "batch-size": "2",
    "grad-accum-steps": "1",
    "model-size": "medium"
}
EOF

echo "Created mock hyperparameters file"

# Build the image if it doesn't exist
if ! docker image inspect ${IMAGE_NAME}:${IMAGE_TAG} > /dev/null 2>&1; then
    echo "Image not found. Building..."
    "$(dirname "$0")/build_docker.sh" ${IMAGE_NAME} ${IMAGE_TAG}
fi

echo ""
echo "========================================="
echo "Running Docker container locally..."
echo "========================================="
echo ""
echo "This will:"
echo "  1. Mount your local data to /opt/ml/input/data/training"
echo "  2. Mount output directories to /opt/ml/output and /opt/ml/model"
echo "  3. Run training with minimal epochs for testing"
echo ""
echo "Press Ctrl+C to stop the container at any time"
echo ""
echo "========================================="
sleep 2


# Run the container with SageMaker-like directory structure
docker run --rm -it \
    --gpus all \
    -v ${LOCAL_DATA_PATH}:/opt/ml/input/data/training \
    -v ${LOCAL_OUTPUT_PATH}:/opt/ml/output \
    -v ${LOCAL_MODEL_PATH}:/opt/ml/model \
    -v /tmp/sagemaker-local/input/config:/opt/ml/input/config \
    -e SM_CHANNEL_TRAINING=/opt/ml/input/data/training \
    -e SM_MODEL_DIR=/opt/ml/model \
    -e SM_OUTPUT_DATA_DIR=/opt/ml/output \
    ${IMAGE_NAME}:${IMAGE_TAG} \
    python /opt/ml/code/train.py \
    --epochs 2 \
    --batch-size 2 \
    --grad-accum-steps 1 \
    --model-size medium


# Run the container with SageMaker-like directory structure interactively for troubleshooting
# docker run --rm -it --entrypoint bash \
#     --gpus all \
#     -v ${LOCAL_DATA_PATH}:/opt/ml/input/data/training \
#     -v ${LOCAL_OUTPUT_PATH}:/opt/ml/output \
#     -v ${LOCAL_MODEL_PATH}:/opt/ml/model \
#     -v /tmp/sagemaker-local/input/config:/opt/ml/input/config \
#     -e SM_CHANNEL_TRAINING=/opt/ml/input/data/training \
#     -e SM_MODEL_DIR=/opt/ml/model \
#     -e SM_OUTPUT_DATA_DIR=/opt/ml/output \
#     ${IMAGE_NAME}:${IMAGE_TAG}

echo ""
echo "========================================="
echo "Test completed!"
echo "========================================="
echo "Check outputs at:"
echo "  - Model: ${LOCAL_MODEL_PATH}"
echo "  - Logs: ${LOCAL_OUTPUT_PATH}"
echo "========================================="
