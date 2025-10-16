#!/bin/bash
# List available PyTorch training images from AWS Deep Learning Containers
# These images can be used as base images in the Dockerfile

set -e

REGION="${AWS_REGION:-us-west-2}"
DLC_ACCOUNT="763104351884"
REPOSITORY="pytorch-training"

echo "========================================="
echo "AWS Deep Learning Container - PyTorch Training Images"
echo "========================================="
echo "Region: ${REGION}"
echo "Account: ${DLC_ACCOUNT}"
echo "Repository: ${REPOSITORY}"
echo "========================================="
echo ""

# Authenticate with AWS DLC ECR
echo "Authenticating with AWS DLC ECR..."
aws ecr get-login-password --region ${REGION} | \
    docker login --username AWS --password-stdin \
    ${DLC_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com 2>/dev/null || true

echo ""
echo "Fetching available PyTorch training images..."
echo ""

# List all image tags
aws ecr list-images \
    --registry-id ${DLC_ACCOUNT} \
    --repository-name ${REPOSITORY} \
    --region ${REGION} \
    --output json | \
    jq -r '.imageIds[].imageTag' | \
    grep -E "^[0-9]" | \
    sort -V | \
    grep "sagemaker" || echo "Error: Could not fetch images. Make sure you have AWS CLI and jq installed."

echo ""
echo "========================================="
echo "Filtering options:"
echo "========================================="
echo ""

# Show PyTorch 2.x images
echo "PyTorch 2.x images (SageMaker-compatible):"
aws ecr list-images \
    --registry-id ${DLC_ACCOUNT} \
    --repository-name ${REPOSITORY} \
    --region ${REGION} \
    --output json 2>/dev/null | \
    jq -r '.imageIds[].imageTag' | \
    grep -E "^2\.[0-9]" | \
    grep "sagemaker" | \
    grep "gpu" | \
    sort -V | \
    tail -20 || echo "Could not filter images"

echo ""
echo "========================================="
echo "Usage in Dockerfile:"
echo "========================================="
echo "FROM ${DLC_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPOSITORY}:<TAG>"
echo ""
echo "Example:"
echo "FROM ${DLC_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPOSITORY}:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker"
echo ""
echo "========================================="
echo "Image tag format:"
echo "========================================="
echo "<pytorch_version>-<processor>-<python_version>-<cuda_version>-<os>-<variant>"
echo ""
echo "Examples:"
echo "  - 2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker"
echo "  - 2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker"
echo "  - 1.13.1-gpu-py39-cu117-ubuntu20.04-sagemaker"
echo ""
echo "Components:"
echo "  - pytorch_version: 2.1.0, 2.0.1, etc."
echo "  - processor: gpu (for training)"
echo "  - python_version: py310, py39, py38"
echo "  - cuda_version: cu121, cu118, cu117"
echo "  - os: ubuntu20.04, ubuntu18.04"
echo "  - variant: sagemaker, ec2"
echo ""
echo "Note: Always use 'sagemaker' variant for SageMaker training jobs"
echo "========================================="
