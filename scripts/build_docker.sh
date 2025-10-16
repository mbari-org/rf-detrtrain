#!/bin/bash
# Build Docker image for SageMaker training
# This script handles authentication with AWS DLC ECR

set -e

IMAGE_NAME="${1:-rfdetr-sagemaker-training}"
IMAGE_TAG="${2:-latest}"
REGION="${AWS_REGION:-us-west-2}"

echo "========================================="
echo "Building Docker Image for SageMaker"
echo "========================================="
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Region: ${REGION}"
echo "========================================="

# Authenticate with AWS Deep Learning Containers ECR
echo "Authenticating with AWS DLC ECR..."
aws ecr get-login-password --region ${REGION} | \
    docker login --username AWS --password-stdin \
    763104351884.dkr.ecr.${REGION}.amazonaws.com

if [ $? -eq 0 ]; then
    echo "✓ Successfully authenticated with AWS DLC ECR"
else
    echo "⚠ Warning: Failed to authenticate with AWS DLC ECR"
    echo "  Continuing anyway - base image might be cached"
fi

# Build the image (from aws/ root directory)
echo ""
echo "Building Docker image..."
cd "$(dirname "$0")/.." || exit 1
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f Dockerfile .

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ Docker image built successfully!"
    echo "========================================="
    echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    echo "To push to ECR:"
    echo "  Run the submit_sagemaker_job.py script"
    echo "  Or manually push with:"
    echo "    ACCOUNT_ID=\$(aws sts get-caller-identity --query Account --output text)"
    echo "    aws ecr create-repository --repository-name ${IMAGE_NAME} --region ${REGION} || true"
    echo "    docker tag ${IMAGE_NAME}:${IMAGE_TAG} \${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}:${IMAGE_TAG}"
    echo "    aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin \${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
    echo "    docker push \${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}:${IMAGE_TAG}"
else
    echo ""
    echo "========================================="
    echo "✗ Docker build failed!"
    echo "========================================="
    exit 1
fi
