#!/bin/bash
# Fast S3 Upload Script
# This script optimizes AWS CLI for maximum upload speed

set -e

# Configuration
LOCAL_PATH="${1:-./test_data/sample}"
S3_BUCKET="${2:-rfdetr-sagemaker-training}"
S3_PREFIX="${3:-rfdetr/training-data}"

echo "========================================="
echo "Fast S3 Upload"
echo "========================================="
echo "Local Path: ${LOCAL_PATH}"
echo "S3 Path: s3://${S3_BUCKET}/${S3_PREFIX}"
echo "========================================="

# Configure AWS CLI for maximum speed
# These settings enable parallel uploads
export AWS_CLI_FILE_ENCODING=UTF-8

# Apply optimized S3 settings
aws configure set default.s3.max_concurrent_requests 100
aws configure set default.s3.max_queue_size 10000
aws configure set default.s3.multipart_threshold 64MB
aws configure set default.s3.multipart_chunksize 16MB
aws configure set default.s3.use_accelerate_endpoint false

echo "AWS CLI Configuration optimized for speed:"
echo "========================================="

# Method 1: AWS CLI sync (fastest for many files)
echo "Uploading with optimized AWS CLI sync..."
time aws s3 sync "${LOCAL_PATH}" "s3://${S3_BUCKET}/${S3_PREFIX}" \
    --storage-class STANDARD \
    --only-show-errors

echo "========================================="
echo "Upload completed!"
echo "Verify with: aws s3 ls s3://${S3_BUCKET}/${S3_PREFIX}/ --recursive --human-readable --summarize"

# Cleanup
rm -f ${AWS_CONFIG_FILE}
