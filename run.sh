#!/bin/bash
# Main entry point for AWS SageMaker training scripts
# This script provides a convenient way to run various tasks

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

show_help() {
    cat << EOF
AWS SageMaker RF-DETR Training - Command Runner

Usage: ./run.sh [command] [args...]

Commands:
  build                  Build Docker image
  test                   Run local CPU tests
  test-gpu               Run local GPU tests
  deploy                 Deploy to SageMaker (full pipeline)
  upload [path]          Upload data to S3
  setup-role             Create SageMaker IAM role
  submit [options]       Submit training job only

Examples:
  ./run.sh build                              # Build Docker image
  ./run.sh test-gpu                           # Full GPU test
  ./run.sh deploy                             # Build, upload, and submit training
  ./run.sh upload /tmp/test_dataset           # Upload data to S3
  ./run.sh setup-role                         # Create IAM role

For more details, see:
  - README.md for complete documentation
  - QUICKSTART.md for quick start guide
EOF
}

case "${1:-}" in
    build)
        echo "Building Docker image..."
        "${SCRIPT_DIR}/scripts/build_docker.sh" "${@:2}"
        ;;

    test-gpu)
        echo "Running local GPU tests..."
        "${SCRIPT_DIR}/scripts/test_docker_local.sh" rfdetr-sagemaker-training latest /tmp/test_dataset
        ;;

    deploy)
        echo "Deploying to SageMaker..."
        "${SCRIPT_DIR}/scripts/example_usage.sh"
        ;;

    upload)
        DATA_PATH="${2:-/tmp/test_dataset}"
        echo "Uploading data from ${DATA_PATH}..."
        "${SCRIPT_DIR}/scripts/fast_s3_upload.sh" "${DATA_PATH}" rfdetr-sagemaker-training rfdetr/training-data
        ;;

    setup-role)
        echo "Setting up SageMaker IAM role..."
        "${SCRIPT_DIR}/scripts/setup_sagemaker_role.sh"
        ;;

    submit)
        echo "Submitting training job..."
        cd "${SCRIPT_DIR}"
        python src/submit_sagemaker_job.py "${@:2}"
        ;;

    help|-h|--help)
        show_help
        ;;

    "")
        echo "Error: No command specified"
        echo ""
        show_help
        exit 1
        ;;

    *)
        echo "Error: Unknown command '${1}'"
        echo ""
        show_help
        exit 1
        ;;
esac
