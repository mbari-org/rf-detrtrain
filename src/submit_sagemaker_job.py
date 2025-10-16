#!/usr/bin/env python
"""
Submit RF-DETR training job to AWS SageMaker.

This script handles:
1. Building and pushing Docker image to ECR
2. Uploading training data to S3
3. Configuring and launching SageMaker training job
"""

import sys
import argparse
import logging
import boto3
import sagemaker
from sagemaker.estimator import Estimator
from pathlib import Path
import subprocess
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_account_id():
    """Get AWS account ID."""
    sts = boto3.client("sts")
    return sts.get_caller_identity()["Account"]


def build_and_push_image(image_name, tag="latest", dockerfile_dir="."):
    """Build Docker image and push to ECR."""
    account_id = get_account_id()
    region = boto3.Session().region_name
    ecr_repository = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{image_name}"

    logger.info(f"Building Docker image: {image_name}:{tag}")

    # Login to AWS DLC ECR (for base image)
    logger.info("Authenticating with AWS Deep Learning Containers ECR...")
    dlc_account = "763104351884"  # AWS DLC account
    dlc_login_cmd = f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {dlc_account}.dkr.ecr.{region}.amazonaws.com"
    try:
        subprocess.run(dlc_login_cmd, shell=True, check=True, capture_output=True)
        logger.info("Successfully authenticated with AWS DLC ECR")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to login to AWS DLC ECR: {e.stderr}")
        logger.warning("Continuing anyway - base image might already be cached")

    # Build Docker image (from aws/ directory, not src/)
    # Note: dockerfile_dir should be the aws/ root directory
    build_cmd = ["docker", "build", "-t", f"{image_name}:{tag}", "-f", "Dockerfile", "."]

    try:
        subprocess.run(build_cmd, check=True)
        logger.info("Docker build successful")
    except subprocess.CalledProcessError as e:
        logger.error(f"Docker build failed: {e}")
        return None

    # Create ECR repository if it doesn't exist
    ecr_client = boto3.client("ecr", region_name=region)
    try:
        ecr_client.create_repository(repositoryName=image_name)
        logger.info(f"Created ECR repository: {image_name}")
    except ecr_client.exceptions.RepositoryAlreadyExistsException:
        logger.info(f"ECR repository already exists: {image_name}")

    # Login to ECR
    login_cmd = (
        f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {ecr_repository}"
    )
    try:
        subprocess.run(login_cmd, shell=True, check=True)
        logger.info("Logged in to ECR")
    except subprocess.CalledProcessError as e:
        logger.error(f"ECR login failed: {e}")
        return None

    # Tag image for ECR
    full_image_name = f"{ecr_repository}:{tag}"
    tag_cmd = ["docker", "tag", f"{image_name}:{tag}", full_image_name]
    subprocess.run(tag_cmd, check=True)

    # Push to ECR
    logger.info(f"Pushing image to ECR: {full_image_name}")
    push_cmd = ["docker", "push", full_image_name]
    try:
        subprocess.run(push_cmd, check=True)
        logger.info("Image pushed successfully")
        return full_image_name
    except subprocess.CalledProcessError as e:
        logger.error(f"Docker push failed: {e}")
        return None


def upload_data_to_s3(local_path, s3_bucket, s3_prefix, use_cli=True):
    """Upload training data to S3 using fast parallel uploads."""
    logger.info(f"Uploading data from {local_path} to s3://{s3_bucket}/{s3_prefix}")

    local_path = Path(local_path)

    if not local_path.exists():
        logger.error(f"Local path does not exist: {local_path}")
        return None

    s3_data_path = f"s3://{s3_bucket}/{s3_prefix}"

    # Method 1: Use AWS CLI sync (fastest, uses parallel uploads)
    if use_cli:
        logger.info("Using AWS CLI with parallel uploads for maximum speed...")
        sync_cmd = [
            "aws",
            "s3",
            "sync",
            str(local_path),
            s3_data_path,
            "--only-show-errors",  # Only show errors, not every file
        ]

        try:
            result = subprocess.run(sync_cmd, check=True, capture_output=True, text=True)
            if result.stdout:
                logger.info(result.stdout)
            logger.info(f"Data uploaded successfully to {s3_data_path}")
            return s3_data_path
        except subprocess.CalledProcessError as e:
            logger.error(f"AWS CLI sync failed: {e.stderr}")
            logger.info("Falling back to boto3 multipart upload...")
            use_cli = False

    # Method 2: Use boto3 with multipart upload and threading
    if not use_cli:
        from boto3.s3.transfer import TransferConfig
        from concurrent.futures import ThreadPoolExecutor

        # Configure for faster uploads
        config = TransferConfig(
            multipart_threshold=1024 * 25,  # 25 MB
            max_concurrency=10,
            multipart_chunksize=1024 * 25,  # 25 MB
            use_threads=True,
        )

        s3_client = boto3.client("s3")

        def upload_file(file_path):
            relative_path = file_path.relative_to(local_path)
            s3_key = f"{s3_prefix}/{relative_path}"
            try:
                s3_client.upload_file(str(file_path), s3_bucket, s3_key, Config=config)
                logger.debug(f"Uploaded {file_path.name}")
                return True
            except Exception as e:
                logger.error(f"Failed to upload {file_path}: {e}")
                return False

        # Collect all files
        files_to_upload = [f for f in local_path.rglob("*") if f.is_file()]
        logger.info(f"Uploading {len(files_to_upload)} files using parallel threads...")

        # Upload in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(upload_file, files_to_upload))

        success_count = sum(results)
        logger.info(f"Successfully uploaded {success_count}/{len(files_to_upload)} files")

        if success_count == len(files_to_upload):
            logger.info(f"Data uploaded to {s3_data_path}")
            return s3_data_path
        else:
            logger.error("Some files failed to upload")
            return None


def submit_training_job(
    image_uri,
    s3_data_path,
    s3_output_path,
    role_arn,
    instance_type="ml.p3.2xlarge",
    instance_count=1,
    job_name=None,
    hyperparameters=None,
    max_run=86400,  # 24 hours
    volume_size=100,  # GB
):
    """Submit SageMaker training job."""

    if job_name is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        job_name = f"rfdetr-training-{timestamp}"

    if hyperparameters is None:
        hyperparameters = {"epochs": 50, "batch-size": 8, "grad-accum-steps": 2, "model-size": "large"}

    logger.info(f"Creating SageMaker training job: {job_name}")
    logger.info(f"Instance type: {instance_type}")
    logger.info(f"Hyperparameters: {hyperparameters}")

    # Create SageMaker session
    sagemaker_session = sagemaker.Session()

    # Create estimator
    estimator = Estimator(
        entry_point="src/train_rfdetr_aws.py",
        image_uri=image_uri,
        role=role_arn,
        instance_count=instance_count,
        instance_type=instance_type,
        output_path=s3_output_path,
        sagemaker_session=sagemaker_session,
        hyperparameters=hyperparameters,
        max_run=max_run,
        volume_size=volume_size,
        use_spot_instances=False,  # Set to True to use spot instances
        base_job_name="rfdetr-training",
    )

    # Start training
    logger.info("Starting training job...")
    estimator.fit({"training": s3_data_path}, job_name=job_name, wait=False)

    logger.info(f"Training job submitted: {job_name}")
    logger.info(
        f"Monitor progress at: https://console.aws.amazon.com/sagemaker/home?region={boto3.Session().region_name}#/jobs/{job_name}"
    )

    return estimator


def main():
    parser = argparse.ArgumentParser(description="Submit RF-DETR training job to SageMaker")

    # Required arguments
    parser.add_argument("--role-arn", type=str, required=True, help="SageMaker execution role ARN")
    parser.add_argument("--s3-bucket", type=str, required=True, help="S3 bucket for data and model artifacts")

    # Data arguments
    parser.add_argument("--local-data-path", type=str, help="Local path to training data (will be uploaded to S3)")
    parser.add_argument(
        "--s3-data-prefix", type=str, default="rfdetr/training-data", help="S3 prefix for training data"
    )
    parser.add_argument("--s3-output-prefix", type=str, default="rfdetr/output", help="S3 prefix for output")

    # Docker image arguments
    parser.add_argument("--image-name", type=str, default="rfdetr-sagemaker-training", help="Docker image name")
    parser.add_argument("--image-tag", type=str, default="latest", help="Docker image tag")
    parser.add_argument("--dockerfile-dir", type=str, default=".", help="Directory containing Dockerfile")
    parser.add_argument("--skip-build", action="store_true", help="Skip building and pushing Docker image")

    # Training job arguments
    parser.add_argument("--instance-type", type=str, default="ml.p3.2xlarge", help="SageMaker instance type")
    parser.add_argument("--instance-count", type=int, default=1, help="Number of instances")
    parser.add_argument("--job-name", type=str, help="Training job name")
    parser.add_argument("--max-run", type=int, default=86400, help="Maximum training time in seconds")
    parser.add_argument("--volume-size", type=int, default=100, help="EBS volume size in GB")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--model-size", type=str, default="large", choices=["large", "medium"])
    parser.add_argument("--learning-rate", type=float)

    args = parser.parse_args()

    # Build and push Docker image
    if not args.skip_build:
        image_uri = build_and_push_image(args.image_name, args.image_tag, args.dockerfile_dir)
        if not image_uri:
            logger.error("Failed to build and push Docker image")
            sys.exit(1)
    else:
        account_id = get_account_id()
        region = boto3.Session().region_name
        image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{args.image_name}:{args.image_tag}"
        logger.info(f"Using existing image: {image_uri}")

    # Upload data to S3 if local path provided
    if args.local_data_path:
        s3_data_path = upload_data_to_s3(args.local_data_path, args.s3_bucket, args.s3_data_prefix)
        if not s3_data_path:
            logger.error("Failed to upload data to S3")
            sys.exit(1)
    else:
        s3_data_path = f"s3://{args.s3_bucket}/{args.s3_data_prefix}"
        logger.info(f"Using existing S3 data: {s3_data_path}")

    # Prepare hyperparameters
    hyperparameters = {
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "grad-accum-steps": args.grad_accum_steps,
        "model-size": args.model_size,
    }
    if args.learning_rate:
        hyperparameters["learning-rate"] = args.learning_rate

    # Submit training job
    s3_output_path = f"s3://{args.s3_bucket}/{args.s3_output_prefix}"

    estimator = submit_training_job(
        image_uri=image_uri,
        s3_data_path=s3_data_path,
        s3_output_path=s3_output_path,
        role_arn=args.role_arn,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        job_name=args.job_name,
        hyperparameters=hyperparameters,
        max_run=args.max_run,
        volume_size=args.volume_size,
    )

    print(estimator)

    logger.info("Training job submission completed")


if __name__ == "__main__":
    main()
