# Training for RF-DETR

This directory contains the necessary files to train the RF-DETR model on AWS SageMaker.

## File Structure

```
aws/
├── src/                          # Python source files
│   ├── train_rfdetr_aws.py      # SageMaker training script
│   ├── train_rfdetr_local.py    # Local training/data prep script
│   └── submit_sagemaker_job.py  # Helper to submit training jobs
├── scripts/                      # Shell scripts
│   ├── build_docker.sh          # Build Docker image
│   ├── example_usage.sh         # Main entry point
│   ├── setup_sagemaker_role.sh  # Create IAM role
│   ├── fast_s3_upload.sh        # Fast S3 upload
│   ├── test_docker_local.sh     # Local GPU testing
├── Dockerfile                    # Container definition
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── QUICKSTART.md                 # Quick start guide
```

## Prerequisites

1. **AWS Account** with SageMaker access
2. **AWS CLI** configured with appropriate credentials
3. **Docker** installed locally (for building images)
4. **Python packages**:
   ```bash
   pip install boto3 sagemaker
   ```

## Setup

### 1. Create SageMaker Execution Role

The SageMaker execution role needs permissions to:
- Access S3 buckets (for data and model artifacts)
- Access ECR (for Docker images)
- Create and manage SageMaker training jobs

Create a role with the `AmazonSageMakerFullAccess` managed policy, or create a custom policy with the required permissions.

```bash
# Get your role ARN
aws iam list-roles | grep SageMaker
```

### 2. Prepare Training Data

Your training data should be in the COCO format with the following structure:

```
dataset/
├── train/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── valid/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   └── ...
└── test/ (optional)
    ├── _annotations.coco.json
    └── ...
```

## Usage

### Option 1: Quick Start (Automated)

Use the `example_usage.sh` script to handle everything:

```bash
# First, configure the script with your AWS details
cp scripts/example_usage.sh.template scripts/example_usage.sh
nano scripts/example_usage.sh  # Update with your AWS account info

# Then run it
./scripts/example_usage.sh
```

Or call the Python script directly:

```bash
python src/submit_sagemaker_job.py \
  --role-arn arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_SAGEMAKER_ROLE \
  --s3-bucket your-sagemaker-bucket \
  --local-data-path test_data/sample \
  --epochs 50 \
  --batch-size 8 \
  --model-size large \
  --instance-type ml.p3.2xlarge
```

**Note**: A sample dataset is provided in `test_data/sample/` for testing. For production, use your own dataset.

This will:
1. Build the Docker image
2. Push it to ECR
3. Upload your dataset to S3
4. Submit the SageMaker training job

### Option 2: Manual Steps

#### Step 1: Build and Push Docker Image

```bash
# Get your AWS account ID and region
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region)
IMAGE_NAME=rfdetr-sagemaker-training
IMAGE_TAG=latest

# Build the image
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f Dockerfile .

# Create ECR repository (if it doesn't exist)
aws ecr create-repository --repository-name ${IMAGE_NAME} --region ${REGION} || true

# Login to ECR
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Tag and push
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}:${IMAGE_TAG}
docker push ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}:${IMAGE_TAG}
```

#### Step 2: Upload Data to S3

```bash
aws s3 sync /path/to/your/dataset s3://your-bucket/rfdetr/training-data/
```

#### Step 3: Submit Training Job Using Python

```python
import boto3
import sagemaker
from sagemaker.estimator import Estimator

# Configuration
role_arn = 'arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_SAGEMAKER_ROLE'
image_uri = 'YOUR_ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/rfdetr-sagemaker-training:latest'
s3_data_path = 's3://your-bucket/rfdetr/training-data/'
s3_output_path = 's3://your-bucket/rfdetr/output/'

# Hyperparameters
hyperparameters = {
    'epochs': 50,
    'batch-size': 8,
    'grad-accum-steps': 2,
    'model-size': 'large'
}

# Create estimator
estimator = Estimator(
    image_uri=image_uri,
    role=role_arn,
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    output_path=s3_output_path,
    hyperparameters=hyperparameters,
    max_run=86400,  # 24 hours
    volume_size=100  # GB
)

# Start training
estimator.fit({'training': s3_data_path})
```

## SageMaker Instance Types

Recommended instance types for training:

- **ml.p3.2xlarge** - 1x V100 GPU, 8 vCPUs, 61 GB RAM (~$3.83/hr)
- **ml.p3.8xlarge** - 4x V100 GPUs, 32 vCPUs, 244 GB RAM (~$14.69/hr)
- **ml.p3.16xlarge** - 8x V100 GPUs, 64 vCPUs, 488 GB RAM (~$28.15/hr)
- **ml.g4dn.xlarge** - 1x T4 GPU, 4 vCPUs, 16 GB RAM (~$0.71/hr) - for testing
- **ml.g5.xlarge** - 1x A10G GPU, 4 vCPUs, 16 GB RAM (~$1.41/hr)

For cost savings, consider using **Managed Spot Training** which can reduce costs by up to 90%.

## Hyperparameters

Available hyperparameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | 50 | Number of training epochs |
| `batch-size` | int | 8 | Batch size per device |
| `grad-accum-steps` | int | 2 | Gradient accumulation steps |
| `model-size` | str | 'large' | Model size: 'large' or 'medium' |
| `learning-rate` | float | None | Learning rate (uses model default if not set) |

## Monitoring

### View Training Progress

1. **AWS Console**:
   - Navigate to SageMaker → Training jobs
   - Click on your job name
   - View metrics and logs

2. **AWS CLI**:
   ```bash
   aws sagemaker describe-training-job --training-job-name YOUR_JOB_NAME
   ```

3. **CloudWatch Logs**:
   ```bash
   aws logs tail /aws/sagemaker/TrainingJobs --follow --filter-pattern YOUR_JOB_NAME
   ```

### Download Model Artifacts

After training completes:

```bash
aws s3 cp s3://your-bucket/rfdetr/output/YOUR_JOB_NAME/output/model.tar.gz .
tar -xzf model.tar.gz
```

## Local Testing

Before deploying to SageMaker (and incurring costs), test your Docker container locally:

### GPU Training Test (GPU required)

To test training locally with a GPU:

```bash
cd aws/
./scripts/test_docker_local.sh rfdetr-sagemaker-training latest test_data/sample
```

This will:
- Run training for 2 epochs with small batch size
- Use your local GPU
- Save outputs to `/tmp/sagemaker-local/`
- Simulate SageMaker's directory structure

**Requirements**:
- NVIDIA Docker runtime installed
- GPU available on host

### Manual Docker Testing

You can also run the container manually:

```bash
# Build the image
./scripts/build_docker.sh

# Run interactively to debug (using sample data from repo)
docker run --rm -it \
  -v $(pwd)/test_data/sample:/opt/ml/input/data/training \
  -v /tmp/output:/opt/ml/output \
  -v /tmp/model:/opt/ml/model \
  --gpus all \
  rfdetr-sagemaker-training:latest \
  bash

# Inside the container, you can:
# - Check Python version: python --version
# - Verify imports: python -c "import torch, rfdetr, transformers"
# - Inspect data: ls /opt/ml/input/data/training
# - Run training: python /opt/ml/code/train.py --epochs 1 --batch-size 2
```

### Test Without GPU

If you don't have a GPU locally, you can still test the container build and imports:

```bash
docker run --rm rfdetr-sagemaker-training:latest \
  python -c "import torch, rfdetr; print('Success!')"
```

## Cost Optimization

1. **Use Spot Instances**: In `submit_sagemaker_job.py`, modify the estimator:
   ```python
   use_spot_instances=True,
   max_wait=90000  # Maximum time to wait for spot instance
   ```

2. **Use Smaller Instances for Testing**: Start with `ml.g4dn.xlarge` for debugging

3. **Checkpoint Regularly**: Modify the training script to save checkpoints to S3

4. **Set Maximum Runtime**: Use `max_run` parameter to avoid runaway costs

## Troubleshooting

### Container Fails to Start

Check CloudWatch logs:
```bash
aws logs tail /aws/sagemaker/TrainingJobs --follow
```

Common issues:
- Missing dependencies in Dockerfile
- Incorrect Python path
- Memory issues

### Training Fails

1. Check the error in CloudWatch logs
2. Verify data format matches expected COCO format
3. Ensure sufficient instance memory for batch size
4. Check S3 permissions for the execution role

### Out of Memory (OOM)

- Reduce `batch-size`
- Increase `grad-accum-steps`
- Use a larger instance type
- Use `model-size: 'medium'` instead of 'large'

## Support

For issues specific to:
- **SageMaker**: Check [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- **RF-DETR**: Check the rfdetr package documentation
- **This Implementation**: Open an issue in the project repository

## References

- [AWS SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [SageMaker Training Toolkit](https://github.com/aws/sagemaker-training-toolkit)
- [SageMaker Docker Containers](https://github.com/aws/deep-learning-containers)
