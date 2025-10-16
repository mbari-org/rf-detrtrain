# Quick Start Guide - RF-DETR SageMaker Training

## Prerequisites

* AWS CLI configured (`aws configure`)
* Docker installed and running
* Training data in COCO format (sample provided in `test_data/sample/`, or use your own)
* S3 bucket created (or will be created automatically)
* SageMaker execution role created (or will be created automatically)

### Test Data

A **sample dataset** (~83MB) is included in `test_data/sample/` for quick testing.

For production training, you'll need your own dataset. See `test_data/README.md` for:
- Using the sample dataset
- Creating your own dataset
- COCO format specifications

## Testing Workflow (Recommended)

### Step 1: Test Locally (FREE - No AWS costs)

```bash
# Full test with GPU (~10 minutes for 2 epochs)
# Using sample data from the repository
./scripts/test_docker_local.sh rfdetr-sagemaker-training latest test_data/sample
```

**What this does:**
- ✓ Validates Docker image builds correctly
- ✓ Checks all Python dependencies
- ✓ Verifies data loading
- ✓ Tests training script (GPU test only)

### Step 2: Deploy to SageMaker

Once local tests pass, deploy to SageMaker:

```bash
# Edit configuration (one time)
nano scripts/example_usage.sh
# Verify: AWS_ACCOUNT_ID, S3_BUCKET, LOCAL_DATA_PATH

# Run the full pipeline
./scripts/example_usage.sh
```

**What this does:**
1. Creates S3 bucket (if needed)
2. Uploads training data to S3 (optimized parallel upload)
3. Builds Docker image
4. Pushes image to ECR
5. Submits SageMaker training job

## Quick Commands Reference

### Build Only
```bash
./scripts/build_docker.sh
```

### Upload Data Only
```bash
# Upload sample data from repo
./scripts/fast_s3_upload.sh test_data/sample rfdetr-sagemaker-training rfdetr/training-data

# Or upload your own dataset
./scripts/fast_s3_upload.sh /path/to/your/dataset rfdetr-sagemaker-training rfdetr/training-data
```

### Submit Training Job Only (skip rebuild/upload)
```bash
python src/submit_sagemaker_job.py \
  --role-arn "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole" \
  --s3-bucket "your-sagemaker-bucket" \
  --skip-build \
  --epochs 50 \
  --batch-size 8 \
  --instance-type ml.p3.2xlarge
```

## Monitor Training

### AWS Console
https://console.aws.amazon.com/sagemaker/home?region=us-west-2#/jobs

### AWS CLI
```bash
# List training jobs
aws sagemaker list-training-jobs --max-results 10

# Describe specific job
aws sagemaker describe-training-job --training-job-name YOUR_JOB_NAME

# View logs
aws logs tail /aws/sagemaker/TrainingJobs --follow --filter-pattern YOUR_JOB_NAME
```

## Download Results

After training completes:

```bash
# Download model
aws s3 cp s3://rfdetr-sagemaker-training/rfdetr/output/YOUR_JOB_NAME/output/model.tar.gz .
tar -xzf model.tar.gz
```

## Cost Estimates

**Instance Costs (approximate, us-west-2)**:
- `ml.g4dn.xlarge`: $0.71/hr (1x T4 GPU) - Good for testing
- `ml.p3.2xlarge`: $3.83/hr (1x V100 GPU) - Recommended for production
- `ml.p3.8xlarge`: $14.69/hr (4x V100 GPUs) - For larger models

**Example Training Cost**:
- 50 epochs @ 8 hours on ml.p3.2xlarge = ~$30

**Tips to Reduce Costs**:
1. Test locally first (FREE)
2. Use spot instances (up to 90% discount)
3. Start with ml.g4dn.xlarge for initial runs
4. Set `max_run` to avoid runaway costs
5. Monitor CloudWatch to catch issues early

## Troubleshooting Quick Fixes

### "Cannot assume role"
```bash
./scripts/setup_sagemaker_role.sh
```

### "Bucket does not exist"
```bash
aws s3 mb s3://rfdetr-sagemaker-training --region us-west-2
```

### "ImportError: cannot import name 'clear_device_cache'"
Already fixed in current Dockerfile. Rebuild:
```bash
./scripts/build_docker.sh
```

### "AttributeError: 'torch.utils._pytree' has no attribute 'register_pytree_node'"
Version mismatch between PyTorch and transformers. Already fixed in Dockerfile. Rebuild:
```bash
./scripts/build_docker.sh
```

### "401 Unauthorized" when building Docker
```bash
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin \
  763104351884.dkr.ecr.us-west-2.amazonaws.com
```

## File Structure

```
aws/
├── src/                          # Python source files
│   ├── train_rfdetr_aws.py      # SageMaker training script
│   ├── train_rfdetr_local.py    # Local training/data prep
│   └── submit_sagemaker_job.py  # Python submission script
├── scripts/                      # Shell scripts
│   ├── example_usage.sh         # Main entry point (USE THIS)
│   ├── build_docker.sh          # Build Docker image
│   ├── setup_sagemaker_role.sh  # Create IAM role
│   ├── fast_s3_upload.sh        # Fast S3 upload
│   ├── test_docker_local.sh     # Local GPU testing
│   └── test_docker_local_cpu.sh # Local CPU testing
├── Dockerfile                    # Container definition
├── requirements.txt              # Python dependencies for local dev
├── README.md                     # Full documentation
└── QUICKSTART.md                 # This file
```

## Support

For detailed documentation, see [README.md](README.md)
