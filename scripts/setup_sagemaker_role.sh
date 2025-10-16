#!/bin/bash
# Create SageMaker Execution Role
# This script creates an IAM role that SageMaker can assume

set -e

ROLE_NAME="${1:-SageMakerExecutionRole}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "========================================="
echo "Creating SageMaker Execution Role"
echo "========================================="
echo "Role Name: ${ROLE_NAME}"
echo "Account ID: ${ACCOUNT_ID}"
echo "========================================="

# Create trust policy document
TRUST_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
)

echo "Creating IAM role with trust policy..."
aws iam create-role \
  --role-name ${ROLE_NAME} \
  --assume-role-policy-document "${TRUST_POLICY}" \
  --description "SageMaker execution role for training jobs" \
  || echo "Role may already exist, continuing..."

# Attach AWS managed policy for SageMaker full access
echo "Attaching AmazonSageMakerFullAccess policy..."
aws iam attach-role-policy \
  --role-name ${ROLE_NAME} \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess \
  || echo "Policy may already be attached"

# Create custom policy for S3 and ECR access
CUSTOM_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::rfdetr-sagemaker-training/*",
        "arn:aws:s3:::rfdetr-sagemaker-training"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"
    }
  ]
}
EOF
)

POLICY_NAME="${ROLE_NAME}CustomPolicy"

echo "Creating custom policy for S3 and ECR access..."
POLICY_ARN=$(aws iam create-policy \
  --policy-name ${POLICY_NAME} \
  --policy-document "${CUSTOM_POLICY}" \
  --query 'Policy.Arn' \
  --output text 2>/dev/null || aws iam list-policies --query "Policies[?PolicyName=='${POLICY_NAME}'].Arn" --output text)

if [ -n "${POLICY_ARN}" ]; then
  echo "Attaching custom policy..."
  aws iam attach-role-policy \
    --role-name ${ROLE_NAME} \
    --policy-arn ${POLICY_ARN} \
    || echo "Policy may already be attached"
fi

# Wait for role to be available
echo "Waiting for role to propagate (this may take a few seconds)..."
sleep 10

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

echo "========================================="
echo "SageMaker Execution Role Created!"
echo "========================================="
echo "Role ARN: ${ROLE_ARN}"
echo ""
echo "Update your example_usage.sh with:"
echo "export SAGEMAKER_ROLE_ARN=\"${ROLE_ARN}\""
echo "========================================="

# Verify role can be assumed
echo "Verifying role..."
aws iam get-role --role-name ${ROLE_NAME} --query 'Role.Arn' --output text
echo "Role is ready to use!"
