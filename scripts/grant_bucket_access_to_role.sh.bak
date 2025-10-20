#!/bin/bash
# Grant full S3 access (s3:*) to a specific bucket for a SageMaker execution role

set -euo pipefail

ROLE_NAME="${1:-SageMakerExecutionRole}"
BUCKET_NAME="${2:-}"

if [ -z "${BUCKET_NAME}" ]; then
  echo "Usage: $0 <RoleName> <BucketName>"
  echo "Example: $0 SageMakerExecutionRole my-sagemaker-bucket"
  exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region)

echo "========================================="
echo "Grant S3 access on bucket to IAM role"
echo "========================================="
echo "Role Name : ${ROLE_NAME}"
echo "Bucket    : s3://${BUCKET_NAME}"
echo "Account   : ${ACCOUNT_ID}"
echo "Region    : ${REGION}"
echo "========================================="

# Validate role exists
if ! aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1; then
  echo "ERROR: IAM role '${ROLE_NAME}' not found. Create it first (e.g., scripts/setup_sagemaker_role.sh)."
  exit 1
fi

# Validate bucket exists
if ! aws s3api head-bucket --bucket "${BUCKET_NAME}" 2>/dev/null; then
  echo "ERROR: S3 bucket '${BUCKET_NAME}' does not exist or is not accessible."
  exit 1
fi

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
POLICY_NAME="${ROLE_NAME}-S3FullAccess-${BUCKET_NAME}"

# Construct policy document granting s3:* on the specific bucket and its contents
POLICY_DOCUMENT=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BucketLevelPerms",
      "Effect": "Allow",
      "Action": [
        "s3:*"
      ],
      "Resource": [
        "arn:aws:s3:::${BUCKET_NAME}",
        "arn:aws:s3:::${BUCKET_NAME}/*"
      ]
    }
  ]
}
EOF
)

echo "Ensuring IAM policy exists: ${POLICY_NAME}"

# Try to create policy; if exists, fetch its ARN
set +e
CREATE_OUT=$(aws iam create-policy \
  --policy-name "${POLICY_NAME}" \
  --policy-document "${POLICY_DOCUMENT}" 2>&1)
STATUS=$?
set -e

if [ ${STATUS} -eq 0 ]; then
  POLICY_ARN=$(echo "$CREATE_OUT" | python -c "import sys, json; print(json.load(sys.stdin)['Policy']['Arn'])")
  echo "Created policy: ${POLICY_ARN}"
else
  echo "Policy may already exist; looking it up..."
  POLICY_ARN=$(aws iam list-policies --scope Local --query "Policies[?PolicyName=='${POLICY_NAME}'].Arn | [0]" --output text)
  if [ -z "${POLICY_ARN}" ] || [ "${POLICY_ARN}" = "None" ]; then
    echo "ERROR: Failed to create or find policy '${POLICY_NAME}'. Output was:" >&2
    echo "$CREATE_OUT" >&2
    exit 1
  fi
  echo "Found policy: ${POLICY_ARN}"
fi

# Attach policy if not already attached
ATTACHED=$(aws iam list-attached-role-policies --role-name "${ROLE_NAME}" \
  --query "AttachedPolicies[?PolicyArn=='${POLICY_ARN}'] | length(@)" --output text)

if [ "${ATTACHED}" != "1" ]; then
  echo "Attaching policy to role..."
  aws iam attach-role-policy --role-name "${ROLE_NAME}" --policy-arn "${POLICY_ARN}"
else
  echo "Policy already attached to role."
fi

echo "Waiting briefly for IAM to propagate..."
sleep 5

echo "========================================="
echo "S3 full access granted to role on bucket"
echo "========================================="
echo "Role ARN  : ${ROLE_ARN}"
echo "Bucket    : s3://${BUCKET_NAME}"
echo "Policy ARN: ${POLICY_ARN}"
echo "Note: This grants s3:* on the bucket and all objects."
echo "========================================="


