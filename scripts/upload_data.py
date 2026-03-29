# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Upload synthetic data to S3 for AWS Clean Rooms ML HCLS ADR Propensity demo.
Creates source and output buckets, uploads CSVs.
Idempotent: safe to re-run — existing buckets and objects are reused.
Reads config from config.py.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import *
validate()

import boto3, json

s3 = boto3.client("s3", region_name=AWS_REGION)

def log(msg):
    print(f"  → {msg}")


def create_bucket(bucket_name):
    """Create an S3 bucket if it doesn't exist, then apply security settings.
    Idempotent: BucketAlreadyOwnedByYou is treated as success.
    """
    try:
        if AWS_REGION == "us-east-1":
            s3_us = boto3.client(
                "s3",
                region_name="us-east-1",
                endpoint_url="https://s3.us-east-1.amazonaws.com",
            )
            s3_us.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": AWS_REGION},
            )
        log(f"Created bucket: {bucket_name}")
    except Exception as e:
        if "BucketAlreadyOwnedByYou" in str(e):
            log(f"Bucket already exists (reusing): {bucket_name}")
        else:
            raise

    # Block all public access
    s3.put_public_access_block(
        Bucket=bucket_name,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True,
            "BlockPublicPolicy": True, "RestrictPublicBuckets": True,
        },
    )
    # Default encryption (SSE-S3)
    s3.put_bucket_encryption(
        Bucket=bucket_name,
        ServerSideEncryptionConfiguration={
            "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}],
        },
    )
    # Versioning
    s3.put_bucket_versioning(
        Bucket=bucket_name,
        VersioningConfiguration={"Status": "Enabled"},
    )
    # Enforce TLS-only
    s3.put_bucket_policy(
        Bucket=bucket_name,
        Policy=json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Sid": "DenyInsecureTransport",
                "Effect": "Deny", "Principal": "*", "Action": "s3:*",
                "Resource": [f"arn:aws:s3:::{bucket_name}", f"arn:aws:s3:::{bucket_name}/*"],
                "Condition": {"Bool": {"aws:SecureTransport": "false"}},
            }],
        }),
    )
    log(f"  Security settings applied to {bucket_name}")


def upload_file(local_path, bucket, key):
    s3.upload_file(local_path, bucket, key)
    log(f"Uploaded {os.path.basename(local_path)} → s3://{bucket}/{key}")


def main():
    print("=" * 60)
    print("Upload Data to S3 — HCLS ADR Propensity Scoring")
    print("=" * 60)
    print(f"Account: {AWS_ACCOUNT_ID}  Region: {AWS_REGION}")
    print(f"Source bucket:  {BUCKET}")
    print(f"Output bucket:  {OUTPUT_BUCKET}")
    print()

    project_root = os.path.join(os.path.dirname(__file__), "..")
    pharma_csv  = os.path.join(project_root, "data", "pharma_drug_exposure.csv")
    insurer_csv = os.path.join(project_root, "data", "insurer_outcomes.csv")

    for path in [pharma_csv, insurer_csv]:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run: python data/generate_synthetic_data.py")
            sys.exit(1)

    # Create buckets (idempotent)
    create_bucket(BUCKET)
    create_bucket(OUTPUT_BUCKET)

    # Upload pharma company data (Party A) — Clean Rooms prefix
    upload_file(pharma_csv,  BUCKET, "pharma/pharma_drug_exposure.csv")
    # Upload health insurer data (Party B) — Clean Rooms prefix
    upload_file(insurer_csv, BUCKET, "insurer/insurer_outcomes.csv")

    # Also upload under data/ prefix for SageMaker / local training channel
    upload_file(pharma_csv,  BUCKET, "data/pharma_drug_exposure.csv")
    upload_file(insurer_csv, BUCKET, "data/insurer_outcomes.csv")

    # Verify
    print("\nVerifying uploads...")
    resp = s3.list_objects_v2(Bucket=BUCKET)
    for obj in resp.get("Contents", []):
        print(f"  {obj['Key']}  ({obj['Size']:,} bytes)")

    print("\nDone! Next: python scripts/build_and_push.py  (or codebuild_containers.py)")


if __name__ == "__main__":
    main()
