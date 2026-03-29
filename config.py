# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
═══════════════════════════════════════════════════════════════
  AWS Clean Rooms ML HCLS ADR Propensity Scoring — Configuration
═══════════════════════════════════════════════════════════════

  Fill in your AWS Account ID and Region below.
  All scripts in this demo read from this file.

  Usage:
    1. Set AWS_ACCOUNT_ID and AWS_REGION
    2. Run the scripts in order (see README)
"""

# ─── REQUIRED: Set these to your values ───────────────────
import os as _os_cfg
AWS_ACCOUNT_ID        = _os_cfg.environ.get("AWS_ACCOUNT_ID", "443514573025")
AWS_REGION            = _os_cfg.environ.get("AWS_REGION", "eu-north-1")
# Required only for Step 6 (QuickSight dashboard). Must be a valid email address.
QS_NOTIFICATION_EMAIL = _os_cfg.environ.get("QS_NOTIFICATION_EMAIL", "your@email.com")

# ─── RUN ID (auto-generated, ensures unique bucket names) ─
import os as _os
from datetime import datetime as _dt

_RUN_ID_FILE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".run_id")

def _get_or_create_run_id():
    """Return a short unique suffix for bucket names.

    Generated once and persisted to .run_id so every script in the same
    demo run shares the same buckets. Delete .run_id to start fresh.
    """
    if _os.path.exists(_RUN_ID_FILE):
        return open(_RUN_ID_FILE).read().strip()
    run_id = _dt.utcnow().strftime("%Y%m%d%H%M")
    with open(_RUN_ID_FILE, "w") as f:
        f.write(run_id)
    return run_id

RUN_ID = _get_or_create_run_id()

# ─── DERIVED (no need to change) ──────────────────────────
# Bucket names use "hcls-adr" prefix — distinct from the FSI fraud demo
BUCKET        = f"cleanrooms-ml-hcls-adr-{AWS_ACCOUNT_ID}-{RUN_ID}"
OUTPUT_BUCKET = f"cleanrooms-ml-hcls-adr-output-{AWS_ACCOUNT_ID}-{RUN_ID}"
PREFIX        = "cleanrooms-ml-hcls-adr"

# ECR image URIs
TRAINING_IMAGE  = f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/cleanrooms-ml-hcls-adr-training:latest"
INFERENCE_IMAGE = f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/cleanrooms-ml-hcls-adr-inference:latest"

# SageMaker DLC registry
SAGEMAKER_REGISTRY       = "763104351884"
SAGEMAKER_TRAINING_BASE  = f"{SAGEMAKER_REGISTRY}.dkr.ecr.{AWS_REGION}.amazonaws.com/pytorch-training:2.3.0-cpu-py311-ubuntu20.04-sagemaker"
SAGEMAKER_INFERENCE_BASE = f"{SAGEMAKER_REGISTRY}.dkr.ecr.{AWS_REGION}.amazonaws.com/pytorch-inference:2.3.0-cpu-py311-ubuntu20.04-sagemaker"

# Glue
GLUE_DB              = "cleanrooms_ml_hcls_adr"
PHARMA_TABLE         = "pharma_drug_exposure"
INSURER_TABLE        = "insurer_outcomes"

# IAM role names
ROLE_DATA_PROVIDER  = f"{PREFIX}-data-provider-role"
ROLE_MODEL_PROVIDER = f"{PREFIX}-model-provider-role"
ROLE_ML_CONFIG      = f"{PREFIX}-ml-config-role"
ROLE_QUERY_RUNNER   = f"{PREFIX}-query-runner-role"


def validate(require_qs_email=False):
    """Call this at the start of any script to catch misconfiguration early."""
    errors = []
    if AWS_ACCOUNT_ID == "CHANGE_ME" or not AWS_ACCOUNT_ID.isdigit() or len(AWS_ACCOUNT_ID) != 12:
        errors.append(f"AWS_ACCOUNT_ID must be a 12-digit number, got: '{AWS_ACCOUNT_ID}'")
    if AWS_REGION == "CHANGE_ME" or not AWS_REGION:
        errors.append(f"AWS_REGION must be set, got: '{AWS_REGION}'")
    if require_qs_email and (not QS_NOTIFICATION_EMAIL or QS_NOTIFICATION_EMAIL == "your@email.com" or "@" not in QS_NOTIFICATION_EMAIL):
        errors.append(f"QS_NOTIFICATION_EMAIL must be a valid email address for Step 6, got: '{QS_NOTIFICATION_EMAIL}'")
    if errors:
        print("=" * 60)
        print("CONFIGURATION ERROR — edit config.py")
        print("=" * 60)
        for e in errors:
            print(f"  ✗ {e}")
        raise SystemExit(1)
