# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
AWS Clean Rooms ML HCLS ADR — Undeploy / Teardown Script
Reads all account/region config from config.py.

Run with: python scripts/undeploy.py  (from the project root folder)

What this script removes (in safe dependency order):
  1. QuickSight dashboard, analysis, dataset, data source
  2. Clean Rooms ML: trained model inference jobs, trained models,
     ML input channels, model algorithm associations, model algorithms,
     ML configuration
  3. Clean Rooms: configured table associations, configured tables,
     membership, collaboration
  4. Glue tables and database
  5. IAM inline policies and roles
  6. S3 buckets (all objects + bucket)
  7. ECR repositories (all images + repo)
  8. CodeBuild project

Idempotent: resources that don't exist are skipped gracefully.
S3 buckets and ECR repos require explicit confirmation before deletion.
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import *
validate()

import boto3
from botocore.exceptions import ClientError

session = boto3.Session(region_name=AWS_REGION)
iam     = session.client("iam")
glue    = session.client("glue")
cr      = session.client("cleanrooms")
crml    = session.client("cleanroomsml")
s3      = session.client("s3")
ecr     = session.client("ecr")
cb      = session.client("codebuild")
qs      = session.client("quicksight")
sts     = session.client("sts")

QS_DATASOURCE_ID = f"{PREFIX}-athena-source"
QS_DS_INFERENCE  = f"{PREFIX}-ds-inference"
QS_ANALYSIS_ID   = f"{PREFIX}-adr-analysis"
QS_DASHBOARD_ID  = f"{PREFIX}-adr-dashboard"
INFERENCE_TABLE  = "adr_inference_output"

TRAINING_REPO    = "cleanrooms-ml-hcls-adr-training"
INFERENCE_REPO   = "cleanrooms-ml-hcls-adr-inference"
CB_PROJECT_NAME  = "cleanrooms-ml-hcls-adr-build"
CB_ROLE_NAME     = "cleanrooms-ml-hcls-adr-codebuild-role"
SM_ROLE_NAME     = "cleanrooms-ml-hcls-adr-sagemaker-role"

IAM_ROLES = [
    ROLE_DATA_PROVIDER,
    ROLE_MODEL_PROVIDER,
    ROLE_ML_CONFIG,
    ROLE_QUERY_RUNNER,
    CB_ROLE_NAME,
    SM_ROLE_NAME,
]


def log(msg):
    print(f"  → {msg}")


def skip(msg):
    print(f"  ✓ {msg} (already gone)")


# ═══ HELPERS ═══

def _delete_s3_bucket(bucket_name):
    """Delete all objects (including versions) then delete the bucket."""
    s3r = boto3.resource("s3", region_name=AWS_REGION)
    try:
        bucket = s3r.Bucket(bucket_name)
        bucket.object_versions.delete()
        bucket.objects.delete()
        bucket.delete()
        log(f"Deleted bucket: {bucket_name}")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchBucket", "404"):
            skip(f"Bucket {bucket_name}")
        else:
            log(f"ERROR deleting bucket {bucket_name}: {e}")


def _delete_ecr_repo(repo_name):
    try:
        ecr.delete_repository(repositoryName=repo_name, force=True)
        log(f"Deleted ECR repo: {repo_name}")
    except ecr.exceptions.RepositoryNotFoundException:
        skip(f"ECR repo {repo_name}")


def _delete_iam_role(role_name):
    try:
        # Remove all inline policies first
        policies = iam.list_role_policies(RoleName=role_name).get("PolicyNames", [])
        for p in policies:
            iam.delete_role_policy(RoleName=role_name, PolicyName=p)
        # Detach managed policies
        attached = iam.list_attached_role_policies(RoleName=role_name).get("AttachedPolicies", [])
        for p in attached:
            iam.detach_role_policy(RoleName=role_name, PolicyArn=p["PolicyArn"])
        iam.delete_role(RoleName=role_name)
        log(f"Deleted IAM role: {role_name}")
    except iam.exceptions.NoSuchEntityException:
        skip(f"IAM role {role_name}")


# ═══ 1. QUICKSIGHT ═══

def undeploy_quicksight():
    print("\n[1/8] Removing QuickSight resources...")
    for resource_id, delete_fn, label in [
        (QS_DASHBOARD_ID, lambda: qs.delete_dashboard(AwsAccountId=AWS_ACCOUNT_ID, DashboardId=QS_DASHBOARD_ID), "dashboard"),
        (QS_ANALYSIS_ID,  lambda: qs.delete_analysis(AwsAccountId=AWS_ACCOUNT_ID, AnalysisId=QS_ANALYSIS_ID, ForceDeleteWithoutRecovery=True), "analysis"),
        (QS_DS_INFERENCE, lambda: qs.delete_data_set(AwsAccountId=AWS_ACCOUNT_ID, DataSetId=QS_DS_INFERENCE), "dataset"),
        (QS_DATASOURCE_ID,lambda: qs.delete_data_source(AwsAccountId=AWS_ACCOUNT_ID, DataSourceId=QS_DATASOURCE_ID), "data source"),
    ]:
        try:
            delete_fn()
            log(f"Deleted QuickSight {label}: {resource_id}")
        except ClientError as e:
            if e.response["Error"]["Code"] in ("ResourceNotFoundException", "UnsupportedUserEditionException"):
                skip(f"QuickSight {label} {resource_id}")
            else:
                log(f"WARNING deleting QuickSight {label}: {e}")


# ═══ 2. CLEAN ROOMS ML ═══

def undeploy_cleanrooms_ml(membership_id):
    print("\n[2/8] Removing Clean Rooms ML resources...")
    if not membership_id:
        log("No membership found — skipping Clean Rooms ML resources")
        return

    # Inference jobs
    try:
        jobs = crml.list_trained_model_inference_jobs(membershipIdentifier=membership_id)
        for j in jobs.get("trainedModelInferenceJobs", []):
            try:
                crml.cancel_trained_model_inference_job(
                    membershipIdentifier=membership_id,
                    trainedModelInferenceJobArn=j["trainedModelInferenceJobArn"]
                )
                log(f"Cancelled inference job: {j['name']}")
            except Exception:
                pass
    except Exception as e:
        log(f"Could not list inference jobs (non-fatal): {e}")

    # Trained models
    try:
        models = crml.list_trained_models(membershipIdentifier=membership_id)
        for m in models.get("trainedModels", []):
            try:
                crml.delete_trained_model(
                    membershipIdentifier=membership_id,
                    trainedModelArn=m["trainedModelArn"]
                )
                log(f"Deleted trained model: {m['name']}")
            except Exception as e:
                log(f"WARNING deleting trained model {m['name']}: {e}")
    except Exception as e:
        log(f"Could not list trained models (non-fatal): {e}")

    # ML input channels
    try:
        channels = crml.list_ml_input_channels(membershipIdentifier=membership_id)
        for ch in channels.get("mlInputChannelsList", []):
            try:
                crml.delete_ml_input_channel(
                    membershipIdentifier=membership_id,
                    mlInputChannelArn=ch["mlInputChannelArn"]
                )
                log(f"Deleted ML input channel: {ch['name']}")
            except Exception as e:
                log(f"WARNING deleting ML input channel {ch['name']}: {e}")
    except Exception as e:
        log(f"Could not list ML input channels (non-fatal): {e}")

    # Model algorithm associations
    try:
        assocs = crml.list_configured_model_algorithm_associations(membershipIdentifier=membership_id)
        for a in assocs.get("configuredModelAlgorithmAssociations", []):
            try:
                crml.delete_configured_model_algorithm_association(
                    membershipIdentifier=membership_id,
                    configuredModelAlgorithmAssociationArn=a["configuredModelAlgorithmAssociationArn"]
                )
                log(f"Deleted algorithm association: {a.get('name', a['configuredModelAlgorithmAssociationArn'])}")
            except Exception as e:
                log(f"WARNING deleting algorithm association: {e}")
    except Exception as e:
        log(f"Could not list algorithm associations (non-fatal): {e}")

    # ML configuration
    try:
        crml.delete_ml_configuration(membershipIdentifier=membership_id)
        log("Deleted ML configuration")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            skip("ML configuration")
        else:
            log(f"WARNING deleting ML configuration: {e}")

    # Configured model algorithms (global, not membership-scoped)
    try:
        algos = crml.list_configured_model_algorithms()
        for algo in algos.get("configuredModelAlgorithms", []):
            if PREFIX in algo.get("name", ""):
                try:
                    crml.delete_configured_model_algorithm(
                        configuredModelAlgorithmArn=algo["configuredModelAlgorithmArn"]
                    )
                    log(f"Deleted model algorithm: {algo['name']}")
                except Exception as e:
                    log(f"WARNING deleting model algorithm: {e}")
    except Exception as e:
        log(f"Could not list model algorithms (non-fatal): {e}")


# ═══ 3. CLEAN ROOMS ═══

def undeploy_cleanrooms():
    print("\n[3/8] Removing Clean Rooms resources...")
    membership_id = collab_id = None

    # Find membership
    try:
        memberships = cr.list_memberships(status="ACTIVE")
        for m in memberships.get("membershipSummaries", []):
            if PREFIX in m.get("collaborationName", ""):
                membership_id = m["id"]
                collab_id     = m["collaborationId"]
                break
    except Exception as e:
        log(f"Could not list memberships (non-fatal): {e}")

    # Run Clean Rooms ML teardown first (needs membership_id)
    undeploy_cleanrooms_ml(membership_id)

    if not membership_id:
        skip("Clean Rooms membership/collaboration (not found)")
        return membership_id

    # Configured table associations
    try:
        assocs = cr.list_configured_table_associations(membershipIdentifier=membership_id)
        for a in assocs.get("configuredTableAssociationSummaries", []):
            try:
                cr.delete_configured_table_association(
                    membershipIdentifier=membership_id,
                    configuredTableAssociationIdentifier=a["id"]
                )
                log(f"Deleted table association: {a['name']}")
            except Exception as e:
                log(f"WARNING deleting table association {a['name']}: {e}")
    except Exception as e:
        log(f"Could not list table associations (non-fatal): {e}")

    # Delete membership
    try:
        cr.delete_membership(membershipIdentifier=membership_id)
        log(f"Deleted membership: {membership_id}")
        time.sleep(5)
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            skip(f"Membership {membership_id}")
        else:
            log(f"WARNING deleting membership: {e}")

    # Delete collaboration
    if collab_id:
        try:
            cr.delete_collaboration(collaborationIdentifier=collab_id)
            log(f"Deleted collaboration: {collab_id}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                skip(f"Collaboration {collab_id}")
            else:
                log(f"WARNING deleting collaboration: {e}")

    # Configured tables (global)
    try:
        tables = cr.list_configured_tables()
        for t in tables.get("configuredTableSummaries", []):
            if t["name"].startswith(PREFIX):
                try:
                    cr.delete_configured_table(configuredTableIdentifier=t["id"])
                    log(f"Deleted configured table: {t['name']}")
                except Exception as e:
                    log(f"WARNING deleting configured table {t['name']}: {e}")
    except Exception as e:
        log(f"Could not list configured tables (non-fatal): {e}")

    return membership_id


# ═══ 4. GLUE ═══

def undeploy_glue():
    print("\n[4/8] Removing Glue resources...")
    for table_name in [PHARMA_TABLE, INSURER_TABLE, INFERENCE_TABLE]:
        try:
            glue.delete_table(DatabaseName=GLUE_DB, Name=table_name)
            log(f"Deleted Glue table: {table_name}")
        except glue.exceptions.EntityNotFoundException:
            skip(f"Glue table {table_name}")
        except Exception as e:
            log(f"WARNING deleting Glue table {table_name}: {e}")

    try:
        glue.delete_database(Name=GLUE_DB)
        log(f"Deleted Glue database: {GLUE_DB}")
    except glue.exceptions.EntityNotFoundException:
        skip(f"Glue database {GLUE_DB}")
    except Exception as e:
        log(f"WARNING deleting Glue database: {e}")


# ═══ 5. IAM ═══

def undeploy_iam():
    print("\n[5/8] Removing IAM roles...")
    for role_name in IAM_ROLES:
        _delete_iam_role(role_name)


# ═══ 6. S3 ═══

def undeploy_s3():
    print("\n[6/8] Removing S3 buckets...")
    _delete_s3_bucket(BUCKET)
    _delete_s3_bucket(OUTPUT_BUCKET)


# ═══ 7. ECR ═══

def undeploy_ecr():
    print("\n[7/8] Removing ECR repositories...")
    _delete_ecr_repo(TRAINING_REPO)
    _delete_ecr_repo(INFERENCE_REPO)


# ═══ 8. CODEBUILD ═══

def undeploy_codebuild():
    print("\n[8/8] Removing CodeBuild project...")
    try:
        cb.delete_project(name=CB_PROJECT_NAME)
        log(f"Deleted CodeBuild project: {CB_PROJECT_NAME}")
    except cb.exceptions.ResourceNotFoundException:
        skip(f"CodeBuild project {CB_PROJECT_NAME}")
    except Exception as e:
        log(f"WARNING deleting CodeBuild project: {e}")


# ═══ MAIN ═══

def main():
    print("=" * 60)
    print("AWS Clean Rooms ML HCLS ADR — Undeploy / Teardown")
    print("=" * 60)
    print(f"Account: {AWS_ACCOUNT_ID}  Region: {AWS_REGION}")
    print(f"Prefix:  {PREFIX}")
    print()
    print("WARNING: This will permanently delete all HCLS ADR demo resources.")
    print("S3 buckets, ECR repos, and all data will be removed.")
    print()

    confirm = input("Type 'yes' to confirm teardown: ").strip().lower()
    if confirm != "yes":
        print("Aborted.")
        sys.exit(0)

    identity = sts.get_caller_identity()
    log(f"Authenticated as: {identity['Arn']}")

    undeploy_quicksight()
    undeploy_cleanrooms()   # also calls undeploy_cleanrooms_ml internally
    undeploy_glue()
    undeploy_iam()
    undeploy_s3()
    undeploy_ecr()
    undeploy_codebuild()

    # Clean up local run ID file so next run gets fresh bucket names
    run_id_file = os.path.join(os.path.dirname(__file__), "..", ".run_id")
    if os.path.exists(run_id_file):
        os.remove(run_id_file)
        log("Removed .run_id — next run will generate fresh bucket names")

    print("\n" + "=" * 60)
    print("Teardown complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
