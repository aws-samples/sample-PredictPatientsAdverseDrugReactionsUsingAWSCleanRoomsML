# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
AWS Clean Rooms ML HCLS ADR — Create QuickSight Dashboard (Step 6)
Reads all account/region config from config.py.

Run with: python scripts/create_dashboard.py  (from the project root folder)

Prerequisites:
  - run_cleanrooms_ml.py must have completed successfully
  - Inference output must exist at s3://{OUTPUT_BUCKET}/cleanrooms-ml-output/

What this script does (idempotent — safe to re-run):
  1. Register QuickSight account (skip if already exists)
  2. Register QuickSight admin user (skip if already exists)
  3. Create Glue table for inference output
  4. Create Athena data source in QuickSight
  5. Create QuickSight dataset
  6. Create analysis + publish dashboard (4 sheets)
  7. Print dashboard URL
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import *
validate(require_qs_email=True)

import boto3
from botocore.exceptions import ClientError

# ─── Clients ──────────────────────────────────────────────
session     = boto3.Session(region_name=AWS_REGION)
session_iam = boto3.Session(region_name="us-east-1")   # QS identity plane is global
qs          = session.client("quicksight")
qs_iam      = session_iam.client("quicksight")
glue        = session.client("glue")
s3          = session.client("s3")
iam         = session.client("iam")
sts         = session.client("sts")

# ─── Resource IDs ─────────────────────────────────────────
QS_DATASOURCE_ID = f"{PREFIX}-athena-source"
QS_DS_INFERENCE  = f"{PREFIX}-ds-inference"
QS_ANALYSIS_ID   = f"{PREFIX}-adr-analysis"
QS_DASHBOARD_ID  = f"{PREFIX}-adr-dashboard"
INFERENCE_TABLE  = "adr_inference_output"


def log(msg):
    print(f"  → {msg}")


# ═══════════════════════════════════════════════════════════
# SECTION 1 — QuickSight account registration
# ═══════════════════════════════════════════════════════════

def ensure_quicksight_account():
    """Register QuickSight ENTERPRISE account. No-op if already registered."""
    print("\n[1/6] Ensuring QuickSight account...")
    try:
        resp = qs_iam.describe_account_subscription(AwsAccountId=AWS_ACCOUNT_ID)
        status = resp["AccountInfo"]["AccountSubscriptionStatus"]
        log(f"QuickSight account already exists (status: {status})")
        if status not in ("ACCOUNT_CREATED", "ACTIVE"):
            print(f"  WARNING: QuickSight account status is '{status}'. Waiting up to 60s...")
            _wait_for_qs_account()
        return
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("ResourceNotFoundException", "AccessDeniedException"):
            raise

    log(f"Registering QuickSight ENTERPRISE account (email: {QS_NOTIFICATION_EMAIL})")
    try:
        qs_iam.create_account_subscription(
            AwsAccountId=AWS_ACCOUNT_ID,
            AccountName=f"{PREFIX}-{AWS_ACCOUNT_ID}",
            Edition="ENTERPRISE",
            AuthenticationMethod="IAM_AND_QUICKSIGHT",
            NotificationEmail=QS_NOTIFICATION_EMAIL,
        )
        log("QuickSight account registration submitted — waiting for activation...")
        _wait_for_qs_account()
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("ResourceExistsException", "ConflictException"):
            log("QuickSight account already exists (race condition) — continuing")
        else:
            raise


def _wait_for_qs_account(max_wait=120):
    for _ in range(max_wait // 10):
        try:
            resp = qs_iam.describe_account_subscription(AwsAccountId=AWS_ACCOUNT_ID)
            status = resp["AccountInfo"]["AccountSubscriptionStatus"]
            if status == "ACCOUNT_CREATED":
                log("QuickSight account is ACTIVE")
                return
            log(f"  Account status: {status} — waiting...")
        except ClientError:
            pass
        time.sleep(10)
    print("  WARNING: QuickSight account did not reach ACCOUNT_CREATED within timeout. Continuing anyway.")


# ═══════════════════════════════════════════════════════════
# SECTION 2 — QuickSight admin user registration
# ═══════════════════════════════════════════════════════════

def ensure_quicksight_user():
    """Register the current IAM caller as a QuickSight ADMIN user. No-op if exists."""
    print("\n[2/6] Ensuring QuickSight admin user...")

    identity = sts.get_caller_identity()
    caller_arn = identity["Arn"]
    arn_parts = caller_arn.split(":")
    raw_name  = arn_parts[-1]
    if raw_name.startswith("assumed-role/"):
        username = "/".join(raw_name.split("/")[1:])
    else:
        username = raw_name.split("/")[-1]

    try:
        qs_iam.describe_user(AwsAccountId=AWS_ACCOUNT_ID, Namespace="default", UserName=username)
        log(f"QuickSight user already exists: {username}")
        return username
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise

    log(f"Registering QuickSight ADMIN user: {username}")
    is_assumed_role = "assumed-role" in caller_arn
    if is_assumed_role:
        parts = caller_arn.split(":")
        role_parts = parts[-1].split("/")
        role_name    = role_parts[1]
        session_name = role_parts[2]
        role_arn = f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/{role_name}"
    else:
        role_arn     = caller_arn
        session_name = None

    qs_iam.register_user(
        AwsAccountId=AWS_ACCOUNT_ID,
        Namespace="default",
        IdentityType="IAM",
        IamArn=role_arn,
        UserRole="ADMIN",
        Email=QS_NOTIFICATION_EMAIL,
        **({"SessionName": session_name} if session_name else {}),
    )
    log(f"Registered user: {username}")
    return username


def _qs_user_arn(username):
    return f"arn:aws:quicksight:us-east-1:{AWS_ACCOUNT_ID}:user/default/{username}"


# ═══════════════════════════════════════════════════════════
# SECTION 3 — Glue table for inference output
# ═══════════════════════════════════════════════════════════

def prepare_glue_tables():
    """Register the inference output CSV as a Glue table (Athena-queryable)."""
    print("\n[3/6] Preparing Glue tables for dashboard data...")
    _ensure_glue_db()
    _register_inference_table()


def _ensure_glue_db():
    try:
        glue.create_database(DatabaseInput={"Name": GLUE_DB, "Description": "AWS Clean Rooms ML HCLS ADR demo"})
        log(f"Created Glue database: {GLUE_DB}")
    except glue.exceptions.AlreadyExistsException:
        log(f"Glue database already exists: {GLUE_DB}")


def _register_inference_table():
    """Register ADR inference output CSV location as a Glue external table."""
    columns = [
        {"Name": "adr_propensity_score",          "Type": "double"},
        {"Name": "predicted_adr",                 "Type": "int"},
        {"Name": "drug_class",                    "Type": "string"},
        {"Name": "therapy_line",                  "Type": "int"},
        {"Name": "known_risk_score",              "Type": "double"},
        {"Name": "black_box_warning",             "Type": "int"},
        {"Name": "patient_age",                   "Type": "int"},
        {"Name": "indication_severity",           "Type": "double"},
        {"Name": "reported_symptom_count",        "Type": "int"},
        {"Name": "symptom_severity_flag",         "Type": "int"},
        {"Name": "er_visits_post_start",          "Type": "int"},
        {"Name": "hospitalizations_post_start",   "Type": "int"},
        {"Name": "lab_abnormality_count",         "Type": "int"},
        {"Name": "comorbidity_index",             "Type": "double"},
        {"Name": "num_concomitant_meds",          "Type": "int"},
        {"Name": "drug_symptom_co_mention",       "Type": "int"},
        {"Name": "chief_complaint_adr_flag",      "Type": "int"},
    ]
    # Note: Glue table is updated via create_dashboard.py (idempotent update_table call)
    table_input = {
        "Name": INFERENCE_TABLE,
        "StorageDescriptor": {
            "Columns": columns,
            "Location": f"s3://{OUTPUT_BUCKET}/cleanrooms-ml-output/",
            "InputFormat":  "org.apache.hadoop.mapred.TextInputFormat",
            "OutputFormat": "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
            "SerdeInfo": {
                "SerializationLibrary": "org.apache.hadoop.hive.serde2.OpenCSVSerde",
                "Parameters": {
                    "separatorChar": ",",
                    "quoteChar": '"',
                    "skip.header.line.count": "1",
                },
            },
        },
        "TableType": "EXTERNAL_TABLE",
        "Parameters": {"classification": "csv"},
    }
    try:
        glue.create_table(DatabaseName=GLUE_DB, TableInput=table_input)
        log(f"Created Glue table: {INFERENCE_TABLE}")
    except glue.exceptions.AlreadyExistsException:
        glue.update_table(DatabaseName=GLUE_DB, TableInput=table_input)
        log(f"Updated Glue table: {INFERENCE_TABLE}")


# ═══════════════════════════════════════════════════════════
# SECTION 3b — Grant QuickSight access to S3 + Athena
# ═══════════════════════════════════════════════════════════

def ensure_quicksight_s3_access():
    """Grant QuickSight permission to read the output S3 bucket via managed service roles."""
    qs_service_role = "aws-quicksight-service-role-v0"
    qs_s3_role      = "aws-quicksight-s3-consumers-role-v0"

    policy_doc = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "HCLSADROutputBucketAccess",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:ListBucket", "s3:GetBucketLocation"],
                "Resource": [
                    f"arn:aws:s3:::{OUTPUT_BUCKET}",
                    f"arn:aws:s3:::{OUTPUT_BUCKET}/*",
                ],
            },
            {
                "Sid": "AthenaAccess",
                "Effect": "Allow",
                "Action": [
                    "athena:BatchGetQueryExecution", "athena:GetQueryExecution",
                    "athena:GetQueryResults", "athena:GetQueryResultsStream",
                    "athena:ListQueryExecutions", "athena:StartQueryExecution",
                    "athena:StopQueryExecution", "athena:GetWorkGroup",
                ],
                "Resource": "*",
            },
            {
                "Sid": "GlueAccess",
                "Effect": "Allow",
                "Action": [
                    "glue:GetDatabase", "glue:GetDatabases",
                    "glue:GetTable", "glue:GetTables",
                    "glue:GetPartition", "glue:GetPartitions",
                    "glue:BatchGetPartition",
                ],
                "Resource": [
                    f"arn:aws:glue:{AWS_REGION}:{AWS_ACCOUNT_ID}:catalog",
                    f"arn:aws:glue:{AWS_REGION}:{AWS_ACCOUNT_ID}:database/{GLUE_DB}",
                    f"arn:aws:glue:{AWS_REGION}:{AWS_ACCOUNT_ID}:table/{GLUE_DB}/*",
                ],
            },
            {
                "Sid": "AthenaResultsBucket",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket",
                           "s3:GetBucketLocation", "s3:AbortMultipartUpload"],
                "Resource": [
                    f"arn:aws:s3:::{OUTPUT_BUCKET}",
                    f"arn:aws:s3:::{OUTPUT_BUCKET}/*",
                ],
            },
        ],
    }

    for role_name in [qs_service_role, qs_s3_role]:
        try:
            iam.put_role_policy(
                RoleName=role_name,
                PolicyName="cleanrooms-ml-hcls-adr-qs-access",
                PolicyDocument=json.dumps(policy_doc),
            )
            log(f"Granted S3/Athena/Glue access to QuickSight role: {role_name}")
        except iam.exceptions.NoSuchEntityException:
            log(f"QuickSight role not found (skipping): {role_name}")
        except Exception as e:
            log(f"Could not update role {role_name} (non-fatal): {e}")


# ═══════════════════════════════════════════════════════════
# SECTION 4 — Athena data source
# ═══════════════════════════════════════════════════════════

def ensure_datasource(user_arn):
    """Create (or verify) the Athena data source in QuickSight."""
    print("\n[4/6] Ensuring QuickSight Athena data source...")
    try:
        qs.describe_data_source(AwsAccountId=AWS_ACCOUNT_ID, DataSourceId=QS_DATASOURCE_ID)
        log(f"Data source already exists: {QS_DATASOURCE_ID}")
        return
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise

    qs.create_data_source(
        AwsAccountId=AWS_ACCOUNT_ID,
        DataSourceId=QS_DATASOURCE_ID,
        Name="CleanRoomsML HCLS ADR — Athena",
        Type="ATHENA",
        DataSourceParameters={"AthenaParameters": {"WorkGroup": "primary"}},
        Permissions=[{
            "Principal": user_arn,
            "Actions": [
                "quicksight:DescribeDataSource",
                "quicksight:DescribeDataSourcePermissions",
                "quicksight:PassDataSource",
                "quicksight:UpdateDataSource",
                "quicksight:DeleteDataSource",
                "quicksight:UpdateDataSourcePermissions",
            ],
        }],
    )
    log(f"Created Athena data source: {QS_DATASOURCE_ID}")
    _wait_for_datasource()


def _wait_for_datasource(max_wait=60):
    for _ in range(max_wait // 5):
        resp = qs.describe_data_source(AwsAccountId=AWS_ACCOUNT_ID, DataSourceId=QS_DATASOURCE_ID)
        status = resp["DataSource"]["Status"]
        if status == "CREATION_SUCCESSFUL":
            log("Data source is ready")
            return
        if "FAILED" in status:
            raise RuntimeError(f"Data source creation failed: {status} — "
                               f"{resp['DataSource'].get('DataSourceErrorInfo', {})}")
        time.sleep(5)
    print("  WARNING: Data source did not reach CREATION_SUCCESSFUL within timeout.")


# ═══════════════════════════════════════════════════════════
# SECTION 5 — QuickSight dataset
# ═══════════════════════════════════════════════════════════

def _dataset_permissions(user_arn):
    return [{
        "Principal": user_arn,
        "Actions": [
            "quicksight:DescribeDataSet", "quicksight:DescribeDataSetPermissions",
            "quicksight:PassDataSet", "quicksight:DescribeIngestion",
            "quicksight:ListIngestions", "quicksight:UpdateDataSet",
            "quicksight:DeleteDataSet", "quicksight:CreateIngestion",
            "quicksight:CancelIngestion", "quicksight:UpdateDataSetPermissions",
        ],
    }]


def ensure_datasets(user_arn):
    """Create (or update) the ADR inference SPICE dataset."""
    print("\n[5/6] Ensuring QuickSight datasets...")
    datasource_arn = f"arn:aws:quicksight:{AWS_REGION}:{AWS_ACCOUNT_ID}:datasource/{QS_DATASOURCE_ID}"

    # Derived fields:
    #   risk_segment  — High/Medium/Low based on adr_propensity_score
    #   score_decile  — 1–10 decile bucket
    #   risk_exposure — lab_abnormality_count * adr_propensity_score (estimated clinical burden)
    sql = (
        f"SELECT adr_propensity_score, predicted_adr, "
        f"drug_class, therapy_line, known_risk_score, black_box_warning, "
        f"patient_age, indication_severity, reported_symptom_count, symptom_severity_flag, "
        f"er_visits_post_start, hospitalizations_post_start, lab_abnormality_count, "
        f"comorbidity_index, num_concomitant_meds, drug_symptom_co_mention, chief_complaint_adr_flag, "
        f"CASE WHEN adr_propensity_score > 0.35 THEN 'High' "
        f"     WHEN adr_propensity_score >= 0.15 THEN 'Medium' "
        f"     ELSE 'Low' END AS risk_segment, "
        f"NTILE(10) OVER (ORDER BY adr_propensity_score) AS score_decile, "
        f"lab_abnormality_count * adr_propensity_score AS risk_exposure "
        f"FROM {GLUE_DB}.{INFERENCE_TABLE}"
    )

    columns = [
        {"Name": "adr_propensity_score",        "Type": "DECIMAL"},
        {"Name": "predicted_adr",               "Type": "INTEGER"},
        {"Name": "drug_class",                  "Type": "STRING"},
        {"Name": "therapy_line",                "Type": "INTEGER"},
        {"Name": "known_risk_score",            "Type": "DECIMAL"},
        {"Name": "black_box_warning",           "Type": "INTEGER"},
        {"Name": "patient_age",                 "Type": "INTEGER"},
        {"Name": "indication_severity",         "Type": "DECIMAL"},
        {"Name": "reported_symptom_count",      "Type": "INTEGER"},
        {"Name": "symptom_severity_flag",       "Type": "INTEGER"},
        {"Name": "er_visits_post_start",        "Type": "INTEGER"},
        {"Name": "hospitalizations_post_start", "Type": "INTEGER"},
        {"Name": "lab_abnormality_count",       "Type": "INTEGER"},
        {"Name": "comorbidity_index",           "Type": "DECIMAL"},
        {"Name": "num_concomitant_meds",        "Type": "INTEGER"},
        {"Name": "drug_symptom_co_mention",     "Type": "INTEGER"},
        {"Name": "chief_complaint_adr_flag",    "Type": "INTEGER"},
        {"Name": "risk_segment",                "Type": "STRING"},
        {"Name": "score_decile",                "Type": "INTEGER"},
        {"Name": "risk_exposure",               "Type": "DECIMAL"},
    ]

    physical_id = f"{QS_DS_INFERENCE}-physical"
    logical_id  = f"{QS_DS_INFERENCE}-logical"
    ds_name     = "ADR Propensity Inference Output"

    physical_table_map = {
        physical_id: {
            "CustomSql": {
                "DataSourceArn": datasource_arn,
                "Name": ds_name,
                "SqlQuery": sql,
                "Columns": columns,
            }
        }
    }
    logical_table_map = {
        logical_id: {
            "Alias": ds_name,
            "Source": {"PhysicalTableId": physical_id},
        }
    }

    kwargs = dict(
        AwsAccountId=AWS_ACCOUNT_ID,
        DataSetId=QS_DS_INFERENCE,
        Name=ds_name,
        PhysicalTableMap=physical_table_map,
        LogicalTableMap=logical_table_map,
        ImportMode="DIRECT_QUERY",
        Permissions=_dataset_permissions(user_arn),
    )

    try:
        qs.describe_data_set(AwsAccountId=AWS_ACCOUNT_ID, DataSetId=QS_DS_INFERENCE)
        update_kwargs = {k: v for k, v in kwargs.items() if k != "Permissions"}
        qs.update_data_set(**update_kwargs)
        log(f"Updated dataset: {ds_name}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        qs.create_data_set(**kwargs)
        log(f"Created dataset: {ds_name}")


# ═══════════════════════════════════════════════════════════
# SECTION 6 — Dashboard visual helpers
# ═══════════════════════════════════════════════════════════

_DS = "adr"   # dataset alias used in DataSetIdentifierDeclarations


def _dataset_declarations():
    return [
        {"Identifier": _DS, "DataSetArn": f"arn:aws:quicksight:{AWS_REGION}:{AWS_ACCOUNT_ID}:dataset/{QS_DS_INFERENCE}"},
    ]


def _col(col_name):
    return {"DataSetIdentifier": _DS, "ColumnName": col_name}


def _num_measure(field_id, col_name, agg="AVERAGE"):
    return {"NumericalMeasureField": {
        "FieldId": field_id,
        "Column": _col(col_name),
        "AggregationFunction": {"SimpleNumericalAggregation": agg},
    }}


def _num_dim(field_id, col_name):
    return {"NumericalDimensionField": {"FieldId": field_id, "Column": _col(col_name)}}


def _cat_dim(field_id, col_name):
    return {"CategoricalDimensionField": {"FieldId": field_id, "Column": _col(col_name)}}


def _title(text):
    return {"Visibility": "VISIBLE", "FormatText": {"PlainText": text}}


def _subtitle(text):
    return {"Visibility": "VISIBLE", "FormatText": {"PlainText": text}}


# ── Sheet 1: Score Distribution ───────────────────────────

def _sheet1():
    histogram = {"BarChartVisual": {
        "VisualId": "bar-score-dist",
        "Title": _title("ADR Score Distribution by Decile"),
        "Subtitle": _subtitle("Patient count per ADR propensity score decile (1=lowest risk, 10=highest risk). A right-skewed distribution concentrates most patients in low-risk deciles. High counts in decile 9–10 indicate patients requiring proactive safety monitoring."),
        "ChartConfiguration": {
            "FieldWells": {
                "BarChartAggregatedFieldWells": {
                    "Category": [_num_dim("decile-dim", "score_decile")],
                    "Values":   [_num_measure("decile-cnt", "adr_propensity_score", "COUNT")],
                }
            },
            "Orientation": "VERTICAL",
            "SortConfiguration": {
                "CategorySort": [{"FieldSort": {"FieldId": "decile-dim", "Direction": "ASC"}}],
            },
        },
    }}

    donut = {"PieChartVisual": {
        "VisualId": "donut-risk-segments",
        "Title": _title("Patients by Risk Segment"),
        "Subtitle": _subtitle("Proportion of patients in High (score >0.7), Medium (0.3–0.7), and Low (<0.3) ADR risk segments. High-segment patients are candidates for enhanced pharmacovigilance outreach and clinical review."),
        "ChartConfiguration": {
            "FieldWells": {
                "PieChartAggregatedFieldWells": {
                    "Category": [_cat_dim("seg-dim", "risk_segment")],
                    "Values":   [_num_measure("seg-cnt", "adr_propensity_score", "COUNT")],
                }
            },
            "DonutOptions": {"ArcOptions": {"ArcThickness": "MEDIUM"}},
        },
    }}

    decile_table = {"TableVisual": {
        "VisualId": "tbl-decile-lift",
        "Title": _title("ADR Score Decile Lift Table"),
        "Subtitle": _subtitle("For each score decile: patient count, average ADR propensity score, and ADR flag rate. Higher deciles should show higher ADR rates — this is the model's lift over random patient selection."),
        "ChartConfiguration": {
            "FieldWells": {
                "TableAggregatedFieldWells": {
                    "GroupBy": [_num_dim("lift-decile", "score_decile")],
                    "Values":  [
                        _num_measure("lift-cnt",   "adr_propensity_score", "COUNT"),
                        _num_measure("lift-score", "adr_propensity_score", "AVERAGE"),
                        _num_measure("lift-adr",   "predicted_adr",        "AVERAGE"),
                    ],
                }
            },
            "SortConfiguration": {
                "RowSort": [{"FieldSort": {"FieldId": "lift-decile", "Direction": "ASC"}}],
            },
        },
    }}

    adr_bar = {"BarChartVisual": {
        "VisualId": "bar-adr-vs-clean",
        "Title": _title("Avg ADR Score: ADR vs Non-ADR Patients"),
        "Subtitle": _subtitle("Validates model discrimination: patients with confirmed ADR (label=1) should have a materially higher average ADR propensity score than non-ADR patients (label=0). A clear gap confirms the model is separating risk correctly."),
        "ChartConfiguration": {
            "FieldWells": {
                "BarChartAggregatedFieldWells": {
                    "Category": [_num_dim("adr-dim", "predicted_adr")],
                    "Values":   [_num_measure("adr-score", "adr_propensity_score", "AVERAGE")],
                }
            },
            "Orientation": "VERTICAL",
        },
    }}

    return {
        "SheetId": "sheet-1",
        "Name": "Score Distribution",
        "Visuals": [histogram, donut, decile_table, adr_bar],
    }


# ── Sheet 2: Risk Breakdown ───────────────────────────────

def _sheet2():
    drug_class_bar = {"BarChartVisual": {
        "VisualId": "bar-drug-class",
        "Title": _title("Avg ADR Score by Drug Class"),
        "Subtitle": _subtitle("Which drug classes are associated with the highest average ADR propensity. Chemotherapy and biologics typically show elevated risk — use this to calibrate drug-class-specific monitoring thresholds and patient support program targeting."),
        "ChartConfiguration": {
            "FieldWells": {
                "BarChartAggregatedFieldWells": {
                    "Category": [_cat_dim("dc-dim", "drug_class")],
                    "Values":   [_num_measure("dc-val", "adr_propensity_score", "AVERAGE")],
                }
            },
            "Orientation": "HORIZONTAL",
            "SortConfiguration": {
                "CategorySort": [{"FieldSort": {"FieldId": "dc-val", "Direction": "DESC"}}],
            },
        },
    }}

    therapy_line_bar = {"BarChartVisual": {
        "VisualId": "bar-therapy-line",
        "Title": _title("Avg ADR Score by Therapy Line"),
        "Subtitle": _subtitle("Average ADR propensity score by line of therapy. Later therapy lines (2nd, 3rd+) typically carry higher ADR risk due to prior treatment exposure and patient fragility — this validates that the model correctly weights therapy line as a risk signal."),
        "ChartConfiguration": {
            "FieldWells": {
                "BarChartAggregatedFieldWells": {
                    "Category": [_num_dim("tl-dim", "therapy_line")],
                    "Values":   [_num_measure("tl-val", "adr_propensity_score", "AVERAGE")],
                }
            },
            "Orientation": "VERTICAL",
            "SortConfiguration": {
                "CategorySort": [{"FieldSort": {"FieldId": "tl-dim", "Direction": "ASC"}}],
            },
        },
    }}

    comorbidity_scatter = {"ScatterPlotVisual": {
        "VisualId": "scatter-comorbidity-score",
        "Title": _title("Comorbidity Index vs ADR Propensity Score"),
        "Subtitle": _subtitle("Each point represents a group of patients with similar comorbidity burden. Patients with higher Charlson Comorbidity Index should cluster toward higher ADR scores — this validates that the model correctly weights patient fragility as a risk signal."),
        "ChartConfiguration": {
            "FieldWells": {
                "ScatterPlotCategoricallyAggregatedFieldWells": {
                    "XAxis":    [_num_measure("sc-x",  "comorbidity_index",      "AVERAGE")],
                    "YAxis":    [_num_measure("sc-y",  "adr_propensity_score",   "AVERAGE")],
                    "Category": [_cat_dim("sc-seg",    "risk_segment")],
                    "Size":     [_num_measure("sc-sz", "adr_propensity_score",   "COUNT")],
                }
            },
        },
    }}

    er_bar = {"BarChartVisual": {
        "VisualId": "bar-er-segment",
        "Title": _title("Avg ER Visits by Risk Segment"),
        "Subtitle": _subtitle("Average ER visits post drug start for each risk segment. High-risk patients should show more ER visits — this validates that the insurer-side outcome signals are correctly driving the risk score upward."),
        "ChartConfiguration": {
            "FieldWells": {
                "BarChartAggregatedFieldWells": {
                    "Category": [_cat_dim("er-seg", "risk_segment")],
                    "Values":   [_num_measure("er-val", "er_visits_post_start", "AVERAGE")],
                }
            },
            "Orientation": "VERTICAL",
        },
    }}

    return {
        "SheetId": "sheet-2",
        "Name": "Risk Breakdown",
        "Visuals": [drug_class_bar, therapy_line_bar, comorbidity_scatter, er_bar],
    }


# ── Sheet 3: Patient & Drug Analysis ─────────────────────

def _sheet3():
    segment_table = {"TableVisual": {
        "VisualId": "tbl-segment-summary",
        "Title": _title("Risk Segment Summary"),
        "Subtitle": _subtitle("Aggregated view of High, Medium, and Low ADR risk segments. Shows patient count, average ADR score, ADR flag rate, average lab abnormalities, and average ER visits. Use filters to drill into specific drug classes or risk segments."),
        "ChartConfiguration": {
            "FieldWells": {
                "TableAggregatedFieldWells": {
                    "GroupBy": [_cat_dim("seg-grp", "risk_segment")],
                    "Values":  [
                        _num_measure("seg-cnt",   "adr_propensity_score",        "COUNT"),
                        _num_measure("seg-score", "adr_propensity_score",        "AVERAGE"),
                        _num_measure("seg-adr",   "predicted_adr",               "AVERAGE"),
                        _num_measure("seg-lab",   "lab_abnormality_count",       "AVERAGE"),
                        _num_measure("seg-er",    "er_visits_post_start",        "AVERAGE"),
                    ],
                }
            },
        },
    }}

    top_records = {"TableVisual": {
        "VisualId": "tbl-top-risk-patients",
        "Title": _title("Highest Risk Patients"),
        "Subtitle": _subtitle("Individual patient records ranked by ADR propensity score (highest first). Each row is one patient-drug combination. Use this list to prioritise patients for pharmacist outreach, clinical review, or patient support program enrollment."),
        "ChartConfiguration": {
            "FieldWells": {
                "TableUnaggregatedFieldWells": {
                    "Values": [
                        {"FieldId": "tr-score",   "Column": _col("adr_propensity_score")},
                        {"FieldId": "tr-adr",     "Column": _col("predicted_adr")},
                        {"FieldId": "tr-class",   "Column": _col("drug_class")},
                        {"FieldId": "tr-line",    "Column": _col("therapy_line")},
                        {"FieldId": "tr-age",     "Column": _col("patient_age")},
                        {"FieldId": "tr-comor",   "Column": _col("comorbidity_index")},
                        {"FieldId": "tr-lab",     "Column": _col("lab_abnormality_count")},
                        {"FieldId": "tr-er",      "Column": _col("er_visits_post_start")},
                        {"FieldId": "tr-hosp",    "Column": _col("hospitalizations_post_start")},
                    ]
                }
            },
            "SortConfiguration": {
                "RowSort": [{"FieldSort": {"FieldId": "tr-score", "Direction": "DESC"}}],
            },
            "PaginatedReportOptions": {"VerticalOverflowVisibility": "VISIBLE"},
        },
    }}

    pivot = {"PivotTableVisual": {
        "VisualId": "pivot-segment-drugclass",
        "Title": _title("ADR Rate: Risk Segment × Drug Class"),
        "Subtitle": _subtitle("Cross-tab of risk segment (rows) vs drug class (columns), showing average ADR flag rate per cell. Identifies which segment + drug class combinations carry the highest ADR density — useful for targeted pharmacovigilance intervention policies."),
        "ChartConfiguration": {
            "FieldWells": {
                "PivotTableAggregatedFieldWells": {
                    "Rows":    [_cat_dim("pt-seg",   "risk_segment")],
                    "Columns": [_cat_dim("pt-class", "drug_class")],
                    "Values":  [_num_measure("pt-adr", "predicted_adr", "AVERAGE")],
                }
            },
        },
    }}

    return {
        "SheetId": "sheet-3",
        "Name": "Patient & Drug Analysis",
        "Visuals": [segment_table, top_records, pivot],
    }

# ── Sheet 4: Business Impact ──────────────────────────────

def _sheet4():
    gains_bar = {"BarChartVisual": {
        "VisualId": "bar-adr-by-decile",
        "Title": _title("ADR Patients Captured by Score Decile"),
        "Subtitle": _subtitle("How many flagged patients (predicted_adr=1) fall in each score decile. A well-calibrated model concentrates ADR patients in the top deciles (9–10). This shows what fraction of all ADR cases you capture if you only review the top N deciles."),
        "ChartConfiguration": {
            "FieldWells": {
                "BarChartAggregatedFieldWells": {
                    "Category": [_num_dim("cg-decile", "score_decile")],
                    "Values":   [_num_measure("cg-adr", "predicted_adr", "SUM")],
                }
            },
            "Orientation": "VERTICAL",
            "SortConfiguration": {
                "CategorySort": [{"FieldSort": {"FieldId": "cg-decile", "Direction": "ASC"}}],
            },
        },
    }}

    exposure_bar = {"BarChartVisual": {
        "VisualId": "bar-risk-exposure-segment",
        "Title": _title("Estimated Clinical Risk Exposure by Segment"),
        "Subtitle": _subtitle("Estimated clinical burden per segment, calculated as lab_abnormality_count × adr_propensity_score. High-segment patients represent the largest potential safety burden — use this to size the business case for enhanced monitoring investment."),
        "ChartConfiguration": {
            "FieldWells": {
                "BarChartAggregatedFieldWells": {
                    "Category": [_cat_dim("exp-seg", "risk_segment")],
                    "Values":   [_num_measure("exp-val", "risk_exposure", "SUM")],
                }
            },
            "Orientation": "HORIZONTAL",
            "SortConfiguration": {
                "CategorySort": [{"FieldSort": {"FieldId": "exp-val", "Direction": "DESC"}}],
            },
        },
    }}

    heatmap = {"HeatMapVisual": {
        "VisualId": "heatmap-drugclass-age",
        "Title": _title("Avg ADR Score: Drug Class × Patient Age Group"),
        "Subtitle": _subtitle("Heatmap of average ADR propensity score across drug classes and patient age groups. Darker cells indicate higher ADR risk. Use this to identify the most vulnerable patient-drug combinations for proactive pharmacovigilance intervention."),
        "ChartConfiguration": {
            "FieldWells": {
                "HeatMapAggregatedFieldWells": {
                    "Rows":    [_cat_dim("hm-class",  "drug_class")],
                    "Columns": [_num_dim("hm-age",    "patient_age")],
                    "Values":  [_num_measure("hm-val", "adr_propensity_score", "AVERAGE")],
                }
            },
        },
    }}

    return {
        "SheetId": "sheet-4",
        "Name": "Business Impact",
        "Visuals": [gains_bar, exposure_bar, heatmap],
    }


# ═══════════════════════════════════════════════════════════
# SECTION 7 — Build dashboard definition + filters
# ═══════════════════════════════════════════════════════════

def _build_definition():
    return {
        "DataSetIdentifierDeclarations": _dataset_declarations(),
        "Sheets": [_sheet1(), _sheet2(), _sheet3(), _sheet4()],
        "FilterGroups": [
            _filter_group("fg-drugclass", "drug_class",   _DS),
            _filter_group("fg-segment",   "risk_segment", _DS),
        ],
    }


def _filter_group(fg_id, col_name, ds_alias):
    """CategoryFilter group scoped to ALL_VISUALS on sheets 2–4."""
    return {
        "FilterGroupId": fg_id,
        "Filters": [{
            "CategoryFilter": {
                "FilterId": f"{fg_id}-filter",
                "Column": {"DataSetIdentifier": ds_alias, "ColumnName": col_name},
                "Configuration": {
                    "FilterListConfiguration": {
                        "MatchOperator": "CONTAINS",
                        "SelectAllOptions": "FILTER_ALL_VALUES",
                    }
                },
            }
        }],
        "ScopeConfiguration": {
            "SelectedSheets": {
                "SheetVisualScopingConfigurations": [
                    {"SheetId": sid, "Scope": "ALL_VISUALS"}
                    for sid in ["sheet-2", "sheet-3", "sheet-4"]
                ]
            }
        },
        "Status": "ENABLED",
        "CrossDataset": "SINGLE_DATASET",
    }


def _analysis_permissions(user_arn):
    return [{
        "Principal": user_arn,
        "Actions": [
            "quicksight:DescribeAnalysis",
            "quicksight:DescribeAnalysisPermissions",
            "quicksight:UpdateAnalysis",
            "quicksight:UpdateAnalysisPermissions",
            "quicksight:DeleteAnalysis",
            "quicksight:RestoreAnalysis",
            "quicksight:QueryAnalysis",
        ],
    }]


def _dashboard_permissions(user_arn):
    return [{
        "Principal": user_arn,
        "Actions": [
            "quicksight:DescribeDashboard",
            "quicksight:ListDashboardVersions",
            "quicksight:UpdateDashboardPermissions",
            "quicksight:QueryDashboard",
            "quicksight:UpdateDashboard",
            "quicksight:DeleteDashboard",
            "quicksight:DescribeDashboardPermissions",
            "quicksight:UpdateDashboardPublishedVersion",
        ],
    }]


# ═══════════════════════════════════════════════════════════
# SECTION 8 — Create analysis + publish dashboard
# ═══════════════════════════════════════════════════════════

def ensure_dashboard(user_arn):
    """Create or update the QuickSight analysis and dashboard."""
    print("\n[6/6] Creating QuickSight analysis and dashboard...")

    definition = _build_definition()
    dashboard_name = "Cross-Party ADR Propensity Scoring"

    # ── Analysis ──
    try:
        qs.describe_analysis(AwsAccountId=AWS_ACCOUNT_ID, AnalysisId=QS_ANALYSIS_ID)
        qs.update_analysis(
            AwsAccountId=AWS_ACCOUNT_ID,
            AnalysisId=QS_ANALYSIS_ID,
            Name=dashboard_name,
            Definition=definition,
        )
        log(f"Updated analysis: {QS_ANALYSIS_ID}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        qs.create_analysis(
            AwsAccountId=AWS_ACCOUNT_ID,
            AnalysisId=QS_ANALYSIS_ID,
            Name=dashboard_name,
            Definition=definition,
            Permissions=_analysis_permissions(user_arn),
        )
        log(f"Created analysis: {QS_ANALYSIS_ID}")

    _wait_for_analysis()

    # ── Dashboard ──
    publish_opts = {
        "AdHocFilteringOption":  {"AvailabilityStatus": "ENABLED"},
        "ExportToCSVOption":     {"AvailabilityStatus": "ENABLED"},
        "VisualPublishOptions":  {"ExportHiddenFieldsOption": {"AvailabilityStatus": "DISABLED"}},
    }
    try:
        qs.describe_dashboard(AwsAccountId=AWS_ACCOUNT_ID, DashboardId=QS_DASHBOARD_ID)
        qs.update_dashboard(
            AwsAccountId=AWS_ACCOUNT_ID,
            DashboardId=QS_DASHBOARD_ID,
            Name=dashboard_name,
            Definition=definition,
            DashboardPublishOptions=publish_opts,
        )
        log(f"Updated dashboard: {QS_DASHBOARD_ID}")
        resp = qs.describe_dashboard(AwsAccountId=AWS_ACCOUNT_ID, DashboardId=QS_DASHBOARD_ID)
        latest = resp["Dashboard"]["Version"]["VersionNumber"]
        qs.update_dashboard_published_version(
            AwsAccountId=AWS_ACCOUNT_ID,
            DashboardId=QS_DASHBOARD_ID,
            VersionNumber=latest,
        )
        log(f"Published dashboard version: {latest}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        qs.create_dashboard(
            AwsAccountId=AWS_ACCOUNT_ID,
            DashboardId=QS_DASHBOARD_ID,
            Name=dashboard_name,
            Definition=definition,
            Permissions=_dashboard_permissions(user_arn),
            DashboardPublishOptions=publish_opts,
        )
        log(f"Created dashboard: {QS_DASHBOARD_ID}")


def _wait_for_analysis(max_wait=120):
    for _ in range(max_wait // 5):
        resp = qs.describe_analysis(AwsAccountId=AWS_ACCOUNT_ID, AnalysisId=QS_ANALYSIS_ID)
        status = resp["Analysis"]["Status"]
        if status in ("CREATION_SUCCESSFUL", "UPDATE_SUCCESSFUL"):
            log(f"Analysis status: {status}")
            return
        if "FAILED" in status:
            errors = resp["Analysis"].get("Errors", [])
            raise RuntimeError(f"Analysis failed ({status}): {errors}")
        time.sleep(5)
    print("  WARNING: Analysis did not reach SUCCESSFUL status within timeout.")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("AWS Clean Rooms ML HCLS ADR — Create QuickSight Dashboard")
    print("=" * 60)
    print(f"Account: {AWS_ACCOUNT_ID}  Region: {AWS_REGION}")
    print(f"Output bucket: {OUTPUT_BUCKET}")

    identity = sts.get_caller_identity()
    log(f"Authenticated as: {identity['Arn']}")

    ensure_quicksight_account()
    username = ensure_quicksight_user()
    user_arn = _qs_user_arn(username)

    prepare_glue_tables()
    ensure_quicksight_s3_access()
    ensure_datasource(user_arn)
    ensure_datasets(user_arn)
    ensure_dashboard(user_arn)

    dashboard_url = (
        f"https://{AWS_REGION}.quicksight.aws.amazon.com"
        f"/sn/dashboards/{QS_DASHBOARD_ID}"
    )
    print("\n" + "=" * 60)
    print("Dashboard ready!")
    print("=" * 60)
    print(f"\n  {dashboard_url}")
    print(f"\n  Note: If visuals show 'No data', wait ~2 min and refresh.")
    print(f"\nNext: open the URL above in your browser.")


if __name__ == "__main__":
    main()
