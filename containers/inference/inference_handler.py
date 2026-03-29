# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Inference handler for HCLS ADR Propensity Scoring model.
Compatible with: local, SageMaker Batch Transform, Clean Rooms ML.
"""

import os, json, logging, io
import pandas as pd
import numpy as np
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
_model = None
_feature_cols = None

# Columns as they arrive from Clean Rooms ML (pre-joined, no patient_id)
# Order matches the SELECT DISTINCT in run_cleanrooms_ml.py
CLEANROOMS_COLUMNS = [
    "drug_id", "drug_class", "dose_mg", "treatment_duration_days",
    "therapy_line", "known_risk_score", "black_box_warning", "patient_age",
    "indication_severity", "reported_symptom_count", "symptom_severity_flag",
    "time_to_onset_days", "concomitant_drug_count_reported", "prior_adr_narrative_flag",
    "er_visits_post_start", "hospitalizations_post_start", "days_to_first_er_visit",
    "drug_discontinuation", "days_to_discontinuation", "num_concomitant_meds",
    "high_risk_combo", "lab_abnormality_count", "comorbidity_index",
    "prior_hospitalization", "symptom_mention_count", "drug_symptom_co_mention",
    "negated_symptom_count", "lab_abnormality_mentioned", "chief_complaint_adr_flag",
    "has_adr",
]

# Columns to pass through to the output for dashboard use
PASSTHROUGH_COLS = [
    "drug_class", "therapy_line", "known_risk_score", "black_box_warning",
    "patient_age", "indication_severity", "reported_symptom_count", "symptom_severity_flag",
    "er_visits_post_start", "hospitalizations_post_start", "lab_abnormality_count",
    "comorbidity_index", "num_concomitant_meds", "drug_symptom_co_mention",
    "chief_complaint_adr_flag",
]


def load_model():
    global _model, _feature_cols
    if _model is not None:
        return _model, _feature_cols

    search_dirs = [MODEL_DIR, "/opt/ml/model", "/opt/ml/input/data/model"]
    model_path = features_path = None

    for d in search_dirs:
        if os.path.exists(d):
            candidate = os.path.join(d, "model.joblib")
            if os.path.exists(candidate):
                model_path = candidate
                features_path = os.path.join(d, "feature_columns.json")
                break

    if model_path is None:
        for root, dirs, files in os.walk("/opt/ml"):
            if "model.joblib" in files:
                model_path = os.path.join(root, "model.joblib")
                features_path = os.path.join(root, "feature_columns.json")
                break

    if model_path is None:
        raise FileNotFoundError(f"model.joblib not found in any of: {search_dirs}")

    _model = joblib.load(model_path)
    if features_path and os.path.exists(features_path):
        with open(features_path, "r") as f:
            _feature_cols = json.load(f)
    else:
        # Fallback feature list matching the pre-joined path in train.py
        _feature_cols = [
            "dose_mg", "treatment_duration_days", "therapy_line", "known_risk_score",
            "black_box_warning", "patient_age", "indication_severity",
            "reported_symptom_count", "symptom_severity_flag", "time_to_onset_days",
            "concomitant_drug_count_reported", "prior_adr_narrative_flag",
            "er_visits_post_start", "hospitalizations_post_start", "days_to_first_er_visit",
            "drug_discontinuation", "days_to_discontinuation", "num_concomitant_meds",
            "high_risk_combo", "lab_abnormality_count", "comorbidity_index",
            "prior_hospitalization", "symptom_mention_count", "drug_symptom_co_mention",
            "negated_symptom_count", "lab_abnormality_mentioned", "chief_complaint_adr_flag",
            "dose_duration_interaction", "comorbidity_drug_burden",
        ]
    logger.info(f"Model loaded. Features: {_feature_cols}")
    return _model, _feature_cols


MAX_INPUT_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB safety limit


def predict(input_data, content_type="text/csv"):
    model, feature_cols = load_model()

    if len(input_data.encode("utf-8")) > MAX_INPUT_SIZE_BYTES:
        raise ValueError(f"Input exceeds maximum allowed size of {MAX_INPUT_SIZE_BYTES} bytes")

    try:
        if content_type == "application/json":
            df = pd.read_json(io.StringIO(input_data))
        else:
            df = pd.read_csv(io.StringIO(input_data))
            # Detect headerless Clean Rooms ML output
            first_col = str(df.columns[0])
            if first_col not in ["drug_id", "drug_class", "dose_mg", "patient_id"] \
                    and len(df.columns) == len(CLEANROOMS_COLUMNS):
                df = pd.read_csv(io.StringIO(input_data), header=None, names=CLEANROOMS_COLUMNS)
    except Exception as e:
        raise ValueError(f"Failed to parse input data ({content_type}): {e}")

    if df.empty:
        raise ValueError("Input data is empty")

    # Compute derived features
    if "dose_duration_interaction" not in df.columns:
        df["dose_duration_interaction"] = (
            df.get("dose_mg", 0) * df.get("treatment_duration_days", 0) / 1000.0
        ).clip(lower=0)
    if "comorbidity_drug_burden" not in df.columns:
        df["comorbidity_drug_burden"] = (
            df.get("comorbidity_index", 0) * df.get("num_concomitant_meds", 0)
        ).clip(lower=0)

    # One-hot encode drug_class if present (must match training encoding)
    if "drug_class" in df.columns:
        df = pd.get_dummies(df, columns=["drug_class"], drop_first=True, dtype=int)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].fillna(0)
    probabilities = model.predict_proba(X)[:, 1]
    # Use a calibrated threshold matching the score distribution (~28% ADR rate)
    # Default sklearn threshold of 0.5 is too high when scores peak at 0.1-0.3
    threshold = 0.25
    predictions = (probabilities >= threshold).astype(int)

    # Build output with score + prediction + passthrough contextual columns
    result = pd.DataFrame({
        "adr_propensity_score": np.round(probabilities, 4),
        "predicted_adr":        predictions.astype(int),
    })

    # Re-read original input to get passthrough columns (before get_dummies)
    try:
        if content_type == "application/json":
            orig_df = pd.read_json(io.StringIO(input_data))
        else:
            orig_df = pd.read_csv(io.StringIO(input_data))
            if str(orig_df.columns[0]) not in ["drug_id", "drug_class", "dose_mg", "patient_id"] \
                    and len(orig_df.columns) == len(CLEANROOMS_COLUMNS):
                orig_df = pd.read_csv(io.StringIO(input_data), header=None, names=CLEANROOMS_COLUMNS)
    except Exception:
        orig_df = pd.DataFrame()

    for col in PASSTHROUGH_COLS:
        if col in orig_df.columns:
            result[col] = orig_df[col].values

    logger.info(f"Output shape: {result.shape}, columns: {list(result.columns)}")
    return result.to_csv(index=False)
