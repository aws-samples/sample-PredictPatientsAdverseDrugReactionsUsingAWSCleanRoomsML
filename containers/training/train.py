# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Training script for HCLS ADR Propensity Scoring model.
Compatible with: local testing, SageMaker AI Training, and AWS Clean Rooms ML.
"""

import argparse, os, sys, json, glob, traceback, logging
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Columns as they arrive from Clean Rooms ML (pre-joined, no patient_id)
# Order matches the SELECT in run_cleanrooms_ml.py
CLEANROOMS_COLUMNS = [
    # Pharma company columns
    "drug_id", "drug_class", "dose_mg", "treatment_duration_days",
    "therapy_line", "known_risk_score", "black_box_warning", "patient_age",
    "indication_severity", "reported_symptom_count", "symptom_severity_flag",
    "time_to_onset_days", "concomitant_drug_count_reported", "prior_adr_narrative_flag",
    # Health insurer columns
    "er_visits_post_start", "hospitalizations_post_start", "days_to_first_er_visit",
    "drug_discontinuation", "days_to_discontinuation", "num_concomitant_meds",
    "high_risk_combo", "lab_abnormality_count", "comorbidity_index",
    "prior_hospitalization", "symptom_mention_count", "drug_symptom_co_mention",
    "negated_symptom_count", "lab_abnormality_mentioned", "chief_complaint_adr_flag",
    # Label
    "has_adr",
]

# Pharma-side identifier columns (not features)
PHARMA_ID_COLS = {"patient_id", "drug_id", "drug_class", "observation_date"}
# Insurer-side identifier columns (not features)
INSURER_ID_COLS = {"patient_id", "drug_id"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output_dir", type=str, default=os.environ.get("SM_OUTPUT_DIR", "/opt/ml/output/data"))
    parser.add_argument("--train_dir", type=str,
                        default=os.environ.get("SM_CHANNEL_TRAIN",
                                os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")))
    parser.add_argument("--train_file_format", type=str, default=os.environ.get("FILE_FORMAT", "csv"))
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.2)
    return parser.parse_args()


def load_data(train_dir, file_format):
    logger.info(f"Loading data from {train_dir} (format: {file_format})")
    if os.path.exists(train_dir):
        for root, dirs, files in os.walk(train_dir):
            logger.info(f"  Dir: {root}, subdirs: {dirs}, files: {files}")
    else:
        alternatives = ["/opt/ml/input/data/training", "/opt/ml/input/data/train", "/opt/ml/input/data"]
        for alt in alternatives:
            if os.path.exists(alt):
                train_dir = alt
                break
        else:
            raise FileNotFoundError(f"No training data directory found. Tried: {train_dir}, {alternatives}")

    all_files = []
    for root, dirs, files in os.walk(train_dir):
        for f in files:
            if f.endswith(f".{file_format}") or not os.path.splitext(f)[1]:
                all_files.append(os.path.join(root, f))
    if not all_files:
        all_files = [f for f in glob.glob(os.path.join(train_dir, "**/*"), recursive=True) if os.path.isfile(f)]
    if not all_files:
        raise FileNotFoundError(f"No data files found in {train_dir}")

    logger.info(f"Found {len(all_files)} files: {all_files}")
    dataframes = {}
    for filepath in all_files:
        name = os.path.basename(filepath).replace(f".{file_format}", "")
        try:
            if file_format == "csv":
                df = pd.read_csv(filepath)
                first_col = str(df.columns[0])
                # Detect headerless Clean Rooms ML output (no patient_id, starts with drug_id)
                is_headerless = (
                    first_col not in ["patient_id", "drug_id", "drug_class",
                                      "dose_mg", "er_visits_post_start"]
                    and len(df.columns) == len(CLEANROOMS_COLUMNS)
                )
                if is_headerless:
                    df = pd.read_csv(filepath, header=None, names=CLEANROOMS_COLUMNS)
                elif len(df.columns) == len(CLEANROOMS_COLUMNS) - 1:
                    df = pd.read_csv(filepath, header=None, names=CLEANROOMS_COLUMNS)
            else:
                df = pd.read_parquet(filepath)
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
            raise ValueError(f"Could not parse data file {filepath} as {file_format}: {e}")
        dataframes[name] = df
        logger.info(f"  Loaded {name}: {df.shape}")
    return dataframes


def engineer_features(dataframes):
    """Route to pre-joined or separate-file feature engineering."""
    pre_joined_df = None
    for name, df in dataframes.items():
        has_pharma  = "known_risk_score" in df.columns or "reported_symptom_count" in df.columns
        has_insurer = "er_visits_post_start" in df.columns or "lab_abnormality_count" in df.columns
        if has_pharma and has_insurer:
            pre_joined_df = df
            break

    if pre_joined_df is not None:
        return _engineer_features_prejoined(pre_joined_df)
    return _engineer_features_separate(dataframes)


def _derive_features(df):
    """Compute derived features common to both code paths."""
    # dose × duration interaction — captures cumulative exposure risk
    df["dose_duration_interaction"] = (
        df["dose_mg"] * df["treatment_duration_days"] / 1000.0
    ).clip(lower=0)
    # comorbidity × drug burden — interaction between patient fragility and polypharmacy
    df["comorbidity_drug_burden"] = (
        df["comorbidity_index"] * df["num_concomitant_meds"]
    ).clip(lower=0)
    return df


def _encode_categoricals(df):
    """One-hot encode low-cardinality categorical columns."""
    cat_cols = [c for c in ["drug_class"] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    return df


def _engineer_features_prejoined(df):
    """Feature engineering when data arrives pre-joined (Clean Rooms ML path)."""
    df = _derive_features(df)
    df = _encode_categoricals(df)

    # All numeric columns except identifiers and label
    exclude = {"patient_id", "drug_id", "observation_date", "has_adr"}
    feature_cols = [c for c in df.columns if c not in exclude
                    and df[c].dtype in [np.int64, np.float64, np.int32, np.float32, int, float]]

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    X = df[feature_cols].fillna(0)
    y = df["has_adr"] if "has_adr" in df.columns else pd.Series(0, index=df.index)
    logger.info(f"Features shape: {X.shape}, Target dist:\n{y.value_counts().to_string()}")
    return X, y, feature_cols, None


def _engineer_features_separate(dataframes):
    """Feature engineering when pharma and insurer CSVs are separate files (local test path)."""
    pharma_df = insurer_df = None
    for name, df in dataframes.items():
        if "known_risk_score" in df.columns or "reported_symptom_count" in df.columns:
            pharma_df = df
        elif "er_visits_post_start" in df.columns or "lab_abnormality_count" in df.columns:
            insurer_df = df
    if pharma_df is None or insurer_df is None:
        raise ValueError(
            f"Could not identify both datasets. Columns: {[list(df.columns) for df in dataframes.values()]}"
        )

    # Aggregate pharma data per patient (multiple drug rows → one row per patient)
    pharma_agg = pharma_df.groupby("patient_id").agg(
        avg_dose_mg=("dose_mg", "mean"),
        max_treatment_duration_days=("treatment_duration_days", "max"),
        max_therapy_line=("therapy_line", "max"),
        avg_known_risk_score=("known_risk_score", "mean"),
        max_black_box_warning=("black_box_warning", "max"),
        patient_age=("patient_age", "first"),
        avg_indication_severity=("indication_severity", "mean"),
        total_reported_symptom_count=("reported_symptom_count", "sum"),
        max_symptom_severity_flag=("symptom_severity_flag", "max"),
        min_time_to_onset_days=("time_to_onset_days", "min"),
        avg_concomitant_drug_count_reported=("concomitant_drug_count_reported", "mean"),
        max_prior_adr_narrative_flag=("prior_adr_narrative_flag", "max"),
        num_drugs=("drug_id", "nunique"),
    ).reset_index()

    # Aggregate insurer data per patient
    insurer_agg = insurer_df.groupby("patient_id").agg(
        total_er_visits=("er_visits_post_start", "sum"),
        total_hospitalizations=("hospitalizations_post_start", "sum"),
        min_days_to_first_er=("days_to_first_er_visit", "min"),
        max_drug_discontinuation=("drug_discontinuation", "max"),
        min_days_to_discontinuation=("days_to_discontinuation", "min"),
        avg_num_concomitant_meds=("num_concomitant_meds", "mean"),
        max_high_risk_combo=("high_risk_combo", "max"),
        total_lab_abnormality_count=("lab_abnormality_count", "sum"),
        avg_comorbidity_index=("comorbidity_index", "mean"),
        max_prior_hospitalization=("prior_hospitalization", "max"),
        total_symptom_mention_count=("symptom_mention_count", "sum"),
        max_drug_symptom_co_mention=("drug_symptom_co_mention", "max"),
        total_negated_symptom_count=("negated_symptom_count", "sum"),
        max_lab_abnormality_mentioned=("lab_abnormality_mentioned", "max"),
        max_chief_complaint_adr_flag=("chief_complaint_adr_flag", "max"),
    ).reset_index()

    # Extract target from insurer data (max across drug rows per patient)
    if "has_adr" in insurer_df.columns:
        target = insurer_df.groupby("patient_id")["has_adr"].max().reset_index()
        insurer_agg = insurer_agg.merge(target, on="patient_id")
    else:
        insurer_agg["has_adr"] = 0

    merged = pharma_agg.merge(insurer_agg, on="patient_id", how="inner")

    # Derived features
    merged["dose_duration_interaction"] = (
        merged["avg_dose_mg"] * merged["max_treatment_duration_days"] / 1000.0
    ).clip(lower=0)
    merged["comorbidity_drug_burden"] = (
        merged["avg_comorbidity_index"] * merged["avg_num_concomitant_meds"]
    ).clip(lower=0)

    feature_cols = [
        "avg_dose_mg", "max_treatment_duration_days", "max_therapy_line",
        "avg_known_risk_score", "max_black_box_warning", "patient_age",
        "avg_indication_severity", "total_reported_symptom_count",
        "max_symptom_severity_flag", "min_time_to_onset_days",
        "avg_concomitant_drug_count_reported", "max_prior_adr_narrative_flag",
        "num_drugs",
        "total_er_visits", "total_hospitalizations", "min_days_to_first_er",
        "max_drug_discontinuation", "min_days_to_discontinuation",
        "avg_num_concomitant_meds", "max_high_risk_combo",
        "total_lab_abnormality_count", "avg_comorbidity_index",
        "max_prior_hospitalization", "total_symptom_mention_count",
        "max_drug_symptom_co_mention", "total_negated_symptom_count",
        "max_lab_abnormality_mentioned", "max_chief_complaint_adr_flag",
        "dose_duration_interaction", "comorbidity_drug_burden",
    ]
    X = merged[feature_cols].fillna(0)
    y = merged["has_adr"]
    logger.info(f"Merged: {merged.shape[0]} patients, Features: {X.shape}")
    return X, y, feature_cols, merged["patient_id"]


def train_model(X, y, args):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    logger.info(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    model = GradientBoostingClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        learning_rate=args.learning_rate, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy":      round(accuracy_score(y_test, y_pred), 4),
        "precision":     round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":        round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":            round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":       round(roc_auc_score(y_test, y_proba), 4),
        "train_samples": X_train.shape[0],
        "test_samples":  X_test.shape[0],
        "n_features":    X_train.shape[1],
    }
    importance = dict(zip(X.columns, [round(float(v), 4) for v in model.feature_importances_]))
    metrics["feature_importance"] = dict(sorted(importance.items(), key=lambda x: -x[1]))
    logger.info(f"Metrics: {json.dumps({k: v for k, v in metrics.items() if k != 'feature_importance'}, indent=2)}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    return model, metrics


def save_artifacts(model, metrics, feature_cols, model_dir, output_dir):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
    with open(os.path.join(model_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Artifacts saved to {model_dir} and {output_dir}")


def main():
    args = parse_args()
    logger.info(f"Arguments: {vars(args)}")
    try:
        sm_vars = {k: v for k, v in os.environ.items() if k.startswith("SM_") or k.startswith("FILE_")}
        logger.info(f"SageMaker env vars: {json.dumps(sm_vars, indent=2)}")
        dataframes = load_data(args.train_dir, args.train_file_format)
        X, y, feature_cols, patient_ids = engineer_features(dataframes)
        model, metrics = train_model(X, y, args)
        save_artifacts(model, metrics, feature_cols, args.model_dir, args.output_dir)
        logger.info("Training completed successfully.")
    except Exception as e:
        failure_path = "/opt/ml/output/failure"
        os.makedirs(os.path.dirname(failure_path), exist_ok=True)
        with open(failure_path, "w") as f:
            f.write(str(e)[:1024])
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
