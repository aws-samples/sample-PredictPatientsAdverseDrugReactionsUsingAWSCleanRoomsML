# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Generate synthetic demo data for AWS Clean Rooms ML HCLS ADR Propensity Scoring.

Scenario: Cross-Party Adverse Drug Reaction (ADR) Propensity Scoring
- Party A (Pharma Company): drug exposure & risk profile data
- Party B (Health Insurer): real-world outcomes & claims data
- Shared key: patient_id (overlapping patient space)

Design principles for realistic ML performance:
- The has_adr label is derived from a COMBINED score using signals from
  BOTH parties, so neither party alone can predict ADR well.
- Each feature has moderate predictive power with significant noise.
- Feature importance is distributed across many features rather than concentrated.
  No single feature should exceed ~12% importance.
- NLP-extracted features (from Comprehend Medical / HealthScribe) are simulated
  as structured outputs — in a real deployment these would be pre-processed
  before contributing to the Clean Rooms collaboration.
- Target class balance is ~25% ADR (realistic for a monitored drug population).
- Rows per party: pharma ~2.2 rows/patient (multiple drugs), insurer ~2.5 rows/patient.
  With 50K patients this yields ~110K+ rows per party.
"""

import csv
import random
import math
import os
from datetime import datetime, timedelta

random.seed(42)

NUM_PATIENTS = 50000
SHARED_PATIENTS = int(NUM_PATIENTS * 0.8)   # 40,000
PHARMA_ONLY    = int(NUM_PATIENTS * 0.1)    # 5,000
INSURER_ONLY   = int(NUM_PATIENTS * 0.1)    # 5,000

shared_ids       = [f"pat_{i:06d}" for i in range(SHARED_PATIENTS)]
pharma_only_ids  = [f"pat_{SHARED_PATIENTS + i:06d}" for i in range(PHARMA_ONLY)]
insurer_only_ids = [f"pat_{SHARED_PATIENTS + PHARMA_ONLY + i:06d}" for i in range(INSURER_ONLY)]

pharma_patient_ids  = shared_ids + pharma_only_ids
insurer_patient_ids = shared_ids + insurer_only_ids

DRUG_CLASSES = ["biologic", "chemotherapy", "NSAID", "statin", "anticoagulant", "immunosuppressant"]
DRUG_IDS = {
    "biologic":          [f"BIO-{i:03d}" for i in range(1, 6)],
    "chemotherapy":      [f"CHE-{i:03d}" for i in range(1, 5)],
    "NSAID":             [f"NSA-{i:03d}" for i in range(1, 5)],
    "statin":            [f"STA-{i:03d}" for i in range(1, 4)],
    "anticoagulant":     [f"ACO-{i:03d}" for i in range(1, 4)],
    "immunosuppressant": [f"IMM-{i:03d}" for i in range(1, 4)],
}
# Black box warning drugs (subset)
BLACK_BOX_DRUGS = {"BIO-001", "BIO-002", "CHE-001", "CHE-002", "CHE-003", "IMM-001", "ACO-001"}

BASE_DATE = datetime(2025, 1, 1)
END_DATE  = BASE_DATE + timedelta(days=180)

# Per-patient latent traits
PATIENT_PHARMA_SCORE  = {}   # pharma-side risk (0-1)
PATIENT_INSURER_SCORE = {}   # insurer-side risk (0-1)


def random_date(start, end):
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def clamp(val, lo=0.0, hi=1.0):
    return max(lo, min(hi, val))


def assign_latent_traits():
    """
    Assign each patient two semi-independent latent risk scores:
    - pharma_risk: drives pharma-side features (known risk score, symptom reports, etc.)
    - insurer_risk: drives insurer-side features (ER visits, lab abnormalities, etc.)

    Moderate correlation (~0.3) between the two scores.
    The has_adr label uses BOTH scores, so neither party alone predicts well.
    """
    all_ids = set(pharma_patient_ids) | set(insurer_patient_ids)
    for pid in all_ids:
        shared    = random.gauss(0, 1)
        pharma_i  = random.gauss(0, 1)
        insurer_i = random.gauss(0, 1)

        pharma_raw  = 0.3 * shared + 0.7 * pharma_i
        insurer_raw = 0.3 * shared + 0.7 * insurer_i

        PATIENT_PHARMA_SCORE[pid]  = 1.0 / (1.0 + math.exp(-pharma_raw))
        PATIENT_INSURER_SCORE[pid] = 1.0 / (1.0 + math.exp(-insurer_raw))


def compute_has_adr(pid):
    """
    Determine if a patient experienced an ADR based on BOTH risk scores.
    Formula: combined = 0.45 * pharma_risk + 0.45 * insurer_risk + 0.10 * noise
    Threshold at 0.57 → ~25% ADR rate.
    """
    pr = PATIENT_PHARMA_SCORE.get(pid, 0.5)
    ir = PATIENT_INSURER_SCORE.get(pid, 0.5)
    noise = random.gauss(0, 0.08)
    combined = 0.45 * pr + 0.45 * ir + 0.10 * noise
    return int(combined > 0.52)


def generate_pharma_data():
    """
    Generate Party A (Pharma Company) drug exposure & risk profile data.

    Each patient has 2-3 drug exposure records (different drugs/therapy lines).
    Features are driven by pharma_risk with substantial noise — no single
    feature dominates. NLP-extracted fields (Comprehend Medical) are simulated
    as structured outputs.

    Target row count: ~110,000+ (50K patients × avg 2.2 drugs)
    """
    rows = []
    for pid in pharma_patient_ids:
        risk = PATIENT_PHARMA_SCORE.get(pid, 0.5)

        # Each patient is on 2-3 drugs (different classes)
        num_drugs = random.choices([1, 2, 3], weights=[0.05, 0.45, 0.50])[0]
        selected_classes = random.sample(DRUG_CLASSES, min(num_drugs, len(DRUG_CLASSES)))

        for i, drug_class in enumerate(selected_classes):
            drug_id = random.choice(DRUG_IDS[drug_class])
            therapy_line = i + 1  # first drug = line 1, second = line 2, etc.

            # dose_mg: moderate signal — higher risk drugs tend toward higher doses
            dose_mg = round(max(5.0, random.gauss(
                100 + 150 * risk + 50 * (therapy_line - 1), 60)), 1)

            # treatment_duration_days: longer duration slightly increases risk
            treatment_duration_days = max(7, int(random.gauss(
                90 + 60 * risk, 45)))

            # known_risk_score: from clinical trial safety data — moderate signal
            # Deliberately noisy: trial conditions ≠ real world
            known_risk_score = round(clamp(random.gauss(
                0.2 + 0.4 * risk, 0.18)), 4)

            # black_box_warning: binary, correlated with drug class
            black_box_warning = int(drug_id in BLACK_BOX_DRUGS)

            # patient_age: older patients have higher ADR risk (non-linear, noisy)
            patient_age = max(18, min(90, int(random.gauss(
                45 + 20 * risk, 15))))

            # indication_severity: severity of underlying condition (0-1)
            indication_severity = round(clamp(random.gauss(
                0.3 + 0.35 * risk, 0.20)), 4)

            # --- NLP-extracted features (Comprehend Medical from spontaneous reports) ---

            # reported_symptom_count: symptoms extracted from ADR report narratives
            # Noisy: many reports mention symptoms unrelated to the drug
            reported_symptom_count = max(0, int(random.gauss(
                0.5 + 2.5 * risk, 1.5)))

            # symptom_severity_flag: 1 if "severe"/"life-threatening" detected
            # Weak-moderate signal with high noise
            sev_prob = clamp(0.05 + 0.25 * risk + random.gauss(0, 0.08))
            symptom_severity_flag = int(random.random() < sev_prob)

            # time_to_onset_days: days from drug start to symptom onset
            # Shorter onset = stronger signal, but very noisy
            time_to_onset_days = max(1, int(random.gauss(
                60 - 35 * risk, 25)))

            # concomitant_drug_count_reported: other drugs mentioned in reports
            concomitant_drug_count_reported = max(0, int(random.gauss(
                1 + 2 * risk, 1.2)))

            # prior_adr_narrative_flag: prior reactions mentioned in report text
            prior_prob = clamp(0.05 + 0.20 * risk + random.gauss(0, 0.07))
            prior_adr_narrative_flag = int(random.random() < prior_prob)

            observation_date = random_date(BASE_DATE, END_DATE).strftime("%Y-%m-%d")

            rows.append({
                "patient_id":                    pid,
                "drug_id":                       drug_id,
                "drug_class":                    drug_class,
                "dose_mg":                       dose_mg,
                "treatment_duration_days":       treatment_duration_days,
                "therapy_line":                  therapy_line,
                "known_risk_score":              known_risk_score,
                "black_box_warning":             black_box_warning,
                "patient_age":                   patient_age,
                "indication_severity":           indication_severity,
                "reported_symptom_count":        reported_symptom_count,
                "symptom_severity_flag":         symptom_severity_flag,
                "time_to_onset_days":            time_to_onset_days,
                "concomitant_drug_count_reported": concomitant_drug_count_reported,
                "prior_adr_narrative_flag":      prior_adr_narrative_flag,
                "observation_date":              observation_date,
            })
    return rows


def generate_insurer_data():
    """
    Generate Party B (Health Insurer) real-world outcomes & claims data.

    Each patient has 2-3 drug monitoring records.
    The has_adr label uses BOTH pharma and insurer risk scores.
    NLP-extracted fields (Comprehend Medical + HealthScribe) are simulated.

    Key design decisions to prevent feature dominance:
    - hospitalizations_post_start: generated with high noise, many zeros even
      for ADR patients (hospitalization has many non-drug causes) → max ~10% weight
    - drug_symptom_co_mention: high noise, counterbalanced by negated_symptom_count
    - All features capped at moderate individual predictive power

    Target row count: ~112,000+ (45K shared + 5K insurer-only × avg 2.5 drugs)
    """
    rows = []
    for pid in insurer_patient_ids:
        risk    = PATIENT_INSURER_SCORE.get(pid, 0.5)
        has_adr = compute_has_adr(pid)

        num_drugs = random.choices([1, 2, 3], weights=[0.05, 0.45, 0.50])[0]
        selected_classes = random.sample(DRUG_CLASSES, min(num_drugs, len(DRUG_CLASSES)))

        for drug_class in selected_classes:
            drug_id = random.choice(DRUG_IDS[drug_class])

            # er_visits_post_start: moderate signal, very noisy
            # Many ER visits are unrelated to the drug
            er_visits_post_start = max(0, int(random.gauss(
                0.3 + 1.2 * risk, 0.9)))

            # hospitalizations_post_start: correlated with ADR but NOT specific
            # Deliberately noisy to prevent dominance — max weight ~10%
            hosp_base = 0.1 + 0.5 * risk
            hosp_noise = random.gauss(0, 0.35)
            hospitalizations_post_start = max(0, int(hosp_base + hosp_noise))

            # days_to_first_er_visit: lower = riskier, 999 if no ER visit
            if er_visits_post_start > 0:
                days_to_first_er_visit = max(1, int(random.gauss(
                    45 - 25 * risk, 20)))
            else:
                days_to_first_er_visit = 999

            # drug_discontinuation: moderate signal
            disc_prob = clamp(0.10 + 0.35 * risk + random.gauss(0, 0.10))
            drug_discontinuation = int(random.random() < disc_prob)

            # days_to_discontinuation: 999 if still on drug
            if drug_discontinuation:
                days_to_discontinuation = max(1, int(random.gauss(
                    50 - 20 * risk, 20)))
            else:
                days_to_discontinuation = 999

            # num_concomitant_meds: polypharmacy increases ADR risk
            num_concomitant_meds = max(0, int(random.gauss(
                3 + 4 * risk, 2.5)))

            # high_risk_combo: 1 if on a known dangerous drug combination
            combo_prob = clamp(0.05 + 0.20 * risk + random.gauss(0, 0.07))
            high_risk_combo = int(random.random() < combo_prob)

            # lab_abnormality_count: out-of-range lab results — moderate signal
            lab_abnormality_count = max(0, int(random.gauss(
                0.5 + 2.0 * risk, 1.2)))

            # comorbidity_index: Charlson score — weak-moderate signal
            comorbidity_index = round(max(0.0, random.gauss(
                1.5 + 3.0 * risk, 1.5)), 2)

            # prior_hospitalization: 1 if hospitalized in 12 months before drug start
            prior_hosp_prob = clamp(0.10 + 0.25 * risk + random.gauss(0, 0.08))
            prior_hospitalization = int(random.random() < prior_hosp_prob)

            # --- NLP-extracted features (Comprehend Medical from clinical notes) ---

            # symptom_mention_count: symptoms in post-visit notes — noisy
            # Notes mention many symptoms unrelated to the monitored drug
            symptom_mention_count = max(0, int(random.gauss(
                1.5 + 3.0 * risk, 2.0)))

            # drug_symptom_co_mention: drug + symptom in same note — weak signal
            # High noise: co-mention doesn't imply causation
            co_prob = clamp(0.15 + 0.30 * risk + random.gauss(0, 0.12))
            drug_symptom_co_mention = int(random.random() < co_prob)

            # negated_symptom_count: "no nausea", "denies chest pain" etc.
            # Acts as counterbalance to symptom_mention_count
            negated_symptom_count = max(0, int(random.gauss(
                1.0 + 0.5 * (1 - risk), 0.8)))

            # lab_abnormality_mentioned: abnormal labs in note text — moderate signal
            lab_mention_prob = clamp(0.10 + 0.30 * risk + random.gauss(0, 0.10))
            lab_abnormality_mentioned = int(random.random() < lab_mention_prob)

            # chief_complaint_adr_flag: from HealthScribe — weak-moderate signal
            # Chief complaint matches known ADR terms for this drug class
            cc_prob = clamp(0.05 + 0.20 * risk + random.gauss(0, 0.08))
            chief_complaint_adr_flag = int(random.random() < cc_prob)

            rows.append({
                "patient_id":               pid,
                "drug_id":                  drug_id,
                "er_visits_post_start":     er_visits_post_start,
                "hospitalizations_post_start": hospitalizations_post_start,
                "days_to_first_er_visit":   days_to_first_er_visit,
                "drug_discontinuation":     drug_discontinuation,
                "days_to_discontinuation":  days_to_discontinuation,
                "num_concomitant_meds":     num_concomitant_meds,
                "high_risk_combo":          high_risk_combo,
                "lab_abnormality_count":    lab_abnormality_count,
                "comorbidity_index":        comorbidity_index,
                "prior_hospitalization":    prior_hospitalization,
                "symptom_mention_count":    symptom_mention_count,
                "drug_symptom_co_mention":  drug_symptom_co_mention,
                "negated_symptom_count":    negated_symptom_count,
                "lab_abnormality_mentioned": lab_abnormality_mentioned,
                "chief_complaint_adr_flag": chief_complaint_adr_flag,
                "has_adr":                  has_adr,
            })
    return rows


def write_csv(filename, rows, fieldnames):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written {len(rows):,} rows to {filename}")


if __name__ == "__main__":
    print("Assigning latent patient traits...")
    assign_latent_traits()

    print("\nGenerating pharma company data (Party A)...")
    pharma_data = generate_pharma_data()
    write_csv(
        "data/pharma_drug_exposure.csv",
        pharma_data,
        ["patient_id", "drug_id", "drug_class", "dose_mg", "treatment_duration_days",
         "therapy_line", "known_risk_score", "black_box_warning", "patient_age",
         "indication_severity", "reported_symptom_count", "symptom_severity_flag",
         "time_to_onset_days", "concomitant_drug_count_reported",
         "prior_adr_narrative_flag", "observation_date"],
    )

    print("\nGenerating health insurer data (Party B)...")
    insurer_data = generate_insurer_data()
    write_csv(
        "data/insurer_outcomes.csv",
        insurer_data,
        ["patient_id", "drug_id", "er_visits_post_start", "hospitalizations_post_start",
         "days_to_first_er_visit", "drug_discontinuation", "days_to_discontinuation",
         "num_concomitant_meds", "high_risk_combo", "lab_abnormality_count",
         "comorbidity_index", "prior_hospitalization", "symptom_mention_count",
         "drug_symptom_co_mention", "negated_symptom_count", "lab_abnormality_mentioned",
         "chief_complaint_adr_flag", "has_adr"],
    )

    # --- Stats ---
    pharma_pats  = set(r["patient_id"] for r in pharma_data)
    insurer_pats = set(r["patient_id"] for r in insurer_data)
    overlap      = pharma_pats & insurer_pats

    print(f"\nPopulation stats:")
    print(f"  Pharma company patients:  {len(pharma_pats):,}")
    print(f"  Health insurer patients:  {len(insurer_pats):,}")
    print(f"  Shared (overlapping):     {len(overlap):,}")
    print(f"  Pharma rows:              {len(pharma_data):,}")
    print(f"  Insurer rows:             {len(insurer_data):,}")

    # ADR rate
    total_adr  = sum(1 for r in insurer_data if r["has_adr"])
    total_rows = len(insurer_data)
    print(f"\nLabel stats:")
    print(f"  Overall ADR rate:         {total_adr:,}/{total_rows:,} ({100*total_adr/total_rows:.1f}%)")

    # ADR rate for shared patients only
    shared_adr   = sum(1 for r in insurer_data if r["patient_id"] in overlap and r["has_adr"])
    shared_total = sum(1 for r in insurer_data if r["patient_id"] in overlap)
    print(f"  Shared patient ADR rate:  {shared_adr:,}/{shared_total:,} ({100*shared_adr/max(1,shared_total):.1f}%)")

    # Feature distribution checks
    print(f"\nFeature distribution checks (pharma):")
    sev_flags = sum(1 for r in pharma_data if r["symptom_severity_flag"])
    bbw_flags = sum(1 for r in pharma_data if r["black_box_warning"])
    print(f"  symptom_severity_flag=1:  {sev_flags:,}/{len(pharma_data):,} ({100*sev_flags/len(pharma_data):.1f}%)")
    print(f"  black_box_warning=1:      {bbw_flags:,}/{len(pharma_data):,} ({100*bbw_flags/len(pharma_data):.1f}%)")

    print(f"\nFeature distribution checks (insurer):")
    hosp_nonzero = sum(1 for r in insurer_data if r["hospitalizations_post_start"] > 0)
    co_mention   = sum(1 for r in insurer_data if r["drug_symptom_co_mention"])
    print(f"  hospitalizations > 0:     {hosp_nonzero:,}/{len(insurer_data):,} ({100*hosp_nonzero/len(insurer_data):.1f}%)")
    print(f"  drug_symptom_co_mention=1:{co_mention:,}/{len(insurer_data):,} ({100*co_mention/len(insurer_data):.1f}%)")
