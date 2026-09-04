"""
model_pipeline.py  —  PRECOMPUTED-RESULTS BACKEND (demo viewer)
================================================================
Replaces the original live-training pipeline. Instead of retraining on
start-up, it loads the team's ALREADY-VALIDATED results from `demo_data.json`
(46 investigation briefs spanning Low/Medium/High risk tiers, 44 of which
passed the human rubric - the 2 that didn't are shown with their real score,
not hidden) and serves them to the existing Gradio front end.

Why: the demo must show the project's REAL reported results — the finalized
XGBoost + isotonic-calibration model (test PR-AUC 0.195), its real risk tiers,
real SHAP drivers, and the real Qwen2.5 briefs the team generated and scored —
not a fresh model retrained live (which would produce different numbers and
could silently run on synthetic fallback data).

The public names imported by app.py are preserved:
  PIPELINE, PIPELINE.fit(), PIPELINE.predict(), FIELD_META, GROUP_ORDER,
  USER_INPUT_FEATURES, ORDINAL_FEATURES, NOMINAL_CATEGORIES.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_PATH = Path(__file__).parent / "demo_data.json"
METRICS_PATH = Path(__file__).parent / "results" / "metrics.json"
OPERATING_THRESHOLD = 0.15
LOW_RISK_MAX = 0.075

# Notebook 06's real finalized test-set evaluation - loaded once at import, never
# recomputed live, same "serve the team's real reported numbers" principle as
# demo_data.json below.
_real_metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
_test = _real_metrics["test"]
_temporal = _real_metrics["temporal_robustness_check"]
_tiers_by_name = {t["tier"]: t for t in _test["risk_tiers"]}

# Notebook 04 cell 30 (validation set): the candidate's own validation PR-AUC and the
# Balanced Random Forest baseline it needs to beat. Per cell 28's markdown, the
# published ~0.16 reference is explicitly "context only, not a hard gate" - the real,
# discussed acceptance bar is beating this baseline, not clearing 0.16.
_CANDIDATE_VAL_PR_AUC = 0.223
_RF_BASELINE_VAL_PR_AUC = 0.204
_PUBLISHED_REFERENCE_PR_AUC = 0.16

ORDINAL_FEATURES: dict[str, list[str]] = {
    "VehiclePrice": ["less than 20000", "20000 to 29000", "30000 to 39000",
                      "40000 to 59000", "60000 to 69000", "more than 69000"],
    "AgeOfVehicle": ["new", "2 years", "3 years", "4 years", "5 years",
                      "6 years", "7 years", "more than 7"],
    "Days_Policy_Accident": ["none", "1 to 7", "8 to 15", "15 to 30", "more than 30"],
    "Days_Policy_Claim": ["none", "8 to 15", "15 to 30", "more than 30"],
    "PastNumberOfClaims": ["none", "1", "2 to 4", "more than 4"],
    "NumberOfSuppliments": ["none", "1 to 2", "3 to 5", "more than 5"],
    "NumberOfCars": ["1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8"],
    "AddressChange_Claim": ["no change", "under 6 months", "1 year", "2 to 3 years", "4 to 8 years"],
    "AgeOfPolicyHolder": ["16 to 17", "18 to 20", "21 to 25", "26 to 30", "31 to 35",
                           "36 to 40", "41 to 50", "51 to 65", "over 65"],
}

NOMINAL_FEATURES = [
    "Make", "BasePolicy", "VehicleCategory", "Fault", "Sex", "MaritalStatus",
    "AccidentArea", "PoliceReportFiled", "WitnessPresent", "AgentType",
    "Month", "DayOfWeek", "MonthClaimed", "DayOfWeekClaimed", "RepNumber",
]

NOMINAL_CATEGORIES: dict[str, list[str]] = {
    "Make": ["Pontiac", "Toyota", "Honda", "Mazda", "Chevrolet", "Accura", "Ford",
              "VW", "Dodge", "Saab", "Mercury", "Saturn", "Nisson", "BMW",
              "Jaguar", "Porche", "Mecedes", "Ferrari", "Lexus"],
    "BasePolicy": ["Collision", "Liability", "All Perils"],
    "VehicleCategory": ["Sedan", "Sport", "Utility"],
    "Fault": ["Policy Holder", "Third Party"],
    "Sex": ["Male", "Female"],
    "MaritalStatus": ["Married", "Single", "Divorced", "Widow"],
    "AccidentArea": ["Urban", "Rural"],
    "PoliceReportFiled": ["No", "Yes"],
    "WitnessPresent": ["No", "Yes"],
    "AgentType": ["External", "Internal"],
    "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "DayOfWeek": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "MonthClaimed": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "DayOfWeekClaimed": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "RepNumber": [str(i) for i in range(1, 17)],
}

NUMERIC_IMPUTE_FEATURES = ["Age", "WeekOfMonth", "WeekOfMonthClaimed"]
PASSTHROUGH_FEATURES = ["Deductible", "DriverRating", "Year"]

ALL_MODEL_FEATURES = (
    list(ORDINAL_FEATURES) + NOMINAL_FEATURES + NUMERIC_IMPUTE_FEATURES + PASSTHROUGH_FEATURES
)
USER_INPUT_FEATURES = list(ALL_MODEL_FEATURES)

FIELD_META = {
    "Make": {"label": "Vehicle Make", "group": "Vehicle", "type": "select"},
    "VehicleCategory": {"label": "Vehicle Category", "group": "Vehicle", "type": "select"},
    "VehiclePrice": {"label": "Vehicle Price Bracket", "group": "Vehicle", "type": "select"},
    "AgeOfVehicle": {"label": "Age of Vehicle", "group": "Vehicle", "type": "select"},
    "NumberOfCars": {"label": "Number of Cars on Policy", "group": "Vehicle", "type": "select"},
    "BasePolicy": {"label": "Base Policy Type", "group": "Policy", "type": "select"},
    "Deductible": {"label": "Deductible ($)", "group": "Policy", "type": "number", "min": 300, "max": 1000, "step": 100, "default": 400},
    "AgentType": {"label": "Agent Type", "group": "Policy", "type": "select"},
    "Year": {"label": "Policy Year", "group": "Policy", "type": "number", "min": 1994, "max": 1996, "step": 1, "default": 1995},
    "RepNumber": {"label": "Claims Rep Number", "group": "Policy", "type": "number", "min": 1, "max": 16, "step": 1, "default": 8},
    "Days_Policy_Accident": {"label": "Days: Policy Start -> Accident", "group": "Policy", "type": "select"},
    "Days_Policy_Claim": {"label": "Days: Policy Start -> Claim Filed", "group": "Policy", "type": "select"},
    "Sex": {"label": "Policyholder Sex", "group": "Policyholder", "type": "select"},
    "MaritalStatus": {"label": "Marital Status", "group": "Policyholder", "type": "select"},
    "Age": {"label": "Policyholder Age", "group": "Policyholder", "type": "number", "min": 16, "max": 90, "step": 1, "default": 35},
    "AgeOfPolicyHolder": {"label": "Policyholder Age Bracket", "group": "Policyholder", "type": "select"},
    "DriverRating": {"label": "Driver Rating (1-4)", "group": "Policyholder", "type": "number", "min": 1, "max": 4, "step": 1, "default": 2},
    "AddressChange_Claim": {"label": "Address Change Since Claim", "group": "Policyholder", "type": "select"},
    "NumberOfSuppliments": {"label": "Number of Supplements Filed", "group": "Claim History", "type": "select"},
    "PastNumberOfClaims": {"label": "Past Number of Claims", "group": "Claim History", "type": "select"},
    "AccidentArea": {"label": "Accident Area", "group": "Accident", "type": "select"},
    "Fault": {"label": "Fault Determination", "group": "Accident", "type": "select"},
    "PoliceReportFiled": {"label": "Police Report Filed", "group": "Accident", "type": "select"},
    "WitnessPresent": {"label": "Witness Present", "group": "Accident", "type": "select"},
    "Month": {"label": "Accident Month", "group": "Timing", "type": "select"},
    "WeekOfMonth": {"label": "Accident Week of Month (1-5)", "group": "Timing", "type": "number", "min": 1, "max": 5, "step": 1, "default": 2},
    "DayOfWeek": {"label": "Accident Day of Week", "group": "Timing", "type": "select"},
    "MonthClaimed": {"label": "Month Claimed", "group": "Timing", "type": "select"},
    "WeekOfMonthClaimed": {"label": "Claim Week of Month (1-5)", "group": "Timing", "type": "number", "min": 1, "max": 5, "step": 1, "default": 2},
    "DayOfWeekClaimed": {"label": "Day of Week Claimed", "group": "Timing", "type": "select"},
}

GROUP_ORDER = ["Accident", "Timing", "Vehicle", "Policy", "Policyholder", "Claim History"]
ACTION_BY_RISK_TIER = {"High": "Fraud Investigation", "Medium": "Manual Review", "Low": "Approve"}

NA = "n/a"

QWEN_BRIEF_RUBRIC = {
    "factual_consistency": "Every fact stated in the brief (probability, tier, feature values) must exactly "
                            "match the model's output — no invented or altered figures.",
    "faithfulness_to_shap": "The brief's stated risk drivers must match the model's actual top SHAP features "
                             "and their direction (increases vs. decreases risk).",
    "actionability": "The brief must clearly justify the recommended action (Approve / Manual Review / Fraud "
                      "Investigation) implied by the risk tier.",
    "no_fabrication": "The brief must not introduce claim details, causes, or conclusions that are not present "
                       "in the underlying data or model output.",
    "scale": "Each criterion is scored 0 (fail), 1 (partial), or 2 (full) per brief.",
    "acceptance_rule": "A brief is accepted only if its total is >=6/8 AND it scores 2 on both "
                        "factual_consistency and no_fabrication; otherwise it's FAILED / NEEDS REVIEW.",
    "note_vs_action_plan": "Scored against the model's actual output (probability, tier, SHAP drivers), not "
                            "against a separately-authored 'ideal' brief — the rubric checks faithfulness to "
                            "the facts, not writing style. Across the full generated batch spanning Low/Medium/"
                            "High risk tiers: 44/46 briefs passed (95.7%, mean score 7.83/8) — the 2 that "
                            "didn't (TEST-06854: no_fabrication scored 1; TEST-06928: factual_consistency and "
                            "faithfulness_to_shap each scored 1) are shown in this demo with their real score, "
                            "not hidden. 100% of recommended actions were tier-consistent, and the automated "
                            "fabrication pre-check flagged 2/46 for human review.",
}

# Finalized, rubric-passed project results (test set). Everything below marked
# "real" is loaded from results/metrics.json (Notebook 06's output) or transcribed
# verbatim from a specific Notebook cell (comment cites which one) - never
# recomputed live. Anything not part of the finalized results summary is shown as
# "n/a" rather than fabricated.
METRICS = {
    "pr_auc_test": round(_test["metrics"]["pr_auc"], 4),
    "roc_auc_test": round(_test["metrics"]["roc_auc"], 4),
    "recall_test": round(_test["metrics"]["recall"], 4),
    "precision_test": round(_test["metrics"]["precision"], 4),
    "macro_f1_test": round(_test["metrics"]["macro_f1"], 4),
    "mcc_test": round(_test["metrics"]["mcc"], 4),
    "brier_score_test": round(_test["metrics"]["brier_score"], 4),
    "no_skill_pr_auc_test": round(_test["no_skill_pr_auc"], 4),
    # Real acceptance criterion per Notebook 04 (not the 0.16 "context only" reference -
    # see the module-level constants above): does the candidate beat the RF baseline on
    # validation PR-AUC?
    "pr_auc_pass": _CANDIDATE_VAL_PR_AUC > _RF_BASELINE_VAL_PR_AUC,
    "pr_auc_status": "PASS" if _CANDIDATE_VAL_PR_AUC > _RF_BASELINE_VAL_PR_AUC else "FAIL",
    "pr_auc_candidate_validation": _CANDIDATE_VAL_PR_AUC,
    "pr_auc_rf_baseline": _RF_BASELINE_VAL_PR_AUC,
    "pr_auc_published_reference": _PUBLISHED_REFERENCE_PR_AUC,
    "calibration_error_test": NA,  # not computed in any notebook - would need Notebook 06 extended
    "confusion_matrix_test": {
        "tp": _test["operating_point"]["tp"], "fp": _test["operating_point"]["fp"],
        "fn": _test["operating_point"]["fn"], "tn": _test["operating_point"]["tn"],
    },
    "operating_threshold": OPERATING_THRESHOLD,
    "risk_tiers": {
        "low_max": LOW_RISK_MAX, "high_min": OPERATING_THRESHOLD,
        "low_share_pct": _tiers_by_name["Low"]["caseload_pct"],
        "low_fraud_rate_pct": _tiers_by_name["Low"]["fraud_rate_pct"],
        "medium_share_pct": _tiers_by_name["Medium"]["caseload_pct"],
        "medium_fraud_rate_pct": _tiers_by_name["Medium"]["fraud_rate_pct"],
        "high_share_pct": _tiers_by_name["High"]["caseload_pct"],
        "high_fraud_rate_pct": _tiers_by_name["High"]["fraud_rate_pct"],
    },
    "data_source": "Vehicle Claim Fraud Detection (Kaggle, CC0 1.0)",
    # Split sizes/overall fraud rate aren't persisted to any local JSON - transcribed
    # verbatim from Notebook 02's split cell (cell 28/35 printed output): train
    # n=10,793 fraud_rate=5.99%, val n=2,313 fraud_rate=6.01%, test n=2,313
    # fraud_rate=5.97%, overall cleaned_df fraud rate=5.99%.
    "n_total": 15419,
    "n_train": 10793,
    "n_val": 2313,
    "n_test": 2313,
    "fraud_rate_pct": 5.99,
    "imbalance_strategy_chosen": "class weighting",
    # Notebook 04's Optuna search converged on class_weight early and barely explored
    # SMOTENC for the winning model (XGBoost): 57 completed trials for class_weight vs.
    # only 2 for SMOTENC (per notebooks/results/experiments.csv). A mean over 2 trials
    # isn't a fair comparison to report as a precise number, so this stays unavailable
    # with an honest reason rather than a misleadingly precise figure.
    "imbalance_comparison": {
        "available": False,
        "note": "Optuna's search converged on class_weight early (57 completed trials vs. only 2 for "
                 "SMOTENC on the winning XGBoost model) — not enough SMOTENC trials for a fair comparison.",
    },
    # Real validation-set comparison from Notebook 04 cell 30 - only 3 models were ever
    # compared there (no LightGBM row exists in that table, despite an earlier version
    # of this file claiming one). All three "pr_auc" values are Val PR-AUC (not CV
    # PR-AUC) so they're apples-to-apples with each other; production_model's CV PR-AUC
    # (0.281, the Optuna selection criterion) is noted separately since it's a different
    # stage of the pipeline than this validation-set snapshot.
    "baseline_comparison": {
        "production_model": {
            "pr_auc": 0.223, "recall_at_050": 0.748, "precision_at_050": 0.151,
            "note": "XGBoost + class weighting, isotonic calibration — production model "
                    "(CV PR-AUC 0.281 was the Optuna selection criterion; 0.223 above is this "
                    "same model's validation-set PR-AUC at the model-selection stage, before "
                    "calibration/threshold-tuning, for apples-to-apples comparison with the "
                    "baselines below)",
        },
        "logistic_regression": {"pr_auc": 0.140, "recall_at_050": 0.871, "precision_at_050": 0.131},
        "balanced_random_forest": {"pr_auc": 0.204, "recall_at_050": 0.842, "precision_at_050": 0.139},
    },
    # Real validation-set sweep from Notebook 05 cell 30.
    "threshold_sweep": [
        {"threshold": 0.05, "recall": 0.914, "precision": 0.135, "fpr": 0.374, "caseload_pct": 40.683},
        {"threshold": 0.10, "recall": 0.719, "precision": 0.159, "fpr": 0.243, "caseload_pct": 27.151},
        {"threshold": 0.15, "recall": 0.353, "precision": 0.236, "fpr": 0.073, "caseload_pct": 8.993},
        {"threshold": 0.20, "recall": 0.216, "precision": 0.303, "fpr": 0.032, "caseload_pct": 4.280},
        {"threshold": 0.25, "recall": 0.115, "precision": 0.381, "fpr": 0.012, "caseload_pct": 1.816},
        {"threshold": 0.30, "recall": 0.065, "precision": 0.562, "fpr": 0.003, "caseload_pct": 0.692},
        {"threshold": 0.35, "recall": 0.065, "precision": 0.562, "fpr": 0.003, "caseload_pct": 0.692},
        {"threshold": 0.40, "recall": 0.065, "precision": 0.562, "fpr": 0.003, "caseload_pct": 0.692},
        {"threshold": 0.45, "recall": 0.065, "precision": 0.562, "fpr": 0.003, "caseload_pct": 0.692},
        {"threshold": 0.50, "recall": 0.065, "precision": 0.562, "fpr": 0.003, "caseload_pct": 0.692},
        {"threshold": 0.55, "recall": 0.007, "precision": 1.000, "fpr": 0.000, "caseload_pct": 0.043},
        {"threshold": 0.60, "recall": 0.007, "precision": 1.000, "fpr": 0.000, "caseload_pct": 0.043},
        {"threshold": 0.65, "recall": 0.007, "precision": 1.000, "fpr": 0.000, "caseload_pct": 0.043},
        {"threshold": 0.70, "recall": 0.007, "precision": 1.000, "fpr": 0.000, "caseload_pct": 0.043},
        {"threshold": 0.75, "recall": 0.007, "precision": 1.000, "fpr": 0.000, "caseload_pct": 0.043},
        {"threshold": 0.80, "recall": 0.007, "precision": 1.000, "fpr": 0.000, "caseload_pct": 0.043},
        {"threshold": 0.85, "recall": 0.007, "precision": 1.000, "fpr": 0.000, "caseload_pct": 0.043},
        {"threshold": 0.90, "recall": 0.007, "precision": 1.000, "fpr": 0.000, "caseload_pct": 0.043},
        {"threshold": 0.95, "recall": 0.007, "precision": 1.000, "fpr": 0.000, "caseload_pct": 0.043},
    ],
    "reliability_curve_test": [],  # not computed anywhere - only a PNG figure exists, no bin data
    "temporal_validation": {
        "available": True,
        "train_period": f"{_temporal['train_years'][0]}-{_temporal['train_years'][-1]}",
        "test_period": str(_temporal["test_year"][0]),
        "train_fraud_rate_pct": _temporal["train_fraud_rate_pct"],
        "test_fraud_rate_pct": _temporal["test_fraud_rate_pct"],
        "pr_auc": round(_temporal["metrics"]["pr_auc"], 4),
        "roc_auc": round(_temporal["metrics"]["roc_auc"], 4),
        "recall": round(_temporal["metrics"]["recall"], 4),
        "precision": round(_temporal["metrics"]["precision"], 4),
        "macro_f1": round(_temporal["metrics"]["macro_f1"], 4),
        "mcc": round(_temporal["metrics"]["mcc"], 4),
        "brier_score": round(_temporal["metrics"]["brier_score"], 4),
        "description": _temporal["description"],
    },
    # Real disparate-impact audit from Notebook 06 cell 62, sliced from the frozen
    # test-set predictions at the production 0.15 threshold. The notebook's own chosen
    # fairness lens is flag rate / false positive rate / false negative rate / the
    # group's own fraud base rate - not recall/precision - preserved as-is rather than
    # force-fit into a different metric shape.
    "fairness": {
        "description": "Disparate-impact screen over the frozen test-set predictions at the production "
                        "operating threshold (0.15): flag rate, false positive rate (among legitimate "
                        "claims), false negative rate (among fraud claims), and the group's own fraud "
                        "base rate.",
        "note": "MaritalStatus 'Divorced' (n=6) and 'Widow' (n=5) are too small to draw reliable "
                "conclusions from - shown for completeness, not as evidence of a real disparity.",
        "groups_by_attribute": {
            "Sex": [
                {"group": "Male", "n": 1930, "pct_of_test": 83.44, "flag_rate_pct": 8.45,
                 "fpr_pct": 6.85, "fnr_pct": 67.23, "fraud_base_rate_pct": 6.17},
                {"group": "Female", "n": 383, "pct_of_test": 16.56, "flag_rate_pct": 3.39,
                 "fpr_pct": 3.30, "fnr_pct": 94.74, "fraud_base_rate_pct": 4.96},
            ],
            "MaritalStatus": [
                {"group": "Married", "n": 1604, "pct_of_test": 69.35, "flag_rate_pct": 7.42,
                 "fpr_pct": 6.26, "fnr_pct": 75.49, "fraud_base_rate_pct": 6.36},
                {"group": "Single", "n": 698, "pct_of_test": 30.18, "flag_rate_pct": 7.88,
                 "fpr_pct": 6.03, "fnr_pct": 57.14, "fraud_base_rate_pct": 5.01},
                {"group": "Divorced", "n": 6, "pct_of_test": 0.26, "flag_rate_pct": 16.67,
                 "fpr_pct": 16.67, "fnr_pct": None, "fraud_base_rate_pct": 0.00},
                {"group": "Widow", "n": 5, "pct_of_test": 0.22, "flag_rate_pct": 20.00,
                 "fpr_pct": 25.00, "fnr_pct": 100.00, "fraud_base_rate_pct": 20.00},
            ],
            "AgeBand": [
                {"group": "<=25", "n": 120, "pct_of_test": 5.19, "flag_rate_pct": 11.67,
                 "fpr_pct": 8.18, "fnr_pct": 50.00, "fraud_base_rate_pct": 8.33},
                {"group": "26-35", "n": 851, "pct_of_test": 36.79, "flag_rate_pct": 8.93,
                 "fpr_pct": 6.97, "fnr_pct": 66.13, "fraud_base_rate_pct": 7.29},
                {"group": "36-45", "n": 607, "pct_of_test": 26.24, "flag_rate_pct": 7.41,
                 "fpr_pct": 6.75, "fnr_pct": 79.31, "fraud_base_rate_pct": 4.78},
                {"group": "46-55", "n": 398, "pct_of_test": 17.21, "flag_rate_pct": 3.52,
                 "fpr_pct": 2.95, "fnr_pct": 88.00, "fraud_base_rate_pct": 6.28},
                {"group": "56-65", "n": 220, "pct_of_test": 9.51, "flag_rate_pct": 6.82,
                 "fpr_pct": 5.21, "fnr_pct": 55.56, "fraud_base_rate_pct": 4.09},
                {"group": "65+", "n": 75, "pct_of_test": 3.24, "flag_rate_pct": 4.00,
                 "fpr_pct": 4.05, "fnr_pct": 100.00, "fraud_base_rate_pct": 1.33},
                {"group": "Missing (Age imputed)", "n": 42, "pct_of_test": 1.82, "flag_rate_pct": 21.43,
                 "fpr_pct": 20.00, "fnr_pct": 50.00, "fraud_base_rate_pct": 4.76},
            ],
            "AgeOfPolicyHolder": [
                {"group": "16 to 17", "n": 42, "pct_of_test": 1.82, "flag_rate_pct": 21.43,
                 "fpr_pct": 20.00, "fnr_pct": 50.00, "fraud_base_rate_pct": 4.76},
                {"group": "18 to 20", "n": 2, "pct_of_test": 0.09, "flag_rate_pct": 50.00,
                 "fpr_pct": 0.00, "fnr_pct": 0.00, "fraud_base_rate_pct": 50.00},
                {"group": "21 to 25", "n": 22, "pct_of_test": 0.95, "flag_rate_pct": 22.73,
                 "fpr_pct": 16.67, "fnr_pct": 50.00, "fraud_base_rate_pct": 18.18},
                {"group": "26 to 30", "n": 96, "pct_of_test": 4.15, "flag_rate_pct": 8.33,
                 "fpr_pct": 6.59, "fnr_pct": 60.00, "fraud_base_rate_pct": 5.21},
                {"group": "31 to 35", "n": 851, "pct_of_test": 36.79, "flag_rate_pct": 8.93,
                 "fpr_pct": 6.97, "fnr_pct": 66.13, "fraud_base_rate_pct": 7.29},
                {"group": "36 to 40", "n": 607, "pct_of_test": 26.24, "flag_rate_pct": 7.41,
                 "fpr_pct": 6.75, "fnr_pct": 79.31, "fraud_base_rate_pct": 4.78},
                {"group": "41 to 50", "n": 398, "pct_of_test": 17.21, "flag_rate_pct": 3.52,
                 "fpr_pct": 2.95, "fnr_pct": 88.00, "fraud_base_rate_pct": 6.28},
                {"group": "51 to 65", "n": 220, "pct_of_test": 9.51, "flag_rate_pct": 6.82,
                 "fpr_pct": 5.21, "fnr_pct": 55.56, "fraud_base_rate_pct": 4.09},
                {"group": "over 65", "n": 75, "pct_of_test": 3.24, "flag_rate_pct": 4.00,
                 "fpr_pct": 4.05, "fnr_pct": 100.00, "fraud_base_rate_pct": 1.33},
            ],
        },
    },
    # Real global SHAP importances from Notebook 07 cell 27 (grouped back to original
    # features, mean |SHAP| across the full held-out test set).
    "global_shap": [
        {"feature": "Fault", "mean_abs_shap": 0.485106},
        {"feature": "BasePolicy", "mean_abs_shap": 0.461850},
        {"feature": "VehicleCategory", "mean_abs_shap": 0.157664},
        {"feature": "MonthClaimed", "mean_abs_shap": 0.071276},
        {"feature": "Deductible", "mean_abs_shap": 0.067459},
        {"feature": "Age", "mean_abs_shap": 0.066902},
        {"feature": "Month", "mean_abs_shap": 0.065123},
        {"feature": "Year", "mean_abs_shap": 0.052633},
        {"feature": "Make", "mean_abs_shap": 0.049600},
        {"feature": "DayOfWeek", "mean_abs_shap": 0.039077},
        {"feature": "DayOfWeekClaimed", "mean_abs_shap": 0.035438},
        {"feature": "AddressChange_Claim", "mean_abs_shap": 0.034683},
        {"feature": "NumberOfSuppliments", "mean_abs_shap": 0.026631},
        {"feature": "VehiclePrice", "mean_abs_shap": 0.025904},
        {"feature": "AgeOfVehicle", "mean_abs_shap": 0.020612},
    ],
    "qwen_status": {
        "mode": "precomputed",
        "model": "Qwen2.5",
        "reason": "Briefs were generated offline by the team's Qwen2.5 pipeline and are served precomputed "
                   "in this demo; no live endpoint is called.",
        "api_base": None,
    },
    "is_real_data": True,
    "dataset_placement_help": "",
}


class FraudPipeline:
    def __init__(self):
        self.claims: list[dict] = []
        self.by_id: dict[str, dict] = {}
        self.operating_threshold = OPERATING_THRESHOLD
        self.low_risk_max = LOW_RISK_MAX
        self.high_risk_min = OPERATING_THRESHOLD
        self.is_real_data = True
        self.metrics = METRICS

    def fit(self):
        data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        self.claims = data["claims"]
        self.by_id = {c["claim_id"]: c for c in self.claims}
        print(f"[FraudPipeline] Loaded {len(self.claims)} precomputed, rubric-passed "
              f"results from {DATA_PATH.name}. No live training — showing finalized outputs.")
        return self

    def list_claim_ids(self) -> list[str]:
        return [c["claim_id"] for c in self.claims]

    def _match_claim(self, claim: dict) -> dict:
        cid = claim.get("claim_id")
        if cid and cid in self.by_id:
            return self.by_id[cid]
        best, best_score = self.claims[0], -1
        for c in self.claims:
            details = c.get("claim_details", {})
            score = sum(1 for k, v in claim.items()
                        if k in details and str(details[k]) == str(v))
            if score > best_score:
                best, best_score = c, score
        return best

    def _to_drivers(self, rec: dict) -> list[dict]:
        """Real per-driver SHAP magnitude from demo_data.json (populated from Notebook
        07's TreeExplainer output) - falls back to a placeholder constant only if a
        record genuinely lacks one, never overwrites a real value with it."""
        drivers = []
        for d in rec.get("increased_risk_factors", []):
            contribution = float(d.get("shap_contribution", 0.10))
            drivers.append({"feature": d["feature"], "value": d["value"],
                            "shap_contribution": contribution, "direction": "up"})
        for d in rec.get("reduced_risk_factors", []):
            contribution = float(d.get("shap_contribution", -0.06))
            drivers.append({"feature": d["feature"], "value": d["value"],
                            "shap_contribution": contribution, "direction": "down"})
        return drivers

    def predict(self, claim: dict) -> dict:
        rec = self._match_claim(claim)
        cal_proba = float(rec["fraud_probability"])
        tier = rec["risk_tier"]
        action = rec.get("recommended_action", ACTION_BY_RISK_TIER.get(tier, "Manual Review"))
        prediction = int(cal_proba >= self.operating_threshold)
        drivers = self._to_drivers(rec)
        # Real per-claim human rubric score from notebooks/rubric/brief_scores.csv,
        # merged into demo_data.json - not a hardcoded perfect score. 44/46 claims
        # pass; the 2 that don't are shown with their real score, not hidden or
        # filtered out.
        rubric = rec.get("rubric", {"factual_consistency": NA, "faithfulness_to_shap": NA,
                                     "actionability": NA, "no_fabrication": NA,
                                     "total": NA, "max_total": 8, "verdict": "N/A"})
        details = rec.get("claim_details", {})
        withheld = [k for k in ("Sex", "MaritalStatus", "Age", "AgeOfPolicyHolder") if k in details]
        fields_sent = [k for k in details if k not in withheld]
        return {
            "claim_id": rec["claim_id"],
            "fraud_probability": round(cal_proba, 4),
            "raw_model_score": round(cal_proba, 4),
            "prediction": prediction,
            "prediction_label": "Fraud Flagged" if prediction else "No Fraud Flagged",
            "risk_tier": tier,
            "operating_threshold": self.operating_threshold,
            "recommended_action": action,
            "top_shap_drivers": drivers,
            "base_value": 0.06,
            "investigation_brief": rec.get("brief", ""),
            "brief_source": "precomputed",
            "brief_rubric_score": rubric,
            "privacy": {"fields_sent_to_qwen": fields_sent,
                        "fields_withheld_from_qwen": withheld,
                        "qwen_enabled": False},
            "human_review_status": "Pending Adjuster Review",
            "claim_details": details,
        }


PIPELINE = FraudPipeline()
