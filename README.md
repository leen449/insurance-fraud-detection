# Insurance Claim Fraud Detection & Action Recommendation System

An explainable machine-learning system that screens vehicle-insurance claims for fraud, assigns each claim a calibrated risk tier, and generates a human-readable investigation brief with a recommended next action. Built as the capstone project for the Samsung Innovation Campus AI Course.

> **Scope note.** This is a proof-of-concept validated on a historical benchmark dataset (1994–1996). It demonstrates a leakage-safe, calibration-focused methodology rather than a system ready for present-day deployment; real-world use would require retraining on current claims data.

---

## Table of Contents
- [Overview](#overview)
- [Methodology](#methodology)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation & Usage](#installation--usage)
- [Team](#team)

---

## Overview

Manual insurance claim processing is slow, resource-intensive, and error-prone, often allowing fraudulent claims through while delaying legitimate payouts. This project builds a decision-support system that:

1. **Classifies** each claim as fraudulent or legitimate using gradient-boosted decision trees.
2. **Calibrates** the predicted probabilities so a "20% risk" score genuinely means ~20%.
3. **Assigns risk tiers** (Low / Medium / High) from the calibrated probabilities.
4. **Explains** each decision with TreeSHAP feature attributions.
5. **Recommends an action** (approve / manual review / investigate) via a generative reasoning model, grounded in the SHAP evidence and subject to a human-in-the-loop check.

The guiding principles throughout are **leakage-safe evaluation**, **honest handling of class imbalance**, and **trustworthy (calibrated) probabilities**.
---

## Methodology

| Stage | Approach |
|-------|----------|
| **Preprocessing** | Ordinal encoding for ordered fields, one-hot for nominal; all transformers fit **inside CV folds only** to prevent leakage |
| **Class imbalance** | Cost-sensitive learning (class weighting) vs. SMOTENC, compared under identical stratified cross-validation |
| **Baselines** | Logistic Regression, Balanced Random Forest (imbalance handled, no hyperparameter tuning) |
| **Candidate models** | XGBoost, LightGBM (tuned with Optuna over model + balancing + hyperparameters) |
| **Selection** | Best configuration chosen on validation **CV PR-AUC**, then evaluated **once** on the locked test set |
| **Calibration** | Reliability curve + Brier score; Platt / isotonic recalibration where it improves reliability |
| **Thresholds & risk tiers** | Operating threshold and Low/Med/High boundaries set on validation from calibrated probabilities — never on the test set |
| **Explainability** | TreeSHAP global + per-claim local attributions |
| **Action recommendation** | Qwen2.5 generates an investigation brief from the SHAP drivers; the classifier makes the decision, the model only explains and recommends |

Primary metric: **PR-AUC**. Reported alongside Recall, Precision, Macro-F1, MCC, and calibration (Brier / reliability).

---

## Results
>Note on Model Comparison & Evaluation:
>Untuned baseline models (Logistic Regression, Balanced Random Forest) are evaluated directly on the left-out validation set. Candidate models >(XGBoost, LightGBM) are tuned and compared using 5-fold cross-validation on the training set. These scores reflect different evaluation setups and are reported in separate columns to ensure fair, transparent comparisons.

| Model | Balancing | PR-AUC | Recall | Precision | Brier |
|-------|-----------|:------:|:------:|:---------:|:-----:|
| Logistic Regression | class_weight | 0.140 | 0.871 | 0.131 | 0.198 |
| Balanced Random Forest | built-in | 0.204 | 0.842 | 0.139 | 0.156 |
| XGBoost |`class_weight`| 0.281 | 0.734 | 0.162 | 0.133 |
| LightGBM | `SMOTENC` | 0.235 | 0.081 | 0.375 | 0.053 |

Final model: `XGBoost + class_weight ` · Test-set PR-AUC: `0.195`

Key figures are in [`results/figures/`](results/figures/).

---

## Repository Structure

```text
insurance-fraud-detection/
│
├── README.md
├── requirements.txt
├── .gitignore                 # ignores data, models, AND prompts/
│
├── notebooks/                 # run order
│   ├── 01_eda_and_statistics.ipynb
│   ├── 02_preprocessing_and_splitting.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_modelling_and_selection.ipynb        # Experiment + Optuna → select on VALIDATION (no test set here)
│   ├── 05_calibration_and_thresholds.ipynb     # calibrate + tune threshold/tiers on VALIDATION
│   ├── 06_final_test_evaluation.ipynb          # LOCKED TEST SET — run once, after model+calibration+thresholds frozen
│   ├── 07_shap_and_qwen.ipynb
│   ├── 08_rubric_evaluation.ipynb              # scores GenAI briefs
│   └── prompts/                # GIT-IGNORED — prompt text lives here, never committed
│       └── brief_prompt.txt
│
├── genai.py                   # loads prompt from git-ignored file; Qwen call + safety rubric
├── pipeline.py                # end-to-end runner
│
├── app/                       # Gradio demo 
│   └── app.py
│
├── data/
│   └── README.md              # dataset source + CC0 license
│
├── results/                   # small final outputs committed for graders
│   ├── metrics.json
│   └── figures/
│
└── report/
    └── final_report.docx
```

---

## Installation & Usage

```bash
git clone https://github.com/leen449/insurance-fraud-detection.git
cd insurance-fraud-detection
pip install -r requirements.txt
```

**Environment:** developed in Google Colab. Key libraries: `scikit-learn`, `xgboost`, `lightgbm`, `imbalanced-learn`, `optuna`, `shap`. See `requirements.txt` for versions.

### Google Drive Setup (Notebooks 01–07)

Notebooks 01–07 run in Colab against a shared Drive folder, mounted as `PROJECT_ROOT`:

```text
InsuranceFraudProject/                 # My Drive - PROJECT_ROOT for Notebooks 01-07
│
├── data/
│   ├── raw/
│   │   └── fraud_oracle.csv           # ADD MANUALLY - the only file you place by hand
│   └── processed/                     # auto-written by Notebook 02
│       ├── cleaned.parquet
│       ├── train.parquet
│       ├── val.parquet
│       ├── test.parquet
│       ├── train_temporal.parquet
│       ├── test_temporal.parquet
│       └── split_manifest.json
│
├── genai/                             # manual backup copy - no notebook reads/writes this folder
│   └── brief_prompt.txt
│
├── models/                            # auto-written by Notebooks 02, 04, 05
│   ├── preprocessing_pipeline.joblib
│   ├── best_model.joblib
│   ├── best_model_config.json
│   ├── calibrated_model.joblib
│   └── risk_config.json
│
├── results/
│   ├── figures/                       # auto-written by Notebooks 01, 03-07
│   │   ├── eda/
│   │   ├── baseline/
│   │   ├── modelling/
│   │   ├── calibration/
│   │   ├── test/
│   │   └── shap/
│   ├── experiments/                   # manual - not produced by any notebook
│   ├── experiments.csv                # auto-appended by Notebooks 03, 04, 05, 06
│   ├── fairness_audit_test.json       # auto-written by Notebook 06
│   ├── generated_briefs.json          # manual backup - Notebook 08 writes this locally, not to Drive
│   ├── metrics.json                   # auto-written by Notebook 06
│   └── shap_claim_briefs.json         # auto-written by Notebook 07 - see handoff note below
│
└── rubric/                            # manual backup - Notebook 08 writes this locally, not to Drive
    └── brief_scores.csv
```

**Setting it up:** create a folder named exactly `InsuranceFraudProject` in `My Drive`,
then add `data/raw/fraud_oracle.csv` — that's the only file you place there by hand.
Everything else under `data/`, `models/`, and `results/` is generated automatically by
running Notebooks 01→07 in order, in Colab, with the Drive mounted; each notebook reads
what the previous one wrote and creates its own output folders as it goes.

**Handoff to Notebook 08** (which runs locally, not in Colab): Notebook 07 writes
`results/shap_claim_briefs.json` to Drive. Copy that one file down to
`notebooks/shap_claim_briefs.json` on your machine before running Notebook 08 — that's
the only Drive-to-local step in the whole pipeline; Notebook 08 needs no Drive access
of its own, and writes its own outputs locally to `notebooks/{prompts,results,rubric}/`
(see that notebook's own setup notes).

**`genai/`, `experiments/`, `generated_briefs.json`, and `rubric/brief_scores.csv`** in
the tree above are manually-kept backups, not something any Colab notebook produces —
safe to ignore when setting up from scratch.

---

## Team

**Innovexa — Team 4 · Samsung Innovation Campus AI Course**

| Name | Role |
|------|------|
| Fatimah Almousa | Project management & coordination, reporting |
| Danah Alessa | Data engineering (acquisition & EDA), training & evaluation |
| Dalal Aldawsari | Preprocessing & feature engineering, calibration |
| Leen Binmueqal | Preprocessing, model development |
| Rawan Asiri | Data engineering (EDA), explainability & reporting |
| Futun Basha | Model development & imbalance comparison, explainability |

---
## References

- Komsrimorakot, P., & Siriborvornratanakul, T. (2025). Enhancing fraud detection in imbalanced motor insurance datasets using CP-SMOTE and Random Under-Sampling. *Journal of Big Data*, 12:172.
- Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16.
- Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*.
