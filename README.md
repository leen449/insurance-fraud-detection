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

| Model | Balancing | PR-AUC | Recall | Precision | Brier |
|-------|-----------|:------:|:------:|:---------:|:-----:|
| Logistic Regression | class_weight | – | – | – | – |
| Balanced Random Forest | built-in | – | – | – | – |
| XGBoost | best (Optuna) | – | – | – | – |
| LightGBM | best (Optuna) | – | – | – | – |

Final model: ` ` · Test-set PR-AUC: ` `

Key figures are in [`results/figures/`](results/figures/).

---

## Repository Structure

```text
insurance-fraud-detection/
├── notebooks/
│   ├── 01_eda_and_statistics.ipynb
│   ├── 02_preprocessing_and_splitting.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_modelling_selection_and_test.ipynb   # Experiment + Optuna → select → locked test set
│   ├── 05_calibration_and_thresholds.ipynb
│   ├── 06_risk_tiers.ipynb
│   ├── 07_shap_and_qwen.ipynb
│   └── 08_rubric_evaluation.ipynb              # scores GenAI briefs
├── genai.py            # prompt + Qwen call + safety rubric (shared)
├── pipeline.py         # end-to-end runner: claim → model → calibrate → tier → SHAP → Qwen → brief
├── app/                # Gradio demo 
├── data/README.md      # dataset source + license
├── results/            # small final outputs (metrics, comparison, key figures)
├── requirements.txt
└── report/
```

---

## Installation & Usage

```bash
git clone https://github.com/leen449/insurance-fraud-detection.git
cd insurance-fraud-detection
pip install -r requirements.txt
```

**Environment:** developed in Google Colab. Key libraries: `scikit-learn`, `xgboost`, `lightgbm`, `imbalanced-learn`, `optuna`, `shap`. See `requirements.txt` for versions.


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
