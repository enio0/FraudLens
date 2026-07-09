# FraudLens

Credit card fraud detection with a SHAP-based explainability layer. Built as a portfolio and learning project, going step by step from raw data to a trained model to human-readable adverse-action reports.

## Overview

Given a highly imbalanced dataset (0.17% fraud), this project builds a fraud detection model, evaluates it honestly (not with accuracy, which is misleading here), tries multiple approaches to correct for the imbalance, and ultimately produces a model that can explain, in plain English, why any individual transaction was flagged.

## Dataset

[Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud). 284,807 transactions, 31 columns, no missing values. Features V1-V28 are anonymized PCA components (original features are confidential); `Amount` and `Time` are the only two features left in their original form. Fraud rate: 0.17% (492 of 284,807 transactions).

## Project structure

```
fraudlens/
├── data/raw/creditcard.csv        # not committed, see .gitignore
├── models/random_forest_fraud_model.joblib
├── 01_explore.ipynb               # Phase 1: EDA
├── 02_baseline.ipynb              # Phase 2: baseline model + honest evaluation
├── 03_balanced.ipynb              # Phase 3: imbalance correction attempts
├── 04_forest.ipynb                # Phase 4: Random Forest
├── 05_shap.ipynb                  # Phase 5: SHAP explainability
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download `creditcard.csv` from Kaggle and place it in `data/raw/`.

## Phase 1: Exploratory Data Analysis

Confirmed no missing values, quantified the severity of the class imbalance (0.17% fraud), and compared transaction `Amount` across fraud vs. legitimate classes (found to be a weak standalone signal, later confirmed by SHAP in Phase 5).

## Phase 2: Baseline model and honest evaluation

Trained a plain logistic regression on a stratified 80/20 train/test split. The key lesson of this phase: **accuracy is meaningless on data this imbalanced.** A model that always predicts "not fraud" scores 99.83% accuracy while catching zero fraud. The trained baseline scored a nearly identical ~99.9% accuracy while actually catching some fraud, accuracy alone couldn't tell the two models apart.

Instead, the project uses precision, recall, and AUPRC (area under the precision-recall curve) throughout, since these actually reflect performance on the rare, important class.

**Baseline results:**

| Metric | Score |
|---|---|
| Precision | 0.8312 |
| Recall | 0.6531 |
| AUPRC | 0.7427 |

(Random-baseline AUPRC = 0.0017, the fraud rate itself, so the model is roughly 430x better than chance.)

Weakness identified: recall. The model misses about a third of real fraud.

## Phase 3: Correcting for imbalance

Two standard techniques were tried to address the recall weakness, plus threshold tuning on each.

| Model | Threshold | Precision | Recall | AUPRC |
|---|---|---|---|---|
| Baseline | 0.50 | 0.8312 | 0.6531 | 0.7427 |
| Baseline, tuned | 0.20 | 0.7327 | 0.7551 | 0.7427 |
| `class_weight='balanced'` | 0.50 | 0.0613 | 0.9184 | 0.7159 |
| `class_weight='balanced'`, tuned | 0.90 | 0.2437 | 0.8878 | 0.7159 |
| SMOTE | 0.50 | 0.1369 | 0.8980 | 0.7389 |
| SMOTE, tuned | 0.90 | 0.4123 | 0.8878 | 0.7389 |

**Key finding:** neither `class_weight='balanced'` nor SMOTE meaningfully improved AUPRC (both stayed close to or below the plain baseline's 0.7427). Both techniques mainly distorted the model's probability calibration, pushing predicted probabilities broadly higher because the model was trained on artificially balanced data, rather than improving its actual ability to rank transactions by suspicion. This showed up as recall jumping sharply at the default threshold while precision collapsed, and as threshold sweeps that stayed unusually flat compared to the baseline's smooth, well-behaved curve.

The strongest result in this phase was simply **threshold tuning on the honestly-trained baseline** (0.20 → precision 0.73, recall 0.76), which beat both correction techniques even after their own tuning.

## Phase 4: Random Forest

Tested whether logistic regression's ~0.74 AUPRC was a ceiling imposed by the dataset, or by the model itself. It was the model.

| Metric | Baseline (LR) | Random Forest |
|---|---|---|
| Precision | 0.8312 | 0.9412 |
| Recall | 0.6531 | 0.8163 |
| AUPRC | 0.7427 | 0.8734 |

Random Forest improved precision, recall, and AUPRC simultaneously, something no logistic regression variant achieved in Phase 3, with no resampling or reweighting needed. This suggests fraud in this dataset involves non-linear feature interactions that a linear model (logistic regression) can't capture, but an ensemble of trees can.

After threshold tuning, the final chosen operating point is **threshold 0.40: precision 0.9333, recall 0.8571**, catching 84 of 98 test-set frauds with very few false alarms. This is the model used for the explainability layer in Phase 5, and the one saved to `models/random_forest_fraud_model.joblib`.

## Phase 5: SHAP explainability

Uses `shap.TreeExplainer` on the Random Forest model to explain individual predictions, then aggregates across a sample of the test set to find global feature importance.

**Top features by average impact on fraud predictions:** V17, V14, V12, V1, V4, V10, V28, V16, V11, V2, confirmed both in single-transaction examples and across a 2,000-row sample. Notably, it is *low* values of V17, V14, and V12 (not high) that are associated with the model's fraud predictions in this dataset, visible in the SHAP summary plot as blue (low-value) outlier points on the far-right (high SHAP value) side for these features.

`Amount` ranked low in importance, consistent with the weak signal found in Phase 1's EDA.

### Adverse-action report function

`generate_adverse_action_report()` takes a single transaction and produces a plain-English explanation: the model's decision, its confidence, the actual outcome, the top contributing features and their direction of impact, and an explicit note that features V1-V28 are anonymized components whose real-world meaning is unknown (rather than fabricating a false interpretation).

Example output for a correctly caught fraud case (probability 0.90):

```
Top 5 contributing factors:
  - V17: pushed toward fraud (impact: +0.3105)
  - V14: pushed toward fraud (impact: +0.1925)
  - V12: pushed toward fraud (impact: +0.1190)
  - V10: pushed toward fraud (impact: +0.0915)
  - V16: pushed toward fraud (impact: +0.0903)
```

Also validated on a false positive case (a legitimate transaction incorrectly flagged at 0.59 probability), where the contributing factors were mixed rather than unanimous, one feature (V14) pushed toward "legitimate" while the rest pushed toward "fraud", a meaningfully different, more honest signature than the confident correct call above, and useful for real dispute scenarios.

## Key takeaways

- Accuracy is the wrong metric for imbalanced classification; precision, recall, and AUPRC tell the real story.
- Reweighting (`class_weight`) and resampling (SMOTE) are not guaranteed to improve a model, they can just shift where a probability threshold happens to fall, without improving the model's actual discriminative skill (as measured by AUPRC).
- Trying a different model family can matter more than trying harder to fix the data balance for the same model. Random Forest broke through a ceiling that no amount of reweighting or resampling could move for logistic regression.
- A "black box" model can still be made explainable: SHAP values let you state, concretely and per-transaction, which features drove a specific decision, and whether the evidence was unanimous or mixed.

## Possible future work

- Feature scaling (StandardScaler on `Amount` and `Time`), flagged early on as a likely convergence issue for logistic regression, never revisited once Random Forest (which doesn't require scaling) became the primary model.
- Gradient boosting models (XGBoost, LightGBM) as a further comparison point against Random Forest.
- A lightweight script or small web demo that loads the saved model and generates an adverse-action report for a user-supplied transaction.
