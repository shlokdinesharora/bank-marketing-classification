# Bank Marketing Subscription Prediction

An end-to-end imbalanced-classification study that predicts whether a customer will subscribe to a term deposit following a Portuguese bank's direct-marketing campaign.

The project covers exploratory analysis, leakage-aware feature selection, feature engineering, model benchmarking, stratified cross-validation, hyperparameter tuning, and decision-threshold optimization.

![Final XGBoost model performance](assets/final_model_performance.png)

## Problem

The target is highly imbalanced: only **11.27%** of the 41,188 customers subscribed. A majority-class model can therefore achieve approximately 88.7% accuracy while identifying no subscribers. Model selection emphasizes precision, recall, F1, ROC-AUC, and especially PR-AUC rather than accuracy alone.

## Highlights

- Excluded `duration` from the deployable model because call duration is unavailable before a marketing call ends.
- Replaced the `pdays = 999` sentinel with two interpretable features:
  - `previously_contacted`
  - `days_since_previous_contact`
- Retained categorical `"unknown"` values as explicit categories instead of guessing replacements.
- Compared Dummy Classifier, Logistic Regression, Decision Tree, Random Forest, and XGBoost.
- Used five-fold stratified cross-validation and PR-AUC-guided hyperparameter tuning.
- Selected a decision threshold from out-of-fold training probabilities rather than test labels.

## Final model

Tuned XGBoost produced the strongest cross-validated PR-AUC and was selected as the final model. Its decision threshold was optimized for subscriber F1 using out-of-fold predictions.

| Metric | Final holdout result |
|---|---:|
| ROC-AUC | 0.816 |
| PR-AUC | 0.489 |
| Subscriber precision | 0.48 |
| Subscriber recall | 0.59 |
| Subscriber F1 | 0.53 |
| Subscribers correctly identified | 552 / 928 |

The threshold-tuned model intentionally accepts more false positives in exchange for identifying substantially more potential subscribers.

## Repository structure

```text
bank-marketing-classification/
├── assets/
│   └── final_model_performance.png
├── bank_marketing_classification.ipynb
├── data/
│   └── README.md
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Dataset

This project uses `bank-additional-full.csv` from the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing).

Download the dataset and place the CSV at:

```text
data/bank-additional-full.csv
```

The dataset itself is excluded from this repository. See [`data/README.md`](data/README.md) for setup details.

## Running the notebook

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Open `bank_marketing_classification.ipynb` and run the cells in order.

> The XGBoost grid search evaluates hundreds of model configurations and may take several minutes depending on available hardware.

## Methodology

1. Inspect target imbalance, categorical unknowns, duplicates, and feature relationships.
2. Create a stratified 80/20 train/test split.
3. Remove post-call `duration` to preserve pre-call deployability.
4. Engineer meaningful previous-contact features from `pdays`.
5. One-hot encode categorical features and standardize numerical features using training data only.
6. Benchmark multiple classification models.
7. Compare model families with stratified cross-validation.
8. Tune Logistic Regression and XGBoost using PR-AUC.
9. Select a decision threshold using out-of-fold training probabilities.
10. Evaluate the selected model on the held-out test set.

## Main findings

- Previous campaign success is strongly associated with current subscription.
- Longer calls are associated with subscription, but call duration is not available for pre-call targeting and is excluded.
- Students and retired customers show the highest job-level subscription rates.
- Cellular contact is associated with a higher conversion rate than telephone contact.
- Economic indicators are strongly correlated, particularly `emp.var.rate`, `euribor3m`, and `nr.employed`.
- Threshold selection materially improves minority-class recall and F1 compared with the default 0.5 threshold.

## Limitations

- The observed relationships are associative rather than causal.
- A production threshold should reflect the bank's real contact costs and subscription value.
- The preprocessing and estimator should be consolidated into a persisted production pipeline before deployment.
- Future work could add probability calibration, SHAP or permutation importance, fairness checks, and monitoring for campaign drift.

## License

This project is licensed under the [MIT License](LICENSE).
