# 💳 Credit Card Fraud Detection

Detecting fraudulent transactions in a highly **imbalanced** dataset (~0.17% fraud) by comparing classifiers, cross-validation strategies and resampling techniques.

## What's inside
| Notebook | Focus |
|---|---|
| `Creditcardfraud.ipynb` | Full pipeline: EDA → skew handling → Logistic Regression (L1/L2), KNN, Decision Tree, Random Forest, XGBoost → RepeatedKFold vs StratifiedKFold → **Random Oversampling, SMOTE, ADASYN** → hyperparameter tuning |
| `credit-card-fraud-detect-under-over-sampling.ipynb` | Under- vs over-sampling comparison with Logistic Regression, Decision Tree and Random Forest (follow-along practice) |

## Result
After oversampling and tuning, **XGBoost** gave the best performance with a **ROC-AUC of ≈ 0.98**.

## Dataset
The 144 MB dataset is not stored in this repo (GitHub's file limit). Download `creditcard.csv` from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it next to the notebooks.

## Tech stack
Python · pandas · scikit-learn · XGBoost · imbalanced-learn · Matplotlib · Seaborn
