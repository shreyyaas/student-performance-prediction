# Student Performance Prediction

Predicting student academic outcomes using ensemble machine learning methods.

## Team
- Member 1 — EDA & Preprocessing
- Member 2 — Feature Engineering & evaluation
- Member 3 — Model Building & report

## Tech Stack
Python, scikit-learn, XGBoost, SHAP, pandas, matplotlib

## Dataset
[UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)

## Models Used
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting
- XGBoost

## Results
## Results

| Model | Accuracy | F1 Score |
|---|---|---|
| Random Forest | 68.4% | 0.797 |
| XGBoost | 68.4% | 0.786 |
| Logistic Regression | 63.3% | 0.739 |
| Gradient Boosting | 55.7% | 0.660 |

**Best Model:** Random Forest (F1 = 0.797)
**Key Finding:** Past failures and absences are the strongest predictors of student performance.
