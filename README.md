# Customer Churn Prediction

An end-to-end machine learning pipeline to predict customer churn for a telecom company. The project covers data cleaning, exploratory data analysis, feature engineering, model training with class imbalance handling, and model evaluation using multiple metrics.

---

## Problem Statement

Customer churn — when a customer stops using a service — is a critical business problem for telecom companies. Acquiring a new customer costs 5–7x more than retaining an existing one. Identifying customers likely to churn before they leave allows the business to take proactive retention action.

**Goal:** Build a classification model that predicts whether a customer will churn (Yes/No) based on their demographics, account information, and service usage patterns.

---

## Dataset

**Source:** Kaggle — Telco Customer Churn
**Size:** 7,043 rows × 21 columns
**Target variable:** `Churn` (Yes / No)

Key features include:
- Demographics: gender, SeniorCitizen, Partner, Dependents
- Account info: tenure, Contract type, PaymentMethod, MonthlyCharges, TotalCharges
- Services: PhoneService, MultipleLines, InternetService, OnlineSecurity, TechSupport, StreamingTV

---

## Project Structure

```
customer-churn/
├── notebooks/
│   ├── 01_data_understanding.ipynb      # Initial data exploration
│   ├── 02_data_cleaning_eda.ipynb       # Data cleaning and EDA
│   └── 03Featuring_Modeling.ipynb       # Feature engineering and ML models
├── data/
│   ├── raw/                             # Original dataset
│   └── processed/                       # Cleaned dataset
└── README.md
```

---

## Approach

### Notebook 1 — Data Understanding
- Shape, dtypes, null check, basic statistics
- Initial understanding of feature distributions

### Notebook 2 — Data Cleaning & EDA
**Cleaning:**
- Converted TotalCharges from object to numeric using pd.to_numeric 
  with errors='coerce' (blank strings became NaN)
- Filled resulting NaN values with median (distribution is right-skewed)
- Filled missing `TotalCharges` values with median
- Encoded binary Yes/No columns to 1/0

**EDA Highlights:**
- Month-to-month contract customers churn at significantly higher rates than annual contract customers
- Electronic check payment method shows highest churn rate among payment types
- Fiber optic internet service customers churn more than DSL customers
- Higher MonthlyCharges correlate with increased churn probability
- Short tenure customers (0–12 months) are most at risk

### Notebook 3 — Feature Engineering & Modeling

**Feature Engineering:**
- `tenure_group` — bucketed tenure into 6 groups (0–12, 13–24, 25–36, 37–48, 49–60, 61–72 months)
- `avg_monthly_spend` — TotalCharges / tenure
- `num_services_used` — count of active services per customer
- `auto_payment_flag` — 1 if customer uses automatic payment method

**Pipeline:**
- Separate `num_pipeline` (imputation + scaling) and `cat_pipeline` (imputation + one-hot encoding)
- Combined via `ColumnTransformer`
- `StratifiedShuffleSplit` used for train/test split to preserve class balance

**Class Imbalance Handling:**
- Dataset is imbalanced (~73% No churn, ~27% Yes churn)
- Applied SMOTE (Synthetic Minority Oversampling) on Gradient Boosting model
- Used `class_weight='balanced'` on Logistic Regression

---

## Model Results

| Model | ROC-AUC | Recall (Churn) | F1 (Churn) |
|-------|---------|----------------|------------|
| Logistic Regression | 0.839 | 0.78 | 0.62 |
| Gradient Boosting | 0.838 | 0.80 | 0.63 |
| Random Forest | 0.805 | 0.50 | 0.53 |

**Selected Model: Gradient Boosting**

Gradient Boosting achieved the highest churn recall (0.80) — meaning it correctly identified 80% of customers who actually churned. In a business context, recall matters more than precision for churn prediction: missing a churner (false negative) is more costly than incorrectly flagging a loyal customer (false positive).

---

## Why Recall Over Accuracy?

Overall accuracy is misleading on imbalanced datasets. A model that predicts "No churn" for every customer would achieve ~73% accuracy but catch zero churners. Recall on the churn class directly measures what matters: how many at-risk customers did we actually identify?

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas, NumPy | Data manipulation and cleaning |
| Matplotlib, Seaborn | Visualizations |
| Scikit-learn | Pipelines, preprocessing, ML models |
| Imbalanced-learn | SMOTE for class imbalance |

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

# Run notebooks in order
# 1. notebooks/01_data_understanding.ipynb
# 2. notebooks/02_data_cleaning_eda.ipynb
# 3. notebooks/03Featuring_Modeling.ipynb
```

---

## Key Business Insights

- **Contract type is the strongest churn predictor** — customers on month-to-month contracts churn 3x more than those on two-year contracts
- **Electronic check users are highest risk** — likely due to lack of payment automation
- **Early tenure is critical** — most churn happens in the first 12 months; early engagement programs would have highest ROI
- **Fiber optic customers need attention** — higher monthly charges combined with higher churn suggests a value perception problem

---

## Author

**Parag Pawar**
- GitHub: https://github.com/paragpawar0077
- Email: paragpawar0077@gmail.com