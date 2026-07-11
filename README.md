# Marketing Campaign Response Predictor

## 1. Overview

This project predicts whether a customer will **respond to a marketing campaign** using Machine Learning. It combines **unsupervised learning (K-Means clustering)** for customer segmentation with **supervised learning (Logistic Regression)** for response prediction. The dataset is the Kaggle *Customer Personality Analysis* marketing campaign dataset. Users can input customer details through a **Streamlit app**, and the model outputs the prediction along with the response probability.

The primary business goal of this model is to help identify customers who are **unlikely to respond**, so marketing resources and campaign costs can be better targeted toward customers with higher response potential.

---

## 2. Files

| File | Description |
|------|-------------|
| `marketing_campaign.csv` | Original marketing campaign dataset |
| `Marketing_EDA_Modeling.ipynb` | Notebook with data cleaning, feature engineering, EDA, clustering, and model training |
| `Marketing_UI.py` | Streamlit app for interactive predictions |
| `campaign_response_lr_model.pkl` | Trained Logistic Regression model |
| `campaign_response_scale.pkl` | StandardScaler used for feature scaling |
| `README.md` | Project description |

---

## 3. Installation

Install Python (recommended ≥ 3.10)

Install required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

---

## 4. Usage

Clone the repository:

```bash
git clone <repository_url>
cd <project_folder>
```

Run the Streamlit app:

```bash
streamlit run Marketing_UI.py
```

Enter customer details in the UI and get the campaign response prediction (Accept / Reject) along with probability score.

---

## 5. Model Information

- **Algorithm:** Logistic Regression (best-performing base model among Logistic Regression, KNN, SVC, Naive Bayes, Decision Tree, and Random Forest)
- **Final Accuracy:** ~90.2% on test data

### Confusion Matrix

```text
[[371  10]
 [ 34  33]]
```

- **Recall (No / Non-Responders):** ~97.4%
- **Recall (Yes / Responders):** ~49.2%

> **Note:** Recall for the "Yes" (Responder) class is intentionally lower relative to the "No" class. Since the business objective is to accurately flag customers who are **unlikely to respond** (so they can be excluded or handled differently in campaign targeting), high recall on the "No" class was prioritized. This trade-off is acceptable for this use case, though it means some genuine responders are missed.

### Key Features Used

#### Demographics
- `Income`
- `age`
- `family_size`
- `children`
- `is_parent`

#### Behavioral
- `Recency`
- `customer_tenure`
- `total_purchases`
- `web_purchase_ratio`

#### Spending
- `MntWines`
- `MntFruits`
- `MntMeatProducts`
- `MntFishProducts`
- `MntSweetProducts`
- `MntGoldProds`
- `total_spent`
- `avg_spend_per_purchase`
- `income_per_person`

#### Purchase Channels
- `NumWebPurchases`
- `NumCatalogPurchases`
- `NumStorePurchases`
- `NumDealsPurchases`
- `NumWebVisitsMonth`

#### Campaign History
- `AcceptedCmp1`–`AcceptedCmp5`
- `total_campaign_acceptance`

#### Categorical (one-hot encoded)
- `Education_*`
- `Marital_Status_*`

---

## 6. Unsupervised Learning – Customer Segmentation

Prior to classification, **K-Means clustering (K=3)** was applied (validated via Elbow Method and Silhouette Score) to segment customers based on income, spending, and purchase behavior:

- **Low Value:** Lowest income and spending, low campaign acceptance
- **High Value (VIP):** Highest income, spending, and campaign acceptance
- **Moderate Value:** Mid-range income/spending with the highest web purchase ratio

These segments provide additional business insight for targeted marketing strategy alongside the response prediction model.

---

## 7. Notes

- Missing `Income` values were imputed using the median.
- Outlier ages (> 100 years) were removed.
- Engineered features (`total_spent`, `family_size`, `customer_tenure`, ratios, etc.) were created to strengthen the feature set.
- Categorical variables (`Education`, `Marital_Status`) were converted to one-hot encoded dummy variables.
- Feature selection was performed using correlation analysis and Chi-Square test (`chi2`) for statistical significance.
- Model selection compared multiple base classifiers (Logistic Regression, KNN, SVC, Naive Bayes, Decision Tree, Random Forest) using accuracy and confusion matrix before finalizing Logistic Regression as the deployed model.
