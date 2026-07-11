import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('campaign_response_lr_model.pkl')
scaler = joblib.load('campaign_response_scale.pkl')

st.set_page_config(page_title="Marketing Campaign Response Predictor", layout="centered")
st.title("🎯 Marketing Campaign Response Predictor")

# Exact feature order the model/scaler were trained on
FEATURE_COLUMNS = [
    'Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
    'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
    'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
    'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2',
    'age', 'children', 'family_size', 'is_parent', 'customer_tenure',
    'total_spent', 'total_purchases', 'total_campaign_acceptance',
    'avg_spend_per_purchase', 'income_per_person', 'web_purchase_ratio',
    'Education_Basic', 'Education_Graduation', 'Education_Master', 'Education_PhD',
    'Marital_Status_Divorced', 'Marital_Status_Married', 'Marital_Status_Single',
    'Marital_Status_Together', 'Marital_Status_Widow'
]

# Reference date used for customer_tenure — set this to df['Dt_Customer'].max() from training
REFERENCE_DATE = pd.to_datetime("2014-06-29")

# ================= UI =================
Year_Birth = st.number_input("Year of Birth", min_value=1900, max_value=2014, value=1980)
Education = st.selectbox("Education", ["2n Cycle", "Basic", "Graduation", "Master", "PhD"])
Marital_Status = st.selectbox("Marital Status", ["Absurd", "Alone", "Divorced", "Married", "Single", "Together", "Widow", "YOLO"])
Income = st.number_input("Income", value=52000)
Kidhome = st.number_input("Kids at Home", min_value=0, max_value=5, value=0)
Teenhome = st.number_input("Teens at Home", min_value=0, max_value=5, value=0)
Dt_Customer = st.date_input("Customer Enrollment Date", value=pd.to_datetime("2013-06-15"))
Recency = st.number_input("Recency (days since last purchase)", value=30)

MntWines = st.number_input("Wine Spend", value=200)
MntFruits = st.number_input("Fruit Spend", value=20)
MntMeatProducts = st.number_input("Meat Spend", value=150)
MntFishProducts = st.number_input("Fish Spend", value=30)
MntSweetProducts = st.number_input("Sweet Spend", value=20)
MntGoldProds = st.number_input("Gold Prod Spend", value=40)

NumDealsPurchases = st.number_input("Num Deals Purchases", value=2)
NumWebPurchases = st.number_input("Num Web Purchases", value=4)
NumCatalogPurchases = st.number_input("Num Catalog Purchases", value=2)
NumStorePurchases = st.number_input("Num Store Purchases", value=5)
NumWebVisitsMonth = st.number_input("Num Web Visits/Month", value=5)

AcceptedCmp1 = st.selectbox("Accepted Campaign 1", ["No", "Yes"])
AcceptedCmp2 = st.selectbox("Accepted Campaign 2", ["No", "Yes"])
AcceptedCmp3 = st.selectbox("Accepted Campaign 3", ["No", "Yes"])
AcceptedCmp4 = st.selectbox("Accepted Campaign 4", ["No", "Yes"])
AcceptedCmp5 = st.selectbox("Accepted Campaign 5", ["No", "Yes"])

# ================= PREDICTION =================
if st.button("Predict"):

    # ---- basic binary mappings ----
    AcceptedCmp1 = 1 if AcceptedCmp1 == "Yes" else 0
    AcceptedCmp2 = 1 if AcceptedCmp2 == "Yes" else 0
    AcceptedCmp3 = 1 if AcceptedCmp3 == "Yes" else 0
    AcceptedCmp4 = 1 if AcceptedCmp4 == "Yes" else 0
    AcceptedCmp5 = 1 if AcceptedCmp5 == "Yes" else 0

    # ---- engineered features (same as training) ----
    age = 2014 - Year_Birth
    children = Kidhome + Teenhome
    family_size = children + 2 if Marital_Status in ["Married", "Together"] else children + 1
    is_parent = 1 if children > 0 else 0
    customer_tenure = (REFERENCE_DATE - pd.to_datetime(Dt_Customer)).days

    total_spent = MntWines + MntFruits + MntMeatProducts + MntFishProducts + MntSweetProducts + MntGoldProds
    total_purchases = NumWebPurchases + NumCatalogPurchases + NumStorePurchases
    total_campaign_acceptance = AcceptedCmp1 + AcceptedCmp2 + AcceptedCmp3 + AcceptedCmp4 + AcceptedCmp5
    avg_spend_per_purchase = total_spent / (total_purchases + 1)
    income_per_person = Income / family_size
    web_purchase_ratio = NumWebPurchases / (total_purchases + 1)

    # ---- one-hot encoding (drop_first=True, same as training) ----
    Education_Basic = 1 if Education == "Basic" else 0
    Education_Graduation = 1 if Education == "Graduation" else 0
    Education_Master = 1 if Education == "Master" else 0
    Education_PhD = 1 if Education == "PhD" else 0

    Marital_Status_Divorced = 1 if Marital_Status == "Divorced" else 0
    Marital_Status_Married = 1 if Marital_Status == "Married" else 0
    Marital_Status_Single = 1 if Marital_Status == "Single" else 0
    Marital_Status_Together = 1 if Marital_Status == "Together" else 0
    Marital_Status_Widow = 1 if Marital_Status == "Widow" else 0
    # note: "Absurd", "Alone", "YOLO" had no dummy column in training (too rare),
    # so selecting them just leaves all Marital_Status_* flags at 0.

    # ---- build feature row in the exact trained order ----
    feat = {
        'Income': Income, 'Recency': Recency,
        'MntWines': MntWines, 'MntFruits': MntFruits, 'MntMeatProducts': MntMeatProducts,
        'MntFishProducts': MntFishProducts, 'MntSweetProducts': MntSweetProducts, 'MntGoldProds': MntGoldProds,
        'NumDealsPurchases': NumDealsPurchases, 'NumWebPurchases': NumWebPurchases,
        'NumCatalogPurchases': NumCatalogPurchases, 'NumStorePurchases': NumStorePurchases,
        'NumWebVisitsMonth': NumWebVisitsMonth,
        'AcceptedCmp3': AcceptedCmp3, 'AcceptedCmp4': AcceptedCmp4, 'AcceptedCmp5': AcceptedCmp5,
        'AcceptedCmp1': AcceptedCmp1, 'AcceptedCmp2': AcceptedCmp2,
        'age': age, 'children': children, 'family_size': family_size, 'is_parent': is_parent,
        'customer_tenure': customer_tenure,
        'total_spent': total_spent, 'total_purchases': total_purchases,
        'total_campaign_acceptance': total_campaign_acceptance,
        'avg_spend_per_purchase': avg_spend_per_purchase,
        'income_per_person': income_per_person, 'web_purchase_ratio': web_purchase_ratio,
        'Education_Basic': Education_Basic, 'Education_Graduation': Education_Graduation,
        'Education_Master': Education_Master, 'Education_PhD': Education_PhD,
        'Marital_Status_Divorced': Marital_Status_Divorced, 'Marital_Status_Married': Marital_Status_Married,
        'Marital_Status_Single': Marital_Status_Single, 'Marital_Status_Together': Marital_Status_Together,
        'Marital_Status_Widow': Marital_Status_Widow,
    }

    row = pd.DataFrame([feat])[FEATURE_COLUMNS]
    final_features = scaler.transform(row)

    prediction = model.predict(final_features)[0]
    probability = model.predict_proba(final_features)[0][1]

    if prediction == 1:
        st.success(f"✅ Likely to ACCEPT the campaign (Probability: {probability:.2f})")
    else:
        st.error(f"❌ Likely to REJECT the campaign (Probability: {probability:.2f})")
