#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =============================================================
# 🛒 SMARTCART CUSTOMER SEGMENTATION APP (FINAL)
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------------------
# Page Config
# -------------------------------------------------------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🛒 SmartCart Customer Segmentation")
st.write("Segment customers based on behavior using Machine Learning")

# -------------------------------------------------------------
# Load Models
# -------------------------------------------------------------
@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    model = joblib.load("cluster_model.pkl")
    return scaler, pca, model

scaler, pca, model = load_models()

# -------------------------------------------------------------
# Input Section
# -------------------------------------------------------------
st.subheader("Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    income = st.number_input("Income", value=50000)
    recency = st.number_input("Recency", value=30)
    age = st.number_input("Age", value=30)

with col2:
    total_spending = st.number_input("Total Spending", value=500)
    total_children = st.number_input("Total Children", value=1)
    tenure = st.number_input("Customer Tenure Days", value=100)

with col3:
    education = st.selectbox("Education", ["Undergraduate", "Graduate", "Postgraduate"])
    living_with = st.selectbox("Living With", ["Alone", "Partner"])

# -------------------------------------------------------------
# Prediction
# -------------------------------------------------------------
if st.button("🔍 Predict Cluster"):

    try:
        # Base DataFrame (exact training features)
        data = pd.DataFrame({
            "Income": [income],
            "Recency": [recency],
            "Age": [age],
            "Customer_Tenure_Days": [tenure],
            "Total_Spending": [total_spending],
            "Total_Children": [total_children],
        })

        # -----------------------------------------------------
        # One-Hot Encoding (MATCH TRAINING EXACTLY)
        # -----------------------------------------------------
        data["Education_Graduate"] = 1 if education == "Graduate" else 0
        data["Education_Postgraduate"] = 1 if education == "Postgraduate" else 0
        data["Education_Undergraduate"] = 1 if education == "Undergraduate" else 0

        data["Living_With_Alone"] = 1 if living_with == "Alone" else 0
        data["Living_With_Partner"] = 1 if living_with == "Partner" else 0

        # -----------------------------------------------------
        # Final Column Order (VERY IMPORTANT)
        # -----------------------------------------------------
        final_cols = [
            'Income','Recency','Age','Customer_Tenure_Days',
            'Total_Spending','Total_Children',
            'Education_Graduate','Education_Postgraduate','Education_Undergraduate',
            'Living_With_Alone','Living_With_Partner'
        ]

        data = data[final_cols]
        data = data.astype(float)

        # -----------------------------------------------------
        # Transformations
        # -----------------------------------------------------
        data_scaled = scaler.transform(data)
        data_pca = pca.transform(data_scaled)

        # Prediction
        cluster = model.predict(data_pca)[0]

        # -----------------------------------------------------
        # Output
        # -----------------------------------------------------
        st.subheader("📊 Result")
        st.success(f"Customer belongs to Cluster {cluster}")

        # Cluster meaning
        if cluster == 3:
            st.success("💰 High Value Customer")
        elif cluster == 1:
            st.success("📉 Low Value Customer")
        elif cluster == 0:
            st.success("🛍️ Medium Customer")
        elif cluster == 2:
            st.success("🎯 Potential Customer")

    except Exception as e:
        st.error("❌ Error in prediction")
        st.exception(e)

# -------------------------------------------------------------
# Footer
# -------------------------------------------------------------
st.markdown("---")
st.caption("© SmartCart Customer Segmentation Project")


# In[ ]:




