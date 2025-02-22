#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('kidney_disease_model.pkl')

# Title of the web app
st.title("Chronic Kidney Disease (CKD) Prediction")

# Input fields for user data
st.header("Enter Patient Details")
age = st.number_input("Age", min_value=0, max_value=120, value=50)
bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80)
sg = st.number_input("Specific Gravity", min_value=1.000, max_value=1.050, value=1.020, step=0.001)
al = st.number_input("Albumin", min_value=0, max_value=5, value=0)
su = st.number_input("Sugar", min_value=0, max_value=5, value=0)
rbc = st.number_input("Red Blood Cells", min_value=0, max_value=1, value=0)
pc = st.number_input("Pus Cells", min_value=0, max_value=1, value=0)
pcc = st.number_input("Pus Cell Clumps", min_value=0, max_value=1, value=0)
ba = st.number_input("Bacteria", min_value=0, max_value=1, value=0)
bgr = st.number_input("Blood Glucose Random", min_value=0, max_value=500, value=100)
bu = st.number_input("Blood Urea", min_value=0, max_value=200, value=50)
sc = st.number_input("Serum Creatinine", min_value=0.0, max_value=20.0, value=1.0, step=0.1)
sod = st.number_input("Sodium", min_value=0, max_value=200, value=140)
pot = st.number_input("Potassium", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
hemo = st.number_input("Hemoglobin", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
pcv = st.number_input("Packed Cell Volume", min_value=0, max_value=60, value=40)
wc = st.number_input("White Blood Cell Count", min_value=0, max_value=20000, value=8000)
rc = st.number_input("Red Blood Cell Count", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
htn = st.number_input("Hypertension", min_value=0, max_value=1, value=0)
dm = st.number_input("Diabetes Mellitus", min_value=0, max_value=1, value=0)
cad = st.number_input("Coronary Artery Disease", min_value=0, max_value=1, value=0)
appet = st.number_input("Appetite", min_value=0, max_value=1, value=0)
pe = st.number_input("Pedal Edema", min_value=0, max_value=1, value=0)
ane = st.number_input("Anemia", min_value=0, max_value=1, value=0)

# Button to make predictions
if st.button("Predict"):
    # Prepare input data
    input_data = np.array([age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display the result
    if prediction[0] == 1:
        st.error("The model predicts that the person has a condition that could lead to CKD.")
    else:
        st.success("The model predicts that the person does not have a condition that could lead to CKD.")


# In[ ]:




