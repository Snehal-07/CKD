#!/usr/bin/env python
# coding: utf-8

# In[3]:


import ipywidgets as widgets
from IPython.display import display
import numpy as np
import joblib

# Load the trained model
model = joblib.load('kidney_disease_model.pkl')

# Create input widgets
age = widgets.FloatText(description="Age:")
bp = widgets.FloatText(description="Blood Pressure:")
sg = widgets.FloatText(description="Specific Gravity:")
al = widgets.FloatText(description="Albumin:")
su = widgets.FloatText(description="Sugar:")
rbc = widgets.FloatText(description="Red Blood Cells:")
pc = widgets.FloatText(description="Pus Cells:")
pcc = widgets.FloatText(description="Pus Cell Clumps:")
ba = widgets.FloatText(description="Bacteria:")
bgr = widgets.FloatText(description="Blood Glucose Random:")
bu = widgets.FloatText(description="Blood Urea:")
sc = widgets.FloatText(description="Serum Creatinine:")
sod = widgets.FloatText(description="Sodium:")
pot = widgets.FloatText(description="Potassium:")
hemo = widgets.FloatText(description="Hemoglobin:")
pcv = widgets.FloatText(description="Packed Cell Volume:")
wc = widgets.FloatText(description="White Blood Cell Count:")
rc = widgets.FloatText(description="Red Blood Cell Count:")
htn = widgets.FloatText(description="Hypertension:")
dm = widgets.FloatText(description="Diabetes Mellitus:")
cad = widgets.FloatText(description="Coronary Artery Disease:")
appet = widgets.FloatText(description="Appetite:")
pe = widgets.FloatText(description="Pedal Edema:")
ane = widgets.FloatText(description="Anemia:")

# Button to make predictions
button = widgets.Button(description="Predict")

# Output widget
output = widgets.Output()

# Define the prediction function
def predict_ckd(b):
    input_data = np.array([age.value, bp.value, sg.value, al.value, su.value, rbc.value, pc.value, pcc.value, ba.value, bgr.value, bu.value, sc.value, sod.value, pot.value, hemo.value, pcv.value, wc.value, rc.value, htn.value, dm.value, cad.value, appet.value, pe.value, ane.value]).reshape(1, -1)
    prediction = model.predict(input_data)
    with output:
        if prediction[0] == 1:
            print("Prediction: The person has a condition that could lead to CKD.")
        else:
            print("Prediction: The person does not have a condition that could lead to CKD.")

# Link the button to the prediction function
button.on_click(predict_ckd)

# Display the widgets
display(age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane, button, output) 


# In[ ]:




