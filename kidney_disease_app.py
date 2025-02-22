#!/usr/bin/env python
# coding: utf-8

# In[9]:


pip install streamlit


# In[3]:


import streamlit as st
import pandas as pd
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
        
streamlit run kidney_disease_app.py        


# In[2]:


streamlit run C:\Users\Lenovo\anaconda3\Lib\site-packages\ipykernel_launcher.py 
    


# In[9]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the dataset
data = pd.read_csv('processed_kidney_disease.csv')

# Display the first few rows
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Handle missing values in features (fill with mean/mode or drop)
data.fillna(data.mean(), inplace=True)

# Convert categorical variables to numerical (if any)
data = pd.get_dummies(data, drop_first=True)

# Separate features and target
X = data.drop('class', axis=1)  # 'class' is the target column
y = data['class']

# Check for missing values in the target variable
print("Missing values in y:", y.isnull().sum())

# Handle missing values in the target variable
# Option 1: Drop rows with missing values in y
data = data.dropna(subset=['class'])
X = data.drop('class', axis=1)
y = data['class']

# Option 2: Fill missing values in y with the most frequent class
# most_frequent_class = y.mode()[0]
# y = y.fillna(most_frequent_class)

# Encode the target column (ckd = 1, notckd = 0)
y = y.map({'ckd': 1, 'notckd': 0})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[10]:


# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Evaluate the model
print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))
print("Classification Report:\n", classification_report(y_test, rf_predictions))

# Save the model
joblib.dump(rf_model, 'ckd_model.pkl')


# In[11]:


# Check for missing values in the target variable
print("Missing values in y:", y.isnull().sum())


# In[12]:


# Drop rows where the target variable is NaN
data = data.dropna(subset=['class'])

# Update X and y after dropping rows
X = data.drop('class', axis=1)
y = data['class']


# In[13]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the dataset
data = pd.read_csv('processed_kidney_disease.csv')

# Display the first few rows
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Handle missing values in features (fill with mean/mode or drop)
data.fillna(data.mean(), inplace=True)

# Convert categorical variables to numerical (if any)
data = pd.get_dummies(data, drop_first=True)

# Separate features and target
X = data.drop('class', axis=1)  # 'class' is the target column
y = data['class']

# Check for missing values in the target variable
print("Missing values in y:", y.isnull().sum())

# Handle missing values in the target variable
# Option 1: Drop rows with missing values in y
data = data.dropna(subset=['class'])
X = data.drop('class', axis=1)
y = data['class']

# Option 2: Fill missing values in y with the most frequent class
# most_frequent_class = y.mode()[0]
# y = y.fillna(most_frequent_class)

# Encode the target column (ckd = 1, notckd = 0)
y = y.map({'ckd': 1, 'notckd': 0})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[14]:


# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Evaluate the model
print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))
print("Classification Report:\n", classification_report(y_test, rf_predictions))

# Save the model
joblib.dump(rf_model, 'ckd_model.pkl')


# In[15]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
data = pd.read_csv('processed_kidney_disease.csv')

# Assuming 'class' is the target variable indicating CKD (1) or not (0)
X = data.drop('class', axis=1)
y = data['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, 'kidney_disease_model.pkl')


# In[17]:


pip install voila ipywidgets


# In[1]:


import pandas as pd
import numpy as np
import joblib
from ipywidgets import widgets, interact, Layout
from IPython.display import display

# Load the trained model
model = joblib.load('kidney_disease_model.pkl')

# Define input fields for the user
age = widgets.FloatText(description="Age:", layout=Layout(width='50%'))
bp = widgets.FloatText(description="Blood Pressure:", layout=Layout(width='50%'))
sg = widgets.FloatText(description="Specific Gravity:", layout=Layout(width='50%'))
al = widgets.FloatText(description="Albumin:", layout=Layout(width='50%'))
su = widgets.FloatText(description="Sugar:", layout=Layout(width='50%'))
rbc = widgets.FloatText(description="Red Blood Cells:", layout=Layout(width='50%'))
pc = widgets.FloatText(description="Pus Cells:", layout=Layout(width='50%'))
pcc = widgets.FloatText(description="Pus Cell Clumps:", layout=Layout(width='50%'))
ba = widgets.FloatText(description="Bacteria:", layout=Layout(width='50%'))
bgr = widgets.FloatText(description="Blood Glucose Random:", layout=Layout(width='50%'))
bu = widgets.FloatText(description="Blood Urea:", layout=Layout(width='50%'))
sc = widgets.FloatText(description="Serum Creatinine:", layout=Layout(width='50%'))
sod = widgets.FloatText(description="Sodium:", layout=Layout(width='50%'))
pot = widgets.FloatText(description="Potassium:", layout=Layout(width='50%'))
hemo = widgets.FloatText(description="Hemoglobin:", layout=Layout(width='50%'))
pcv = widgets.FloatText(description="Packed Cell Volume:", layout=Layout(width='50%'))
wc = widgets.FloatText(description="White Blood Cell Count:", layout=Layout(width='50%'))
rc = widgets.FloatText(description="Red Blood Cell Count:", layout=Layout(width='50%'))
htn = widgets.FloatText(description="Hypertension:", layout=Layout(width='50%'))
dm = widgets.FloatText(description="Diabetes Mellitus:", layout=Layout(width='50%'))
cad = widgets.FloatText(description="Coronary Artery Disease:", layout=Layout(width='50%'))
appet = widgets.FloatText(description="Appetite:", layout=Layout(width='50%'))
pe = widgets.FloatText(description="Pedal Edema:", layout=Layout(width='50%'))
ane = widgets.FloatText(description="Anemia:", layout=Layout(width='50%'))

# Define a function to make predictions
def predict_ckd(age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane):
    input_data = np.array([age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane]).reshape(1, -1)
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        print("Prediction: The person has a condition that could lead to CKD.")
    else:
        print("Prediction: The person does not have a condition that could lead to CKD.")

# Create an interactive widget
interact(predict_ckd, 
         age=age, bp=bp, sg=sg, al=al, su=su, rbc=rbc, pc=pc, pcc=pcc, ba=ba, bgr=bgr, 
         bu=bu, sc=sc, sod=sod, pot=pot, hemo=hemo, pcv=pcv, wc=wc, rc=rc, htn=htn, 
         dm=dm, cad=cad, appet=appet, pe=pe, ane=ane);


# In[8]:


voila kidney_disease_app.py


# In[ ]:




