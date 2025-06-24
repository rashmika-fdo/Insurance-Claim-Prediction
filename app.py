import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://img.freepik.com/free-vector/blue-curve-frame-template_53876-116707.jpg?semt=ais_items_boosted&w=740");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)







st.title('Insurance Claim Fraud Detection')
st.write('Enter the following details')

colx, spacer, coly = st.columns([1,0.3,1])


with colx:
    genderoptions = ['Female','Male']
    gender = st.selectbox('Select the Gender', genderoptions)

    policyoptions = ['IL','IN','OH']
    policy = st.selectbox('Select the policy state', policyoptions)

    st.write('Enter Customer Tenure Period')
    col1, col2 = st.columns(2)
    with col1:
        years = st.number_input('Years', min_value=0, step=1)
    with col2:
        months = st.number_input('Months', min_value=0, max_value=11, step=1)


    months_as_customer = (years*12) + months
    st.write(months_as_customer)


    umbrella_limit = st.number_input('Enter umbrella limit')

    relation = ['Husband','Not-in-family','Other-relative','Own-child','Unmarried','Wife']
    relationship = st.selectbox('Enter the insured relationship', relation)

with coly:
    number_of_vehicles_involved = st.slider('Select the number of vehicles involved', min_value=0, max_value=10 )
    bodily_injuries = st.slider('Select the number of bodily injuries', min_value=0, max_value=10 )
    witnesses = st.slider('Select the number of witnesses', min_value=0, max_value=10 )

    total_claim_amount = st.number_input('Enter the total claim amount')
    injury_claim = st.number_input('Enter the injury claim amount')
    property_claim = st.number_input('Enter the property claim amount')
    vehicle_claim = st.number_input('Enter the vehicle claim amount')


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

insured_sex = le.fit_transform([gender])[0]
policy_state = le.fit_transform([policy])[0]
insured_relationship = le.fit_transform([relationship])[0]

input_values = [vehicle_claim,total_claim_amount,property_claim,injury_claim,umbrella_limit,
number_of_vehicles_involved,witnesses,bodily_injuries,insured_sex,policy_state,insured_relationship,
months_as_customer]

input_scaled = np.array(input_values).reshape(1,-1)


with open('best_xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)


xgb_pred = xgb_model.predict(input_scaled)

if xgb_pred==0:
    st.success('No Fraud Detected')
else:
    st.error('Fraud Detected')