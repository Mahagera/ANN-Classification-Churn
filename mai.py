

import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import pickle

# Load trained model
model = tf.keras.models.load_model('model.h5')

# Load gender encoder
with open('gender_incoder.pkl', 'rb') as file:
    gender_incoder = pickle.load(file)

# Load scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load OneHotEncoder
with open('one_hot_encoder.pkl', 'rb') as file:
    one_hot_encoder = pickle.load(file)


st.title('Customer Churn Prediction')

st.write('Enter customer information below.')

# User inputs
geography = st.selectbox(
    'Geography',
    one_hot_encoder.categories_[0]
)

gender = st.selectbox(
    'Gender',
    gender_incoder.classes_
)

age = st.slider('Age', 18, 92)
balance = st.number_input('Balance', min_value=0.0)
credit_score = st.number_input('CreditScore', min_value=0)
estimated_salary = st.number_input('EstimatedSalary', min_value=0.0)
tenure = st.slider('Tenure', 0, 10)
number_of_products = st.slider('NumOfProducts', 1, 4)
has_cr_card = st.selectbox('HasCrCard', [0, 1])
is_an_active_member = st.selectbox('IsActiveMember', [0, 1])


if st.button('Predict Churn'):

    # Step 1: Create numerical and encoded gender data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [gender_incoder.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [number_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_an_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # Step 2: Encode Geography
    geo_encoded = one_hot_encoder.transform(
        pd.DataFrame({'Geography': [geography]})
    )

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=one_hot_encoder.get_feature_names_out(['Geography'])
    )

    # Step 3: Combine features
    input_diffe = pd.concat(
        [input_data.reset_index(drop=True),
         geo_encoded_df.reset_index(drop=True)],
        axis=1
    )

    # Step 4: Scale all features
    input_scaled = scaler.transform(input_diffe)

    # Step 5: Make prediction
    prediction = model.predict(input_scaled)

    prediction_proba = float(prediction[0][0])

    # Step 6: Display result
    st.write(f'Churn Probability: {prediction_proba:.2%}')

    if prediction_proba > 0.5:
        st.error('The customer is likely to churn.')
    else:
        st.success('The customer is not likely to churn.')