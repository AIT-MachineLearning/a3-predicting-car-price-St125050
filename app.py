import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
with open('Ridge_Logistic_Regression_Model.pkl', 'rb') as file:
    model = pickle.load(file)

# If using scaling, load the scaler as well
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define function to predict with the model
def predict(features):
    # Scale features if necessary
    features_scaled = scaler.transform([features])
    # Predict
    prediction = model.predict(features_scaled)
    return prediction[0]

# Define the Streamlit app
st.title('Car Price Prediction App')

# Input fields for user data
st.sidebar.header('User Input')

def user_input_features():
    name = st.sidebar.selectbox('Car Brand', ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault', 'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz', 'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force', 'Ambassador', 'Ashok', 'Isuzu', 'Opel'])
    mileage = st.sidebar.number_input('Mileage (km/l)', min_value=0.0, value=0.0)
    max_power = st.sidebar.number_input('Max Power (bhp)', min_value=0.0, value=0.0)
    engine = st.sidebar.number_input('Engine (cc)', min_value=0.0, value=0.0)
    seats = st.sidebar.slider('Seats', 2, 10, 5)
    fuel = st.sidebar.selectbox('Fuel Type', ['Diesel', 'Petrol', 'LPG', 'CNG'])
    transmission = st.sidebar.selectbox('Transmission', ['Manual', 'Automatic'])
    seller_type = st.sidebar.selectbox('Seller Type', ['Individual', 'Dealer', 'Trustmark Dealer'])
    owner = st.sidebar.selectbox('Owner', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
    
    data = {
        'name': name,
        'mileage': mileage,
        'max_power': max_power,
        'engine': engine,
        'seats': seats,
        'fuel': fuel,
        'transmission': transmission,
        'seller_type': seller_type,
        'owner': owner
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Display the user input data
st.write('User Input features:')
st.write(df)

# Preprocess input data
df['name'] = df['name'].map({
    'Maruti': 1, 'Skoda': 2, 'Honda': 3, 'Hyundai': 4, 'Toyota': 5,
    'Ford': 6, 'Renault': 7, 'Mahindra': 8, 'Tata': 9, 'Chevrolet': 10,
    'Datsun': 11, 'Jeep': 12, 'Mercedes-Benz': 13, 'Mitsubishi': 14, 
    'Audi': 15, 'Volkswagen': 16, 'BMW': 17, 'Nissan': 18, 'Lexus': 19,
    'Jaguar': 20, 'Land': 21, 'MG': 22, 'Volvo': 23, 'Daewoo': 24, 
    'Kia': 25, 'Fiat': 26, 'Force': 27, 'Ambassador': 28, 'Ashok': 29,
    'Isuzu': 30, 'Opel': 31
})

df['fuel'] = df['fuel'].map({'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4})
df['transmission'] = df['transmission'].map({'Manual': 1, 'Automatic': 2})
df['seller_type'] = df['seller_type'].map({'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3})
df['owner'] = df['owner'].map({'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 5})

# Predict
prediction = predict(df.iloc[0].values)

# Display the prediction result
st.write('Prediction:')
st.write(f'The predicted price category is: {prediction}')
