import streamlit as st
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import pickle

# Load the models
@st.cache(allow_output_mutation=True)
def load_model(model_name):
    model_path = f"{model_name}.pkl"
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

logistic_model = load_model("Logistic_Regression_Model")
ridge_model = load_model("Ridge_Logistic_Regression_Model")

# Set the title of the app
st.title("Car Price Prediction App")

# Input features
st.header("Input Features")

mileage = st.number_input("Mileage (km/l)", min_value=0.0)
engine = st.number_input("Engine (cc)", min_value=0)
max_power = st.number_input("Max Power (bhp)", min_value=0)
km_driven = st.number_input("KM Driven", min_value=0)
seats = st.selectbox("Number of Seats", options=[2, 4, 5, 6, 7])
fuel = st.selectbox("Fuel Type", options=["Diesel", "Petrol", "LPG", "CNG"])
transmission = st.selectbox("Transmission Type", options=["Manual", "Automatic"])
seller_type = st.selectbox("Seller Type", options=["Individual", "Dealer", "Trustmark Dealer"])
owner = st.selectbox("Owner Type", options=["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
name = st.selectbox("Brand Name", options=[
    "Maruti", "Skoda", "Honda", "Hyundai", "Toyota", "Ford", "Renault",
    "Mahindra", "Tata", "Chevrolet", "Datsun", "Jeep", "Mercedes-Benz",
    "Mitsubishi", "Audi", "Volkswagen", "BMW", "Nissan", "Lexus",
    "Jaguar", "Land", "MG", "Volvo", "Daewoo", "Kia", "Fiat", "Force",
    "Ambassador", "Ashok", "Isuzu", "Opel"
])

# Preprocess input data
def preprocess_input(mileage, engine, max_power, km_driven, seats, fuel, transmission, seller_type, owner, name):
    fuel_mapping = {"Diesel": 1, "Petrol": 2, "LPG": 3, "CNG": 4}
    transmission_mapping = {"Manual": 1, "Automatic": 2}
    seller_type_mapping = {"Individual": 1, "Dealer": 2, "Trustmark Dealer": 3}
    owner_mapping = {
        "First Owner": 1,
        "Second Owner": 2,
        "Third Owner": 3,
        "Fourth & Above Owner": 4,
        "Test Drive Car": 5
    }
    brand_mapping = {
        "Maruti": 1, "Skoda": 2, "Honda": 3, "Hyundai": 4, "Toyota": 5,
        "Ford": 6, "Renault": 7, "Mahindra": 8, "Tata": 9, "Chevrolet": 10,
        "Datsun": 11, "Jeep": 12, "Mercedes-Benz": 13, "Mitsubishi": 14,
        "Audi": 15, "Volkswagen": 16, "BMW": 17, "Nissan": 18, "Lexus": 19,
        "Jaguar": 20, "Land": 21, "MG": 22, "Volvo": 23, "Daewoo": 24,
        "Kia": 25, "Fiat": 26, "Force": 27, "Ambassador": 28, "Ashok": 29,
        "Isuzu": 30, "Opel": 31
    }
    
    return np.array([
        mileage,
        engine,
        max_power,
        km_driven,
        seats,
        fuel_mapping[fuel],
        transmission_mapping[transmission],
        seller_type_mapping[seller_type],
        owner_mapping[owner],
        brand_mapping[name]
    ]).reshape(1, -1)

input_data = preprocess_input(mileage, engine, max_power, km_driven, seats, fuel, transmission, seller_type, owner, name)

# Predict the price using both models
if st.button("Predict"):
    logistic_price = logistic_model.predict(input_data)
    ridge_price = ridge_model.predict(input_data)
    
    st.subheader("Predicted Prices")
    st.write(f"Logistic Regression Model: ₹{logistic_price[0] * 1000:.2f}")
    st.write(f"Ridge Logistic Regression Model: ₹{ridge_price[0] * 1000:.2f}")
