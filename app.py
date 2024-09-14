import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

# Load the model from a pickle file
@st.cache(allow_output_mutation=True)
def load_model():
    # You may need to adjust the path to where your model is stored
    with open('Ridge_Logistic_Regression_Model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load the scaler from a pickle file
@st.cache(allow_output_mutation=True)
def load_scaler():
    # You may need to adjust the path to where your scaler is stored
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return scaler

def main():
    st.title("Car Price Prediction using Ridge Logistic Regression")

    # Load model and scaler
    model = load_model()
    scaler = load_scaler()

    # Input fields
    st.sidebar.header("Input Features")

    mileage = st.sidebar.number_input("Mileage (km/l)", min_value=0.0, max_value=100.0, value=15.0)
    engine = st.sidebar.number_input("Engine (cc)", min_value=0, max_value=5000, value=1500)
    max_power = st.sidebar.number_input("Max Power (bhp)", min_value=0, max_value=500, value=100)
    km_driven = st.sidebar.number_input("KM Driven (in thousands)", min_value=0, max_value=1000, value=50)
    seats = st.sidebar.number_input("Seats", min_value=1, max_value=7, value=5)

    fuel = st.sidebar.selectbox("Fuel Type", ["Diesel", "Petrol", "LPG", "CNG"])
    transmission = st.sidebar.selectbox("Transmission Type", ["Manual", "Automatic"])
    seller_type = st.sidebar.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
    owner = st.sidebar.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
    name = st.sidebar.selectbox("Car Brand", [
        'Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
        'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
        'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
        'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
        'Ambassador', 'Ashok', 'Isuzu', 'Opel'
    ])

    # Mapping categorical inputs to numeric values
    fuel_map = {'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4}
    transmission_map = {'Manual': 1, 'Automatic': 2}
    seller_map = {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3}
    owner_map = {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 5}
    name_map = {
        'Maruti': 1, 'Skoda': 2, 'Honda': 3, 'Hyundai': 4, 'Toyota': 5, 'Ford': 6, 'Renault': 7,
        'Mahindra': 8, 'Tata': 9, 'Chevrolet': 10, 'Datsun': 11, 'Jeep': 12, 'Mercedes-Benz': 13,
        'Mitsubishi': 14, 'Audi': 15, 'Volkswagen': 16, 'BMW': 17, 'Nissan': 18, 'Lexus': 19,
        'Jaguar': 20, 'Land': 21, 'MG': 22, 'Volvo': 23, 'Daewoo': 24, 'Kia': 25, 'Fiat': 26, 'Force': 27,
        'Ambassador': 28, 'Ashok': 29, 'Isuzu': 30, 'Opel': 31
    }

    # Prepare the input data for prediction
    input_data = {
        'mileage': mileage,
        'engine': engine,
        'max_power': max_power,
        'km_driven': km_driven * 1000,  # Convert to absolute km
        'seats': seats,
        'fuel': fuel_map[fuel],
        'transmission': transmission_map[transmission],
        'seller_type': seller_map[seller_type],
        'owner': owner_map[owner],
        'name': name_map[name]
    }

    input_df = pd.DataFrame([input_data])

    # Scale the input data
    input_scaled = scaler.transform(input_df)

    # Make predictions
    prediction = model.predict(input_scaled)

    # Display results
    st.subheader("Predicted Selling Price")
    st.write(f"The predicted selling price is approximately ₹{prediction[0]:,.2f}")

if __name__ == "__main__":
    main()
