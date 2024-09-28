import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from mainfile import RidgeLogisticRegression

# Load the trained Ridge Logistic Regression model
with open('Ridge_Logistic_Regression_Model.pkl', 'rb') as file:
    ridge_model = pickle.load(file)

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the price ranges for each category
price_ranges = {
    0: (0, 200000),      # Category 0: Low price
    1: (200001, 500000), # Category 1: Medium price
    2: (500001, 1000000),# Category 2: High price
    3: (1000001, float('inf')) # Category 3: Very high price
}

# Define the Streamlit app
def main():
    st.title("Car Selling Price Prediction")

    # Input fields for the features
    mileage = st.number_input("Mileage (kmpl)", min_value=0.0, step=0.1)
    engine = st.number_input("Engine (CC)", min_value=0.0, step=0.1)
    max_power = st.number_input("Max Power (bhp)", min_value=0.0, step=0.1)
    km_driven = st.number_input("KM Driven", min_value=0)
    seats = st.number_input("Seats", min_value=1, max_value=10, step=1)
    
    fuel = st.selectbox("Fuel Type", options=[1, 2, 3, 4],
                         format_func=lambda x: ["Diesel", "Petrol", "LPG", "CNG"][x-1])
    
    transmission = st.selectbox("Transmission", options=[1, 2],
                                  format_func=lambda x: ["Manual", "Automatic"][x-1])
    
    seller_type = st.selectbox("Seller Type", options=[1, 2, 3],
                                format_func=lambda x: ["Individual", "Dealer", "Trustmark Dealer"][x-1])
    
    owner = st.selectbox("Owner", options=[1, 2, 3, 4, 5],
                         format_func=lambda x: ["First Owner", "Second Owner", "Third Owner", 
                                                "Fourth & Above Owner", "Test Drive Car"][x-1])
    
    brand_names = [
        'Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
        'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
        'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
        'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
        'Ambassador', 'Ashok', 'Isuzu', 'Opel'
    ]
    brand = st.selectbox("Brand", options=list(range(1, len(brand_names) + 1)),
                         format_func=lambda x: brand_names[x-1])

    # Predict button
    if st.button("Predict"):
        # Prepare the input data
        input_data = np.array([[mileage, engine, max_power, km_driven, seats, fuel, 
                                transmission, seller_type, owner, brand]])
        input_data_scaled = scaler.transform(input_data)

        # Predict the selling price category
        prediction = ridge_model.predict(input_data_scaled)

        # Get the price range based on the predicted category
        category = int(prediction[0])
        price_range = price_ranges.get(category, (0, 0))
        
        # Display the prediction
        st.success(f"Predicted Selling Price Category: {category}")
        st.success(f"Estimated Selling Price Range: ₹{price_range[0]:,.2f} - ₹{price_range[1]:,.2f}")

if __name__ == "__main__":
    main()
