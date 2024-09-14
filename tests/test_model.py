import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the model and scaler from pickle files
def load_model():
    with open('Ridge_Logistic_Regression_Model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def load_scaler():
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return scaler

# Test the model with sample inputs
def test_model(model, scaler):
    # Sample input data
    sample_data = {
        'mileage': 15.0,
        'engine': 1500,
        'max_power': 100,
        'km_driven': 50000,  # 50,000 km
        'seats': 5,
        'fuel': 2,  # Petrol
        'transmission': 1,  # Manual
        'seller_type': 1,  # Individual
        'owner': 1,  # First Owner
        'name': 1   # Maruti
    }
    
    # Create DataFrame for input data
    input_df = pd.DataFrame([sample_data])
    
    # Load the scaler and scale the input data
    input_scaled = scaler.transform(input_df)
    
    # Predict using the loaded model
    prediction = model.predict(input_scaled)
    
    return prediction[0]

def main():
    model = load_model()
    scaler = load_scaler()
    
    predicted_price = test_model(model, scaler)
    
    print(f"The predicted selling price for the sample input is approximately ₹{predicted_price:,.2f}")

if __name__ == "__main__":
    main()
