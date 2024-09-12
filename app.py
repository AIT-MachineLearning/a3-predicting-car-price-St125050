import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the Ridge Logistic Regression model
with open('Ridge_Logistic_Regression_Model.pkl', 'rb') as file:
    ridge_model = pickle.load(file)

# Ideally, you should load the scaler used during model training
# For this example, we're initializing a new scaler, which may not have the same parameters
scaler = StandardScaler()

def preprocess_data(data):
    """Preprocess the input data."""
    # Convert categorical variables to numeric
    data['fuel'] = data['fuel'].map({'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4})
    data['transmission'] = data['transmission'].map({'Manual': 1, 'Automatic': 2})
    data['seller_type'] = data['seller_type'].map({'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3})
    data['owner'] = data['owner'].map({'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 5})
    data['name'] = data['name'].map({
        'Maruti': 1, 'Skoda': 2, 'Honda': 3, 'Hyundai': 4, 'Toyota': 5, 'Ford': 6, 'Renault': 7,
        'Mahindra': 8, 'Tata': 9, 'Chevrolet': 10, 'Datsun': 11, 'Jeep': 12, 'Mercedes-Benz': 13,
        'Mitsubishi': 14, 'Audi': 15, 'Volkswagen': 16, 'BMW': 17, 'Nissan': 18, 'Lexus': 19,
        'Jaguar': 20, 'Land': 21, 'MG': 22, 'Volvo': 23, 'Daewoo': 24, 'Kia': 25, 'Fiat': 26, 'Force': 27,
        'Ambassador': 28, 'Ashok': 29, 'Isuzu': 30, 'Opel': 31
    })
    
    # Extract features and scale
    features = ['mileage', 'engine', 'max_power', 'km_driven', 'seats']
    X = data[features]
    
    # Scale the features (make sure scaler is fitted with training data in a real scenario)
    X_scaled = scaler.fit_transform(X)  # Use the actual scaler fitted on training data
    return X_scaled

def predict(data):
    """Predict using the Ridge Logistic Regression model."""
    try:
        X_scaled = preprocess_data(data)
        return ridge_model.predict(X_scaled)
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return [None]

# Streamlit app layout
st.title('Car Price Prediction Using Ridge Logistic Regression')

# Input fields
name = st.text_input('Car Brand Name', '')
fuel = st.selectbox('Fuel Type', ['Diesel', 'Petrol', 'LPG', 'CNG'])
transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer', 'Trustmark Dealer'])
owner = st.selectbox('Owner Type', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
mileage = st.number_input('Mileage (kmpl)', step=0.1, min_value=0.0)
engine = st.number_input('Engine (cc)', min_value=0)
max_power = st.number_input('Max Power (bhp)', min_value=0)
km_driven = st.number_input('KM Driven', min_value=0)
seats = st.number_input('Seats', step=1, min_value=1)

# Create a DataFrame for prediction
data = pd.DataFrame({
    'name': [name],
    'mileage': [mileage],
    'engine': [engine],
    'max_power': [max_power],
    'km_driven': [km_driven],
    'seats': [seats],
    'fuel': [fuel],
    'transmission': [transmission],
    'seller_type': [seller_type],
    'owner': [owner]
})

if st.button('Predict'):
    prediction = predict(data)
    if prediction[0] is not None:
        st.write(f'Predicted Category: {prediction[0]}')
    else:
        st.write('Prediction could not be made. Please check your inputs.')
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the Ridge Logistic Regression model
with open('Ridge_Logistic_Regression_Model.pkl', 'rb') as file:
    ridge_model = pickle.load(file)

# Ideally, you should load the scaler used during model training
# For this example, we're initializing a new scaler, which may not have the same parameters
scaler = StandardScaler()

def preprocess_data(data):
    """Preprocess the input data."""
    # Convert categorical variables to numeric
    data['fuel'] = data['fuel'].map({'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4})
    data['transmission'] = data['transmission'].map({'Manual': 1, 'Automatic': 2})
    data['seller_type'] = data['seller_type'].map({'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3})
    data['owner'] = data['owner'].map({'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 5})
    data['name'] = data['name'].map({
        'Maruti': 1, 'Skoda': 2, 'Honda': 3, 'Hyundai': 4, 'Toyota': 5, 'Ford': 6, 'Renault': 7,
        'Mahindra': 8, 'Tata': 9, 'Chevrolet': 10, 'Datsun': 11, 'Jeep': 12, 'Mercedes-Benz': 13,
        'Mitsubishi': 14, 'Audi': 15, 'Volkswagen': 16, 'BMW': 17, 'Nissan': 18, 'Lexus': 19,
        'Jaguar': 20, 'Land': 21, 'MG': 22, 'Volvo': 23, 'Daewoo': 24, 'Kia': 25, 'Fiat': 26, 'Force': 27,
        'Ambassador': 28, 'Ashok': 29, 'Isuzu': 30, 'Opel': 31
    })
    
    # Extract features and scale
    features = ['mileage', 'engine', 'max_power', 'km_driven', 'seats']
    X = data[features]
    
    # Scale the features (make sure scaler is fitted with training data in a real scenario)
    X_scaled = scaler.fit_transform(X)  # Use the actual scaler fitted on training data
    return X_scaled

def predict(data):
    """Predict using the Ridge Logistic Regression model."""
    try:
        X_scaled = preprocess_data(data)
        return ridge_model.predict(X_scaled)
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return [None]

# Streamlit app layout
st.title('Car Price Prediction Using Ridge Logistic Regression')

# Input fields
name = st.text_input('Car Brand Name', '')
fuel = st.selectbox('Fuel Type', ['Diesel', 'Petrol', 'LPG', 'CNG'])
transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer', 'Trustmark Dealer'])
owner = st.selectbox('Owner Type', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
mileage = st.number_input('Mileage (kmpl)', step=0.1, min_value=0.0)
engine = st.number_input('Engine (cc)', min_value=0)
max_power = st.number_input('Max Power (bhp)', min_value=0)
km_driven = st.number_input('KM Driven', min_value=0)
seats = st.number_input('Seats', step=1, min_value=1)

# Create a DataFrame for prediction
data = pd.DataFrame({
    'name': [name],
    'mileage': [mileage],
    'engine': [engine],
    'max_power': [max_power],
    'km_driven': [km_driven],
    'seats': [seats],
    'fuel': [fuel],
    'transmission': [transmission],
    'seller_type': [seller_type],
    'owner': [owner]
})

if st.button('Predict'):
    prediction = predict(data)
    if prediction[0] is not None:
        st.write(f'Predicted Category: {prediction[0]}')
    else:
        st.write('Prediction could not be made. Please check your inputs.')
