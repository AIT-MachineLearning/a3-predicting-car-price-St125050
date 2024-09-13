import pytest
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the Ridge Logistic Regression model and scaler
with open('Ridge_Logistic_Regression_Model.pkl', 'rb') as file:
    ridge_model = pickle.load(file)

# Ideally, you should load the scaler used during model training
scaler = StandardScaler()

def preprocess_data(data):
    """Preprocess the input data."""
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
    
    # Scale the features
    X_scaled = scaler.fit_transform(X)  # Use the actual scaler fitted on training data
    return X_scaled

def test_model_input():
    """Test if the model takes the expected input."""
    data = pd.DataFrame({
        'name': ['Toyota'],
        'mileage': [15.0],
        'engine': [1500],
        'max_power': [100],
        'km_driven': [50000],
        'seats': [5],
        'fuel': ['Petrol'],
        'transmission': ['Manual'],
        'seller_type': ['Individual'],
        'owner': ['First Owner']
    })
    processed_data = preprocess_data(data)
    assert processed_data.shape[1] == 5  # Ensure that the number of features is as expected

def test_model_output():
    """Test if the model's output has the expected shape."""
    data = pd.DataFrame({
        'name': ['Toyota'],
        'mileage': [15.0],
        'engine': [1500],
        'max_power': [100],
        'km_driven': [50000],
        'seats': [5],
        'fuel': ['Petrol'],
        'transmission': ['Manual'],
        'seller_type': ['Individual'],
        'owner': ['First Owner']
    })
    processed_data = preprocess_data(data)
    prediction = ridge_model.predict(processed_data)
    assert prediction.shape == (1,)  # Ensure the output shape is as expected

if __name__ == "__main__":
    pytest.main()
