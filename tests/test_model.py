import unittest
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from streamlit_app import preprocess_data, predict  # Adjust import based on file structure

class TestCarPricePrediction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the Ridge Logistic Regression model
        with open('Ridge_Logistic_Regression_Model.pkl', 'rb') as file:
            cls.ridge_model = pickle.load(file)

        # Load the scaler used during model training
        with open('scaler.pkl', 'rb') as file:
            cls.scaler = pickle.load(file)

    def test_preprocess_data(self):
        """Test the preprocessing of data."""
        data = pd.DataFrame({
            'name': [1],
            'mileage': [15.0],
            'engine': [1500],
            'max_power': [100],
            'km_driven': [50000],
            'seats': [5],
            'fuel': [1],
            'transmission': [1],
            'seller_type': [1],
            'owner': [1]
        })
        X_scaled = preprocess_data(data)
        self.assertEqual(X_scaled.shape[0], 1)  # Ensure the shape is correct

    def test_predict(self):
        """Test the prediction function."""
        data = pd.DataFrame({
            'name': [1],
            'mileage': [15.0],
            'engine': [1500],
            'max_power': [100],
            'km_driven': [50000],
            'seats': [5],
            'fuel': [1],
            'transmission': [1],
            'seller_type': [1],
            'owner': [1]
        })
        prediction = predict(data)
        self.assertIsNotNone(prediction[0])  # Ensure prediction is not None

if __name__ == '__main__':
    unittest.main()
