import unittest
import pandas as pd
import pickle
from data_processing import preprocess_data
from models import RidgeLogisticRegression  # Import the correct class

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
        # Ensure preprocess_data is applied correctly
        preprocessed_data = preprocess_data(data)
        # Use the ridge model for prediction
        prediction = self.ridge_model.predict(preprocessed_data)
        self.assertIsNotNone(prediction[0])  # Ensure prediction is not None

if __name__ == '__main__':
    unittest.main()
