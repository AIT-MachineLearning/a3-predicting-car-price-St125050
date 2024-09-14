import unittest
import pandas as pd
import pickle
from data_processing import preprocess_data
from mainfile import RidgeLogisticRegression  # Import the correct class

class TestCarPricePrediction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load the Ridge Logistic Regression model and scaler."""
        with open('Ridge_Logistic_Regression_Model.pkl', 'rb') as file:
            cls.ridge_model = pickle.load(file)

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
        self.assertTrue(np.all(np.isfinite(X_scaled)))  # Check for finite values

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
        preprocessed_data = preprocess_data(data)
        prediction = self.ridge_model.predict(preprocessed_data)
        self.assertEqual(prediction.shape[0], 1)  # Ensure prediction has correct shape
        self.assertIn(prediction[0], [0, 1])  # Check that prediction is binary

if __name__ == '__main__':
    unittest.main()
