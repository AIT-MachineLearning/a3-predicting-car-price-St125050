import unittest
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from data_processing import preprocess_data
from models import RidgeLogisticRegression  # Import the class instead of the function

class TestCarPricePrediction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the Ridge Logistic Regression model
        with open('Ridge_Logistic_Regression_Model.pkl', 'rb') as file:
            cls.ridge_model_params = pickle.load(file)  # Load parameters instead of the model

        # Load the scaler used during model training
        with open('scaler.pkl', 'rb') as file:
            cls.scaler = pickle.load(file)

        # Instantiate the RidgeLogisticRegression with loaded parameters
        cls.ridge_model = RidgeLogisticRegression(
            learning_rate=cls.ridge_model_params.get('learning_rate', 0.01),
            num_iterations=cls.ridge_model_params.get('num_iterations', 1000),
            fit_intercept=cls.ridge_model_params.get('fit_intercept', True),
            verbose=cls.ridge_model_params.get('verbose', False),
            lambda_=cls.ridge_model_params.get('lambda_', 0.1)
        )
        cls.ridge_model.theta = cls.ridge_model_params.get('theta', None)  # Set model parameters

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
        data_preprocessed = preprocess_data(data)  # Ensure data is preprocessed
        prediction = self.ridge_model.predict(data_preprocessed)
        self.assertIsNotNone(prediction)  # Ensure prediction is not None

if __name__ == '__main__':
    unittest.main()
