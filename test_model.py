import unittest
import numpy as np
from mainfile import LogisticRegression  # Import your model class 

class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        self.model = LogisticRegression()
        self.X_test = np.array([[1, 2, 3], [4, 5, 6]])  # Example input
        self.y_test = np.array([0, 1])  # Example output

    def test_model_input(self):
        self.assertEqual(self.X_test.shape[1], 3)  # Check the expected number of features

    def test_model_output_shape(self):
        self.model.fit(self.X_test, self.y_test)
        predictions = self.model.predict(self.X_test)
        self.assertEqual(predictions.shape[0], self.X_test.shape[0])  # Check if output shape matches input samples

if __name__ == "__main__":
    unittest.main()
