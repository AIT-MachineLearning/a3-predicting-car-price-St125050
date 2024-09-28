import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess the data
df = pd.read_csv('Cars.csv')  # Use relative path

# Data cleaning
df.drop(columns=['torque'], inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

def get_brand_name(car_name):
    return car_name.split(' ')[0].strip()

def clean_data(value):
    if isinstance(value, str):
        value = value.split(' ')[0].strip()
        if value == '':
            value = 0
    if isinstance(value, float):
        return value
    return float(value)

# Apply cleaning functions
df['name'] = df['name'].apply(get_brand_name)
df['mileage'] = df['mileage'].apply(clean_data)
df['max_power'] = df['max_power'].apply(clean_data)
df['engine'] = df['engine'].apply(clean_data)

# Replace categorical values with numeric
brand_names = [
    'Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
    'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
    'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
    'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
    'Ambassador', 'Ashok', 'Isuzu', 'Opel'
]
brand_ids = list(range(1, len(brand_names) + 1))
df['name'].replace(brand_names, brand_ids, inplace=True)

df['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
df['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
df['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
df['owner'].replace(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'], [1, 2, 3, 4, 5], inplace=True)

df.reset_index(drop=True, inplace=True)

# Create price categories for classification
price_bins = [0, 200000, 500000, 1000000, float('inf')]
price_labels = [0, 1, 2, 3]
df['price_category'] = pd.cut(df['selling_price'], bins=price_bins, labels=price_labels)

# Convert price_category to integer type
df['price_category'] = df['price_category'].astype(int)

# Define features and target variable
features = ['mileage', 'engine', 'max_power', 'km_driven', 'seats', 'fuel', 'transmission', 'seller_type', 'owner', 'name']
X = df[features]
y = df['price_category']  # Use price_category for classification

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 # Use price_category for classification

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Class
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, fit_intercept=True, lambda_=0):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.lambda_ = lambda_

    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.add_intercept(X)
        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.learning_rate * gradient

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.add_intercept(X)
        return self.sigmoid(np.dot(X, self.theta))

    def predict(self, X):
      probs = self.predict_prob(X)
      return np.floor(probs * len(np.unique(y_train)))  # Convert to category labels
 # Return the index of the highest probability

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def precision(self, y_true, y_pred, class_):
        TP = np.sum((y_true == class_) & (y_pred == class_))
        FP = np.sum((y_true != class_) & (y_pred == class_))
        return TP / (TP + FP) if TP + FP > 0 else 0

    def recall(self, y_true, y_pred, class_):
        TP = np.sum((y_true == class_) & (y_pred == class_))
        FN = np.sum((y_true == class_) & (y_pred != class_))
        return TP / (TP + FN) if TP + FN > 0 else 0

    def f1_score(self, y_true, y_pred, class_):
        precision = self.precision(y_true, y_pred, class_)
        recall = self.recall(y_true, y_pred, class_)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    def macro_precision(self, y_true, y_pred):
        classes = np.unique(y_true)
        return np.mean([self.precision(y_true, y_pred, c) for c in classes])

    def macro_recall(self, y_true, y_pred):
        classes = np.unique(y_true)
        return np.mean([self.recall(y_true, y_pred, c) for c in classes])

    def macro_f1(self, y_true, y_pred):
        classes = np.unique(y_true)
        return np.mean([self.f1_score(y_true, y_pred, c) for c in classes])

    def weighted_precision(self, y_true, y_pred):
        classes = np.unique(y_true)
        weights = np.array([np.sum(y_true == c) / len(y_true) for c in classes])
        return np.sum(weights * [self.precision(y_true, y_pred, c) for c in classes])

    def weighted_recall(self, y_true, y_pred):
        classes = np.unique(y_true)
        weights = np.array([np.sum(y_true == c) / len(y_true) for c in classes])
        return np.sum(weights * [self.recall(y_true, y_pred, c) for c in classes])

    def weighted_f1(self, y_true, y_pred):
        classes = np.unique(y_true)
        weights = np.array([np.sum(y_true == c) / len(y_true) for c in classes])
        return np.sum(weights * [self.f1_score(y_true, y_pred, c) for c in classes])

# Ridge Logistic Regression Class
class RidgeLogisticRegression(LogisticRegression):
    def __init__(self, learning_rate=0.01, num_iterations=1000, fit_intercept=True, lambda_=0.1):
        super().__init__(learning_rate, num_iterations, fit_intercept, lambda_)

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.add_intercept(X)
        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            gradient += (self.lambda_ / y.size) * self.theta
            gradient[0] -= (self.lambda_ / y.size) * self.theta[0]  # Don't regularize intercept
            self.theta -= self.learning_rate * gradient

# Save model as pickle file
def save_model_as_pkl(model, model_name):
    with open(f'{model_name}.pkl', 'wb') as file:
        pickle.dump(model, file)

# Initialize and train Logistic Regression model
logistic_model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
logistic_model.fit(X_train_scaled, y_train)
save_model_as_pkl(logistic_model, "Logistic_Regression_Model")

# Initialize and train Ridge Logistic Regression model
ridge_model = RidgeLogisticRegression(learning_rate=0.01, num_iterations=1000, lambda_=0.1)
ridge_model.fit(X_train_scaled, y_train)
save_model_as_pkl(ridge_model, "Ridge_Logistic_Regression_Model")

# Evaluate the models
y_pred_logistic = logistic_model.predict(X_test_scaled)
y_pred_ridge = ridge_model.predict(X_test_scaled)

print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_logistic))
print("Ridge Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_ridge))

# Function to load the model and predict
def load_model_and_predict(model_path, test_data):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model.predict(test_data)

# Creating test data for each output category
test_data_samples = [
    [15, 1200, 90, 50000, 5, 1, 1, 1, 1, 1],  # Category 0: Low price
    [12, 1400, 100, 60000, 5, 2, 2, 2, 2, 1], # Category 1: Medium price
    [10, 1600, 120, 70000, 5, 3, 1, 2, 3, 1], # Category 2: High price
    [8, 1800, 150, 80000, 5, 4, 2, 3, 4, 1],  # Category 3: Very high price
]

# Scale the test data
test_data_scaled = scaler.transform(test_data_samples)

# Predicting categories
predictions_logistic = load_model_and_predict("Logistic_Regression_Model.pkl", test_data_scaled)
predictions_ridge = load_model_and_predict("Ridge_Logistic_Regression_Model.pkl", test_data_scaled)

# Display predictions
for i, (pred_logistic, pred_ridge) in enumerate(zip(predictions_logistic, predictions_ridge)):
    print(f"Test data {i+1} predicted category (Logistic): {pred_logistic}, (Ridge): {pred_ridge}")
