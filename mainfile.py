import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

# Load and preprocess the data
df = pd.read_csv('/content/Cars.csv')

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

df['name'] = df['name'].apply(get_brand_name)
df['mileage'] = df['mileage'].apply(clean_data)
df['max_power'] = df['max_power'].apply(clean_data)
df['engine'] = df['engine'].apply(clean_data)

# Replace categorical values with numeric
df['name'].replace([
    'Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
    'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
    'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
    'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
    'Ambassador', 'Ashok', 'Isuzu', 'Opel'
], range(1, 32), inplace=True)
df['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
df['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
df['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
df['owner'].replace(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'], [1, 2, 3, 4, 5], inplace=True)

df.reset_index(drop=True, inplace=True)

# Plotting
plt.figure(figsize=(10, 6))
sns.histplot(df['selling_price'], bins=30, kde=True)
plt.title('Distribution of Selling Prices')
plt.xlabel('Selling Price')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='fuel', data=df)
plt.title('Count of Cars by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Diesel', 'Petrol', 'LPG', 'CNG'])
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='selling_price', data=df)
plt.title('Box Plot of Selling Prices')
plt.xlabel('Selling Price')
plt.show()

corr = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

numerical_features = ['selling_price', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
plt.figure(figsize=(12, 10))
sns.pairplot(df[numerical_features])
plt.suptitle('Pair Plot of Selected Features', y=1.02)
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['km_driven'], bins=30, kde=True)
plt.title('Distribution of KM Driven')
plt.xlabel('KM Driven')
plt.ylabel('Frequency')
plt.show()

df['price_category'] = pd.cut(df['selling_price'], bins=4, labels=[0, 1, 2, 3])
# Define features and target variable
features = ['mileage', 'engine', 'max_power', 'km_driven', 'seats', 'fuel', 'transmission', 'seller_type', 'owner', 'name']
X = df[features]
# Change y to be the selling_price instead of price_category
y = df['selling_price']

# Split and scale the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define Logistic Regression class
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, fit_intercept=True, verbose=False, lambda_=0):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.lambda_ = lambda_  # Ridge regularization parameter

    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.add_intercept(X)

        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size

            # Add regularization term to gradient
            gradient += (self.lambda_ / y.size) * self.theta
            gradient[0] -= (self.lambda_ / y.size) * self.theta[0]  # Don't regularize intercept

            self.theta -= self.learning_rate * gradient

            if self.verbose and i % 100 == 0:
                print(f'Iteration {i}: loss = {self.loss(h, y)}')

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.add_intercept(X)
        return self.sigmoid(np.dot(X, self.theta))

    def predict(self, X):
        return (self.predict_prob(X) >= 0.5).astype(int)

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def precision(self, y_true, y_pred, class_):
        true_positives = np.sum((y_true == class_) & (y_pred == class_))
        predicted_positives = np.sum(y_pred == class_)
        return true_positives / predicted_positives if predicted_positives > 0 else 0

    def recall(self, y_true, y_pred, class_):
        true_positives = np.sum((y_true == class_) & (y_pred == class_))
        actual_positives = np.sum(y_true == class_)
        return true_positives / actual_positives if actual_positives > 0 else 0

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

# Define Ridge Logistic Regression class
class RidgeLogisticRegression(LogisticRegression):
    def __init__(self, learning_rate=0.01, num_iterations=1000, fit_intercept=True, verbose=False, lambda_=0.1):
        super().__init__(learning_rate, num_iterations, fit_intercept, verbose, lambda_)

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.add_intercept(X)

        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size

            # Add regularization term to gradient
            gradient += (self.lambda_ / y.size) * self.theta
            gradient[0] -= (self.lambda_ / y.size) * self.theta[0]  # Don't regularize intercept

            self.theta -= self.learning_rate * gradient

            if self.verbose and i % 100 == 0:
                print(f'Iteration {i}: loss = {self.loss(h, y)}')

# Save model as pickle file
def save_model_as_pkl(model, model_name):
    with open(f'{model_name}.pkl', 'wb') as file:
        pickle.dump(model, file)

# Log model to MLflow
def log_model(model, model_name, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    with mlflow.start_run(run_name=model_name) as run:
        params = {
            "learning_rate": getattr(model, "learning_rate", None),
            "num_iterations": getattr(model, "num_iterations", None),
            "lambda_": getattr(model, "lambda_", None)
        }
        mlflow.log_params({k: v for k, v in params.items() if v is not None})

        mlflow.log_metric("accuracy", model.accuracy(y_test, y_pred))
        mlflow.log_metric("macro_precision", model.macro_precision(y_test, y_pred))
        mlflow.log_metric("macro_recall", model.macro_recall(y_test, y_pred))
        mlflow.log_metric("macro_f1", model.macro_f1(y_test, y_pred))
        mlflow.log_metric("weighted_precision", model.weighted_precision(y_test, y_pred))
        mlflow.log_metric("weighted_recall", model.weighted_recall(y_test, y_pred))
        mlflow.log_metric("weighted_f1", model.weighted_f1(y_test, y_pred))

        mlflow.log_artifact(f'{model_name}.pkl')
        
        return run.info.run_id

# Initialize and train Logistic Regression model
logistic_model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
logistic_model.fit(X_train_scaled, y_train)
save_model_as_pkl(logistic_model, "Logistic_Regression_Model")

# Initialize and train Ridge Logistic Regression model
ridge_model = RidgeLogisticRegression(learning_rate=0.01, num_iterations=1000, lambda_=0.1)
ridge_model.fit(X_train_scaled, y_train)
save_model_as_pkl(ridge_model, "Ridge_Logistic_Regression_Model")

# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri("http://mlflow.ml.brain.cs.ait.ac.th/")
mlflow.set_experiment("st125050a3car_pred")

# Log models and metrics to MLflow
logistic_run_id = log_model(logistic_model, "Logistic_Regression_Model", X_train_scaled, y_train, X_test_scaled, y_test)
ridge_run_id = log_model(ridge_model, "Ridge_Logistic_Regression_Model", X_train_scaled, y_train, X_test_scaled, y_test)

def get_run_metrics(model_name):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("st125050a3car_pred")
    runs = client.search_runs(experiment.experiment_id)
    for run in runs:
        if run.info.run_name == model_name:
            run_id = run.info.run_id
            metrics = client.get_run(run_id).data.metrics
            return metrics
    return None

logistic_metrics = get_run_metrics("Logistic_Regression_Model")
ridge_metrics = get_run_metrics("Ridge_Logistic_Regression_Model")

print("Logistic Regression Metrics:", logistic_metrics)
print("Ridge Logistic Regression Metrics:", ridge_metrics)

# Compare accuracy
logistic_accuracy = logistic_metrics.get("accuracy", 0)
ridge_accuracy = ridge_metrics.get("accuracy", 0)

if logistic_accuracy > ridge_accuracy:
    print("Logistic Regression performed better.")
else:
    print("Ridge Logistic Regression performed better.")

# Register and transition model in MLflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Define model name
model_name = "st125050-a3carpred"

# Check if model already exists
try:
    client.get_registered_model(model_name)
    print(f"Model '{model_name}' already exists.")
except mlflow.exceptions.RestException:
    client.create_registered_model(model_name)
    print(f"Model '{model_name}' created.")

# Register Logistic Regression model
logistic_model_uri = f"runs:/{logistic_run_id}/model"
logistic_model_version = client.create_model_version(model_name, logistic_model_uri, "Logistic_Regression_Model")

# Register Ridge Logistic Regression model
ridge_model_uri = f"runs:/{ridge_run_id}/model"
ridge_model_version = client.create_model_version(model_name, ridge_model_uri, "Ridge_Logistic_Regression_Model")