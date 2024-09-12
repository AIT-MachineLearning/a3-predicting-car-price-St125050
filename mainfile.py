import numpy as np
import pandas as pd
df = pd.read_csv('/content/Cars.csv')

df.head()

df.shape

df.drop(columns=['torque'], inplace=True)

df.dropna(inplace=True)
df.duplicated().sum()
df.drop_duplicates(inplace=True)

df.shape
df.info()

def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

def clean_data(value):
    value = value.split(' ')[0]
    value = value.strip()
    if value == '':
        value = 0
    return float(value)

get_brand_name('Maruti  Swift Dzire VDI')


df['name'] = df['name'].apply(get_brand_name)
df['name'].unique()


def clean_data(value):
    if isinstance(value, str): # Check if the value is a string
        value = value.split(' ')[0]
        value = value.strip()
        if value == '':
            value = 0
    if isinstance(value, float): # Check if the value is a float and return as is
        return value
    return float(value) # Convert to float if not already a float



df['mileage'] = df['mileage'].apply(clean_data)
df['max_power'] = df['max_power'].apply(clean_data)
df['engine'] = df['engine'].apply(clean_data)
df['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
       'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
       'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
       'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
       'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                          [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
                          ,inplace=True)
df['transmission'].unique()
df['transmission'].replace(['Manual', 'Automatic'],[1,2], inplace=True)
df['seller_type'].unique() # correct
df['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'],[1,2,3], inplace=True)
df['fuel'].unique()
df['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'],[1,2,3,4], inplace=True)
df.reset_index(inplace=True)

df.info()

df['owner'].unique()
df['owner'].replace(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'],[1,2,3,4,5], inplace=True)
df['owner'].unique()
df.head()

df.isnull().sum()

import matplotlib.pyplot as plt
import seaborn as sns

# Distribution plot for selling_price
plt.figure(figsize=(10, 6))
sns.histplot(df['selling_price'], bins=30, kde=True)
plt.title('Distribution of Selling Prices')
plt.xlabel('Selling Price')
plt.ylabel('Frequency')
plt.show()


# Count plot for fuel types
plt.figure(figsize=(10, 6))
sns.countplot(x='fuel', data=df)
plt.title('Count of Cars by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Diesel', 'Petrol', 'LPG', 'CNG'])  # Customize labels if necessary
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='selling_price', data=df)
plt.title('Box Plot of Selling Prices')
plt.xlabel('Selling Price')
plt.show()
# Calculate the correlation matrix
corr = df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Selecting numerical features for pair plot
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
# Convert selling price to discrete variable
df['price_category'] = pd.cut(df['selling_price'], bins=4, labels=[0, 1, 2, 3])
print(df['price_category'].unique())


import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

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

# Define function to save model as .pkl file
def save_model_as_pkl(model, model_name):
    with open(f'{model_name}.pkl', 'wb') as file:
        pickle.dump(model, file)

# Define function to log model and metrics to MLflow
def log_model(model, model_name, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    with mlflow.start_run(run_name=model_name):
        # Log model parameters
        params = {
            "learning_rate": getattr(model, "learning_rate", None),
            "num_iterations": getattr(model, "num_iterations", None),
            "lambda_": getattr(model, "lambda_", None)
        }
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        
        # Log metrics
        mlflow.log_metric("accuracy", model.accuracy(y_test, y_pred))
        mlflow.log_metric("macro_precision", model.macro_precision(y_test, y_pred))
        mlflow.log_metric("macro_recall", model.macro_recall(y_test, y_pred))
        mlflow.log_metric("macro_f1", model.macro_f1(y_test, y_pred))
        mlflow.log_metric("weighted_precision", model.weighted_precision(y_test, y_pred))
        mlflow.log_metric("weighted_recall", model.weighted_recall(y_test, y_pred))
        mlflow.log_metric("weighted_f1", model.weighted_f1(y_test, y_pred))
        
        # Save the model to MLflow
        mlflow.log_artifact(f'{model_name}.pkl')



# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train Logistic Regression model
logistic_model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
logistic_model.fit(X_train_scaled, y_train)

# Save Logistic Regression model locally
save_model_as_pkl(logistic_model, "Logistic_Regression_Model")

# Initialize and train Ridge Logistic Regression model
ridge_model = RidgeLogisticRegression(learning_rate=0.01, num_iterations=1000, lambda_=0.1)
ridge_model.fit(X_train_scaled, y_train)

# Save Ridge Logistic Regression model locally
save_model_as_pkl(ridge_model, "Ridge_Logistic_Regression_Model")


# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri("http://mlflow.ml.brain.cs.ait.ac.th/")
mlflow.set_experiment("st125050a3carpred")

# Log models and metrics to MLflow
log_model(logistic_model, "Logistic_Regression_Model", X_train_scaled, y_train, X_test_scaled, y_test)
log_model(ridge_model, "Ridge_Logistic_Regression_Model", X_train_scaled, y_train, X_test_scaled, y_test)

def get_run_metrics(model_name):
    client = mlflow.tracking.MlflowClient()
    # Use the correct experiment name "st125050a3carpred" 
    experiment = client.get_experiment_by_name("st125050a3carpred")
    # Use search_runs() instead of list_run_infos()
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


#download models as pkl files
import pickle
with open('Logistic_Regression_Model.pkl', 'wb') as f:
    pickle.dump(logistic_model, f)
with open('Ridge_Logistic_Regression_Model.pkl', 'wb') as f:
    pickle.dump(ridge_model, f)


import mlflow
mlflow.set_tracking_uri("http://mlflow.ml.brain.cs.ait.ac.th/")
mlflow.set_experiment("st125050a3carpred")


model_uri = f"runs:/907bc2ad0ebf4922a1604ff659c4ddbf/model"

from mlflow.tracking import MlflowClient

client = MlflowClient()
 
model_name = "st125050-a3-model"
run_id = "907bc2ad0ebf4922a1604ff659c4ddbf" # replace with your run ID

# Register the model
result = client.create_registered_model(model_name)

# Create a new version of the model
client.create_model_version(
    name=model_name,
    source=f"runs:/{run_id}/model",
    run_id=run_id
)


from mlflow.tracking import MlflowClient

# Initialize the MLflow client
client = MlflowClient()

model_name = "st125050-a3-model"
version = 1  # The version number you created

# Transition the model version to the staging stage
client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage="staging"
)

print(f"Model version {version} transitioned to staging stage.")
from mlflow.tracking import MlflowClient

# Initialize the MLflow client
client = MlflowClient()

model_name = "st125050-a3-model"
version = 1  # The version number you created

# Retrieve the model version information
model_version = client.get_model_version(name=model_name, version=version)

# Print current stage
print(f"Model version {version} is in stage: {model_version.current_stage}")
