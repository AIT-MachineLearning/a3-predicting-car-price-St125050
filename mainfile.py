import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess the data
df = pd.read_csv('Cars.csv')

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
df['name'].replace([...], range(1, 32), inplace=True)
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

# Define features and target variable
features = ['mileage', 'engine', 'max_power', 'km_driven', 'seats', 'fuel', 'transmission', 'seller_type', 'owner', 'name']
X = df[features]
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
        self.lambda_ = lambda_

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

# Define Ridge Logistic Regression class
class RidgeLogisticRegression(LogisticRegression):
    def __init__(self, learning_rate=0.01, num_iterations=1000, fit_intercept=True, verbose=False, lambda_=0.1):
        super().__init__(learning_rate, num_iterations, fit_intercept, verbose, lambda_)

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

# Evaluate models
y_pred_logistic = logistic_model.predict(X_test_scaled)
y_pred_ridge = ridge_model.predict(X_test_scaled)

logistic_accuracy = logistic_model.accuracy(y_test, y_pred_logistic)
ridge_accuracy = ridge_model.accuracy(y_test, y_pred_ridge)

print("Logistic Regression Accuracy:", logistic_accuracy)
print("Ridge Logistic Regression Accuracy:", ridge_accuracy)

if logistic_accuracy > ridge_accuracy:
    print("Logistic Regression performed better.")
else:
    print("Ridge Logistic Regression performed better.")
