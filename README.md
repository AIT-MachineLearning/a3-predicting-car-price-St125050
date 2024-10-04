# a3-predicting-car-price-St125050
a3-predicting-car-price-St125050 created by GitHub Classroom


## Live Demo

You can view the live demo of the application at the following link:

[Car Selling Price Prediction App](https://app3.st125050.ml.brain.cs.ait.ac.th/)


# Car Price Prediction

This repository contains a machine learning project aimed at predicting car prices based on various features of the cars using logistic regression techniques. The project implements both standard logistic regression and ridge logistic regression to classify cars into different price categories.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Data Processing](#data-processing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Making Predictions](#making-predictions)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this project is to classify cars into different price categories based on their specifications, such as mileage, engine capacity, and power. By employing machine learning algorithms, we can provide insights into car prices, which can be beneficial for both sellers and buyers in the automotive market.

## Dataset

The dataset used in this project is `Cars.csv`, which contains the following features:
- **name**: Car brand and model
- **mileage**: Distance the car can cover per liter of fuel
- **engine**: Engine capacity (in cc)
- **max_power**: Maximum power of the car (in bhp)
- **km_driven**: Total kilometers driven by the car
- **seats**: Number of seats in the car
- **fuel**: Type of fuel used
- **transmission**: Type of transmission (Manual/Automatic)
- **seller_type**: Type of seller (Individual/Dealer)
- **owner**: Number of previous owners
- **selling_price**: Price at which the car is being sold

The target variable is `price_category`, which categorizes the selling price into four classes.

## Installation

To run this project, ensure you have Python installed along with the required libraries. You can install the necessary libraries using pip:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Usage

1. **Load the Data**: Load the dataset using Pandas.
2. **Preprocess the Data**: Clean the data and encode categorical variables.
3. **Train the Model**: Use the provided classes to train the Logistic Regression and Ridge Logistic Regression models.
4. **Evaluate the Models**: Use the classification report to assess model performance.
5. **Make Predictions**: Use the trained models to make predictions on new data.

## Data Processing

### Step 1: Data Cleaning
The data is first cleaned by:
- Dropping unnecessary columns (like `torque`).
- Removing any rows with missing or duplicate values.

### Step 2: Feature Extraction
The `get_brand_name` function extracts the brand name from the car name, and a `clean_data` function processes the numerical values to ensure they are in the correct format.

### Step 3: Encoding Categorical Variables
Categorical variables are replaced with numeric values to prepare the data for model training:
- Car brands are assigned unique integer values.
- The transmission type, seller type, fuel type, and owner categories are also encoded similarly.

### Step 4: Creating Price Categories
The `selling_price` is binned into categories (0, 1, 2, 3) based on price ranges, making it suitable for classification tasks.

### Step 5: Feature Scaling
The features are then scaled using `StandardScaler` to standardize the data, which improves the performance of the logistic regression algorithms.

## Model Training

Two classes are defined for model training:
1. **Logistic Regression**
2. **Ridge Logistic Regression** (with L2 regularization)

Both models implement methods for:
- Fitting the model to the training data (`fit` method).
- Predicting probabilities and class labels (`predict` method).
- Calculating performance metrics such as accuracy, precision, recall, and F1 score.

### Code Example
```python
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def fit(self, X, y):
        # Model training logic here
```

## Evaluation

After training, the models are evaluated using the classification report from `sklearn`, which provides metrics like precision, recall, and F1-score. This helps in understanding the performance of each model in predicting the price categories.

### Code Example
```python
from sklearn.metrics import classification_report

y_pred_logistic = logistic_model.predict(X_test_scaled)
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_logistic))
```

## Making Predictions

The trained models can be used to predict price categories for new car data. The `load_model_and_predict` function demonstrates how to load a saved model and make predictions on sample data.

### Code Example
```python
def load_model_and_predict(model_path, test_data):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model.predict(test_data)

# Example test data
test_data_samples = [
    [15, 1200, 90, 50000, 5, 1, 1, 1, 1, 1],  # Category 0
    ...
]

# Predict categories
predictions_logistic = load_model_and_predict("Logistic_Regression_Model.pkl", test_data_scaled)
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests. Please ensure to follow the code of conduct and contribution guidelines.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
```

### Customization Suggestions
- Add more specific explanations for any complex logic in your code.
- If your project includes any additional features or libraries, mention those in the relevant sections.
- Consider adding visualizations or screenshots to illustrate your results.
- If applicable, include references to any research papers or datasets you used.





### Code Breakdown

```python
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
```
- **Imports**: The code begins by importing necessary libraries. `mlflow` is the primary library for managing machine learning workflows, while `mlflow.sklearn` is specifically for logging Scikit-learn models. The `MlflowClient` is used to interact with the MLflow server.

### MLflow Setup

```python
mlflow.set_tracking_uri("http://mlflow.ml.brain.cs.ait.ac.th/")
mlflow.set_experiment("st125050a3car_pred")
```
- **Tracking URI**: Sets the URI for the MLflow tracking server, which is where all experiment data will be logged.
- **Experiment**: Specifies the name of the experiment, which helps organize and manage runs related to this project.

### Logging Function

```python
def log_model(model, model_name, X_train, y_train, X_test, y_test):
```
- **Function Definition**: This function takes in a model, its name, and training/testing datasets to log the model's training process and performance metrics.

#### Model Training and Logging

```python
    with mlflow.start_run(run_name=model_name):
```
- **Start a Run**: Initiates a new run within the MLflow experiment, allowing for logging related to this specific model.

```python
        model.fit(X_train, y_train)
```
- **Model Training**: The model is trained using the training dataset.

#### Logging Parameters

```python
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("learning_rate", model.learning_rate)
        mlflow.log_param("num_iterations", model.num_iterations)
```
- **Log Parameters**: The model type and relevant hyperparameters (like learning rate and number of iterations) are logged for later reference.

#### Making Predictions and Logging Metrics

```python
        y_pred = model.predict(X_test)
        accuracy = model.accuracy(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
```
- **Predictions**: The trained model makes predictions on the test dataset.
- **Log Metrics**: Accuracy is calculated and logged.

#### Class-Specific Metrics

```python
        classes = np.unique(y_test)
        for class_ in classes:
            precision = model.precision(y_test, y_pred, class_)
            recall = model.recall(y_test, y_pred, class_)
            f1 = model.f1_score(y_test, y_pred, class_)
            mlflow.log_metric(f"precision_class_{class_}", precision)
            mlflow.log_metric(f"recall_class_{class_}", recall)
            mlflow.log_metric(f"f1_score_class_{class_}", f1)
```
- **Metrics Loop**: For each unique class in the target variable, precision, recall, and F1 score are calculated and logged.

#### Logging Macro and Weighted Metrics

```python
        macro_precision = model.macro_precision(y_test, y_pred)
        macro_recall = model.macro_recall(y_test, y_pred)
        macro_f1 = model.macro_f1(y_test, y_pred)
        weighted_precision = model.weighted_precision(y_test, y_pred)
        weighted_recall = model.weighted_recall(y_test, y_pred)
        weighted_f1 = model.weighted_f1(y_test, y_pred)
```
- **Macro and Weighted Metrics**: Additional metrics are computed to provide a broader view of model performance, especially for imbalanced classes.

#### Logging the Model

```python
        mlflow.sklearn.log_model(model, "model")
```
- **Log Model**: The trained model is saved to the MLflow tracking server for later retrieval.

#### Return Run ID

```python
        return mlflow.active_run().info.run_id
```
- **Return Run ID**: The function returns the unique identifier for the current run, which can be used for model registration.

### Logging Models

```python
logistic_run_id = log_model(logistic_model, "Logistic_Regression_Model", X_train_scaled, y_train, X_test_scaled, y_test)
ridge_run_id = log_model(ridge_model, "Ridge_Logistic_Regression_Model", X_train_scaled, y_train, X_test_scaled, y_test)
```
- **Log Models**: Calls the `log_model` function for both the logistic regression and ridge logistic regression models, capturing their training processes and performance metrics.

### Retrieving Metrics

```python
def get_run_metrics(model_name):
    client = MlflowClient()
    experiment = client.get_experiment_by_name("st125050a3car_pred")
    runs = client.search_runs(experiment.experiment_id)
    for run in runs:
        if run.info.run_name == model_name:
            run_id = run.info.run_id
            metrics = client.get_run(run_id).data.metrics
            return metrics
    return None
```
- **Function Definition**: Retrieves logged metrics for a specified model by searching through the experiment runs.
- **Search Runs**: Looks for runs that match the given model name and fetches metrics associated with the run.

### Fetching and Comparing Metrics

```python
logistic_metrics = get_run_metrics("Logistic_Regression_Model")
ridge_metrics = get_run_metrics("Ridge_Logistic_Regression_Model")

print("Logistic Regression Metrics:", logistic_metrics)
print("Ridge Logistic Regression Metrics:", ridge_metrics)
```
- **Fetch Metrics**: Retrieves and prints metrics for both models.

### Accuracy Comparison

```python
logistic_accuracy = logistic_metrics.get("accuracy", 0)
ridge_accuracy = ridge_metrics.get("accuracy", 0)

if logistic_accuracy > ridge_accuracy:
    print("Logistic Regression performed better.")
else:
    print("Ridge Logistic Regression performed better.")
```
- **Comparison**: Compares the accuracy of both models and prints which model performed better.

### Model Registration

```python
client = MlflowClient()
model_name = "st125050-a3carpred"

try:
    client.get_registered_model(model_name)
    print(f"Model '{model_name}' already exists.")
except mlflow.exceptions.RestException:
    client.create_registered_model(model_name)
    print(f"Model '{model_name}' created.")
```
- **Model Registration**: Checks if the model already exists in the MLflow model registry. If not, it creates a new registered model.

### Creating Model Versions

```python
logistic_model_uri = f"runs:/{logistic_run_id}/model"
client.create_model_version(model_name, logistic_model_uri, "Logistic_Regression_Model")

ridge_model_uri = f"runs:/{ridge_run_id}/model"
client.create_model_version(model_name, ridge_model_uri, "Ridge_Logistic_Regression_Model")
```
- **Registering Model Versions**: Both models are registered under the defined model name, associating each version with the respective run ID.




Code Breakdown

```python
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from mainfile import RidgeLogisticRegression
```
- **Imports**: 
  - `streamlit` is used for building the web application.
  - `numpy` and `pandas` are libraries for data manipulation and numerical operations.
  - `pickle` is used for loading the trained machine learning model and the scaler.
  - The custom `RidgeLogisticRegression` class is imported from `mainfile`.

### Loading the Model and Scaler

```python
# Load the trained Ridge Logistic Regression model
with open('Ridge_Logistic_Regression_Model.pkl', 'rb') as file:
    ridge_model = pickle.load(file)

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
```
- **Model Loading**: The trained Ridge Logistic Regression model and the `StandardScaler` used for feature scaling are loaded from their respective `.pkl` files.

### Price Ranges Definition

```python
# Define the price ranges for each category
price_ranges = {
    0: (0, 200000),      # Category 0: Low price
    1: (200001, 500000), # Category 1: Medium price
    2: (500001, 1000000),# Category 2: High price
    3: (1000001, float('inf')) # Category 3: Very high price
}
```
- **Price Ranges**: A dictionary is created to map price categories to their respective price ranges. This will be used later to display the estimated selling price based on the predicted category.

### Streamlit App Definition

```python
def main():
    st.title("Car Selling Price Prediction")
```
- **Main Function**: The `main` function defines the main body of the Streamlit application and sets the title of the web app.

### Input Fields for Features

```python
    # Input fields for the features
    mileage = st.number_input("Mileage (kmpl)", min_value=0.0, step=0.1)
    engine = st.number_input("Engine (CC)", min_value=0.0, step=0.1)
    max_power = st.number_input("Max Power (bhp)", min_value=0.0, step=0.1)
    km_driven = st.number_input("KM Driven", min_value=0)
    seats = st.number_input("Seats", min_value=1, max_value=10, step=1)
```
- **Input Fields**: Various input fields are created using Streamlit's `number_input` and `selectbox` functions to collect features necessary for prediction, including mileage, engine capacity, max power, kilometers driven, number of seats, fuel type, transmission type, seller type, owner category, and brand.

### Fuel, Transmission, Seller Type, Owner Type, and Brand Selection

```python
    fuel = st.selectbox("Fuel Type", options=[1, 2, 3, 4],
                         format_func=lambda x: ["Diesel", "Petrol", "LPG", "CNG"][x-1])
    
    transmission = st.selectbox("Transmission", options=[1, 2],
                                  format_func=lambda x: ["Manual", "Automatic"][x-1])
    
    seller_type = st.selectbox("Seller Type", options=[1, 2, 3],
                                format_func=lambda x: ["Individual", "Dealer", "Trustmark Dealer"][x-1])
    
    owner = st.selectbox("Owner", options=[1, 2, 3, 4, 5],
                         format_func=lambda x: ["First Owner", "Second Owner", "Third Owner", 
                                                "Fourth & Above Owner", "Test Drive Car"][x-1])
    
    brand_names = [
        'Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
        'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
        'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
        'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
        'Ambassador', 'Ashok', 'Isuzu', 'Opel'
    ]
    brand = st.selectbox("Brand", options=list(range(1, len(brand_names) + 1)),
                         format_func=lambda x: brand_names[x-1])
```
- **Categorical Inputs**: The fuel type, transmission, seller type, owner category, and brand are collected through dropdown menus, with labels mapped to numerical values for consistency with the model's expected input.

### Prediction Button

```python
    # Predict button
    if st.button("Predict"):
        # Prepare the input data
        input_data = np.array([[mileage, engine, max_power, km_driven, seats, fuel, 
                                transmission, seller_type, owner, brand]])
        input_data_scaled = scaler.transform(input_data)

        # Predict the selling price category
        prediction = ridge_model.predict(input_data_scaled)

        # Get the price range based on the predicted category
        category = int(prediction[0])
        price_range = price_ranges.get(category, (0, 0))
        
        # Display the prediction
        st.success(f"Predicted Selling Price Category: {category}")
        st.success(f"Estimated Selling Price Range: ₹{price_range[0]:,.2f} - ₹{price_range[1]:,.2f}")
```
- **Prediction Logic**: 
  - When the "Predict" button is clicked, the input data is collected into a NumPy array.
  - The data is then scaled using the preloaded `scaler`.
  - The Ridge Logistic Regression model makes a prediction based on the scaled input.
  - The predicted category is used to retrieve the corresponding price range from the `price_ranges` dictionary.
  - Finally, the predicted category and estimated price range are displayed as success messages on the app.

### Main Function Execution

```python
if __name__ == "__main__":
    main()
```
- **Execution Check**: The script checks if it is being run as the main module and calls the `main` function to start the application.


