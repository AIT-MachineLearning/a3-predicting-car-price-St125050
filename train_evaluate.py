import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models import LogisticRegression, RidgeLogisticRegression
from mlflow_utils import log_model

def train_and_save_models(df):
    X = df.drop('selling_price', axis=1).values
    y = df['selling_price'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logistic_model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
    logistic_model.fit(X_train_scaled, y_train)
    save_model_as_pkl(logistic_model, "Logistic_Regression_Model")

    ridge_model = RidgeLogisticRegression(learning_rate=0.01, num_iterations=1000, lambda_=0.1)
    ridge_model.fit(X_train_scaled, y_train)
    save_model_as_pkl(ridge_model, "Ridge_Logistic_Regression_Model")

    log_model(logistic_model, "Logistic_Regression_Model", X_train_scaled, y_train, X_test_scaled, y_test)
    log_model(ridge_model, "Ridge_Logistic_Regression_Model", X_train_scaled, y_train, X_test_scaled, y_test)
