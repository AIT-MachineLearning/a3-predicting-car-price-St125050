import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri("http://mlflow.ml.brain.cs.ait.ac.th/")
mlflow.set_experiment("st125050a3car_pred")

def log_model(model, model_name, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=model_name):
        # Train the model
        model.fit(X_train, y_train)

        # Log parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("learning_rate", model.learning_rate)
        mlflow.log_param("num_iterations", model.num_iterations)

        # Make predictions and log metrics
        y_pred = model.predict(X_test)
        accuracy = model.accuracy(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # Log additional metrics
        classes = np.unique(y_test)
        for class_ in classes:
            precision = model.precision(y_test, y_pred, class_)
            recall = model.recall(y_test, y_pred, class_)
            f1 = model.f1_score(y_test, y_pred, class_)
            mlflow.log_metric(f"precision_class_{class_}", precision)
            mlflow.log_metric(f"recall_class_{class_}", recall)
            mlflow.log_metric(f"f1_score_class_{class_}", f1)

        macro_precision = model.macro_precision(y_test, y_pred)
        macro_recall = model.macro_recall(y_test, y_pred)
        macro_f1 = model.macro_f1(y_test, y_pred)
        weighted_precision = model.weighted_precision(y_test, y_pred)
        weighted_recall = model.weighted_recall(y_test, y_pred)
        weighted_f1 = model.weighted_f1(y_test, y_pred)

        mlflow.log_metric("macro_precision", macro_precision)
        mlflow.log_metric("macro_recall", macro_recall)
        mlflow.log_metric("macro_f1", macro_f1)
        mlflow.log_metric("weighted_precision", weighted_precision)
        mlflow.log_metric("weighted_recall", weighted_recall)
        mlflow.log_metric("weighted_f1", weighted_f1)

        # Log the model
        mlflow.sklearn.log_model(model, "model")

        # Return run ID
        return mlflow.active_run().info.run_id

# Log models and metrics to MLflow
logistic_run_id = log_model(logistic_model, "Logistic_Regression_Model", X_train_scaled, y_train, X_test_scaled, y_test)
ridge_run_id = log_model(ridge_model, "Ridge_Logistic_Regression_Model", X_train_scaled, y_train, X_test_scaled, y_test)

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
client.create_model_version(model_name, logistic_model_uri, "Logistic_Regression_Model")

# Register Ridge Logistic Regression model
ridge_model_uri = f"runs:/{ridge_run_id}/model"
client.create_model_version(model_name, ridge_model_uri, "Ridge_Logistic_Regression_Model")
