import mlflow
import pickle
from mlflow.tracking import MlflowClient

def log_model(model, model_name, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({
            "learning_rate": model.learning_rate,
            "num_iterations": model.num_iterations,
            "lambda_": model.lambda_
        })
        mlflow.log_metric("accuracy", model.accuracy(y_test, y_pred))
        mlflow.log_metric("macro_precision", model.macro_precision(y_test, y_pred))
        mlflow.log_metric("macro_recall", model.macro_recall(y_test, y_pred))
        mlflow.log_metric("macro_f1", model.macro_f1(y_test, y_pred))
        mlflow.log_metric("weighted_precision", model.weighted_precision(y_test, y_pred))
        mlflow.log_metric("weighted_recall", model.weighted_recall(y_test, y_pred))
        mlflow.log_metric("weighted_f1", model.weighted_f1(y_test, y_pred))
        mlflow.log_artifact(f'{model_name}.pkl')

def register_model(model_name, run_id):
    client = MlflowClient()
    result = client.create_registered_model(model_name)
    client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/model",
        run_id=run_id
    )

def transition_model_stage(model_name, version):
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="staging"
    )

def get_run_metrics(model_name):
    client = MlflowClient()
    experiment = client.get_experiment_by_name("st125050a3carpred")
    runs = client.search_runs(experiment.experiment_id)
    for run in runs:
        if run.info.run_name == model_name:
            run_id = run.info.run_id
            metrics = client.get_run(run_id).data.metrics
            return metrics
    return None
