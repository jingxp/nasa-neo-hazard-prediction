import os
import pandas as pd
import mlflow
from prefect import task, flow
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import logging

@task(log_prints=True, retries=3, tags=["read_data"])
def read_data(file_name: str) -> pd.DataFrame:
    logging.info("Reading data from csv ...")
    df = pd.read_csv(file_name)
    return df

@task(log_prints=True, tags=["data_preprocessing"])
def data_prep(data: pd.DataFrame) -> pd.DataFrame:
    logging.info("Data prep ...")
    data = data.dropna()
    data = data.drop(['neo_id', 'name', 'orbiting_body'], axis=1)
    
    data['average_diameter'] = (data['estimated_diameter_min'] + data['estimated_diameter_max']) / 2
    data['scaled_relative_velocity'] = (data['relative_velocity'] - data['relative_velocity'].min()) / (data['relative_velocity'].max() - data['relative_velocity'].min())
    data['momentum'] = data['relative_velocity'] * data['average_diameter']
    data['velocity_distance_ratio'] = data['relative_velocity'] / data['miss_distance']
    data['diameter_magnitude_ratio'] =data['average_diameter'] /data['absolute_magnitude']
    
    label_encoder = LabelEncoder()
    data['hazardous_label'] = label_encoder.fit_transform(data["is_hazardous"])
    data = data.drop(["is_hazardous"], axis=1)
    return data

@task(log_prints=True, tags=["split_data"])
def data_split(data: pd.DataFrame):
    X = data.drop(['hazardous_label'], axis=1)
    y = data['hazardous_label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=89)
    return X_train, X_test, y_train, y_test

@task(log_prints=True, tags=["train_model"])
def train_model(data_path: str):
    mlflow.sklearn.autolog()
    data = read_data(data_path)
    prepared_data = data_prep(data)
    X_train, X_test, y_train, y_test = data_split(prepared_data)

    rf = RandomForestClassifier(random_state=99)
    param_grid = {'n_estimators': [10, 50, 100],
                  'max_depth': [2, 5, 10],
                  'min_samples_split': [2, 3, 4],}
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    
    with mlflow.start_run():
        # Fit the model
        grid_search.fit(X_train, y_train)
        # Log the best parameters
        mlflow.log_params(grid_search.best_params_)

        # Predict on the test set
        y_pred = grid_search.predict(X_test)
        y_score = grid_search.predict_proba(X_test)[:, 1]

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric('accuracy', accuracy)
        recall = recall_score(y_test, y_pred)
        mlflow.log_metric('recall', recall)
        auc = roc_auc_score(y_test, y_score)
        mlflow.log_metric('auc', auc)
        rfc_cm = RocCurveDisplay.from_estimator(grid_search.best_estimator_, X_test, y_test)
        plt.title(f"ROC Curve (AUC = {auc:.2f})")
        roc_fig_path = "roc_curve.png"
        rfc_cm.figure_.savefig(roc_fig_path)
        mlflow.log_artifact(roc_fig_path)
        # Log the model
        mlflow.sklearn.log_model(grid_search.best_estimator_, "model")
        print(f'Best Parameters: {grid_search.best_params_}')
        print(f'Accuracy: {accuracy}')
        mlflow.end_run()

@flow
def train_model_flow():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment('neo-harzard')
    data_path = 'nearest-earth-objects(1910-2024).csv'
    train_model(data_path)

if __name__ == "__main__":
    train_model_flow()
