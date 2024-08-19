import os
import time
import pandas as pd
import mlflow
from flask import Flask, request, jsonify
from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import DatasetDriftMetric, ColumnQuantileMetric
import psycopg

create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
	num_drifted_columns integer,
    fare_half_quantile float
)
"""
num_features = ['absolute_magnitude', 'estimated_diameter_min',
       'estimated_diameter_max', 'relative_velocity', 'miss_distance',
       'average_diameter', 'scaled_relative_velocity', 'momentum',
       'velocity_distance_ratio', 'diameter_magnitude_ratio']

column_mapping = ColumnMapping(
    numerical_features=num_features,
)

report = Report(metrics = [
    DatasetDriftMetric(),
    ColumnQuantileMetric(column_name='relative_velocity', quantile=0.5)
])

reference_data = pd.read_csv('ref_data.csv')

def prep_db():
    """
    Create a new database in postgres server if it's not exist
    """
    with psycopg.connect("host=postgres port=5432 user=postgres password=example", autocommit=True) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='nasa_neo'")
        if len(res.fetchall()) == 0:
            conn.execute("create database nasa_neo;")
        with psycopg.connect("host=postgres port=5432 dbname=nasa_neo user=postgres password=example") as conn:
            conn.execute(create_table_statement)

def calculate_metrics_postgresql(current_data, reference_data=reference_data):
    """
    Calculate dataset drift and quantile metrics using evently
    Args:
        current_data (pd.DataFrame): data for predcition
        reference_data (pd.DataFrame, optional): data to compare with. Defaults to reference_data.

    Returns:
        num_drifted_columns: Number of drifted columns
        fare_half_quantile: half quantile value of selected feature
    """
    report.run(reference_data = reference_data, current_data = current_data,
               column_mapping=column_mapping)
    result = report.as_dict()
    num_drifted_columns = result['metrics'][0]['result']['number_of_drifted_columns']
    fare_half_quantile = result['metrics'][1]['result']['current']['value']
    return num_drifted_columns, fare_half_quantile

def send_monitoring_metrics(current_data):
    with psycopg.connect("host=postgres port=5432 dbname=nasa_neo user=postgres password=example", autocommit=True) as conn:
        with conn.cursor() as curr:
            num_drifted_columns, fare_half_quantile = calculate_metrics_postgresql(current_data)
            curr.execute("insert into dummy_metrics(num_drifted_columns, fare_half_quantile) values (%s, %s)",
                         (num_drifted_columns,fare_half_quantile)
                         )

def load_production_model(model_name, stage="Production", polling_interval=60, max_retries=10):
    """
    Load the production model from MLflow.
    
    Parameters:
    - model_name (str): The name of the model to load.
    - stage (str): The stage of the model to load, default is 'Production'.
    - polling_interval (int): The time to wait before checking again if the model isn't available, default is 60 seconds.
    
    Returns:
    - model: The loaded MLflow model.
    """
    # Set the tracking URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI environment variable is not set.")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    
    attempts = 0
    while attempts < max_retries:
        try:
            # Attempt to load the model from the specified stage
            model_uri = f"models:/{model_name}/{stage}"
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"Model '{model_name}' loaded successfully from stage '{stage}'")
            return model
        except mlflow.exceptions.MlflowException as e:
            # If model is not found or cannot be loaded, wait and retry
            print(f"Attempt {attempts + 1}: Model '{model_name}' not found in stage '{stage}'. Waiting for {polling_interval} seconds.")
            time.sleep(polling_interval)
            attempts += 1
        except Exception as e:
            # Handle unexpected exceptions
            print(f"Unexpected error: {e}")
            raise
    raise RuntimeError(f"Failed to load model '{model_name}' from stage '{stage}' after {max_retries} attempts.")

def data_process(data: pd.DataFrame):
    """
    Data preprocessing
    Args:
        data (pd.DataFrame): batch data for prediction
    Returns:
        data: preprocessed dataframe
    """
    data = data.dropna()
    data = data.drop(['neo_id', 'name', 'orbiting_body', 'is_hazardous'], axis=1)
    data['average_diameter'] = (data['estimated_diameter_min'] + data['estimated_diameter_max']) / 2
    data['scaled_relative_velocity'] = (data['relative_velocity'] - data['relative_velocity'].min()) / (data['relative_velocity'].max() - data['relative_velocity'].min())
    data['momentum'] = data['relative_velocity'] * data['average_diameter']
    data['velocity_distance_ratio'] = data['relative_velocity'] / data['miss_distance']
    data['diameter_magnitude_ratio'] = data['average_diameter'] /data['absolute_magnitude']
    return data

app = Flask('neo-hazard-prediction')

@app.route('/')
def welcome():
    return "Welcome to the Neo-Hazardous Prediction API!"

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    json_data = request.get_json()
    data = pd.read_json(json_data, orient='records')
    model = load_production_model(os.getenv("MODEL_NAME"))
    processed_data = data_process(data)
    send_monitoring_metrics(current_data=processed_data)
    prediction = model.predict(processed_data)
    result = prediction.tolist()
    return jsonify(result)

if __name__ == "__main__":
    prep_db()
    app.run(debug=True, host='0.0.0.0', port=9696)
