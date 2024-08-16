import os
import time
import pandas as pd
import mlflow
from flask import Flask, request, jsonify

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
    data = data.dropna()
    data = data.drop(['neo_id', 'name', 'orbiting_body'], axis=1)
    data['average_diameter'] = (data['estimated_diameter_min'] + data['estimated_diameter_max']) / 2
    data['scaled_relative_velocity'] = (data['relative_velocity'] - data['relative_velocity'].min()) / (data['relative_velocity'].max() - data['relative_velocity'].min())
    data['momentum'] = data['relative_velocity'] * data['average_diameter']
    data['velocity_distance_ratio'] = data['relative_velocity'] / data['miss_distance']
    data['diameter_magnitude_ratio'] = data['average_diameter'] /data['absolute_magnitude']
    return data

app = Flask('neo-hazard-prediction')

@app.route('/')
def welcome():
    return "Welcome to the Neo-Hazard Prediction API!"

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    json_data = request.get_json()
    data = pd.read_json(json_data, orient='records')
    processed_data = data_process(data)
    model = load_production_model(os.getenv("MODEL_NAME"))
    prediction = model.predict(processed_data)
    result = prediction.tolist()
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
