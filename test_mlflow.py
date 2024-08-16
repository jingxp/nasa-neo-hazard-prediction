import mlflow
import time

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

if __name__ == "__main__":
    #app.run(debug=True, host='0.0.0.0', port=9696)
    mlflow.set_tracking_uri("http://localhost:5000")
    model_name = 'neo-prediction'
    model = load_production_model(model_name)
    print(model)