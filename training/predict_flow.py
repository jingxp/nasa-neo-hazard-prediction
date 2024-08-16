import prefect
from prefect import Flow
from training_flow import data_prep

with Flow("predict-flow") as flow:
    data = prefect.Parameter("data")
    processed_data = data_prep(data)
    model = prefect.Parameter("model")
    prediction = model.predict(processed_data)
    result = prediction[0]