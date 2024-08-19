# NASA-neo-hazard-prediction
Outer space is teeming with an infinite number of objects, some of which are much closer to Earth than we realize. While a distance of 70,000 km might seem harmless, in astronomical terms, it’s incredibly close and capable of disrupting natural phenomena. These near-Earth objects, including asteroids, can pose significant risks. Therefore, it's crucial to understand what surrounds us and identify potential threats.

This final project for the MLOps Zoom Camp, focuses on developing and deploying a machine learning model for predicting the hazards of near-Earth objects.

# Data
The dataset is from kaggle: 
The NASA certified asteroids that are classified as the nearest earth object are contained. 
Features including:
- neo_id: A unique identifier for each near-Earth object in the dataset.
- name: The name or designation of the near-Earth object.
- absolute_magnitude: The absolute magnitude (brightness) of the NEO, which is a measure of its intrinsic brightness as seen from a standard distance.
- estimated_diameter_min: The minimum estimated diameter of the NEO, typically measured in kilometers.
- estimated_diameter_max: The maximum estimated diameter of the NEO, also measured in kilometers.
- orbiting_body: The celestial body around which the NEO orbits, usually Earth or the Sun.
- relative_velocity: The velocity of the NEO relative to Earth, measured in kilometers per second (km/s).
miss_distance: The distance at which the NEO will pass by Earth, typically measured in kilometers or astronomical units (AU).
- is_hazardous: A binary indicator (True/False) that denotes whether the NEO is classified as potentially hazardous to Earth based on its size, proximity, and other factors.
# Solution
Technologies Used:
## Docker:
Purpose: Containerization of applications and services to ensure consistency across development, testing, and production environments.
Usage: Used to package and deploy MLflow, Prefect, and other necessary services in isolated containers.
## MLflow:
Purpose: Manage the machine learning lifecycle, including experimentation, reproducibility, and deployment.
Usage: Tracks experiments, manages model versions, and facilitates the deployment of predictive models.
## Localstack
Purpose: simulating AWS cloud services locally
Usage: Store Mlflow artifacts by simulating a S3 service
## Prefect:
Purpose: Orchestrate and automate workflows and data pipelines.
Usage: Handles the scheduling, execution, and monitoring of tasks related to data processing and model training.
## Evently:
Purpose: Event-driven architecture and workflow automation.
Usage: Manages event-based triggers and integrations to automate responses and actions based on specific conditions or updates.
## Grafana:
Purpose: Visualization and monitoring of metrics and logs.
Usage: Creates dashboards to monitor the performance of models, visualize prediction results, and track system metrics in real time.

# Reproduce
Here’s a step-by-step guide to reproduce a web-serving implementation for Neo hazardous prediction.

## Pre-requisites
Ensure Docker and Docker Compose are installed in your environment.

## Quick start

1. **Clone the repository:**
 ```bash
 git clone https://github.com/jingxp/nasa-neo-hazard-prediction.git
```
2. **Navigate to the project directory:**
```bash
cd nasa_nearest_earth_objects_prediction/
```
3. **Training and deployment:**
First build images:
```bash
docker-compose build
```
after building, start all the components:
```bash
docker-compose up
```
Training will start automaticly and may take a few minutes

4 **Publish the model:**
After training finished (indicated by prefect contoiner exit safely), we can go to mlflow ui and:
1. go to the Experiments "neo-harzard", the latest running and register the model from "Artifacts" use the model name 
"neo-prediction"
2. go to Models, find the registered models and change the stage to "Production"

5 **Use case test:**
In the local terminal. Run
```bash
python test_app.py
```
This would send 10 batches of data (1000 each batch) to the wed-server and the grafana dash board will also start work.
The predictions will be printed in the terminal.

