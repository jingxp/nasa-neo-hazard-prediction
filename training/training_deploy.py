from prefect.client.schemas.schedules import CronSchedule
from prefect.deployments import Deployment

from training_flow import train_model_flow

deployment_train = Deployment.build_from_flow(
    flow=train_model_flow,
    name="model_training",
    schedule=CronSchedule(cron="0 0 * * 1"),
    work_queue_name="main",
)

if __name__ == "__main__":
    deployment_train.apply()