#!/bin/bash

# Wait for LocalStack to be ready
until awslocal s3 ls; do
  echo "Waiting for LocalStack to be ready..."
  sleep 2
done

# Create the S3 bucket
awslocal s3 mb s3://mlflow-model-neo