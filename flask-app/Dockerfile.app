FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

#ENV MLFLOW_TRACKING_URI=http://mlflow:5000
#ENV MODEL_NAME=neo-prediction

EXPOSE 9696

CMD ["python", "app.py"]