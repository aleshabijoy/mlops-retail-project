import os
os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/mlruns"
import mlflow
mlflow.set_tracking_uri("file:///tmp/mlruns")

from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Retail Analytics API. Go to /docs for endpoints."}

@patch('main.mlflow')
@patch('analysis_logic.mlflow')
def test_trigger_training_endpoint(mock_mlflow_main, mock_mlflow_analysis):
    response = client.post("/trigger-training")
    assert response.status_code == 200
    response_data = response.json()
    assert "message" in response_data
    assert "task_id" in response_data
    assert response_data["message"].startswith("Model training task has been triggered")
