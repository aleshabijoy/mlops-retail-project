from fastapi.testclient import TestClient
from main import app 
from unittest.mock import patch

client = TestClient(app)

def test_read_root():
    """Tests the root endpoint '/'."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome!"}

# The @patch decorator replaces the real 'mlflow' object in your 'main.py'
# with a fake "mock" object just for the duration of this test.
@patch('main.mlflow')
def test_trigger_training_endpoint(mock_mlflow):
    """
    Tests the '/trigger-training' endpoint.
    The 'mock_mlflow' argument is the fake MLflow object.
    """
    response = client.post("/trigger-training")
    
    # We are no longer using Celery in this simplified main.py,
    # so the API will run the training synchronously.
    # The test should now check for the final response.
    assert response.status_code == 200
    response_data = response.json()
    
    # Check that the response contains the keys from a completed training run
    assert "message" in response_data
    assert "run_id" in response_data
    assert "mse" in response_data
    assert response_data["message"] == "Model training complete."

    # You can also assert that the mock mlflow functions were called
    mock_mlflow.set_experiment.assert_called_once_with("Retail Spending Predictor")
    mock_mlflow.log_metric.assert_called_once()
