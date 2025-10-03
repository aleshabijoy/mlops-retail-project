# test_main.py

from fastapi.testclient import TestClient
from main import app # Import the 'app' object from your main.py file

# Create a client to interact with your app
client = TestClient(app)

def test_read_root():
    """Tests the root endpoint '/'."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Retail Analytics API. Go to /docs for endpoints."}

def test_get_rfm_segmentation():
    """Tests the '/rfm-segmentation' endpoint."""
    response = client.get("/rfm-segmentation")
    assert response.status_code == 200
    # Check that the response is a dictionary and is not empty
    assert isinstance(response.json(), dict)
    assert len(response.json()) > 0

def test_trigger_training_endpoint():
    """Tests the '/trigger-training' endpoint."""
    response = client.post("/trigger-training")
    assert response.status_code == 200
    response_data = response.json()
    # Check for the correct keys in the response
    assert "message" in response_data
    assert "task_id" in response_data # This is the correct key to check

    assert response_data["message"] == "Model training task has been triggered."
