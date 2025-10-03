from fastapi.testclient import TestClient
from main import app # This will now import correctly

client = TestClient(app)

def test_read_root():
    """Tests the root endpoint '/'."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome!"}

def test_trigger_training_endpoint():
    """Tests the '/trigger-training' endpoint."""
    response = client.post("/trigger-training")
    assert response.status_code == 200
    response_data = response.json()
    assert "message" in response_data
    assert "task_id" in response_data
    assert response_data["message"] == "Model training task has been triggered."
