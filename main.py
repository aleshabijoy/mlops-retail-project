from fastapi import FastAPI
import sys
import os

# This allows the script to find and import from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis_logic import load_and_process_data, calculate_rfm
from celery_app.tasks import trigger_training_task

app = FastAPI(title="Retail Analytics & MLOps API")

# Load data once on startup for quick API responses
data = load_and_process_data()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Retail Analytics API. Go to /docs for endpoints."}

@app.get("/rfm-segmentation")
def get_rfm_segmentation():
    """Returns the RFM segmentation for all customers."""
    rfm_data = calculate_rfm(data)
    return rfm_data.to_dict(orient='index')

@app.post("/trigger-training")
def trigger_training():
    """
    This endpoint triggers the model training task asynchronously via Celery.
    It returns immediately with a task ID.
    """
    task = trigger_training_task.delay()
    return {"message": "Model training task has been triggered.", "task_id": task.id}