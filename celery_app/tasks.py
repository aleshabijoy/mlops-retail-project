from celery import Celery
import sys
import os

# This allows the script to find and import from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis_logic import load_and_process_data, train_spending_model

# Define the Celery app, using Redis as the message broker
app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

@app.task
def trigger_training_task():
    """
    Celery task to load data and execute the real model training function.
    """
    print("Celery task started: Loading data...")
    df = load_and_process_data()
    print("Data loaded. Starting model training...")
    result = train_spending_model(df)
    return result
