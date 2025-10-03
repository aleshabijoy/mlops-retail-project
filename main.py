import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI
import uvicorn
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
from celery import Celery

# --- Celery Configuration ---
# This defines the Celery app, using Redis as the message broker.
celery_app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Retail Analytics & MLOps API",
    description="An API to serve customer segmentation and sales insights."
)

# --- Helper Functions (Your Core Logic) ---

def load_and_process_data(filepath="customer_shopping_data.csv"):
    df = pd.read_csv(filepath)
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], format='%d/%m/%Y')
    df['total_price'] = df['quantity'] * df['price']
    return df

def train_spending_model(df):
    mlflow.set_experiment("Retail Spending Predictor")
    with mlflow.start_run():
        model_data = df[['age', 'total_price']].dropna()
        X = model_data[['age']]
        y = model_data['total_price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression().fit(X_train, y_train)
        mse = mean_squared_error(y_test, model.predict(X_test))
        
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "spending-predictor-model")
        
        return {"message": "Model training complete.", "run_id": mlflow.active_run().info.run_id, "mse": mse}

# --- Celery Task Definition ---
@celery_app.task
def trigger_training_task():
    """Celery task to load data and trigger the model training function."""
    print("Celery task started: Loading data...")
    df = load_and_process_data()
    print("Data loaded. Starting model training...")
    result = train_spending_model(df)
    return result

# --- Load data on startup ---
data = load_and_process_data()

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome!"}

@app.post("/trigger-training")
def trigger_training():
    """This endpoint triggers the model training task asynchronously via Celery."""
    task = trigger_training_task.delay()
    return {"message": "Model training task has been triggered.", "task_id": task.id}

# --- Main Execution Block (for local running) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
