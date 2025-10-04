from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 9, 29),
    'retries': 1,
}

with DAG(
    dag_id='daily_model_retraining_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False
) as dag:
    trigger_retraining_task = BashOperator(
        task_id='trigger_fastapi_retraining_endpoint',
        # This command sends a POST request to your FastAPI server.
        # Use host.docker.internal if running Airflow in Docker and the API on your host machine.
        # If both are running locally, you can use localhost.
        bash_command='curl -X POST http://localhost:8000/trigger-training'
    )