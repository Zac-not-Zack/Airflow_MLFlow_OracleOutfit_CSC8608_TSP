from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime
import os

print(os.getcwd())
default_args = {'start_date': datetime(2025, 4, 11)}

with DAG('pipeline_ml', default_args=default_args, schedule=None) as dag:
    ingestion = BashOperator(task_id='ingestion', bash_command='python /Users/zac/mini-projet/src/data_ingestion.py')
    preprocessing = BashOperator(task_id='preprocessing', bash_command='python /Users/zac/mini-projet/src/preprocessing.py')
    training_knn = BashOperator(task_id='training_knn', bash_command='python /Users/zac/mini-projet/src/train_knn.py')
    validation = BashOperator(task_id='validation', bash_command='python /Users/zac/mini-projet/src/validation.py')

    ingestion >> preprocessing >> training_knn >> validation