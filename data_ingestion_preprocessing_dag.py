from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

# Define the DAG
dag = DAG(
    'data_ingestion_preprocessing',
    default_args=default_args,
    description='A simple data ingestion and preprocessing DAG',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
)

# Define the task functions
def ingest_data(**kwargs):
    df = pd.read_csv('/path/to/your/dataset.csv')
    df.to_csv('/path/to/ingested_data.csv', index=False)

def preprocess_data(**kwargs):
    df = pd.read_csv('/path/to/ingested_data.csv')
    
    # Example preprocessing steps
    X = df.drop(columns=['Cycle Time'])
    y = df['Cycle Time']
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the processed data
    processed_df = pd.DataFrame(X_scaled, columns=X.columns)
    processed_df['Cycle Time'] = y.values
    processed_df.to_csv('/path/to/processed_data.csv', index=False)
    
    # Save the scaler
    joblib.dump(scaler, '/path/to/scaler.joblib')

# Create the tasks
ingest_task = PythonOperator(
    task_id='ingest_data',
    provide_context=True,
    python_callable=ingest_data,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    provide_context=True,
    python_callable=preprocess_data,
    dag=dag,
)

# Set the task dependencies
ingest_task >> preprocess_task
