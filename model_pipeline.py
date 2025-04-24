from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator, ShortCircuitOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta

from model_utils.model_engineering import train_model
from model_utils.model_validation import evaluate_model
# from model_utils.model_deployment import register_model, compare_models
import logging
# ==== Default DAG Config ====
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

# ==== DAG Definition ====
with DAG(
    dag_id="model_pipeline",
    default_args=default_args,
    description="Train, evaluate, and register LSTM model",
    schedule_interval=None,
    start_date=datetime(2025, 4, 1),
    catchup=False,
    tags=["model_pipeline", "distilbert"],
) as dag:

    # 1. Train model
    def _train_model(**kwargs):
        result = train_model(
            model_name="LSTM",
            db_uri="postgresql+psycopg2://huyvu:password@localhost:5432/raw_data",
            query="SELECT * FROM cleaned_df"
        )
        kwargs['ti'].xcom_push(key="run_id", value=result["run_id"])
        kwargs['ti'].xcom_push(key="model_uri", value=result["model_uri"])

    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
        provide_context=True
    )

    # 2. Evaluate model 
    def _evaluate_model(**kwargs):
        model_uri = kwargs['ti'].xcom_pull(task_ids="train_model", key="model_uri")
        run_id = kwargs['ti'].xcom_pull(task_ids="train_model", key="run_id")
        evaluate_model(
            model_path=model_uri,
            run_id = run_id,
            db_uri="postgresql+psycopg2://huyvu:password@localhost:5432/raw_data",
            query="SELECT * FROM test_data"
        )

    evaluate_model_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=_evaluate_model
    )

    # # 3. Compare model performance
    # def _compare_models(**kwargs):
    #     run_id = kwargs['ti'].xcom_pull(key="run_id")
    #     return "register_model" if compare_models(
    #         new_run_id=run_id,
    #         experiment_name="SentimentAnalysis",
    #         metric_key="accuracy"
    #     ) else "skip_register"

    # compare_model_task = BranchPythonOperator(
    #     task_id="compare_model",
    #     python_callable=_compare_models,
    #     provide_context=True
    # )

    # # 4a. Register model if it's better
    # def _register_model(**kwargs):
    #     model_uri = kwargs['ti'].xcom_pull(key="pyfunc_model_uri")
    #     register_model(
    #         model_uri=model_uri,
    #         model_name="distilbert_sentiment",
    #         tags={"version": "auto", "source": "airflow"}
    #     )

    # register_model_task = PythonOperator(
    #     task_id="register_model",
    #     python_callable=_register_model,
    #     provide_context=True
    # )

    # # 4b. Dummy skip task if model not better
    # skip_register_task = PythonOperator(
    #     task_id="skip_register",
    #     python_callable=lambda: print("Skip register - model not better")
    # )

    # ==== Flow ==== #
    train_model_task >> evaluate_model_task 
    # >> compare_model_task
    # compare_model_task >> register_model_task
    # compare_model_task >> skip_register_task
    logging.info('Done.')