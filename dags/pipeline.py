from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import csv
import requests
import json
from trainingValidation import import_data , load_schema , validate_data
from featureEngineeringAndCreatingFeatureStore import preprocessing,creating_feature_store
from modelTraining import  model_training ,load_data_from_s3
from modelEvaluation import find_model_with_lowest_accuracy
# from evidently import monitoring 




default_args = {
    "owner": "airflow",
    "email_on_failure": False,   ### receive an email when a dag is fail
    "email_on_retry": False,    ### receive an email at each time dag retry a dag 
    "email": "admin@localhost.com",
    "retries": 1,
    "retry_delay": timedelta(minutes=5)   
}

#### Lets create our dag object . The first input is the dag id  ("forex_data_pipeline") .This is the unique id that we use to idenify our dag 
with DAG("liver_disease_pipeline_2", start_date=datetime(2024, 4 ,7), 
    schedule_interval="@daily", default_args=default_args, catchup=False) as dag:

       import_data_task = PythonOperator(
       task_id='import_training_data_from_s3',
       python_callable=import_data,
       provide_context=True,
       )

       load_schema_task = PythonOperator(
       task_id='load_schema',
       python_callable=load_schema,
       provide_context=True,
       )


       
       validate_data_task = PythonOperator(
       task_id='validate_data',
       python_callable=validate_data,
       provide_context=True,
       
       )



       preprocessing_task = PythonOperator(
       task_id='preprocessing',
       python_callable=preprocessing,
       provide_context=True,
      
       )

       creating_feature_store_task = PythonOperator(
       task_id='creating_feature_store',
       python_callable=creating_feature_store,
       provide_context=True,
      
       )




###################################################################################################################   Model Training  #########################################################

      

       download_and_load_data_task = PythonOperator(
       task_id='download_and_load_data',
       python_callable=load_data_from_s3,  
       provide_context=True,
       
       )


       model_training_task = PythonOperator(
       task_id='model_training',
       python_callable=model_training,
       provide_context=True,
       
       )



################################################################################################
#####################    Model Evaluation  ####################################################


       model_evaluation = PythonOperator(
              task_id='model_evaluator',
              python_callable=find_model_with_lowest_accuracy,
              provide_context = True,
       )


'''
       model_monitoring = PythonOperator(
              task_id='model_monitoring',
              python_callable=monitoring,
              provide_context = True,
       )

'''



import_data_task >> load_schema_task >> validate_data_task >> preprocessing_task >> creating_feature_store_task  >> download_and_load_data_task >>  model_training_task >> model_evaluation  # >> model_monitoring

