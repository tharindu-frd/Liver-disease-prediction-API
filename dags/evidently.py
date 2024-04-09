
import os
import pandas as pd 
from datetime import datetime, timedelta
from sklearn import datasets, ensemble
from airflow import DAG
from airflow.operators.python_operator import PythonOperator 
from evidently.test_suite import TestSuite 
from evidently.test_preset import DataDriftTestPreset, NoTargetPerformanceTestPreset, BinaryClassificationTestPreset
from evidently import ColumnMapping, Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, ClassificationPreset
import boto3
import pickle 
import numpy as np
from utils.utils import read_yaml


def monitoring(**context):

    def load_model_from_s3_with_credentials(access_key_id, secret_access_key, bucket_name, key):    
        s3 = boto3.client('s3', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
        response = s3.get_object(Bucket=bucket_name, Key=key)
        model_bytes = response['Body'].read()
        model = pickle.loads(model_bytes)
        return model

   
    config = read_yaml('config.yaml')
    access_key = config['modelStore']['Access_key']
    secret_access_key = config['modelStore']['Secret_Access_Key']
   
    
    bucket_name_1 = 'liverfeaturestore'
    file_key_1 = 'feature_store/X_train.csv'
    s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_access_key)
    obj = s3.get_object(Bucket=bucket_name_1, Key=file_key_1)
    reference_data = pd.read_csv(obj['Body']).loc[:500]

   
    bucket_name_2 = 'livermonitoring'
    file_key_2 = 'production.csv'
    obj = s3.get_object(Bucket=bucket_name_2, Key=file_key_2)
    production_data = pd.read_csv(obj['Body']).loc[:500]

    
    bucket_name = 'liverdatamodelserving'
    model_key = 'lowest_accuracy_model.pkl'
    model = load_model_from_s3_with_credentials(access_key, secret_access_key, bucket_name, model_key)

    prediction_curr = model.predict(production_data)
    prediction_ref = model.predict(reference_data)

   
    prediction_curr_df = pd.DataFrame(prediction_curr, columns=['prediction_curr'])
    prediction_ref_df = pd.DataFrame(prediction_ref, columns=['prediction_ref'])
    prediction_curr_df = pd.DataFrame({'prediction': prediction_curr})
    prediction_ref_df = pd.DataFrame({'prediction': prediction_ref})
    
    
   

    prediction_curr_df['prediction'] = prediction_curr_df['prediction'].replace({1: 0, 2: 1})
    prediction_ref_df['prediction'] = prediction_ref_df['prediction'].replace({1: 0, 2: 1})
    
    
    result = 'result'
    prediction = 'prediction'
    column_mapping = ColumnMapping()
    column_mapping.target = 'result'
    column_mapping.prediction = 'prediction'
  

    print(prediction_ref_df )
   
    model_performance_suite = TestSuite(tests=[BinaryClassificationTestPreset()])
    model_performance_suite.run(reference_data=prediction_ref_df, current_data=prediction_curr_df)
    model_performance_suite.show(mode='inline')

     







