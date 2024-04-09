from utils.utils import read_yaml
from utils.utils import  download_csv_from_s3
import boto3
import json
import os
import shutil
import pandas as pd 



def import_data(**kwargs):
    try:

        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, 'config.yaml')

        
        config = read_yaml(config_file)
        local_folder = os.path.join(config['validation']['destination_before_validation']['mainfolder'],
                                    config['validation']['destination_before_validation']['subfolder'])
        aws_access_key_id = config['validation']['source']['Access_key']
        aws_secret_access_key = config['validation']['source']['Secret_Access_Key']
        bucket_name = config['validation']['source']['bucketname']

        
        s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

       
        response = s3_client.list_objects_v2(Bucket=bucket_name)

        
        os.makedirs(local_folder, exist_ok=True)

       
        for obj in response.get('Contents', []):
            key = obj['Key']
            if key.endswith('.csv'):  # Check if the object is a CSV file
                local_filename = os.path.join(local_folder, f"{key.replace('/', '_')}")

                # Download the CSV file from S3
                download_csv_from_s3(s3_client, bucket_name, key, local_filename)

                print(f"Downloaded and saved: {key} as {local_filename}")

    
        return local_folder

    except Exception as e:
        print(f"Error saving CSV files from S3: {str(e)}")
        raise

def load_schema(**kwargs):
    try:
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        schema_file = os.path.join(current_dir, 'schema_training.json')
        
        with open(schema_file, 'r') as file:
            schema = json.load(file)

        
        return schema

    except Exception as e:
        print(f"Error loading schema from {schema_file}: {e}")
        raise

def validate_data(**kwargs):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(current_dir, 'config.yaml')

    
    config = read_yaml(config_file)
    try:
       
        
        local_folder = os.path.join(config['validation']['destination_before_validation']['mainfolder'],
                                       config['validation']['destination_before_validation']['subfolder'])
        
      
        current_dir = os.path.dirname(os.path.abspath(__file__))
        schema_file = os.path.join(current_dir, 'schema_training.json')
        
        with open(schema_file, 'r') as file:
            schema = json.load(file)

        
        files = os.listdir(local_folder)
        csv_files = [f for f in files if f.endswith('.csv')]

        valid_csv_files = []

        for csv_file in csv_files:
            csv_file_path = os.path.join(local_folder, csv_file)
            df = pd.read_csv(csv_file_path, encoding='ISO-8859-1')

            df.columns = map(lambda x: x.strip().lower(), df.columns)

            if len(df.columns) != schema['NumberofColumns']:
                print(f"Error: Number of columns in {csv_file} doesn't match the schema.")
                continue

            for col_name, col_dtype in schema['ColName'].items():
                if col_name not in df.columns:
                    print(f"Error: Column '{col_name}' is missing in {csv_file}.")
                    break

                if df[col_name].dtype.name != col_dtype:
                    print(f"Error: Data type of column '{col_name}' in {csv_file} is incorrect.")
                    break
            else:
                valid_csv_files.append(csv_file)

        validated_folder = os.path.join(config['validation']['destination_after_validation']['mainfolder'],
                                         config['validation']['destination_after_validation']['subfolder'])
        os.makedirs(validated_folder, exist_ok=True)

        for valid_csv_file in valid_csv_files:
            src = os.path.join(local_folder, valid_csv_file)
            dst = os.path.join(validated_folder, valid_csv_file)
            shutil.move(src, dst)
            print(f"Moved {valid_csv_file} to {validated_folder}")

        
        return len(valid_csv_files)

    except Exception as e:
        print(f"Error validating CSV files: {e}")
        raise





if __name__ == "__main__":
     import_data()
     load_schema()
     validate_data()

 
     
   

'''
cd src
python trainingValidation.py

'''
