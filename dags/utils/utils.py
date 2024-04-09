import yaml
from botocore.exceptions import NoCredentialsError
import boto3

def read_yaml(path_to_yaml:str)-> dict:
       '''
        this function is used to read yaml files and it takes a string as an input
        and returns a disctionary as an output 
       '''

       with open(path_to_yaml) as yaml_file:
              content = yaml.safe_load(yaml_file)

       return content 


def download_csv_from_s3(s3_client, bucket_name, key, local_filename):
    """
    Download a CSV file from S3.

    :param s3_client: The S3 client.
    :param bucket_name: The name of the S3 bucket.
    :param key: The key of the object in S3.
    :param local_filename: The local filename to save the CSV file.
    """
    try:
        response = s3_client.download_file(bucket_name, key, local_filename)
    except NoCredentialsError as e:
        print(f"Error downloading CSV file from S3: {str(e)}")
        raise