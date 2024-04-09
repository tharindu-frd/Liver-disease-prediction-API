from flask import Flask, request, jsonify
import boto3
import pickle
import numpy as np 
from io import StringIO


app = Flask(__name__)


@app.route('/')
def hello():
    return "Prediction Services "

@app.route('/predict', methods=['POST'])
def predict_from_model():
    
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

    def load_model_from_s3_with_credentials(access_key_id, secret_access_key, bucket_name, key):    
        s3 = boto3.client('s3', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
        response = s3.get_object(Bucket=bucket_name, Key=key)
        model_bytes = response['Body'].read()
        model = pickle.loads(model_bytes)
        return model



    config = read_yaml('config.yaml')
    access_key_id = config['modelStore']['Access_key']
    secret_access_key = config['modelStore']['Secret_Access_Key']
    bucket_name = config['modelStore']['bucketname']
    model_key = 'lowest_accuracy_model.pkl'

    
    model = load_model_from_s3_with_credentials(access_key_id, secret_access_key, bucket_name, model_key)

    
    data = request.json
    gender_mapping = {'Female': 0, 'Male': 1}
    data['Gender of the patient'] = gender_mapping.get(data.get('Gender of the patient', '').capitalize(), -1)
    
   
    features = [
        data['age of the patient'],
        data['Gender of the patient'],
        data['total bilirubin'],
        data['direct bilirubin'],
        data['alkphos alkaline phosphotase'],
        data['sgpt alamine aminotransferase'],
        data['sgot aspartate aminotransferase'],
        data['total protiens'],
        data['alb albumin'],
        data['a/g ratio albumin and globulin ratio']
    ]
    features_array = np.array(features).reshape(1, -1)  # Reshape to a single-row array
    

   
    prediction = model.predict(features_array)

    
   
    csv_data = ','.join(map(str, features)) + f',{prediction[0]}\n'
    csv_buffer = StringIO()
    csv_buffer.write(csv_data)
    csv_buffer.seek(0)
    s3 = boto3.client('s3', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
    s3.put_object(Bucket=bucket_name, Key='production.csv', Body=csv_buffer.getvalue())
   

    prediction = model.predict(features_array)
    print(prediction)
    return jsonify({'prediction': prediction.tolist()})




'''
curl -X POST \
  http://localhost:5050/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "age of the patient": 65,
    "gender of the patient": "Female",
    "total bilirubin": 0.7,
    "direct bilirubin": 0.1,
    "alkphos alkaline phosphotase": 187.0,
    "sgpt alamine aminotransferase": 16.0,
    "sgot aspartate aminotransferase": 18.0,
    "total protiens": 6.8,
    "alb albumin": 3.3,
    "a/g ratio albumin and globulin ratio": 0.9
  }'


'''
